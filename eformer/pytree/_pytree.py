from __future__ import annotations

import dataclasses
import json
import types
import typing as tp
from functools import wraps, lru_cache

import typing_extensions
from jax import tree_util as tu

T = tp.TypeVar("T")
FnDict = tp.Dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = tp.Dict[tp.Any, tp.Any]
Path = tp.Tuple[tp.Any, ...]
FilterSpec = tp.Union[bool, tp.Callable[[tp.Any], bool]]
IsLeafFn = tp.Callable[[tp.Any], bool]


# Cache for type checking to avoid repeated expensive operations
@lru_cache(maxsize=1024)
def _is_non_jax_type(typ):
	"""Check if a type is not JAX-compatible with caching for performance."""
	# Types that are not compatible with JAX operations
	NON_JAX_TYPES = (
		str,
		bytes,
		types.FunctionType,
		types.MethodType,
		type,
		tp.Callable,
	)

	if typ is tp.Any:
		return False

	origin = tp.get_origin(typ)
	if origin is tp.Union:
		args = tp.get_args(typ)
		return any(_is_non_jax_type(arg) for arg in args)

	for non_jax_type in NON_JAX_TYPES:
		try:
			if issubclass(typ, non_jax_type):
				return True
		except TypeError:
			pass

	return False


def field(pytree_node=True, *, metadata=None, **kwargs):
	"""Define a field with pytree_node metadata."""
	metadata_dict = (metadata or {}).copy()
	metadata_dict["pytree_node"] = pytree_node
	return dataclasses.field(metadata=metadata_dict, **kwargs)


# Pre-compute field information only once per class
class PyTreeClassInfo:
	"""Cache for class metadata to avoid repeated computation."""

	__slots__ = ["data_fields", "meta_fields", "frozen", "type_hints"]

	def __init__(self, data_fields, meta_fields, frozen, type_hints):
		self.data_fields = data_fields
		self.meta_fields = meta_fields
		self.frozen = frozen
		self.type_hints = type_hints


# Global registry to store class information
_CLASS_INFO_REGISTRY = {}


@typing_extensions.dataclass_transform(field_specifiers=(field,))
def auto_pytree(
	cls=None,
	meta_fields: tp.Optional[tp.Tuple[str, ...]] = None,
	json_serializable: bool = True,
	frozen: bool = False,
):
	"""
	Register a class as a JAX PyTree with performance optimizations.
	"""

	def wrap(cls):
		# Create a dataclass with the frozen option if specified
		cls = dataclasses.dataclass(cls, frozen=frozen)

		# Get all fields that are included in initialization
		fields = [f for f in dataclasses.fields(cls) if f.init]
		all_field_names = tuple(f.name for f in fields)

		# Start with explicitly provided meta fields
		final_meta_fields: tp.Set[str] = set(meta_fields or ())

		# First pass: check explicit field metadata (highest priority)
		for field_obj in fields:
			field_metadata = field_obj.metadata
			if field_metadata and "pytree_node" in field_metadata:
				# If explicitly marked, respect that marking
				if field_metadata["pytree_node"] is False:
					final_meta_fields.add(field_obj.name)
				elif (
					field_metadata["pytree_node"] is True and field_obj.name in final_meta_fields
				):
					final_meta_fields.remove(field_obj.name)

		# Get type hints once to avoid repeated lookups
		type_hints = tp.get_type_hints(cls)

		# Second pass: auto-detect non-JAX types, but don't override explicit settings
		for field_obj in fields:
			if field_obj.name in final_meta_fields:
				continue

			# Skip fields with explicit pytree_node=True marking
			if field_obj.metadata and field_obj.metadata.get("pytree_node") is True:
				continue

			field_type = type_hints.get(field_obj.name)
			if field_type is not None and _is_non_jax_type(field_type):
				final_meta_fields.add(field_obj.name)

		# All fields not marked as meta are data fields
		data_fields = tuple(f for f in all_field_names if f not in final_meta_fields)
		meta_fields_tuple = tuple(final_meta_fields)

		# Optimized replace method using __new__ directly for better performance
		def fast_replace(self, **kwargs):
			if not kwargs:  # No changes, return self
				return self

			# Create new instance directly without re-validation
			if frozen:
				# For frozen dataclasses, we need to use the underlying __new__ pattern
				# to avoid validation errors when setting fields
				new_values = {**{f: getattr(self, f) for f in all_field_names}, **kwargs}
				new_instance = cls.__new__(cls)
				object.__setattr__(new_instance, "__dict__", new_values)
				return new_instance
			else:
				# For non-frozen, dataclasses.replace is fine
				return dataclasses.replace(self, **kwargs)

		cls.replace = fast_replace

		# Enhanced string representation with length limiting
		def enhanced_repr(self):
			cls_name = self.__class__.__name__
			items = []

			for k, v in self.__dict__.items():
				if not k.startswith("_"):
					try:
						repr_str = str(v)
						if len(repr_str) > 200:  # Limit long representations
							repr_str = f"{v.__class__.__name__}(...)"
						items.append(f"  {k} : {repr_str}")
					except TypeError:
						pass

			return f"{cls_name}(\n" + "\n".join(items) + "\n)"

		cls.__repr__ = enhanced_repr
		cls.__str__ = enhanced_repr

		# Store class info in registry for faster lookups
		class_info = PyTreeClassInfo(
			data_fields=data_fields,
			meta_fields=meta_fields_tuple,
			frozen=frozen,
			type_hints=type_hints,
		)
		_CLASS_INFO_REGISTRY[cls] = class_info

		# Store pytree metadata for introspection
		cls.__pytree_meta__ = {
			"data_fields": data_fields,
			"meta_fields": meta_fields_tuple,
			"frozen": frozen,
		}

		# Add JSON serialization capabilities if requested
		if json_serializable:

			def to_dict(self):
				"""Convert the instance to a dictionary for JSON serialization."""
				result = {}
				for field in dataclasses.fields(self):
					value = getattr(self, field.name)
					# Skip Ellipsis values
					if value is Ellipsis:
						continue
					# Convert tuples to lists for JSON compatibility
					if isinstance(value, tuple):
						result[field.name] = list(value)
					# Handle None values
					elif value is None:
						result[field.name] = None
					# Try to convert other objects that might have to_dict
					elif hasattr(value, "to_dict") and callable(value.to_dict):
						result[field.name] = value.to_dict()
					else:
						# For primitive types or those without to_dict method
						try:
							# Check if value is JSON serializable
							json.dumps(value)
							result[field.name] = value
						except (TypeError, OverflowError):
							# If not serializable, convert to string representation
							result[field.name] = str(value)
				return result

			cls.to_dict = to_dict

			@classmethod
			def from_dict(cls, data):
				"""Create an instance from a dictionary (deserialization)."""
				# Process the data to convert lists back to tuples where needed
				processed_data = {}

				# Get cached type hints for fields to handle conversion
				class_info = _CLASS_INFO_REGISTRY.get(cls)
				type_hints = class_info.type_hints if class_info else tp.get_type_hints(cls)

				for field in dataclasses.fields(cls):
					field_name = field.name
					if field_name not in data:
						# Skip missing fields
						continue

					value = data[field_name]
					field_type = type_hints.get(field_name)

					# Convert lists back to tuples if the field type is a tuple
					if (
						value is not None
						and isinstance(value, list)
						and field_type is not None
						and tp.get_origin(field_type) is tuple
					):
						processed_data[field_name] = tuple(value)
					else:
						processed_data[field_name] = value

				return cls(**processed_data)

			cls.from_dict = from_dict

			def to_json(self, **kwargs):
				"""Convert the instance to a JSON string."""
				return json.dumps(self.to_dict(), **kwargs)

			cls.to_json = to_json

			@classmethod
			def from_json(cls, json_str):
				"""Create an instance from a JSON string."""
				data = json.loads(json_str)
				return cls.from_dict(data)

			cls.from_json = from_json

			# Custom JSON encoder support (only patch once)
			if not hasattr(json.JSONEncoder, "_pytree_patched"):
				original_default = json.JSONEncoder.default

				@wraps(original_default)
				def json_default(self, obj):
					if hasattr(obj, "to_dict") and callable(obj.to_dict):
						return obj.to_dict()
					return original_default(self, obj)

				json.JSONEncoder.default = json_default
				json.JSONEncoder._pytree_patched = True

		# Register with JAX tree utilities
		return tu.register_dataclass(
			cls,
			data_fields=data_fields,
			meta_fields=meta_fields_tuple,
		)

	# Handle both @auto_pytree and @auto_pytree(meta_fields=(...))
	if cls is None:
		return wrap
	return wrap(cls)


# Base implementation for PyTree - not a dataclass itself
class _PyTreeNodeBase:
	"""Base implementation for PyTree classes."""

	def replace(self, **kwargs):
		"""Return a new instance with the specified fields replaced with new values."""
		return dataclasses.replace(self, **kwargs)


@typing_extensions.dataclass_transform(field_specifiers=(field,))
class PyTree(_PyTreeNodeBase):
	"""Base class for dataclasses that should act like a JAX pytree node."""

	def __init_subclass__(
		cls,
		*,
		frozen: bool = False,
		json_serializable: bool = True,
		meta_fields: tp.Optional[tp.Tuple[str, ...]] = None,
		**kwargs,
	):
		"""Initialize a PyTree subclass."""
		auto_pytree(
			cls,
			meta_fields=meta_fields,
			json_serializable=json_serializable,
			frozen=frozen,
			**kwargs,
		)


@typing_extensions.dataclass_transform(field_specifiers=(field,))
class FrozenPyTree(_PyTreeNodeBase):
	"""Immutable base class for JAX pytree nodes."""

	def __init_subclass__(
		cls,
		*,
		json_serializable: bool = True,
		meta_fields: tp.Optional[tp.Tuple[str, ...]] = None,
		**kwargs,
	):
		"""Initialize a FrozenPyTree subclass."""
		auto_pytree(
			cls,
			meta_fields=meta_fields,
			json_serializable=json_serializable,
			frozen=True,
			**kwargs,
		)
