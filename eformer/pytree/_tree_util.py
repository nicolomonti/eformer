# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2023 The Equinox Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Portions of this code are derived from the Equinox library
# (https://github.com/patrick-kidger/equinox)

import dataclasses
import json
import types
import typing as tp
from functools import wraps

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import tree_util as tu

from ._pytree import PyTree

T = tp.TypeVar("T")
_CLS = tp.TypeVar("_CLS")


FnDict = tp.Dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = tp.Dict[tp.Any, tp.Any]
Path = tp.Tuple[tp.Any, ...]
FilterSpec = tp.Union[bool, tp.Callable[[tp.Any], bool]]
IsLeafFn = tp.Callable[[tp.Any], bool]


def _array_equal(x, y, npi, rtol, atol):
	assert x.dtype == y.dtype
	if (
		isinstance(rtol, (int, float))
		and isinstance(atol, (int, float))
		and rtol == 0
		and atol == 0
	) or not npi.issubdtype(x.dtype, npi.inexact):
		return npi.all(x == y)
	else:
		return npi.allclose(x, y, rtol=rtol, atol=atol)


def is_array(element: tp.Any) -> bool:
	"""Returns `True` if `element` is a JAX array or NumPy array."""
	return isinstance(element, (np.ndarray, np.generic, Array))


def is_array_like(element: tp.Any) -> bool:
	return isinstance(
		element,
		(
			Array,
			np.ndarray,
			np.generic,
			float,
			complex,
			bool,
			int,
		),
	) or hasattr(element, "__jax_array__")


class TreeFilter(tp.Protocol):
	"""tp.Protocol for tree filter functions."""

	def __call__(self, mask: tp.Any, arg: tp.Any) -> TreeDict: ...


def split(
	pytree: PyTree,
	filter_spec: FilterSpec,
	replace: tp.Any = None,
	is_leaf: tp.Optional[IsLeafFn] = None,
) -> tp.Tuple[PyTree, PyTree]:
	def _make_filter_tree(il):
		def _filter_tree(mask: FilterSpec, arg: tp.Any) -> TreeDict:
			if isinstance(mask, bool):
				return tu.tree_map(lambda _: mask, arg, is_leaf=il)
			elif callable(mask):
				return tu.tree_map(mask, arg, is_leaf=il)
			else:
				raise ValueError(f"filter_spec must be bool or callable, got {type(mask)}")

		return _filter_tree

	filter_tree = tu.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
	return (
		tu.tree_map(lambda mask, x: x if mask else replace, filter_tree, pytree),
		tu.tree_map(lambda mask, x: replace if mask else x, filter_tree, pytree),
	)


def merge(*pytrees: PyTree, is_leaf: tp.Optional[IsLeafFn] = None) -> PyTree:
	"""
	Combines multiple PyTrees into a single PyTree.

	Args:
	    *pytrees: PyTrees to merge
	    is_leaf: tp.Optional function to determine if a node is a leaf

	Returns:
	    Combined PyTree
	"""

	def _combine(*args: tp.Any) -> tp.Any:
		"""Returns first non-None value from args."""
		return next((arg for arg in args if arg is not None), None)

	def _is_none(x: tp.Any) -> bool:
		"""Checks if value is None."""
		return x is None

	if is_leaf is None:
		_is_leaf = _is_none
	else:

		def _is_leaf(x: tp.Any) -> bool:
			return _is_none(x) or is_leaf(x)

	return tu.tree_map(_combine, *pytrees, is_leaf=_is_leaf)


def tree_equal(
	*pytrees: PyTree,
	typematch: bool = False,
	rtol=0.0,
	atol=0.0,
) -> bool:
	flat, treedef = tu.tree_flatten(pytrees[0])
	traced_out = True
	for pytree in pytrees[1:]:
		flat_, treedef_ = tu.tree_flatten(pytree)
		if treedef_ != treedef:
			return False
		assert len(flat) == len(flat_)
		for elem, elem_ in zip(flat, flat_):  # noqa
			if typematch:
				if type(elem) != type(elem_):  # noqa
					return False
			if isinstance(elem, (np.ndarray, np.generic)) and isinstance(
				elem_, (np.ndarray, np.generic)
			):
				if (
					(elem.shape != elem_.shape)
					or (elem.dtype != elem_.dtype)
					or not _array_equal(elem, elem_, np, rtol, atol)
				):
					return False
			elif is_array(elem):
				if is_array(elem_):
					if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype):
						return False
					traced_out = traced_out & _array_equal(elem, elem_, jax.numpy, rtol, atol)
				else:
					return False
			else:
				if is_array(elem_):
					return False
				else:
					if elem != elem_:
						return False
	return traced_out


def tree_map_with_path(
	f: tp.Callable,
	tree: PyTree,
	is_leaf: tp.Optional[IsLeafFn] = None,
) -> PyTree:
	"""Maps a function over a pytree while providing the path to each leaf.

	Args:
	    f: Function that takes (path, leaf_value) as arguments
	    tree: Input pytree
	    is_leaf: Optional function to determine if a node is a leaf

	Returns:
	    PyTree with mapped values
	"""

	def _walk(path: tp.Tuple[str, ...], x):
		if is_leaf is not None and is_leaf(x):
			return f(path, x)
		elif isinstance(x, (list, tuple)):
			return type(x)([_walk(path + (str(i),), v) for i, v in enumerate(x)])
		elif isinstance(x, dict):
			return {k: _walk(path + (str(k),), v) for k, v in x.items()}
		else:
			return f(path, x)

	return _walk((), tree)


def tree_flatten_with_paths(
	tree: PyTree,
	is_leaf: tp.Optional[IsLeafFn] = None,
) -> tp.Tuple[tp.List[tp.Tuple[tp.Tuple, tp.Any]], tu.PyTreeDef]:
	"""Flattens a pytree while keeping track of paths to leaves.

	Args:
	    tree: Input pytree
	    is_leaf: Optional function to determine if a node is a leaf

	Returns:
	    Tuple of (list of (path, value) pairs, treedef)
	"""
	paths_and_vals = []

	def _record_path(path, x):
		paths_and_vals.append((path, x))
		return x

	tree_map_with_path(_record_path, tree, is_leaf=is_leaf)
	treedef = tu.tree_structure(tree)
	return paths_and_vals, treedef


def tree_leaves_with_paths(
	tree: PyTree, is_leaf: tp.Optional[IsLeafFn] = None
) -> tp.List[tp.Tuple[tp.Tuple, tp.Any]]:
	"""Returns list of (path, leaf_value) pairs in the pytree."""
	paths_and_vals, _ = tree_flatten_with_paths(tree, is_leaf=is_leaf)
	return paths_and_vals


def tree_structure_equal(tree1: PyTree, tree2: PyTree) -> bool:
	"""Returns True if two pytrees have the same structure."""
	try:
		return tu.tree_structure(tree1) == tu.tree_structure(tree2)
	except Exception:
		return False


def tree_filter(tree: PyTree, predicate: tp.Callable[[tp.Any], bool]) -> PyTree:
	"""Filters a pytree keeping only leaves that satisfy the predicate."""
	flat, treedef = tu.tree_flatten(tree)
	filtered = [x for x in flat if predicate(x)]
	return tu.tree_unflatten(treedef, filtered)


def tree_concatenate(trees: tp.List[PyTree], axis: int = 0) -> PyTree:
	"""Concatenates corresponding arrays in a list of pytrees."""
	return tu.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *trees)


def tree_stack(trees: tp.List[PyTree], axis: int = 0) -> PyTree:
	"""Stacks corresponding arrays in a list of pytrees."""
	return tu.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *trees)


def tree_where(condition: PyTree, x: PyTree, y: PyTree) -> PyTree:
	"""Element-wise where operation on pytrees."""
	return tu.tree_map(lambda c, a, b: jnp.where(c, a, b), condition, x, y)


def tree_zeros_like(tree: PyTree) -> PyTree:
	"""Creates a pytree of zeros with the same structure and shapes."""
	return tu.tree_map(lambda x: jnp.zeros_like(x) if is_array_like(x) else x, tree)


def tree_ones_like(tree: PyTree) -> PyTree:
	"""Creates a pytree of ones with the same structure and shapes."""
	return tu.tree_map(lambda x: jnp.ones_like(x) if is_array_like(x) else x, tree)


def is_flatten(tree: dict) -> bool:
	"""Checks if a dictionary represents a flattened tree.

	A flattened tree is a dictionary where the keys are tuples representing
	the path to the leaf nodes. This function checks if any of the keys in the
	input dictionary is a tuple, indicating a flattened tree.

	Args:
	    tree: The dictionary to check.

	Returns:
	    bool: True if the dictionary is a flattened tree, False otherwise.
	"""
	return True in set(isinstance(k, tuple) for k in tree.keys())


def tree_apply(fns: FnDict, tree: TreeDict) -> TreeDict:
	"""
	Apply a dictionary of functions to a corresponding PyTree.

	Args:
		fns: A dictionary where keys match the PyTree structure and values are functions.
		tree: The PyTree to apply functions to.

	Returns:
		A new PyTree with the same structure as `tree`, but with values modified by the functions in `fns`.
	"""
	return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def tree_path_to_string(path: Path, sep: tp.Optional[str] = None) -> str:
	"""
	Convert a JAX tree path to a string representation.

	Args:
		path: The JAX tree path tuple.
		sep: Separator to use when joining path elements.

	Returns:
		The string representation of the path.
	"""
	keys = []
	for key in path:
		if isinstance(key, jax.tree_util.SequenceKey):
			keys.append(str(key.idx))
		elif isinstance(key, jax.tree_util.DictKey):
			keys.append(str(key.key))
		elif isinstance(key, jax.tree_util.GetAttrKey):
			keys.append(str(key.name))
		elif isinstance(key, jax.tree_util.FlattenedIndexKey):
			keys.append(str(key.key))
		else:
			keys.append(str(key))
	if sep is None:
		return tuple(keys)  # Return a tuple of strings if no separator
	return sep.join(keys)


def flatten_tree(
	xs: PyTree,
	is_leaf: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
	sep: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
	"""
	Flatten a JAX tree and convert paths to strings.

	Args:
		xs: The JAX tree to flatten.
		is_leaf: Optional function to determine leaf nodes.
		sep: Separator to use when joining path elements.

	Returns:
		A flattened dictionary with string keys representing the tree paths.
	"""
	flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
	output = {}
	for key, val in flattened:
		output[tree_path_to_string(key, sep=sep)] = val
	return output


def named_tree_map(
	f: tp.Callable[[str, tp.Any, tp.Any], tp.Any],
	tree: PyTree,
	*rest: tp.Any,
	is_leaf: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
	sep: tp.Optional[str] = None,
) -> PyTree:
	"""
	An extended version of `jax.tree_util.tree_map`.

	This function extends `jax.tree_util.tree_map` by providing the path
	(as a string) to the current leaf node as an argument to the mapped function `f`.

	Args:
		f: The function to apply to each leaf node, taking the path and value as input.
		tree: The JAX tree to map over.
		*rest: Additional arguments to be passed to `f`.
		is_leaf: Optional function to determine leaf nodes.
		sep: Separator to use when joining path elements.

	Returns:
		A new tree with the same structure as `tree` but with the values modified by `f`.
	"""
	return jax.tree_util.tree_map_with_path(
		lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
		tree,
		*rest,
		is_leaf=is_leaf,
	)


def auto_pytree(
	cls: _CLS = None,
	meta_fields: tp.Optional[tp.Tuple[str, ...]] = None,
	json_serializable: bool = True,
) -> _CLS:
	"""
	Register a class as a JAX PyTree with automatic field inference and JSON serialization.

	This function wraps jax.tree_util.register_dataclass to automatically infer
	data_fields based on the provided meta_fields. It first converts the class
	to a dataclass if it isn't already one, then determines which fields should
	be treated as data fields (traversed by JAX) and which should be treated as
	metadata fields (not traversed).

	Args:
	    cls: The class to be registered as a PyTree.
	    meta_fields: A tuple of field names to be treated as metadata fields.
	                 These fields will not be traversed by JAX's PyTree functions.
	                 Defaults to None (auto-detection).
	    json_serializable: If True, adds JSON serialization methods to the class.

	Returns:
	    The registered dataclass that can be used with JAX's PyTree operations and JSON serialization.

	Example:
	    # Fully automatic inference
	    >>> @auto_pytree
	    >>> class Vector:
	    >>>     x: float  # Automatically a data field
	    >>>     y: float  # Automatically a data field
	    >>>     name: str  # Automatically a meta field (str is not JAX-compatible)

	    # With explicit meta_fields
	    >>> @auto_pytree(meta_fields=("z",))
	    >>> class Vector3D:
	    >>>     x: float
	    >>>     y: float
	    >>>     z: float  # Explicitly marked as meta field despite being a JAX-compatible type
	"""

	NON_JAX_TYPES = (
		str,
		bytes,
		types.FunctionType,
		types.MethodType,
		type,
		tp.Callable,
	)

	def is_non_jax_type(typ):
		"""Check if a type is not JAX-compatible."""
		if typ is tp.Any:
			return False
		origin = tp.get_origin(typ)
		if origin is tp.Union:
			args = tp.get_args(typ)
			return any(is_non_jax_type(arg) for arg in args)

		for non_jax_type in NON_JAX_TYPES:
			try:
				if issubclass(typ, non_jax_type):
					return True
			except TypeError:
				pass

		return False

	def wrap(cls):
		cls = dataclasses.dataclass(cls)
		fields = [f for f in dataclasses.fields(cls) if f.init]
		all_field_names = tuple(f.name for f in fields)
		final_meta_fields: tp.Set[str] = set(meta_fields or ())

		# Get meta fields from metadata
		metadata_meta_fields = {
			f.name for f in fields if f.metadata and f.metadata.get("pytree_node") is False
		}
		final_meta_fields.update(metadata_meta_fields)

		# Get meta fields from type
		for field in fields:
			if field.name in final_meta_fields:
				continue

			if hasattr(field, "type") and field.type is not None:
				if is_non_jax_type(field.type):
					final_meta_fields.add(field.name)

		data_fields = tuple(f for f in all_field_names if f not in final_meta_fields)

		# Fix the replace method to properly handle the class
		def replace_method(self, **kwargs):
			return dataclasses.replace(self, **kwargs)

		cls.replace = replace_method

		# Improve __repr__ to show which fields are data vs meta
		original_repr = cls.__repr__

		def enhanced_repr(self):
			base_repr = original_repr(self)
			if hasattr(self, "__pytree_meta__"):
				meta_info = (
					f" [data_fields={data_fields}, meta_fields={tuple(final_meta_fields)}]"
				)
				return base_repr[:-1] + meta_info + base_repr[-1:]
			return base_repr

		cls.__repr__ = enhanced_repr
		cls.__pytree_meta__ = {
			"data_fields": data_fields,
			"meta_fields": tuple(final_meta_fields),
		}

		# Add JSON serialization capabilities
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

				# Get type hints for fields to handle conversion
				type_hints = tp.get_type_hints(cls)

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

			# Custom JSON encoder support
			original_default = json.JSONEncoder.default

			@wraps(original_default)
			def json_default(self, obj):
				if hasattr(obj, "to_dict") and callable(obj.to_dict):
					return obj.to_dict()
				return original_default(self, obj)

			# Monkey patch the JSONEncoder if not already patched
			if not hasattr(json.JSONEncoder, "_pytree_patched"):
				json.JSONEncoder.default = json_default
				json.JSONEncoder._pytree_patched = True

		return tu.register_dataclass(
			cls,
			data_fields=data_fields,
			meta_fields=tuple(final_meta_fields),
		)

	# Handle both @auto_pytree and @auto_pytree(meta_fields=(...))
	if cls is None:
		return wrap
	return wrap(cls)
