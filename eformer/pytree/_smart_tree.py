"""
This module provides utilities for creating JAX PyTree-compatible dataclasses
with enhanced serialization capabilities.

It includes a custom `dataclass` decorator that automatically registers
dataclasses with JAX's tree utility and provides serialization/deserialization
helpers. It also defines a base class `xTree` for convenience.
"""

# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF tp.Any KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
import threading
import types
import typing as tp
from collections.abc import Callable

from contextlib2 import contextmanager
import jax
from typing_extensions import dataclass_transform

_T = tp.TypeVar("_T")

_PRIMITIVE_TYPES = (
	str,
	bytes,
	types.FunctionType,
	types.MethodType,
	type,
	tp.Callable,
)

_STATE_DICT_REGISTRY: dict[tp.Any, tp.Any] = {}


class _NamedTuple:
	"""Fake type marker for namedtuple for registry."""


def _is_namedtuple(x: tp.Any) -> bool:
	"""
	Duck typing test for namedtuple factory-generated objects.

	Args:
		x: The object to test.

	Returns:
		True if the object appears to be a namedtuple, False otherwise.
	"""
	return isinstance(x, tuple) and hasattr(x, "_fields")


class _ErrorContext(threading.local):
	"""
	Thread-local context for tracking the path during deserialization for error messages.
	"""

	def __init__(self):
		"""Initializes the error context with an empty path."""
		self.path = []


def register_serialization_state(
	ty: tp.Any,
	ty_to_state_dict: Callable[[tp.Any], dict[str, tp.Any]],
	ty_from_state_dict: Callable[[tp.Any, dict[str, tp.Any]], tp.Any],
	override: bool = False,
):
	"""
	Registers serialization and deserialization functions for a given type.

	Args:
		ty: The type to register handlers for.
		ty_to_state_dict: A callable that converts an instance of `ty` to a state dictionary.
		ty_from_state_dict: A callable that updates an instance of `ty` from a state dictionary.
		override: If True, overrides an existing registration for the type.
				  If False and a registration exists, raises a ValueError.

	Raises:
		ValueError: If a handler for the type is already registered and `override` is False.
	"""
	if ty in _STATE_DICT_REGISTRY and not override:
		raise ValueError(
			f'a serialization handler for "{ty.__name__}" is already registered'
		)
	_STATE_DICT_REGISTRY[ty] = (ty_to_state_dict, ty_from_state_dict)


_error_context = _ErrorContext()


@contextmanager
def _record_path(name: str):
	"""
	Context manager to record the current path component during deserialization.

	Args:
		name: The name of the current path component (e.g., a field name).

	Yields:
		None. The context manager pushes the name onto the path on entry
		and pops it on exit.
	"""
	try:
		_error_context.path.append(name)
		yield
	finally:
		_error_context.path.pop()


def xfrom_state_dict(target: _T, state: dict[str, tp.Any], name: str = ".") -> _T:
	"""
	Recursively deserializes the state dictionary into the target object.

	Uses the registered `from_state_dict` function for the target's type if available,
	otherwise returns the state directly.

	Args:
		target: The object to deserialize into.
		state: The state dictionary.
		name: The name of the current object in the parent structure (used for error reporting).

	Returns:
		The deserialized object.
	"""
	if _is_namedtuple(target):
		ty = _NamedTuple
	else:
		ty = type(target)
	if ty not in _STATE_DICT_REGISTRY:
		return state
	ty_from_state_dict = _STATE_DICT_REGISTRY[ty][1]
	with _record_path(name):
		return ty_from_state_dict(target, state)


def xto_state_dict(target: tp.Any) -> dict[str, tp.Any]:
	"""
	Recursively converts the target object into a state dictionary.

	Uses the registered `to_state_dict` function for the target's type if available,
	otherwise returns the target directly.

	Args:
		target: The object to serialize.

	Returns:
		A dictionary representing the state of the target object, or the target itself
		if no serialization handler is registered.
	"""
	if _is_namedtuple(target):
		ty = _NamedTuple
	else:
		ty = type(target)
	if ty not in _STATE_DICT_REGISTRY:
		return target

	ty_to_state_dict = _STATE_DICT_REGISTRY[ty][0]
	state_dict = ty_to_state_dict(target)
	if isinstance(state_dict, dict):
		for key in state_dict.keys():
			assert isinstance(key, str), "A state dict must only have string keys."
	return state_dict


def _is_pytree_node_annotation(annotation: tp.Any) -> bool:
	"""
	Determines whether a type annotation should be treated as a JAX PyTree node.

	Primitive types and simple containers of primitives are considered leaves.
	More complex types, custom classes, and containers of non-primitives are
	considered nodes.

	Args:
		annotation: The type annotation to check.

	Returns:
		True if the annotation indicates a PyTree node, False if it indicates a leaf.
	"""
	origin = tp.get_origin(annotation)
	args = tp.get_args(annotation)

	if annotation in _PRIMITIVE_TYPES:
		return False

	if origin is tp.Union:
		return any(_is_pytree_node_annotation(arg) for arg in args if arg is not type(None))

	if origin in (list, tuple, set, frozenset):
		return not all(arg in _PRIMITIVE_TYPES for arg in args)

	return True


def field(*, pytree_node: bool | None = None, metadata: dict | None = None, **kwargs):
	"""
	Define a dataclass field and optionally mark it explicitly as a PyTree node.

	This function is a wrapper around `dataclasses.field` that adds a `pytree_node`
	option to the metadata.

	Args:
		pytree_node: Explicitly mark the field as a PyTree node (True) or leaf (False).
					 If None, the type annotation will be used to infer behavior.
		metadata: A dictionary of metadata for the field. The `pytree_node` key
				  will be added or updated in this dictionary.
		**kwargs: Additional keyword arguments passed to `dataclasses.field`.

	Returns:
		A `dataclasses.Field` object.
	"""
	md = dict(metadata or {})
	if pytree_node is not None:
		md["pytree_node"] = pytree_node
	return dataclasses.field(metadata=md, **kwargs)


@dataclass_transform(field_specifiers=(field,))
@tp.overload
def dataclass(clz: _T, **kwargs) -> _T:
	"""
	Overload for `dataclass` when used as a decorator with a class argument.
	"""
	...


@dataclass_transform(field_specifiers=(field,))
@tp.overload
def dataclass(**kwargs) -> Callable[[_T], _T]:
	"""
	Overload for `dataclass` when used as a decorator factory with keyword arguments.
	"""
	...


@dataclass_transform(field_specifiers=(field,))
def dataclass(clz: _T | None = None, **kwargs) -> _T | Callable[[_T], _T]:
	"""
	A decorator that enhances standard dataclasses to be JAX PyTree compatible
	and adds serialization/deserialization capabilities.

	It automatically registers the dataclass with `jax.tree_util` and defines
	`to_state_dict` and `from_state_dict` methods based on the field types
	and explicit `pytree_node` markings.

	Args:
		clz: The class to decorate.
		**kwargs: Additional keyword arguments passed to `dataclasses.dataclass`.
				  Defaults to `frozen=True`.

	Returns:
		The decorated class.
	"""
	if clz is None:
		return functools.partial(dataclass, **kwargs)
	if getattr(clz, "_eformer_dataclass", False):
		return clz

	kwargs.setdefault("frozen", True)
	data_clz = dataclasses.dataclass(**kwargs)(clz)
	data_fields: list[str] = []
	meta_fields: list[str] = []

	annotations = getattr(data_clz, "__annotations__", {})
	for field_info in dataclasses.fields(data_clz):
		if "pytree_node" in field_info.metadata:
			is_node = field_info.metadata["pytree_node"]
		else:
			ann = annotations.get(field_info.name, tp.Any)
			is_node = _is_pytree_node_annotation(ann)
		(data_fields if is_node else meta_fields).append(field_info.name)

	def replace(self, **updates):
		"""
		Returns a new instance of the dataclass with specified fields updated.

		Args:
			**updates: Keyword arguments where keys are field names and values
					   are the new values for those fields.

		Returns:
			A new instance of the dataclass with the updated fields.
		"""
		return dataclasses.replace(self, **updates)

	data_clz.replace = replace
	jax.tree_util.register_dataclass(data_clz, data_fields, meta_fields)

	def to_state_dict(x):
		"""
		Converts the dataclass instance to a state dictionary.

		Args:
			x: The dataclass instance.

		Returns:
			A dictionary containing the state of the dataclass instance.
		"""
		return {name: xto_state_dict(getattr(x, name)) for name in data_fields}

	def from_state_dict(x, state):
		"""
		Updates the dataclass instance from a state dictionary.

		Args:
			x: The dataclass instance to update.
			state: The state dictionary.

		Returns:
			A new dataclass instance with the state loaded from the dictionary.

		Raises:
			ValueError: If a required field is missing in the state dictionary
						or if unknown fields are present.
		"""
		state = state.copy()
		updates = {}
		for name in data_fields:
			if name not in state:
				raise ValueError(f"Missing field {name} in state dict for {clz.__name__}")
			value = getattr(x, name)
			updates[name] = xfrom_state_dict(value, state.pop(name), name=name)
		if state:
			raise ValueError(
				f"Unknown field(s) {list(state.keys())} in state dict for {clz.__name__}"
			)
		return x.replace(**updates)

	register_serialization_state(data_clz, to_state_dict, from_state_dict)

	setattr(data_clz, "_eformer_dataclass", True)  # noqa

	return data_clz


STree = tp.TypeVar("STree", bound="xTree")


@dataclass_transform(field_specifiers=(field,))
class xTree:
	"""
	Base class for dataclasses acting as JAX PyTree nodes with built-in
	serialization support.

	Classes inheriting from `xTree` are automatically processed by the
	`dataclass` decorator upon definition, making them JAX PyTree compatible
	and adding `to_state_dict` and `from_state_dict` methods.
	"""

	def __init_subclass__(cls, **kwargs):
		"""
		Automatically applies the `dataclass` decorator to subclasses.
		"""
		dataclass(cls, **kwargs)

	def __init__(self, *args, **kwargs):
		raise NotImplementedError

	def replace(self: STree, **overrides) -> STree:
		"""
		Returns a new instance of the xTree subclass with specified fields updated.

		This method is added dynamically by the `dataclass` decorator.

		Args:
			**overrides: Keyword arguments where keys are field names and values
						 are the new values for those fields.

		Returns:
			A new instance of the xTree subclass with the updated fields.
		"""
		raise NotImplementedError
