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
import types
import typing as tp
from collections.abc import Callable

import jax
from typing_extensions import dataclass_transform

from . import serialization

_T = tp.TypeVar("_T")

_PRIMITIVE_TYPES = (
	str,
	bytes,
	types.FunctionType,
	types.MethodType,
	type,
	tp.Callable,
)


def _is_pytree_node_annotation(annotation: tp.Any) -> bool:
	"""
	Determines whether a type annotation should be treated as a JAX PyTree node.
	Primitive types and simple containers of primitives are considered leaves.
	"""
	origin = tp.get_origin(annotation)
	args = tp.get_args(annotation)

	# Direct primitive
	if annotation in _PRIMITIVE_TYPES:
		return False

	# Optional[...] or tp.Union[...] annotations
	if origin is tp.Union:
		# If tp.Any option is a node, treat the union as a node
		return tp.Any(
			_is_pytree_node_annotation(arg) for arg in args if arg is not type(None)
		)

	# Simple homogeneous containers of primitives: List[str], Tuple[int, ...], etc.
	if origin in (list, tuple, set, frozenset):
		return not all(arg in _PRIMITIVE_TYPES for arg in args)

	# Fallback: assume a node (including custom classes, numpy arrays, etc.)
	return True


def field(*, pytree_node: bool | None = None, metadata: dict | None = None, **kwargs):
	"""
	Define a dataclass field and optionally mark it explicitly as a PyTree node.
	If pytree_node is None, the type annotation will be used to infer behavior.
	"""
	md = dict(metadata or {})
	if pytree_node is not None:
		md["pytree_node"] = pytree_node
	return dataclasses.field(metadata=md, **kwargs)


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@tp.overload
def dataclass(clz: _T, **kwargs) -> _T: ...


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@tp.overload
def dataclass(**kwargs) -> Callable[[_T], _T]: ...


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(clz: _T | None = None, **kwargs) -> _T | Callable[[_T], _T]:
	if clz is None:
		return functools.partial(dataclass, **kwargs)
	if getattr(clz, "_eformer_dataclass", False):
		return clz

	# Default to frozen dataclasses
	kwargs.setdefault("frozen", True)
	data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore

	# Separate PyTree node fields vs. metadata-only fields
	data_fields: list[str] = []
	meta_fields: list[str] = []

	annotations = getattr(data_clz, "__annotations__", {})
	for field_info in dataclasses.fields(data_clz):
		# Explicit override
		if "pytree_node" in field_info.metadata:
			is_node = field_info.metadata["pytree_node"]
		else:
			ann = annotations.get(field_info.name, tp.Any)
			is_node = _is_pytree_node_annotation(ann)
		(data_fields if is_node else meta_fields).append(field_info.name)

	# Register replace method
	def replace(self, **updates):
		return dataclasses.replace(self, **updates)

	data_clz.replace = replace  # type: ignore[attr-defined]

	# Register with JAX tree utilities
	jax.tree_util.register_dataclass(data_clz, data_fields, meta_fields)

	# Serialization helpers
	def to_state_dict(x):
		return {name: serialization.to_state_dict(getattr(x, name)) for name in data_fields}

	def from_state_dict(x, state):
		state = state.copy()
		updates = {}
		for name in data_fields:
			if name not in state:
				raise ValueError(f"Missing field {name} in state dict for {clz.__name__}")
			value = getattr(x, name)
			updates[name] = serialization.from_state_dict(value, state.pop(name), name=name)
		if state:
			raise ValueError(
				f"Unknown field(s) {list(state.keys())} in state dict for {clz.__name__}"
			)
		return x.replace(**updates)

	serialization.register_serialization_state(data_clz, to_state_dict, from_state_dict)

	# Mark class as a eformer dataclass
	setattr(data_clz, "_eformer_dataclass", True)  # noqa

	return data_clz  # type: ignore


# Base class for PyTree-enabled dataclasses
STree = tp.TypeVar("STree", bound="xTree")


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
class xTree:
	"""
	Base class for dataclasses acting as JAX PyTree nodes.
	"""

	def __init_subclass__(cls, **kwargs):
		dataclass(cls, **kwargs)  # pytype: disable=wrong-arg-types

	def __init__(self, *args, **kwargs):
		# stub for type checkers
		raise NotImplementedError

	def replace(self: STree, **overrides) -> STree:
		# stub for type checkers
		raise NotImplementedError
