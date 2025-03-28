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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for tree mappin"""

import dataclasses
import types
import typing as tp

import jax
from jax import tree_util as jtu

PyTree = tp.Dict
FnDict = tp.Dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = tp.Dict[tp.Any, tp.Any]
Path = tp.Tuple[tp.Any, ...]


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


_CLS = tp.TypeVar("_CLS")


def auto_pytree(
	cls: _CLS = None,
	meta_fields: tp.Optional[tp.Tuple[str, ...]] = None,
) -> _CLS:
	"""
	Register a class as a JAX PyTree with automatic field inference.

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

	Returns:
	    The registered dataclass that can be used with JAX's PyTree operations.

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
		return jtu.register_dataclass(
			cls,
			data_fields=data_fields,
			meta_fields=tuple(final_meta_fields),
		)

	# Handle both @auto_pytree and @auto_pytree(meta_fields=(...))

	if cls is None:
		return wrap
	return wrap(cls)
