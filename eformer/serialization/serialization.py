# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey
from jax.experimental.array_serialization import serialization as array_ser
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding
from jaxtyping import PyTree

from eformer.loggings import get_logger
from eformer.paths import ePath

from . import fsspec_utils

logger = get_logger(__name__)

__all__ = [
    "is_array_like",
    "leaf_key_paths",
    "tree_deserialize_leaves",
    "tree_serialize_leaves",
]


def join_key(prefix: str | None, k: str | None) -> str:
    """Join a prefix and key with a dot separator.

    Args:
        prefix: Optional prefix string.
        k: Optional key string.

    Returns:
        Joined string with dot separator, or empty string if both are None.
    """
    if not k:
        return prefix or ""
    return f"{prefix}.{k}" if prefix else k


def _keyentry_to_str(path_elem: Any) -> str:
    """Convert a JAX tree path element to a string representation.

    Handles various JAX tree path element types including DictKey, SequenceKey,
    GetAttrKey, and FlattenedIndexKey. Falls back to string conversion and
    cleaning for unknown types.

    Args:
        path_elem: A JAX tree path element.

    Returns:
        String representation of the path element.
    """
    try:
        if isinstance(path_elem, DictKey):
            return str(path_elem.key)
        elif isinstance(path_elem, SequenceKey):
            return str(path_elem.idx)
        elif isinstance(path_elem, GetAttrKey):
            s = str(path_elem)
            return s[1:] if s.startswith(".") else s
        elif isinstance(path_elem, FlattenedIndexKey):
            return str(path_elem.idx)
    except Exception:
        pass

    s = str(path_elem)
    if s.startswith("."):
        s = s[1:]
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        inner = s[1:-1]
        if len(inner) >= 2 and ((inner[0] == "'" and inner[-1] == "'") or (inner[0] == '"' and inner[-1] == '"')):
            inner = inner[1:-1]
        s = inner
    return s


def leaf_key_paths(pytree: Any, prefix: str | None = "", *, is_leaf: Callable[[Any], bool] | None = None):
    """Create dotted key paths for each leaf in a pytree.

    Returns a pytree of the same structure where each leaf is replaced by its
    key path (prefixed by `prefix` if provided). Uses jax.tree_util.tree_flatten_with_path
    for robust handling of dicts, sequences, dataclasses, namedtuples, and custom PyTree nodes.

    Args:
        pytree: The pytree to create key paths for.
        prefix: Optional prefix to add to all key paths.
        is_leaf: Optional function to determine if a node is a leaf.

    Returns:
        PyTree with same structure where leaves are replaced by their dotted key paths.
    """
    path_value_pairs, treedef = jtu.tree_flatten_with_path(pytree, is_leaf=is_leaf)

    def path_to_str(path: Sequence[Any]) -> str:
        if not path:
            return prefix or ""
        parts = [_keyentry_to_str(pe) for pe in path]
        return join_key(prefix, ".".join(parts))

    leaf_paths = [path_to_str(path) for path, _ in path_value_pairs]
    return jtu.tree_unflatten(treedef, leaf_paths)


def is_array_like(x: Any) -> bool:
    """Check if an object is array-like.

    Minimal check similar to equinox.is_array_like, checking for shape and dtype attributes.

    Args:
        x: Object to check.

    Returns:
        True if object has both shape and dtype attributes, False otherwise.
    """
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _is_none(x):
    """Check if a value is None.

    Args:
        x: Value to check.

    Returns:
        True if x is None, False otherwise.
    """
    return x is None


def tree_serialize_leaves(
    checkpoint_dir,
    pytree,
    manager: array_ser.GlobalAsyncCheckpointManager | None = None,
    *,
    prefix: str | None = None,
    commit_callback: Callable | None = None,
    write_index: bool = True,
):
    """Serialize a pytree's leaves to TensorStore format.

    Serializes arrays in a pytree to TensorStore format with optional prefixing
    for organizing multiple trees in the same checkpoint directory.

    Args:
        checkpoint_dir: Directory to save the checkpoint.
        pytree: PyTree structure containing arrays to serialize.
        manager: Optional GlobalAsyncCheckpointManager. If None, creates a new one.
        prefix: Optional prefix for organizing arrays (e.g., 'model', 'optimizer').
        commit_callback: Optional callback to run after committing the checkpoint.
        write_index: Whether to write an index file for the checkpoint.

    Returns:
        None

    Note:
        Uses a unified index file (tensorstore_index.json) that supports multiple
        prefixes in version 2.0 format.
    """
    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()
        manager_was_none = True
    else:
        manager_was_none = False

    leaf_path = leaf_key_paths(pytree, prefix=prefix, is_leaf=_is_none)
    assert len(jax.tree.leaves(leaf_path, is_leaf=_is_none)) == len(jax.tree.leaves(pytree, is_leaf=_is_none))

    paths = _fs_paths_from_key_paths(checkpoint_dir, leaf_path)

    @dataclass
    class Pair:
        path: str
        leaf: Any

    zipped = jax.tree.map(lambda x, y: Pair(x, y), paths, pytree, is_leaf=_is_none)
    paired_leaves = jax.tree.leaves(zipped)
    paths = [p.path for p in paired_leaves]
    leaves = [p.leaf for p in paired_leaves]

    def _ensure_is_array(x):
        if isinstance(x, int | float | bool | complex):
            return jnp.array(x)
        else:
            return x

    arrays = [_ensure_is_array(x) for x in leaves]

    array_info = []
    arrays_filtered = []
    paths_filtered = []

    for a, p in zip(arrays, paths, strict=False):
        if is_array_like(a):
            arrays_filtered.append(a)
            paths_filtered.append(p)

            rel_path = os.path.relpath(p, checkpoint_dir)
            array_info.append(
                {
                    "path": rel_path,
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                }
            )

    arrays = arrays_filtered
    paths = paths_filtered

    if commit_callback is None:
        commit_callback = lambda: logger.info("Committed checkpoint to Tensorstore")  # noqa

    manager.serialize_with_paths(arrays, paths, on_commit_callback=commit_callback)

    if manager_was_none:
        manager.wait_until_finished()

    if write_index and array_info:
        index_path = ePath(checkpoint_dir) / "tensorstore_index.json"
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
        else:
            index_data = {
                "format": "tensorstore",
                "version": "2.0",
                "prefixes": {},
            }

        if prefix:
            if "prefixes" not in index_data:
                index_data = {
                    "format": "tensorstore",
                    "version": "2.0",
                    "prefixes": {},
                }
            index_data["prefixes"][prefix] = array_info
        else:
            index_data["arrays"] = array_info

        index_path.write_text(json.dumps(index_data, indent=2))


def _fs_paths_from_key_paths(checkpoint_dir, leaf_path):
    """Convert dotted key paths to filesystem paths.

    Args:
        checkpoint_dir: Base directory for checkpoint.
        leaf_path: PyTree of dotted key paths.

    Returns:
        PyTree with filesystem paths corresponding to the key paths.
    """

    def path_from_key_path(key_path):
        path = ePath(checkpoint_dir)
        for part in key_path.split("."):
            path = path / part
        return str(path)

    paths = jtu.tree_map(path_from_key_path, leaf_path)
    return paths


def _fully_replicated_sharding(mesh: Mesh | None) -> Sharding:
    """Create a fully replicated sharding.

    Args:
        mesh: Optional JAX mesh. If None, uses single device sharding.

    Returns:
        Sharding that replicates data across all devices.
    """
    if mesh is None:
        return SingleDeviceSharding(jax.devices()[0])
    else:
        return NamedSharding(mesh, PartitionSpec())


def _sharding_from_leaf(leaf, mesh) -> Sharding | None:
    """Determine appropriate sharding for a leaf value.

    Args:
        leaf: Leaf value from a pytree.
        mesh: JAX mesh for distributed computation.

    Returns:
        Appropriate Sharding for the leaf, or None if type is unknown.
    """
    if hasattr(leaf, "sharding") and leaf.sharding is not None:
        return leaf.sharding
    elif is_array_like(leaf):
        return _fully_replicated_sharding(mesh)
    elif isinstance(leaf, bool | float | complex | int | np.ndarray):
        return _fully_replicated_sharding(mesh)
    else:
        logger.warning(f"Unknown leaf type {type(leaf)}")
        return None


def tree_deserialize_leaves(
    checkpoint_dir,
    pytree=None,
    mesh: Mesh | None = None,
    manager: array_ser.GlobalAsyncCheckpointManager | None = None,
    *,
    prefix: str | None = None,
    shard_fns: dict[Callable] | None = None,
    allow_missing: bool = False,
    mismatch_allowed: bool = True,
):
    """Deserialize a PyTree of arrays from a TensorStore checkpoint.

    If pytree is provided, returns a pytree with the same structure as the template.
    If pytree is None, discovers the structure from the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing the TensorStore checkpoint.
        pytree: Optional pytree template for structure. If None, structure is discovered
            from the checkpoint.
        mesh: Optional JAX mesh for distributed arrays.
        manager: Optional GlobalAsyncCheckpointManager. If None, creates a new one.
        prefix: Optional prefix to filter/load specific tree (e.g., 'model', 'optimizer').
        shard_fns: Dictionary mapping array paths to sharding functions.
        allow_missing: Whether to allow missing arrays in checkpoint.
        mismatch_allowed: Whether to allow missing shard functions without error.

    Returns:
        Deserialized pytree structure with loaded arrays.

    Raises:
        ValueError: If checkpoint format is unsupported or prefix not found.
        FileNotFoundError: If required arrays are missing and allow_missing is False.

    Note:
        Supports both v1.0 (single prefix) and v2.0 (multi-prefix) index formats.
        When using v2.0 format with multiple prefixes, you must specify which prefix
        to load or an error will be raised listing available prefixes.
    """
    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()

    if pytree is None:
        index_path = ePath(checkpoint_dir) / "tensorstore_index.json"

        if index_path.exists():
            index_data = json.loads(index_path.read_text())

            if index_data.get("format") != "tensorstore":
                raise ValueError(f"Unsupported index format: {index_data.get('format')}")

            version = index_data.get("version", "1.0")

            if version == "2.0" and "prefixes" in index_data:
                if prefix:
                    if prefix not in index_data["prefixes"]:
                        available = list(index_data["prefixes"].keys())
                        raise ValueError(f"Prefix '{prefix}' not found. Available: {available}")
                    array_info = index_data["prefixes"][prefix]
                else:
                    if "arrays" in index_data:
                        array_info = index_data["arrays"]
                    else:
                        available = list(index_data["prefixes"].keys())
                        raise ValueError(f"No prefix specified. Available prefixes: {available}")
            else:
                array_info = index_data.get("arrays", [])
            paths_to_load = []
            keys = []

            for info in array_info:
                rel_path = info["path"]
                abs_path = str(ePath(checkpoint_dir) / rel_path)
                paths_to_load.append(abs_path)

                key = rel_path.replace("/", ".").replace("\\\\", ".")

                if prefix and key.startswith(f"{prefix}."):
                    key = key[len(prefix) + 1 :]
                keys.append(key)

            shardings_to_load = [_fully_replicated_sharding(mesh) for _ in paths_to_load]

            if paths_to_load:
                deser_leaves = manager.deserialize_with_paths(shardings=shardings_to_load, paths=paths_to_load)
            else:
                deser_leaves = []

            if shard_fns:
                from eformer.pytree import flatten_dict, is_flatten

                if not is_flatten(shard_fns):
                    shard_fns = flatten_dict(shard_fns, sep=".")

                processed_leaves = []
                for key, array in zip(keys, deser_leaves, strict=False):
                    if key in shard_fns:
                        array = shard_fns[key](array)
                    elif not mismatch_allowed:
                        logger.warning(f"No shard function found for key: {key}")
                    processed_leaves.append(array)
                deser_leaves = processed_leaves

            result = {}
            for key, array in zip(keys, deser_leaves, strict=False):
                parts = key.split(".")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = array

            return result
        else:
            import glob

            logger.warning(f"No tensorstore_index.json found in {checkpoint_dir}, attempting filesystem discovery")

            checkpoint_path = ePath(checkpoint_dir)
            if prefix:
                pattern = str(checkpoint_path / prefix / "**" / ".zarray")
            else:
                pattern = str(checkpoint_path / "**" / ".zarray")
            zarray_paths = glob.glob(pattern, recursive=True)

            array_paths = [str(ePath(p).parent) for p in zarray_paths]

            paths_to_load = []
            keys = []
            for path in array_paths:
                path_obj = ePath(path)
                checkpoint_obj = ePath(checkpoint_dir)
                try:
                    rel_path = str(path_obj.relative_to(checkpoint_obj))
                except (ValueError, TypeError):
                    rel_path = os.path.relpath(path, checkpoint_dir)

                key = rel_path.replace("/", ".").replace("\\\\", ".")

                if prefix and key.startswith(f"{prefix}."):
                    key = key[len(prefix) + 1 :]
                keys.append(key)
                paths_to_load.append(path)

            shardings_to_load = [_fully_replicated_sharding(mesh) for _ in paths_to_load]

            if paths_to_load:
                deser_leaves = manager.deserialize_with_paths(shardings=shardings_to_load, paths=paths_to_load)
            else:
                deser_leaves = []

            if shard_fns:
                from eformer.pytree import flatten_dict, is_flatten

                if not is_flatten(shard_fns):
                    shard_fns = flatten_dict(shard_fns, sep=".")

                processed_leaves = []
                for key, array in zip(keys, deser_leaves, strict=False):
                    if key in shard_fns:
                        array = shard_fns[key](array)
                    elif not mismatch_allowed:
                        logger.warning(f"No shard function found for key: {key}")
                    processed_leaves.append(array)
                deser_leaves = processed_leaves

            result = {}
            for key, array in zip(keys, deser_leaves, strict=False):
                parts = key.split(".")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = array

            return result

    shardings: PyTree[Sharding | None] = jtu.tree_map(partial(_sharding_from_leaf, mesh=mesh), pytree, is_leaf=_is_none)

    leaf_path = leaf_key_paths(shardings, prefix=prefix, is_leaf=_is_none)
    paths = _fs_paths_from_key_paths(checkpoint_dir, leaf_path)
    paths = jtu.tree_leaves(paths, is_leaf=_is_none)

    shardings_leaves, shardings_structure = jtu.tree_flatten(shardings, is_leaf=_is_none)
    assert len(shardings_leaves) == len(paths)

    real_indices = [i for i, x in enumerate(shardings_leaves) if x is not None]
    paths_to_load = []
    indices_to_load = []
    shardings_to_load = []

    missing_paths = []
    missing_indices = []

    for i in real_indices:
        path = paths[i]

        if not fsspec_utils.exists(path):
            missing_paths.append(path)
            missing_indices.append(i)
            continue

        paths_to_load.append(path)
        indices_to_load.append(i)
        shardings_to_load.append(shardings_leaves[i])

    if missing_paths:
        if not allow_missing:
            raise FileNotFoundError(f"Missing paths: {missing_paths}")
        else:
            to_log = f"Several keys were missing from the checkpoint directory {checkpoint_dir}:"
            leaf_paths = jtu.tree_leaves(leaf_path, is_leaf=_is_none)
            for i in missing_indices:
                to_log += f"\n  - {leaf_paths[i]}"
            logger.warning(to_log)

    deser_leaves = manager.deserialize_with_paths(shardings=shardings_to_load, paths=paths_to_load)

    if shard_fns:
        from eformer.pytree import flatten_dict, is_flatten

        if not is_flatten(shard_fns):
            shard_fns = flatten_dict(shard_fns, sep=".")

        leaf_paths = jtu.tree_leaves(leaf_path, is_leaf=_is_none)
        processed_leaves = []

        for idx, array in zip(indices_to_load, deser_leaves, strict=False):
            key = leaf_paths[idx]
            if key in shard_fns:
                array = shard_fns[key](array)
            elif not mismatch_allowed:
                logger.warning(f"No shard function found for key: {key}")
            processed_leaves.append(array)
        deser_leaves = processed_leaves

    out_leaves = jax.tree.leaves(pytree, is_leaf=_is_none)
    assert len(out_leaves) == len(shardings_leaves)
    for i, x in zip(indices_to_load, deser_leaves, strict=False):
        out_leaves[i] = x

    deser_arrays = jtu.tree_unflatten(shardings_structure, out_leaves)
    return deser_arrays
