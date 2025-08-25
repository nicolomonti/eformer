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


import asyncio
import hashlib
import json
import os
import threading
import typing as tp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from safetensors import flax as safe_flax
from tqdm.autonotebook import tqdm

from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.pytree import PyTree, flatten_dict, is_flatten, serialization, unflatten_dict

from .base_manager import CheckpointManager
from .serialization import (
    tree_deserialize_leaves,
    tree_serialize_leaves,
)
from .utils import derive_base_prefix_from_path, group_keys_by_shard_size, index_filename, shard_filename
from .utils import read_process_array as _read_process_array
from .utils import to_host as _to_host

logger = get_logger(__name__)

try:
    import tensorstore as ts

    TENSORSTORE_AVAILABLE = True
except ImportError:
    TENSORSTORE_AVAILABLE = False
    logger.warning("Tensorstore not available. Install with: pip install tensorstore")


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for checkpoints with versioning and validation.

    Stores comprehensive metadata about a checkpoint including version information,
    timestamps, checksums for validation, and custom user metadata.

    Attributes:
        version: Version string for the checkpoint format.
        timestamp: ISO format timestamp of when checkpoint was created.
        checksum: Dictionary mapping array keys to SHA256 checksums.
        array_metadata: Dictionary mapping array keys to shape/dtype info.
        framework_version: Version of the framework used to create checkpoint.
        custom_metadata: User-defined metadata dictionary.
    """

    version: str = "0.0.51"
    timestamp: str = None
    checksum: dict[str, str] = None
    array_metadata: dict[str, dict] = None
    framework_version: str = None
    custom_metadata: dict = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary format.

        Returns:
            Dictionary representation of the metadata.
        """
        return {
            "version": self.version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "checksum": self.checksum or {},
            "array_metadata": self.array_metadata or {},
            "framework_version": self.framework_version,
            "custom_metadata": self.custom_metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create CheckpointMetadata from dictionary.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            CheckpointMetadata instance.
        """
        return cls(
            version=data.get("version", "0.0.52"),
            timestamp=data.get("timestamp"),
            checksum=data.get("checksum", {}),
            array_metadata=data.get("array_metadata", {}),
            framework_version=data.get("framework_version"),
            custom_metadata=data.get("custom_metadata", {}),
        )


class AsyncCheckpointManager:
    """Async-capable checkpoint manager with concurrent operations.

    This manager provides asynchronous checkpoint saving and loading with support
    for parallel operations, tensorstore backend, validation, and compression.

    Attributes:
        float_dtype: Default data type for floating point arrays.
        enable: Whether checkpointing is enabled.
        verbose: Enable verbose output.
        gcs_bucket: Google Cloud Storage bucket name.
        max_workers: Maximum number of worker threads.
        enable_validation: Enable checksum validation.
        enable_compression: Enable compression for tensorstore.
        use_tensorstore: Use tensorstore backend when available.
    """

    def __init__(
        self,
        enable: bool | None = None,
        float_dtype: jnp.dtype = jnp.bfloat16,
        verbose: bool = False,
        gcs_bucket: str | None = None,
        gcs_credentials_path: str | None = None,
        max_workers: int = 4,
        enable_validation: bool = True,
        enable_compression: bool = False,
        use_tensorstore: bool = True,
    ):
        self.float_dtype = float_dtype
        self.enable = enable
        self.verbose = verbose
        self.gcs_bucket = gcs_bucket
        self.max_workers = max_workers
        self.enable_validation = enable_validation
        self.enable_compression = enable_compression
        self.use_tensorstore = use_tensorstore and TENSORSTORE_AVAILABLE

        self.gcs_client = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_saves = []
        self._save_lock = threading.Lock()

        if gcs_bucket:
            self.gcs_client = CheckpointManager.create_gcs_client(gcs_credentials_path)

    def __del__(self):
        """Cleanup executor on deletion.

        Ensures the thread pool executor is properly shutdown when the
        manager is destroyed.
        """
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def _run_async(self, coro):
        """Helper to run async code in sync context.

        Attempts to create a task if an event loop is running, otherwise
        runs the coroutine in a new event loop.

        Args:
            coro: Coroutine to run.

        Returns:
            Result of the coroutine execution.
        """
        try:
            return asyncio.create_task(coro)
        except RuntimeError:
            return asyncio.run(coro)

    @staticmethod
    def _estimate_nbytes(array: jax.Array) -> int:
        """Estimate the number of bytes in an array.

        Args:
            array: JAX array to estimate size for.

        Returns:
            Estimated number of bytes in the array.
        """
        if hasattr(array, "nbytes"):
            return array.nbytes
        elif hasattr(array, "shape") and hasattr(array, "dtype"):
            return np.prod(array.shape) * np.dtype(array.dtype).itemsize
        else:
            return 0

    def _calculate_optimal_chunks(self, shape: tuple, dtype: jnp.dtype) -> list[int] | None:
        """Calculate optimal chunk sizes for an array.

        Aims for chunks of ~64MB for optimal I/O performance. Balances between
        chunk size and number of chunks to optimize read/write operations.

        Args:
            shape: Shape of the array to chunk.
            dtype: Data type of the array.

        Returns:
            List of chunk sizes for each dimension, or None for small arrays
            that don't need chunking.

        Note:
            For very large dimensions (>10000), limits chunk size to 2000 elements
            to avoid overly large chunks.
        """
        if not shape:
            return None

        target_chunk_bytes = 64 * 1024 * 1024

        dtype_size = np.dtype(dtype).itemsize

        total_elements = np.prod(shape)
        total_bytes = total_elements * dtype_size

        if total_bytes < target_chunk_bytes:
            return None

        chunks = []
        remaining_bytes = target_chunk_bytes

        for dim_size in shape:
            elements_per_chunk = min(dim_size, max(1, remaining_bytes // dtype_size))

            if dim_size > 10000:
                elements_per_chunk = min(2000, elements_per_chunk)

            chunks.append(int(elements_per_chunk))
            remaining_bytes = remaining_bytes // max(1, (dim_size // elements_per_chunk))

        return chunks

    @staticmethod
    def compute_checksum(array: jax.Array) -> str:
        """Compute SHA256 checksum for validation.

        Converts array to bytes and computes SHA256 hash for data integrity
        verification.

        Args:
            array: JAX array to compute checksum for.

        Returns:
            SHA256 checksum as hexadecimal string.

        Note:
            Arrays are converted to numpy before hashing for consistency.
        """
        array_bytes = np.asarray(array).tobytes()
        return hashlib.sha256(array_bytes).hexdigest()

    def _validate_checkpoint(self, tree: dict, metadata: CheckpointMetadata) -> bool:
        """Validate checkpoint integrity using checksums.

        Compares computed checksums of loaded arrays against stored checksums
        in metadata to ensure data integrity.

        Args:
            tree: Dictionary containing checkpoint data.
            metadata: Checkpoint metadata containing checksums.

        Returns:
            True if validation passes, False otherwise.

        Note:
            Validation is skipped if enable_validation is False or no checksums
            are present in metadata.
        """
        if not self.enable_validation or not metadata.checksum:
            return True

        flat_tree = flatten_dict(tree) if not is_flatten(tree) else tree
        for key, array in flat_tree.items():
            if key in metadata.checksum:
                computed = self.compute_checksum(array)
                if computed != metadata.checksum[key]:
                    logger.error(f"Checksum mismatch for {key}")
                    return False
        return True

    def save(
        self,
        tree: PyTree,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh = None,
        gather_fns: dict[tp.Callable] | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        metadata: dict[str, str] | None = None,
        shard_size_gb: float | None = 5.00,
        callback: tp.Callable[[str], None] | None = None,
        shard_prefix: str | None = None,
        prefix: str | None = None,
        do_all_gather: bool = False,
    ) -> str:
        """Synchronous wrapper for save_tree_async.

        This method can be called without async/await and handles the async runtime internally.

        Args:
            tree: PyTree structure to save.
            path: Path where the checkpoint will be saved.
            mesh: JAX mesh for distributed computation.
            gather_fns: Dictionary of gather functions or bool for device gathering.
            float_dtype: Data type for floating point arrays.
            metadata: Additional metadata to save with checkpoint.
            shard_size_gb: Maximum size of each shard in GB.
            callback: Optional callback function called after save.
            shard_prefix: Custom prefix for shard files.
            prefix: Optional prefix for saving specific tree (e.g., 'model', 'optimizer').
            do_all_gather: Whether to gather all arrays to host.

        Returns:
            Path where the checkpoint was saved.
        """
        return self._run_async(
            self.save_tree_async(
                tree=tree,
                path=path,
                mesh=mesh,
                gather_fns=gather_fns,
                float_dtype=float_dtype,
                metadata=metadata,
                shard_size_gb=shard_size_gb,
                callback=callback,
                shard_prefix=shard_prefix,
                prefix=prefix,
                do_all_gather=do_all_gather,
            )
        )

    async def save_tree_async(
        self,
        tree: PyTree,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh,
        gather_fns: dict[tp.Callable] | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        metadata: dict[str, str] | None = None,
        shard_size_gb: float | None = 5.00,
        callback: tp.Callable[[str], None] | None = None,
        shard_prefix: str | None = None,
        prefix: str | None = None,
        do_all_gather: bool = False,
    ) -> str:
        """Asynchronously save checkpoint with parallel shard writing.

        Saves a PyTree structure to disk using either TensorStore or SafeTensors format,
        with support for sharding large checkpoints and parallel I/O operations.

        Args:
            tree: PyTree structure to save.
            path: Path where the checkpoint will be saved.
            mesh: JAX mesh for distributed computation.
            gather_fns: Dictionary of gather functions or bool for device gathering.
                If True, uses jax.device_get for all arrays.
            float_dtype: Data type for floating point arrays. Defaults to self.float_dtype.
            metadata: Additional metadata to save with checkpoint.
            shard_size_gb: Maximum size of each shard in GB. If total size exceeds this,
                checkpoint is split into multiple files.
            callback: Optional callback function called after save completes.
            shard_prefix: Custom prefix for shard files (default: 'easystore').
            prefix: Optional prefix for saving specific tree (e.g., 'model', 'optimizer').
                Used for organizing multiple trees in same directory.
            do_all_gather: Whether to gather all arrays to host before saving.

        Returns:
            Path where the checkpoint was saved.

        Note:
            Automatically chooses between TensorStore (if available and enabled) or
            SafeTensors format based on configuration and checkpoint size.
        """
        if float_dtype is None:
            float_dtype = self.float_dtype

        tree = serialization.to_state_dict(tree)
        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")

        if gather_fns:
            tree = await self._gather_async(tree, gather_fns)
        if do_all_gather:
            tree = jax.tree_util.tree_map(
                lambda x: _to_host(x, float_dtype, mesh),
                tree,
                is_leaf=lambda x: isinstance(x, jax.Array | np.generic | float | int),
            )

        checkpoint_meta = CheckpointMetadata(timestamp=datetime.now().isoformat(), custom_metadata=metadata)

        if self.enable_validation:
            checkpoint_meta.checksum = {k: self.compute_checksum(v) for k, v in tree.items()}
            checkpoint_meta.array_metadata = {
                k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in tree.items()
            }

        path_str = str(path)

        total_size_gb = sum(self._estimate_nbytes(arr) / (1024**3) for arr in jax.tree_util.tree_leaves(tree))

        if self.use_tensorstore:
            if shard_size_gb and total_size_gb > shard_size_gb:
                return await self._save_tensorstore_sharded_async(
                    tree,
                    path_str,
                    checkpoint_meta,
                    shard_size_gb,
                    shard_prefix,
                    prefix,
                )
            return await self._save_tensorstore_async(tree, path_str, checkpoint_meta, prefix)

        if shard_size_gb and shard_size_gb > 0:
            return await self._save_sharded_async(tree, path_str, checkpoint_meta, shard_size_gb)

        await self._save_single_async(tree, path_str, checkpoint_meta.to_dict())

        if callback:
            callback(path_str)

        return path_str

    async def _gather_async(self, tree: dict, gather_fns: dict[tp.Callable] | bool) -> dict:
        """Asynchronously gather distributed arrays.

        Performs parallel gathering of distributed arrays using provided gather
        functions or device_get.

        Args:
            tree: Dictionary of arrays to gather.
            gather_fns: Dictionary mapping keys to gather functions, or bool.
                If True, uses jax.device_get for all arrays.
                If dict, applies specific gather function for matching keys.

        Returns:
            Dictionary with gathered arrays.

        Note:
            Arrays without matching gather functions are returned unchanged.
        """
        if isinstance(gather_fns, bool):
            loop = asyncio.get_event_loop()
            futures = []
            for key, value in tree.items():
                future = loop.run_in_executor(self.executor, jax.device_get, value)
                futures.append((key, future))

            results = {}
            for key, future in futures:
                results[key] = await future
            return results

        if not is_flatten(gather_fns):
            gather_fns = flatten_dict(gather_fns, sep=".")

        loop = asyncio.get_event_loop()
        futures = []

        for key, value in tree.items():
            if key in gather_fns:
                future = loop.run_in_executor(self.executor, gather_fns[key], value)
                futures.append((key, future))
            else:
                futures.append((key, asyncio.create_task(asyncio.sleep(0, value))))

        results = {}
        for key, future in futures:
            results[key] = await future

        return results

    async def _save_sharded_async(
        self,
        flat_state: dict,
        base_path: str,
        metadata: CheckpointMetadata,
        shard_size_gb: float,
    ) -> str:
        """Save sharded checkpoint with parallel writes.

        Args:
            flat_state: Flattened state dictionary to save.
            base_path: Base path for shard files.
            metadata: Checkpoint metadata.
            shard_size_gb: Maximum size of each shard in GB.

        Returns:
            Path to the index file.
        """
        max_bytes = int(shard_size_gb * (1024**3))
        shards = group_keys_by_shard_size(flat_state, max_bytes)
        base_prefix = derive_base_prefix_from_path(base_path)

        loop = asyncio.get_event_loop()
        shard_futures = []

        for i, shard_keys in enumerate(shards, start=1):
            shard_path = shard_filename(base_prefix, i, len(shards))
            shard_tensors = {k: flat_state[k] for k in shard_keys}

            meta_dict = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.to_dict().items()}

            future = loop.run_in_executor(self.executor, safe_flax.save_file, shard_tensors, shard_path, meta_dict)
            shard_futures.append(future)

        await asyncio.gather(*shard_futures)

        weight_map = {}
        for i, shard_keys in enumerate(shards, start=1):
            shard_name = os.path.basename(shard_filename(base_prefix, i, len(shards)))
            for k in shard_keys:
                weight_map[k] = shard_name

        index_data = {"metadata": metadata.to_dict(), "weight_map": weight_map}

        index_path = index_filename(base_prefix)
        ePath(index_path).write_text(json.dumps(index_data, ensure_ascii=False))

        return index_path

    async def _save_single_async(self, tree: dict, path: str, metadata: dict):
        """Save single checkpoint file asynchronously.

        Args:
            tree: Dictionary to save.
            path: Path where the checkpoint will be saved.
            metadata: Metadata to save with the checkpoint.
        """

        if metadata:
            metadata = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()}

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, safe_flax.save_file, tree, path, metadata)

    async def _save_tensorstore_async(
        self, tree: dict, path: str, metadata: CheckpointMetadata, prefix: str | None = None
    ) -> str:
        """Save using tensorstore via the core serialization module.

        Leverages TensorStore for efficient array serialization with support for
        zarr format and concurrent writes.

        Args:
            tree: Dictionary of arrays to save (flattened).
            path: Path where the checkpoint will be saved.
            metadata: Checkpoint metadata.
            prefix: Optional prefix for saving specific tree.

        Returns:
            Path where the checkpoint was saved.

        Note:
            Creates a unified index file (tensorstore_index.json) that supports
            multiple prefixes in v2.0 format. Also saves checkpoint metadata
            separately.
        """

        from eformer.pytree import unflatten_dict

        pytree = unflatten_dict(tree, sep=".")

        loop = asyncio.get_event_loop()
        from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

        manager = GlobalAsyncCheckpointManager()

        def commit_with_metadata():
            logger.info("Committed checkpoint to Tensorstore")
            meta_path = ePath(path) / "checkpoint_metadata.json"
            meta_path.write_text(json.dumps(metadata.to_dict()))

        await loop.run_in_executor(
            self.executor,
            lambda: tree_serialize_leaves(
                checkpoint_dir=path,
                pytree=pytree,
                manager=manager,
                prefix=prefix,
                commit_callback=lambda: logger.info("Committed checkpoint to Tensorstore"),
                write_index=True,
            ),
        )

        await loop.run_in_executor(self.executor, commit_with_metadata)

        return path

    async def _save_tensorstore_sharded_async(
        self,
        tree: dict,
        path: str,
        metadata: CheckpointMetadata,
        shard_size_gb: float,
        shard_prefix: str | None = None,
        prefix: str | None = None,
    ) -> str:
        """Save large checkpoint as multiple SafeTensors files with custom prefix.

        Args:
            tree: Dictionary of arrays to save.
            path: Path where the checkpoint will be saved.
            metadata: Checkpoint metadata.
            shard_size_gb: Maximum size of each shard in GB (typically 3-5GB).
            shard_prefix: Custom prefix for shard files (default: "easystore").
            prefix: Optional prefix for saving specific tree.

        Returns:
            Path where the checkpoint was saved.
        """
        ePath(path).mkdir(parents=True, exist_ok=True)

        shard_size_bytes = shard_size_gb * (1024**3)
        shards = []
        current_shard = {}
        current_size = 0

        for key, array in tree.items():
            array_size = self._estimate_nbytes(array)

            if array_size > shard_size_bytes:
                if current_shard:
                    shards.append(current_shard)
                    current_shard = {}
                    current_size = 0
                shards.append({key: array})

            elif current_size + array_size > shard_size_bytes:
                if current_shard:
                    shards.append(current_shard)
                current_shard = {key: array}
                current_size = array_size
            else:
                current_shard[key] = array
                current_size += array_size

        if current_shard:
            shards.append(current_shard)

        write_futures = []
        shard_info = {}
        total_shards = len(shards)

        shard_pref = shard_prefix or "easystore"
        if prefix:
            shard_pref = f"{prefix}_{shard_pref}"

        loop = asyncio.get_event_loop()

        for i, shard_data in enumerate(shards, 1):
            shard_filename = str(ePath(path) / f"{shard_pref}-{i:05d}-{total_shards:05d}.params")

            shard_metadata = {
                "shard_index": str(i),
                "total_shards": str(total_shards),
                "prefix": shard_pref,
                "tree_prefix": prefix,
                "timestamp": metadata.timestamp,
            }

            if metadata.custom_metadata:
                for k, v in metadata.custom_metadata.items():
                    shard_metadata[f"custom_{k}"] = json.dumps(v) if not isinstance(v, str) else v

            future = loop.run_in_executor(self.executor, safe_flax.save_file, shard_data, shard_filename, shard_metadata)
            write_futures.append(future)

            for key in shard_data.keys():
                shard_info[key] = f"{shard_pref}-{i:05d}-{total_shards:05d}.params"

        await asyncio.gather(*write_futures)

        meta_path = ePath(path) / "metadata.json"
        metadata.custom_metadata = metadata.custom_metadata or {}
        metadata.custom_metadata["shard_info"] = shard_info
        metadata.custom_metadata["num_shards"] = total_shards
        metadata.custom_metadata["shard_size_gb"] = shard_size_gb

        meta_path.write_text(json.dumps(metadata.to_dict()))

        index_file = f"{prefix}_index.json" if prefix else f"{shard_pref}-index.json"
        prefix_index_path = ePath(path) / index_file
        index_data = {
            "format": "tensorstore_sharded",
            "num_shards": total_shards,
            "shard_size_gb": shard_size_gb,
            "shard_map": shard_info,
            "total_size_gb": sum(self._estimate_nbytes(arr) / (1024**3) for arr in tree.values()),
            "shard_prefix": shard_pref,
            "tree_prefix": prefix,
        }

        prefix_index_path.write_text(json.dumps(index_data, indent=2))

        master_index_path = ePath(path) / "index.json"
        if master_index_path.exists():
            master_index = json.loads(master_index_path.read_text())
        else:
            master_index = {
                "format": "tensorstore_multi_sharded",
                "prefixes": {},
            }

        index_key = prefix or shard_pref
        master_index["prefixes"][index_key] = {
            "index_file": index_file,
            "num_shards": total_shards,
            "total_size_gb": index_data["total_size_gb"],
        }

        master_index_path.write_text(json.dumps(master_index, indent=2))

        return path

    async def _load_tensorstore_sharded_async(
        self,
        path: str,
        prefix: str | None = None,
        shard_fns: dict[tp.Callable] | None = None,
    ) -> tuple[dict, dict]:
        """Load sharded tensorstore checkpoint with optional filtering.

        Args:
            path: Path to the checkpoint or index file.
            prefix: Optional prefix to filter shards.
            shard_fns: Dictionary of functions to apply to shards.

        Returns:
            Tuple of (loaded tree dictionary, metadata dictionary).

        Raises:
            ImportError: If tensorstore is not available.
            FileNotFoundError: If index file is not found.
            ValueError: If format is unknown or prefix not found.
        """
        if not TENSORSTORE_AVAILABLE:
            raise ImportError("Tensorstore not available")

        if path.endswith(".json"):
            index_path = path
            base_path = str(ePath(path).parent)
        else:
            base_path = path
            index_path = str(ePath(path) / "index.json")

        index_path_obj = ePath(index_path)
        if not index_path_obj.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        index_data = json.loads(index_path_obj.read_text())

        format_type = index_data.get("format")

        if format_type == "tensorstore_multi_sharded":
            if not prefix:
                prefixes = list(index_data.get("prefixes", {}).keys())
                raise ValueError(
                    f"Multiple checkpoints found. Specify prefix_filter or load specific index. Available: {prefixes}"
                )

            prefix_info = index_data["prefixes"].get(prefix)
            if not prefix_info:
                available = list(index_data.get("prefixes", {}).keys())
                raise ValueError(f"Prefix '{prefix}' not found. Available: {available}")

            prefix_index_path = ePath(base_path) / prefix_info["index_file"]
            index_data = json.loads(prefix_index_path.read_text())

        if index_data.get("format") != "tensorstore_sharded":
            raise ValueError(f"Unknown format: {index_data.get('format')}")

        shard_map = index_data.get("shard_map", {})

        if prefix:
            matching_shards = set()
            filtered_keys = {}
            for key, shard_file in shard_map.items():
                if shard_file.startswith(prefix):
                    matching_shards.add(shard_file)
                    filtered_keys[key] = shard_file

            if not matching_shards:
                raise ValueError(f"No shards found with prefix: {prefix}")

            shard_map = filtered_keys

        if shard_fns and not is_flatten(shard_fns):
            shard_fns = flatten_dict(shard_fns, sep=".")

        tree = {}
        loaded_shards = set()

        for key, shard_file in shard_map.items():
            if shard_file not in loaded_shards:
                loaded_shards.add(shard_file)

            shard_path = str(ePath(base_path) / shard_file)
            clean_key = key.replace("/", "_").replace(".", "_")
            array_path = str(ePath(shard_path) / clean_key)

            spec = {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": array_path},
            }

            store = await ts.open(spec)
            array = await store.read()
            array = jnp.array(array)

            if shard_fns and key in shard_fns:
                array = shard_fns[key](array)

            tree[key] = array

        meta_path = ePath(base_path) / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {}

        print(f"✓ Loaded {len(loaded_shards)} shards, {len(tree)} arrays")
        if prefix:
            print(f"  Filtered by prefix: {prefix}")

        return tree, metadata

    async def _load_tensorstore_async(
        self, path: str, shard_fns: dict[tp.Callable] | None = None, prefix: str | None = None
    ) -> tuple[dict, dict]:
        """Load checkpoint saved with tensorstore using core deserialization.

        Args:
            path: Path to the tensorstore checkpoint.
            shard_fns: Dictionary of functions to apply to shards.
            prefix: Optional prefix for loading specific tree.

        Returns:
            Tuple of (loaded tree dictionary, metadata dictionary).
        """
        loop = asyncio.get_event_loop()
        from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

        manager = GlobalAsyncCheckpointManager()

        tree = await loop.run_in_executor(
            self.executor,
            lambda: tree_deserialize_leaves(
                checkpoint_dir=path,
                pytree=None,
                mesh=None,
                manager=manager,
                prefix=prefix,
                shard_fns=shard_fns,
                allow_missing=False,
                mismatch_allowed=True,
            ),
        )

        meta_path = ePath(path) / "checkpoint_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {}

        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")

        return tree, metadata

    def load(
        self,
        path: ePathLike | str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        validate: bool | None = None,
        prefix_filter: str | None = None,
        prefix: str | None = None,
        use_async: bool = True,
    ) -> tuple[PyTree | dict, dict]:
        """Synchronous load method that can work with or without async.

        Automatically detects checkpoint format (TensorStore or SafeTensors) and
        loads accordingly. Can be called without async/await.

        Args:
            path: Path to the checkpoint directory or file.
            shard_fns: Dictionary mapping keys to functions that process/reshard arrays
                after loading.
            mismatch_allowed: Whether to allow missing shard functions without error.
            callback: Optional callback to process each array after loading.
                Receives (array, key) and returns processed array.
            dtype: Data type to cast arrays to after loading.
            validate: Whether to validate checksums. If None, uses self.enable_validation.
            prefix_filter: Deprecated. Use 'prefix' instead.
            prefix: Optional prefix for loading specific tree (e.g., 'model', 'optimizer').
                Required when checkpoint contains multiple prefixes.
            use_async: Whether to use async loading (faster) or sync loading.

        Returns:
            Tuple of (loaded tree, metadata dictionary).
            Tree is unflattened to nested structure.

        Raises:
            ValueError: If validation fails or prefix not found.
            FileNotFoundError: If checkpoint doesn't exist.

        Note:
            Automatically detects TensorStore format by checking for .zarray files
            or tensorstore_index.json.
        """
        path_str = str(path)

        is_tensorstore = False
        path_obj = ePath(path_str)
        if path_obj.is_dir():
            if (path_obj / "tensorstore_index.json").exists():
                is_tensorstore = True
            elif any((path_obj / d / ".zarray").exists() for d in os.listdir(path_str) if (path_obj / d).is_dir()):
                is_tensorstore = True

        if is_tensorstore:
            if use_async:
                tree, metadata = self._run_async(self._load_tensorstore_async(path_str, shard_fns, prefix))
            else:
                from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

                manager = GlobalAsyncCheckpointManager()
                tree = tree_deserialize_leaves(
                    checkpoint_dir=path_str,
                    pytree=None,
                    mesh=None,
                    manager=manager,
                    prefix=prefix,
                    shard_fns=shard_fns,
                    allow_missing=False,
                    mismatch_allowed=mismatch_allowed,
                )
                meta_path = path_obj / "checkpoint_metadata.json"
                if meta_path.exists():
                    metadata = json.loads(meta_path.read_text())
                else:
                    metadata = {}

            if not is_flatten(tree):
                tree = flatten_dict(tree, sep=".")
            tree = unflatten_dict(tree, sep=".")
            return tree, metadata
        else:
            return self.load_tree_parallel(
                path=path,
                shard_fns=shard_fns,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
                validate=validate,
                prefix_filter=prefix_filter,
            )

    def load_tree_parallel(
        self,
        path: ePathLike | str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        validate: bool | None = None,
        prefix_filter: str | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load checkpoint with parallel shard reading.

        Args:
            path: Path to the checkpoint.
            shard_fns: Dictionary of functions to apply to shards.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.
            validate: Whether to validate checksums.
            prefix_filter: Optional prefix to filter shards.

        Returns:
            Tuple of (loaded tree, metadata dictionary).

        Raises:
            ValueError: If checkpoint validation fails.
        """
        validate = validate if validate is not None else self.enable_validation

        path_str = str(path)
        base_prefix = derive_base_prefix_from_path(path_str)
        index_path_str = index_filename(base_prefix)

        if ePath(index_path_str).exists():
            return self._load_sharded_parallel(index_path_str, shard_fns, mismatch_allowed, callback, dtype, validate)

        tree, metadata = CheckpointManager.load_checkpoint(
            path, shard_fns, self.verbose, mismatch_allowed, callback, dtype, self.gcs_client
        )

        if validate and metadata:
            meta = CheckpointMetadata.from_dict(metadata)
            if not self._validate_checkpoint(tree, meta):
                raise ValueError("Checkpoint validation failed")

        return tree, metadata

    def _load_sharded_parallel(
        self,
        index_path: str,
        shard_fns: dict[tp.Callable] | None,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
        validate: bool,
    ) -> tuple[PyTree | dict, dict]:
        """Load sharded checkpoint with parallel reads.

        Args:
            index_path: Path to the index file.
            shard_fns: Dictionary of functions to apply to shards.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.
            validate: Whether to validate checksums.

        Returns:
            Tuple of (loaded tree, metadata dictionary).

        Raises:
            ValueError: If checkpoint validation fails.
        """
        index_data = json.loads(ePath(index_path).read_text())

        weight_map: dict[str, str] = index_data.get("weight_map", {})
        directory = str(ePath(index_path).parent)

        file_to_keys: dict[str, list[str]] = defaultdict(list)
        for k, shard_name in weight_map.items():
            file_to_keys[shard_name].append(k)

        if shard_fns and not is_flatten(shard_fns):
            shard_fns = flatten_dict(shard_fns, sep=".")

        tree = {}
        futures = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for shard_name, keys in file_to_keys.items():
                shard_path = str(ePath(directory) / shard_name)
                future = executor.submit(
                    self._load_shard_file, shard_path, keys, shard_fns, mismatch_allowed, callback, dtype
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Loading shards (parallel)", disable=not self.verbose
            ):
                shard_tree = future.result()
                tree.update(shard_tree)

        tree = unflatten_dict(tree, sep=".")
        metadata = index_data.get("metadata", {})

        if validate and metadata:
            meta = CheckpointMetadata.from_dict(metadata)
            if not self._validate_checkpoint(tree, meta):
                raise ValueError("Checkpoint validation failed")

        return tree, metadata

    def _load_shard_file(
        self,
        shard_path: str,
        keys: list[str],
        shard_fns: dict | None,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
    ) -> dict:
        """Load a single shard file.

        Args:
            shard_path: Path to the shard file.
            keys: List of keys to load from the shard.
            shard_fns: Dictionary of functions to apply to shards.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.

        Returns:
            Dictionary with loaded tensors.
        """
        shard_tree = {}
        with safe_flax.safe_open(shard_path, framework="flax") as manager:
            process_func = partial(
                _read_process_array,
                shard_fns=shard_fns,
                mismatch_allowed=mismatch_allowed,
                manager=manager,
                callback=callback,
                dtype=dtype,
            )
            for key in keys:
                k, tensor, _ = process_func(key)
                shard_tree[k] = tensor
        return shard_tree

    async def wait_for_pending_saves(self):
        """Wait for all pending async saves to complete.

        Ensures all asynchronous save operations tracked by this manager are
        finished before continuing. Useful for ensuring data consistency before
        shutdown or when synchronization is needed.

        Note:
            Clears the pending saves list after all operations complete.
        """
        if self._pending_saves:
            await asyncio.gather(*self._pending_saves)
            self._pending_saves.clear()
