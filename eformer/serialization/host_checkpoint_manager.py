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
import typing as tp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
from safetensors import flax as safe_flax
from tqdm.autonotebook import tqdm

from eformer.loggings import get_logger
from eformer.pytree import flatten_dict, is_flatten, serialization, unflatten_dict

from .checkpoint_manager import CheckpointManager
from .loader import SerializationLoader
from .utils import (
    apply_gather_functions,
    derive_base_prefix_from_path,
    flatten_for_broadcast,
    group_keys_by_shard_size,
    index_filename,
    optimize_shard_layout,
    put_dtype,
    shard_filename,
    to_host,
)

logger = get_logger(__name__)


class HostCheckpointManager(CheckpointManager):
    def __init__(
        self,
        checkpoint_dir: str | os.PathLike,
        enable: bool | None = None,
        float_dtype: jnp.dtype = jnp.bfloat16,
        save_optimizer_state: bool = True,
        verbose: bool = False,
        gcs_bucket: str | None = None,
        gcs_credentials_path: str | None = None,
        enable_single_replica_loading: bool = True,
        memory_limit_mb: int | None = None,
        max_concurrent_shards: int = 4,
        use_async_io: bool = True,
        optimize_shard_layout: bool = True,
    ):
        """
        Initialize host-aware checkpoint manager.

        New parameters:
            enable_single_replica_loading: Use single-replica broadcast optimization
            memory_limit_mb: Memory limit in MB for chunking operations
            max_concurrent_shards: Maximum shards to load/save concurrently
            use_async_io: Enable async I/O operations
            optimize_shard_layout: Optimize tensor grouping in shards
        """
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            enable=enable,
            float_dtype=float_dtype,
            save_optimizer_state=save_optimizer_state,
            verbose=verbose,
            gcs_bucket=gcs_bucket,
            gcs_credentials_path=gcs_credentials_path,
        )

        self.enable_single_replica_loading = enable_single_replica_loading
        self.memory_limit_bytes = (memory_limit_mb * 1024 * 1024) if memory_limit_mb else None
        self.max_concurrent_shards = max_concurrent_shards
        self.use_async_io = use_async_io
        self.optimize_shard_layout = optimize_shard_layout
        self.loader = SerializationLoader(
            memory_limit_bytes=self.memory_limit_bytes,
            enable_single_replica_loading=enable_single_replica_loading,
            max_workers=max_concurrent_shards,
        )

    @classmethod
    def load_checkpoint_optimized(
        cls,
        path: str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        gcs_client=None,
        # New optimization parameters
        enable_single_replica: bool = True,
        memory_limit_mb: int | None = None,
        max_concurrent: int = 4,
    ) -> tuple[dict, dict]:
        """
        Load checkpoint with optimizations.

        Key optimizations:
        1. Single-replica loading and broadcasting
        2. Concurrent shard loading
        3. Memory-aware streaming
        """
        # Create optimized loader instance
        loader = SerializationLoader(
            memory_limit_bytes=(memory_limit_mb * 1024 * 1024) if memory_limit_mb else None,
            enable_single_replica_loading=enable_single_replica,
            max_workers=max_concurrent,
        )

        # Check if we should use single-replica optimization
        if enable_single_replica and jax.process_count() > 1:
            return cls._load_with_single_replica_broadcast(
                path=path,
                loader=loader,
                shard_fns=shard_fns,
                verbose=verbose,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
                gcs_client=gcs_client,
            )

        # Fall back to standard loading with concurrency optimizations
        return cls._load_with_concurrency(
            path=path,
            shard_fns=shard_fns,
            verbose=verbose,
            mismatch_allowed=mismatch_allowed,
            callback=callback,
            dtype=dtype,
            gcs_client=gcs_client,
            max_concurrent=max_concurrent,
        )

    @classmethod
    def _load_with_single_replica_broadcast(
        cls,
        path: str,
        loader: SerializationLoader,
        shard_fns: dict | None,
        verbose: bool,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
        gcs_client,
    ) -> tuple[dict, dict]:
        """Load checkpoint using single-replica broadcast optimization."""

        # Only process 0 loads from disk
        if jax.process_index() == 0:
            if verbose:
                logger.info("Process 0 loading checkpoint for broadcast")

            # Load checkpoint metadata and structure
            state, metadata = cls.load_checkpoint(
                path=path,
                shard_fns=shard_fns,
                verbose=verbose,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
                gcs_client=gcs_client,
            )

            # Prepare state for broadcasting
            flat_state = flatten_for_broadcast(state)
        else:
            if verbose:
                logger.info(f"Process {jax.process_index()} waiting for broadcast")
            flat_state = None
            metadata = None

        # Broadcast state structure first (lightweight)
        if jax.process_index() == 0:
            state_structure = {k: (v.shape, v.dtype) for k, v in flat_state.items()}
        else:
            state_structure = None

        state_structure = jax.experimental.multihost_utils.broadcast_one_to_all(state_structure)

        metadata = jax.experimental.multihost_utils.broadcast_one_to_all(metadata)

        broadcasted_state = {}
        for key, (shape, tensor_dtype) in state_structure.items():
            if jax.process_index() == 0:
                tensor = flat_state[key]
            else:
                tensor = jnp.zeros(shape, dtype=tensor_dtype)

            broadcasted_state[key] = loader.broadcast_tensor(tensor)

        state = unflatten_dict(broadcasted_state, sep=".")

        return state, metadata

    @classmethod
    def _load_with_concurrency(
        cls,
        path: str,
        shard_fns: dict | None,
        verbose: bool,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
        gcs_client,
        max_concurrent: int = 4,
    ) -> tuple[dict, dict]:
        """Load checkpoint with concurrent shard loading."""

        path_str = str(path)
        base_prefix = derive_base_prefix_from_path(path_str)
        index_path = index_filename(base_prefix)

        # Check if sharded
        if os.path.exists(index_path) or path_str.endswith(".safetensors.index.json"):
            return cls._load_sharded_concurrent(
                index_path if not path_str.endswith(".safetensors.index.json") else path_str,
                shard_fns=shard_fns,
                verbose=verbose,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
                max_concurrent=max_concurrent,
            )

        # Single file - use standard loading
        return cls.load_checkpoint(
            path=path,
            shard_fns=shard_fns,
            verbose=verbose,
            mismatch_allowed=mismatch_allowed,
            callback=callback,
            dtype=dtype,
            gcs_client=gcs_client,
        )

    @classmethod
    def _load_sharded_concurrent(
        cls,
        index_path: str,
        shard_fns: dict | None,
        verbose: bool,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
        max_concurrent: int = 4,
    ) -> tuple[dict, dict]:
        """Load sharded checkpoint with concurrent shard loading."""

        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        directory = os.path.dirname(index_path)

        # Group keys by shard
        file_to_keys = defaultdict(list)
        for k, shard_name in weight_map.items():
            file_to_keys[shard_name].append(k)

        if shard_fns and not is_flatten(shard_fns):
            shard_fns = flatten_dict(shard_fns, sep=".")

        state = {}
        mismatch_count = 0

        def load_shard(shard_name: str, keys: list[str]) -> tuple[dict, int]:
            """Load a single shard."""
            shard_path = os.path.join(directory, shard_name)
            shard_state = {}
            shard_mismatch = 0

            with safe_flax.safe_open(shard_path, framework="flax") as manager:
                for key in keys:
                    tensor = manager.get_tensor(key)

                    # Apply shard function if available
                    if shard_fns:
                        func = shard_fns.get(key)
                        if func:
                            tensor = func(tensor)
                        elif not mismatch_allowed:
                            raise KeyError(f"Shard function for {key} not found")
                        else:
                            shard_mismatch += 1

                    # Apply callback
                    if callback:
                        tensor = callback(tensor, key)

                    # Apply dtype conversion
                    if dtype:
                        tensor = put_dtype(tensor, dtype)

                    shard_state[key] = tensor

            return shard_state, shard_mismatch

        # Load shards concurrently
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(load_shard, shard_name, keys): shard_name for shard_name, keys in file_to_keys.items()
            }

            pbar = tqdm(total=len(file_to_keys), desc="Loading shards concurrently", disable=not verbose)

            for future in as_completed(futures):
                shard_name = futures[future]
                try:
                    shard_state, shard_mismatch = future.result()
                    state.update(shard_state)
                    mismatch_count += shard_mismatch
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to load shard {shard_name}: {e}")
                    raise

            pbar.close()

        if verbose and mismatch_count:
            logger.info(f"Total sharding mismatches: {mismatch_count}")

        state = unflatten_dict(state, sep=".")
        metadata = index_data.get("metadata", {})

        return state, metadata

    @classmethod
    def save_checkpoint_optimized(
        cls,
        state: dict,
        path: str | os.PathLike,
        gather_fns: dict | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        metadata: dict | None = None,
        shard_size_gb: float = 5.0,
        optimize_layout: bool = True,
        max_concurrent: int = 4,
    ) -> str:
        """
        Save checkpoint with optimizations.

        Key optimizations:
        1. Optimized shard layout for better loading
        2. Concurrent shard writing
        3. Direct device-to-disk streaming
        """

        state = serialization.to_state_dict(state)
        if not is_flatten(state):
            state = flatten_dict(state, sep=".")

        # Apply gather functions if needed
        if gather_fns:
            state = apply_gather_functions(state, gather_fns, mismatch_allowed, verbose)

        # Apply dtype conversion
        if float_dtype:
            state = jax.tree_util.tree_map(
                lambda x: to_host(x, float_dtype),
                state,
                is_leaf=lambda x: isinstance(x, (jax.Array, np.generic, float, int)),  # noqa
            )

        # Optimize shard layout if requested
        if optimize_layout and shard_size_gb:
            max_bytes = int(shard_size_gb * (1024**3))
            shards = optimize_shard_layout(state, max_bytes)
        else:
            # Use standard sharding
            max_bytes = int(shard_size_gb * (1024**3)) if shard_size_gb else None
            shards = group_keys_by_shard_size(state, max_bytes) if max_bytes else [[k for k in state.keys()]]

        # Save shards concurrently
        base_prefix = derive_base_prefix_from_path(str(path))
        total_shards = len(shards)

        if total_shards > 1:
            cls._save_sharded_concurrent(
                state=state,
                base_prefix=base_prefix,
                shards=shards,
                total_shards=total_shards,
                metadata=metadata,
                verbose=verbose,
                max_concurrent=max_concurrent,
            )

            # Create index file
            weight_map = {}
            for i, shard_keys in enumerate(shards, start=1):
                shard_name = os.path.basename(shard_filename(base_prefix, i, total_shards))
                for k in shard_keys:
                    weight_map[k] = shard_name

            index_data = {"metadata": metadata or {}, "weight_map": weight_map}
            index_path = index_filename(base_prefix)

            with open(index_path, "w") as f:
                json.dump(index_data, f, ensure_ascii=False)

            return index_path
        else:
            # Single shard - save directly
            safe_flax.save_file(tensors=state, filename=str(path), metadata=metadata)
            return str(path)

    @classmethod
    def _save_sharded_concurrent(
        cls,
        state: dict,
        base_prefix: str,
        shards: list[list[str]],
        total_shards: int,
        metadata: dict | None,
        verbose: bool,
        max_concurrent: int = 4,
    ):
        """Save shards concurrently."""

        def save_shard(shard_idx: int, shard_keys: list[str]):
            """Save a single shard."""
            shard_path = shard_filename(base_prefix, shard_idx, total_shards)
            shard_tensors = {k: state[k] for k in shard_keys}
            safe_flax.save_file(tensors=shard_tensors, filename=shard_path, metadata=metadata)
            return shard_path

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(save_shard, i, keys): i for i, keys in enumerate(shards, start=1)}

            pbar = tqdm(total=total_shards, desc="Saving shards concurrently", disable=not verbose)

            for future in as_completed(futures):
                shard_idx = futures[future]
                try:
                    shard_path = future.result()
                    pbar.update(1)
                    if verbose:
                        pbar.set_postfix({"last_shard": os.path.basename(shard_path)})
                except Exception as e:
                    logger.error(f"Failed to save shard {shard_idx}: {e}")
                    raise

            pbar.close()
