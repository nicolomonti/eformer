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

import typing as tp
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.experimental.multihost_utils
from jax.sharding import NamedSharding
from safetensors import flax as safe_flax

from eformer.loggings import get_logger

from .utils import broadcast_tensor as _broadcast_tensor
from .utils import estimate_available_memory, optimize_shard_layout

logger = get_logger(__name__)


class SerializationLoader:
    def __init__(
        self,
        memory_limit_bytes: int | None = None,
        enable_single_replica_loading: bool = True,
        prefetch_shards: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize optimized checkpoint loader.

        Args:
            memory_limit_bytes: Maximum memory to use for broadcasting chunks
            enable_single_replica_loading: Use single-replica broadcasting optimization
            prefetch_shards: Enable prefetching of next shard while processing current
            max_workers: Maximum thread workers for concurrent operations
        """
        self.memory_limit_bytes = memory_limit_bytes or estimate_available_memory()
        self.enable_single_replica_loading = enable_single_replica_loading
        self.prefetch_shards = prefetch_shards
        self.max_workers = max_workers

    def broadcast_tensor(
        self,
        tensor: jax.Array,
        target_sharding: NamedSharding | None = None,
    ) -> jax.Array:
        """
        Efficiently broadcast tensor from single replica to all devices.

        Uses chunking if tensor is too large for available memory.
        """
        if not self.enable_single_replica_loading:
            return tensor

        return _broadcast_tensor(tensor, self.memory_limit_bytes, target_sharding)

    def load_sharded_checkpoint_optimized(
        self,
        shard_paths: list[str],
        shard_keys: dict[str, list[str]],
        target_shardings: dict[str, NamedSharding] | None = None,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
    ) -> dict[str, jax.Array]:
        """
        Load sharded checkpoint with optimizations.

        Features:
        - Single-replica loading (only process 0 reads from disk)
        - Concurrent shard prefetching
        - Memory-aware chunking
        - Efficient broadcasting
        """
        state = {}

        if self.enable_single_replica_loading and jax.process_index() != 0:
            logger.info("Process %d waiting for checkpoint broadcast", jax.process_index())
            jax.experimental.multihost_utils.sync_global_devices("checkpoint_load_sync")

            for shard_path in shard_paths:
                for _ in shard_keys[shard_path]:
                    ...  # TODO:Impl this
            return state

        logger.info("Process 0 loading checkpoint for broadcast")

        def load_shard(shard_path: str) -> dict[str, jax.Array]:
            """Load a single shard file."""
            shard_state = {}
            with safe_flax.safe_open(shard_path, framework="flax") as f:
                for key in shard_keys[shard_path]:
                    tensor = f.get_tensor(key)
                    if callback:
                        tensor = callback(tensor, key)
                    shard_state[key] = tensor
            return shard_state

        if self.prefetch_shards and len(shard_paths) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(shard_paths))) as executor:
                futures = [executor.submit(load_shard, path) for path in shard_paths]

                for future in futures:
                    shard_state = future.result()
                    for key, tensor in shard_state.items():
                        target_sharding = target_shardings.get(key) if target_shardings else None
                        state[key] = self.broadcast_tensor(tensor, target_sharding)
        else:
            for shard_path in shard_paths:
                shard_state = load_shard(shard_path)
                for key, tensor in shard_state.items():
                    target_sharding = target_shardings.get(key) if target_shardings else None
                    state[key] = self.broadcast_tensor(tensor, target_sharding)

        jax.experimental.multihost_utils.sync_global_devices("checkpoint_load_complete")

        return state

    optimize_shard_layout = staticmethod(optimize_shard_layout)
