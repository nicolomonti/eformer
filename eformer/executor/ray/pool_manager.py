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

"""Resource pool management for distributed Ray actors.

This module provides abstractions for managing pools of Ray actors,
particularly focused on TPU/GPU slice management for distributed computing.
It includes health monitoring, automatic scaling, and resource lifecycle
management.

Key Components:
    - **ActorPoolMember**: Wrapper for actor handles with metadata
    - **ResourcePoolManager**: Abstract base for managing actor pools
    - **SlicePoolManager**: Specialized manager for TPU/GPU slices
    - **_SliceActor**: Ray actor for managing individual compute slices

Example:
    Managing a multi-slice TPU configuration:

    >>> from eformer.executor.ray import SlicePoolManager
    >>>
    >>> # Create a pool manager for TPU v4 slices
    >>> manager = SlicePoolManager(tpu_type="v4-8")
    >>>
    >>> # Scale to 4 slices
    >>> manager.scale_multislice(num_slices=4)
    >>>
    >>> # Get all slice actors
    >>> actors = manager.get_all_actors_in_pool()
    >>>
    >>> # Clean up when done
    >>> manager.drain_actor_pool()
"""

from __future__ import annotations

import logging
import socket
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import ray
from ray._private.accelerators import TPUAcceleratorManager
from ray.actor import ActorHandle
from ray.exceptions import ActorDiedError, ActorUnavailableError, GetTimeoutError
from ray.remote_function import RemoteFunction
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .types import MultisliceInfo, SliceInfo

logger = logging.getLogger("ray")

_HEALTH_CHECK_TIMEOUT = 60
ActorInfoT = TypeVar("ActorInfoT")


@dataclass(frozen=True)
class ActorPoolMember(Generic[ActorInfoT]):
    """Wrapper for Ray actor handles with associated metadata.

    Provides a typed container for pairing Ray actors with their
    configuration and runtime information.

    Attributes:
        actor (ActorHandle): The Ray actor handle for remote execution.
        actor_info (ActorInfoT): Generic metadata about the actor,
            typically containing configuration and identification information.

    Type Parameters:
        ActorInfoT: Type of the actor information/metadata.

    Example:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        >>> class WorkerInfo:
        ...     worker_id: int
        ...     resource_type: str
        >>>
        >>> worker = ray.remote(MyWorker).remote()
        >>> info = WorkerInfo(worker_id=1, resource_type="gpu")
        >>> member = ActorPoolMember(actor=worker, actor_info=info)
    """

    actor: ActorHandle
    actor_info: ActorInfoT


class ResourcePoolManager(ABC, Generic[ActorInfoT]):
    """Abstract base class for managing pools of Ray actors.

    Provides a framework for managing distributed resources with automatic
    health monitoring, scaling, and lifecycle management. Subclasses must
    implement specific actor creation and naming strategies.

    This class handles:
        - Actor pool scaling (up and down)
        - Health monitoring and automatic cleanup
        - Placement group management
        - Resource lifecycle management

    Type Parameters:
        ActorInfoT: Type of metadata associated with each actor.

    Attributes:
        _actor_pool (list[ActorPoolMember[ActorInfoT]]): Current pool members.
        _placement_groups (dict[str, PlacementGroup]): Placement groups by name.
    """

    def __init__(self):
        """Initialize an empty resource pool."""
        self._actor_pool: list[ActorPoolMember[ActorInfoT]] = []
        self._placement_groups: dict[str, PlacementGroup] = {}

    @abstractmethod
    def get_actor_pool_name(self) -> str:
        """Get the unique name identifying this actor pool.

        Returns:
            str: Unique pool identifier.
        """
        pass

    @abstractmethod
    def get_actor_name_from_actor_info(self, actor_info: ActorInfoT) -> str:
        """Generate a unique name for an actor based on its metadata.

        Args:
            actor_info (ActorInfoT): Actor metadata.

        Returns:
            str: Unique actor name.
        """
        pass

    @abstractmethod
    def create_actor(self, actor_info: ActorInfoT) -> ActorHandle:
        """Create a new Ray actor with the specified configuration.

        Args:
            actor_info (ActorInfoT): Configuration and metadata for the actor.

        Returns:
            ActorHandle: Handle to the newly created Ray actor.
        """
        pass

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        """Get all Ray actor handles currently in the pool.

        Returns:
            list[ActorHandle]: List of all active actor handles.
        """
        return [member.actor for member in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember[ActorInfoT]]:
        """Get all pool members including actors and their metadata.

        Returns:
            list[ActorPoolMember[ActorInfoT]]: List of all pool members
                with both actor handles and associated metadata.
        """
        return self._actor_pool

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        """Remove unhealthy or unresponsive actors from the pool.

        Performs health checks on all actors and removes those that:
            - Have died (ActorDiedError)
            - Are unavailable (ActorUnavailableError)
            - Don't respond within the timeout period

        Also cleans up associated placement groups for removed actors.
        """
        healthy_members = []
        for member in self._actor_pool:
            try:
                # Check if actor is still alive
                ray.get(member.actor.healthy.remote(), timeout=_HEALTH_CHECK_TIMEOUT)
                healthy_members.append(member)
            except (ActorDiedError, ActorUnavailableError, GetTimeoutError) as e:
                logger.warning(f"Removing unhealthy actor {self.get_actor_name_from_actor_info(member.actor_info)}: {e}")
                # Clean up placement group if exists
                actor_name = self.get_actor_name_from_actor_info(member.actor_info)
                if actor_name in self._placement_groups:
                    try:
                        remove_placement_group(self._placement_groups[actor_name])
                    except Exception as cleanup_error:
                        logger.error(f"Failed to remove placement group for {actor_name}: {cleanup_error}")
                    del self._placement_groups[actor_name]

        self._actor_pool = healthy_members

    def _add_members_to_actor_pool(self, desired_num_actors: int, actor_infos: list[ActorInfoT]) -> None:
        """Add new actors to the pool to reach the desired count.

        Creates and adds new actors until the pool size matches the desired
        number. Actors are created based on the provided configurations.

        Args:
            desired_num_actors (int): Target number of actors in the pool.
            actor_infos (list[ActorInfoT]): Configurations for creating new actors.
                Should contain at least (desired_num_actors - current_count) items.
        """
        current_count = len(self._actor_pool)
        if current_count >= desired_num_actors:
            return

        num_to_add = desired_num_actors - current_count
        for i in range(num_to_add):
            if i < len(actor_infos):
                actor_info = actor_infos[i]
                try:
                    actor = self.create_actor(actor_info)
                    self._actor_pool.append(ActorPoolMember(actor, actor_info))
                    logger.info(f"Added actor {self.get_actor_name_from_actor_info(actor_info)} to pool")
                except Exception as e:
                    logger.error(f"Failed to create actor: {e}")

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        """Remove excess actors from the pool to reach the desired count.

        Removes actors from the end of the pool (LIFO) until the pool size
        matches the desired number. Also cleans up associated resources.

        Args:
            desired_num_actors (int): Target number of actors in the pool.
        """
        while len(self._actor_pool) > desired_num_actors:
            member = self._actor_pool.pop()
            actor_name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                ray.kill(member.actor)
            except Exception as e:
                logger.error(f"Failed to kill actor {actor_name}: {e}")

            # Clean up placement group
            if actor_name in self._placement_groups:
                try:
                    remove_placement_group(self._placement_groups[actor_name])
                except Exception as e:
                    logger.error(f"Failed to remove placement group for {actor_name}: {e}")
                del self._placement_groups[actor_name]

    def _scale_actor_pool(self, desired_num_actors: int, actor_infos: list[ActorInfoT]) -> None:
        """Scale the actor pool to the desired size.

        Adjusts the pool size by:
            1. Removing unhealthy actors
            2. Adding new actors if below target
            3. Removing excess actors if above target

        Args:
            desired_num_actors (int): Target number of actors.
            actor_infos (list[ActorInfoT]): Configurations for new actors if scaling up.
        """
        self._remove_unhealthy_members_from_actor_pool()
        current_count = len(self._actor_pool)

        if current_count < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors, actor_infos)
        elif current_count > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)

    def drain_actor_pool(self) -> None:
        """Remove all actors from the pool and clean up resources.

        Completely empties the actor pool by:
            1. Killing all actors
            2. Removing all placement groups
            3. Clearing internal data structures

        This method is typically called during shutdown or reset.
        """
        for member in self._actor_pool:
            actor_name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                ray.kill(member.actor)
            except Exception as e:
                logger.error(f"Failed to kill actor {actor_name}: {e}")

        # Clean up all placement groups
        for pg_name, pg in self._placement_groups.items():
            try:
                remove_placement_group(pg)
            except Exception as e:
                logger.error(f"Failed to remove placement group {pg_name}: {e}")

        self._actor_pool = []
        self._placement_groups = {}


class SlicePoolManager(ResourcePoolManager[SliceInfo]):
    """Manager for TPU/GPU slice actors in multi-slice configurations.

    Specialized resource pool manager for handling distributed TPU/GPU slices,
    providing coordination for multi-slice workloads and automatic scaling.

    This manager handles:
        - TPU/GPU slice creation and configuration
        - Multi-slice coordination setup
        - Automatic resource discovery and allocation
        - Health monitoring across slices

    Attributes:
        tpu_type (str): Type of TPU being managed (e.g., "v4-8", "v4-32").
        _last_scale_check_time (float): Timestamp of last scaling check.
        _last_scale_up_time (float): Timestamp of last scale-up operation.

    Example:
        >>> manager = SlicePoolManager("v4-32")
        >>> manager.scale_multislice(4)  # Create 4 TPU slices
        >>> actors = manager.get_all_actors_in_pool()
    """

    def __init__(self, tpu_type: str):
        """Initialize the slice pool manager.

        Args:
            tpu_type (str): TPU type identifier (e.g., "v4-8", "v4-32").
        """
        super().__init__()
        self.tpu_type = tpu_type
        self._last_scale_check_time = 0
        self._last_scale_up_time = 0

    def get_actor_pool_name(self) -> str:
        """Get the unique name for this slice pool.

        Returns:
            str: Pool name in format "slice_pool_{tpu_type}".
        """
        return f"slice_pool_{self.tpu_type}"

    def get_actor_name_from_actor_info(self, actor_info: SliceInfo) -> str:
        """Get the unique name for a slice actor.

        Args:
            actor_info (SliceInfo): Slice information.

        Returns:
            str: The slice name from the SliceInfo.
        """
        return actor_info.slice_name

    def create_actor(self, actor_info: SliceInfo) -> ActorHandle:
        """Create a SliceActor for managing a TPU/GPU slice.

        Creates a Ray actor with appropriate resource requirements and
        placement group for the slice. The actor is scheduled with
        strict spread strategy across hosts.

        Args:
            actor_info (SliceInfo): Configuration for the slice.

        Returns:
            ActorHandle: Handle to the created SliceActor.
        """
        # Create placement group for the slice
        bundles = []
        for _ in range(actor_info.num_hosts):
            bundles.append(
                {
                    "CPU": 1,
                    self.tpu_type: actor_info.num_accelerators_per_host,
                }
            )

        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        self._placement_groups[actor_info.slice_name] = pg

        # Create the slice actor with placement group scheduling
        SliceActor = ray.remote(
            num_cpus=0,
            resources={f"{actor_info.slice_name}-head": 1},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
            ),
        )(_SliceActor)

        return SliceActor.remote(actor_info, self.tpu_type, pg)

    def scale_multislice(self, num_slices: int | Sequence[int]) -> None:
        """Scale the multi-slice configuration to the desired number.

        Adjusts the number of active slices in the pool, creating or removing
        slices as needed. Also sets up multi-slice coordination if needed.

        Args:
            num_slices (int | Sequence[int]): Either a specific target number
                of slices, or a sequence of acceptable slice counts (will choose
                the maximum feasible value).

        Example:
            >>> manager.scale_multislice(4)  # Scale to exactly 4 slices
            >>> manager.scale_multislice([2, 4, 8])  # Use best available option
        """
        if isinstance(num_slices, Sequence):
            # Find the optimal slice count based on available resources
            available_slices = self._get_available_slice_count()
            target_slices = max(n for n in num_slices if n <= available_slices)
        else:
            target_slices = num_slices

        # Generate slice infos for the target configuration
        slice_infos = self._generate_slice_infos(target_slices)

        # Scale the actor pool
        self._scale_actor_pool(target_slices, slice_infos)

        # Update coordination information for multi-slice setup
        if target_slices > 1:
            self._setup_multislice_coordination()

    def _generate_slice_infos(self, num_slices: int) -> list[SliceInfo]:
        """Generate SliceInfo configurations for the specified number of slices.

        Creates slice configurations based on available TPU/GPU resources,
        with the first slice using current pod information.

        Args:
            num_slices (int): Number of slices to generate info for.

        Returns:
            list[SliceInfo]: List of slice configurations.
        """
        slice_infos = []
        for i in range(num_slices):
            slice_name = ray.util.accelerators.tpu.get_current_pod_name() if i == 0 else f"tpu-{self.tpu_type}-{i}"
            num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
            ip_address = socket.gethostbyname(socket.gethostname()) if i == 0 else None
            num_accelerators = TPUAcceleratorManager.get_current_node_num_accelerators()

            slice_infos.append(
                SliceInfo(
                    slice_name=slice_name,
                    num_hosts=num_hosts,
                    ip_address=ip_address or "unknown",
                    num_accelerators_per_host=num_accelerators // num_hosts if num_hosts > 0 else num_accelerators,
                )
            )

        return slice_infos

    def _get_available_slice_count(self) -> int:
        """Determine the number of slices available in the cluster.

        Returns:
            int: Number of available slices (current pool size + 1).
        """
        return len(self._actor_pool) + 1

    def _setup_multislice_coordination(self) -> None:
        """Configure coordination between multiple slices.

        Sets up the multi-slice communication by:
            1. Designating the first slice as coordinator
            2. Distributing coordinator information to all slices
            3. Assigning slice IDs and port configurations

        The coordinator slice handles cross-slice synchronization.
        """
        if not self._actor_pool:
            return

        coordinator_member = self._actor_pool[0]
        coordinator_info = coordinator_member.actor_info

        for i, member in enumerate(self._actor_pool):
            multislice_info = MultisliceInfo(
                coordinator_ip=coordinator_info.ip_address,
                slice_id=i,
                num_slices=len(self._actor_pool),
                port=8081,
            )
            try:
                ray.get(member.actor.configure_multislice.remote(multislice_info))
            except Exception as e:
                logger.error(f"Failed to configure multislice for {member.actor_info.slice_name}: {e}")


@ray.remote
class _SliceActor:
    """Ray actor managing a single TPU/GPU compute slice.

    Handles resource allocation, health monitoring, task execution, and
    multi-slice coordination for a single slice in a distributed setup.
    Each slice actor manages its own placement group and can execute
    tasks with slice-specific environment variables.

    Attributes:
        slice_info (SliceInfo): Configuration and metadata for this slice.
        resource_type (str): Type of accelerator resource (e.g., "v4-8").
        placement_group (PlacementGroup): Ray placement group for this slice.
        multislice_info (MultisliceInfo | None): Multi-slice coordination info.
        _is_healthy (bool): Current health status of the actor.
        _tasks (list): List of running tasks on this slice.

    Note:
        This class is decorated with @ray.remote and should not be
        instantiated directly. Use through SlicePoolManager instead.
    """

    def __init__(self, slice_info: SliceInfo, resource_type: str, placement_group: PlacementGroup):
        """Initialize the slice actor.

        Args:
            slice_info (SliceInfo): Configuration for this slice.
            resource_type (str): Type of accelerator resource.
            placement_group (PlacementGroup): Placement group for resource allocation.
        """
        self.slice_info = slice_info
        self.resource_type = resource_type
        self.placement_group = placement_group
        self.multislice_info: MultisliceInfo | None = None
        self._is_healthy = True
        self._tasks = []

    def healthy(self) -> bool:
        """Check the health status of this slice actor.

        Returns:
            bool: True if the actor is healthy, False otherwise.
        """
        return self._is_healthy

    def is_being_preempted(self) -> bool:
        """Check if this slice is being preempted.

        Returns:
            bool: True if preemption is detected, False otherwise.

        Note:
            Currently returns False. Override for preemption detection.
        """
        # This would check for preemption signals
        # For now, return False
        return False

    def configure_multislice(self, multislice_info: MultisliceInfo) -> None:
        """Configure this slice for multi-slice coordination.

        Sets up the necessary information for this slice to participate
        in a multi-slice execution, including coordinator address and
        slice ID assignment.

        Args:
            multislice_info (MultisliceInfo): Coordination information including
                coordinator IP, slice ID, and port configuration.
        """
        self.multislice_info = multislice_info

    def get_slice_info(self) -> tuple[str, int, str]:
        """Get slice information for multi-slice coordination.

        Returns:
            tuple[str, int, str]: Tuple containing:
                - slice_name: Unique identifier for this slice
                - num_hosts: Number of hosts in this slice
                - ip_address: IP address of this slice
        """
        return (
            self.slice_info.slice_name,
            self.slice_info.num_hosts,
            self.slice_info.ip_address,
        )

    def run_task(self, remote_fn: RemoteFunction, runtime_env: dict, **kwargs) -> ray.ObjectRef:
        """Execute a Ray task on this slice.

        Runs the provided remote function with slice-specific environment
        variables and placement group scheduling. Automatically adds
        multi-slice coordination environment variables if configured.

        Args:
            remote_fn (RemoteFunction): Ray remote function to execute.
            runtime_env (dict): Runtime environment configuration.
            **kwargs: Arguments to pass to the remote function.

        Returns:
            ray.ObjectRef: Future representing the task result.

        Example:
            >>> future = slice_actor.run_task.remote(
            ...     my_remote_fn,
            ...     runtime_env={"pip": ["numpy"]},
            ...     data=training_data
            ... )
            >>> result = ray.get(future)
        """
        env_vars = {}
        if self.multislice_info:
            env_vars.update(
                {
                    "MEGASCALE_COORDINATOR_ADDRESS": f"{self.multislice_info.coordinator_ip}:"
                    f"{self.multislice_info.port}",
                    "MEGASCALE_NUM_SLICES": str(self.multislice_info.num_slices),
                    "MEGASCALE_PORT": str(self.multislice_info.port),
                    "MEGASCALE_SLICE_ID": str(self.multislice_info.slice_id),
                }
            )

        final_runtime_env = {**runtime_env}
        if env_vars:
            final_runtime_env["env_vars"] = {**final_runtime_env.get("env_vars", {}), **env_vars}

        task_options = {
            "runtime_env": final_runtime_env,
            "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=self.placement_group),
        }

        if isinstance(remote_fn, RemoteFunction):
            task = remote_fn.options(**task_options).remote(**kwargs)
        else:
            task = ray.remote(remote_fn).options(**task_options).remote(**kwargs)

        self._tasks.append(task)
        return task

    def cancel_tasks(self) -> None:
        """Cancel all running tasks on this slice.

        Attempts to cancel all tasks that were started through run_task().
        Errors during cancellation are logged but not raised.
        """
        for task in self._tasks:
            try:
                ray.cancel(task)
            except Exception as e:
                logger.error(f"Failed to cancel task: {e}")
        self._tasks = []
