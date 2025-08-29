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

"""Ray-based executor for distributed machine learning workloads.

This module provides the core execution framework for running distributed
workloads on accelerators (TPUs, GPUs) using Ray. It supports single-pod,
multi-slice, and fault-tolerant execution patterns with automatic retry
mechanisms.

Key Features:
    - Single-pod and multi-slice execution on TPUs/GPUs
    - Automatic retry mechanisms for preemption and failures
    - Resource management and allocation via Ray
    - Support for both synchronous and asynchronous execution
    - Decorator-based API for easy integration

Example:
    Basic single-pod execution:

    >>> import ray
    >>> from eformer.executor.ray import RayExecutor, TpuAcceleratorConfig
    >>>
    >>> @ray.remote
    >>> def train_model(data):
    ...     # Training logic here
    ...     return trained_model
    >>>
    >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
    >>> result = RayExecutor.execute_resumable(
    ...     train_model,
    ...     tpu_config,
    ...     max_retries_preemption=10,
    ...     max_retries_failure=3
    ... )

    Multi-slice execution with decorator:

    >>> from eformer.executor.ray import execute_multislice_resumable
    >>>
    >>> @execute_multislice_resumable(tpu_config)
    >>> @ray.remote
    >>> def distributed_train(slice_data):
    ...     # Distributed training logic
    ...     return slice_results
    >>>
    >>> results = distributed_train(training_data)
"""

import functools
import logging

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from .pool_manager import SlicePoolManager
from .resource_manager import (
    AcceleratorConfigType,
    RayResources,
    TpuAcceleratorConfig,
)
from .types import (
    JobError,
    JobFailed,
    JobInfo,
    JobPreempted,
    JobStatus,
    JobSucceeded,
    handle_ray_error,
)

ENV_CALL_INDEX = "EXECUTOR_CALL_INDEX"
ENV_CALL_SLICE = "EXECUTOR_CALL_SLICE"


MEGASCALE_DEFAULT_PORT = 8081

logger = logging.getLogger("ray")


class RayExecutor:
    """Core executor for Ray-based distributed workloads.

    Provides static methods to execute Ray remote functions on various
    accelerators (TPUs, GPUs) with support for single-pod, multi-slice,
    and fault-tolerant execution patterns.

    This class serves as the main interface for running distributed ML
    workloads with automatic resource allocation, retry mechanisms, and
    failure handling.

    Note:
        All methods are static and can be called directly on the class.
    """

    @staticmethod
    def execute(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        **kwargs,
    ):
        """Execute a Ray remote function on a single pod or slice.

        Runs a Ray remote function on a single accelerator pod (TPU/GPU)
        with the specified resource configuration. For multi-slice TPU
        workloads, use execute_multislice instead.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute.
                Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources (TPU, GPU, or CPU).
            **kwargs: Additional keyword arguments passed to the remote function.

        Returns:
            ray.ObjectRef: A Ray future representing the result of execution.
                Call ray.get() on this to retrieve the actual result.

        Raises:
            AssertionError: If pod_count in accelerator_config is not 1,
                indicating that execute_multislice should be used instead.

        Example:
            >>> @ray.remote
            >>> def compute(x):
            ...     return x * 2
            >>>
            >>> config = GpuAcceleratorConfig(count=1, type="v100")
            >>> future = RayExecutor.execute(compute, config, x=10)
            >>> result = ray.get(future)  # Returns JobStatus object
        """
        assert getattr(accelerator_config, "pod_count", 1) == 1, (
            "Multi-slice workloads on TPUs should use 'execute_multislice'."
        )

        def do_run(
            remote_fn,
            accelerator_config: AcceleratorConfigType,
            kwargs,
        ) -> JobStatus:
            """Internal function to run the remote function with proper resource allocation.

            Args:
                remote_fn: The remote function to execute.
                accelerator_config: Accelerator configuration.
                kwargs: Arguments for the remote function.

            Returns:
                JobStatus: Status object indicating success, failure, or preemption.
            """
            info = JobInfo(
                accelerator_config.runtime_name,
                "running",
                accelerator_config.resource_name,
            )
            futures = []
            for idx in range(accelerator_config.worker_count):
                _call = accelerator_config.redecorate_remote_fn_for_call(
                    remote_fn=remote_fn,
                    env_vars={ENV_CALL_INDEX: str(idx)},
                )
                futures.append(_call.remote(**kwargs))
            try:
                out = ray.get(futures)
                return JobSucceeded(info, out)
            except RayError as e:
                RayResources.cancel_all_futures(futures)
                return handle_ray_error(info, e)
            except Exception as e:
                RayResources.cancel_all_futures(futures)
                return JobFailed(info, e)

        if accelerator_config.head_name is None and not isinstance(
            accelerator_config,
            TpuAcceleratorConfig,
        ):
            do_run = ray.remote(do_run)
        else:
            default_name = f"TPU-{accelerator_config.tpu_version}-head"
            resources = {accelerator_config.head_name or default_name: accelerator_config.head_workers}
            do_run = ray.remote(resources=resources)(do_run)
        return do_run.remote(remote_fn, accelerator_config, kwargs)

    @staticmethod
    def execute_multislice(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        **kwargs,
    ) -> list[ray.ObjectRef]:
        """Execute a Ray remote function across multiple TPU slices.

        Distributes execution of a remote function across multiple TPU slices
        for large-scale parallel processing.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute
                on each slice. Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources, must include multi-slice details
                (pod_count > 1).
            **kwargs: Additional keyword arguments passed to the remote
                function on each slice.

        Returns:
            list[ray.ObjectRef]: List of Ray futures, one per slice.
                Each future represents the execution result on that slice.
                The order corresponds to slice IDs (0-indexed).

        Raises:
            RayError: If slice actor creation fails, coordinator IP cannot
                be determined, or remote function calls fail.

        Example:
            >>> @ray.remote
            >>> def train_on_slice(data, slice_id):
            ...     # Training logic for this slice
            ...     return model_weights
            >>>
            >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
            >>> futures = RayExecutor.execute_multislice(
            ...     train_on_slice,
            ...     tpu_config,
            ...     data=training_data
            ... )
            >>> results = ray.get(futures)  # List of JobStatus objects
        """
        num_slices = getattr(accelerator_config, "pod_count", 1)
        pool_manager = SlicePoolManager(accelerator_config.tpu_version)
        pool_manager.scale_multislice(num_slices)
        pool_members = pool_manager.get_all_pool_members()
        if not pool_members:
            raise RayError("Failed to create slice actors in pool")
        futures = []
        for member in pool_members:
            slice_actor = member.actor
            future = slice_actor.run_task.remote(
                remote_fn=remote_fn,
                runtime_env=accelerator_config.execution_env,
                **kwargs,
            )
            futures.append(future)

        return futures

    @classmethod
    def execute_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """Execute a remote function with automatic retry on failures.

        Provides fault-tolerant execution of Ray remote functions with
        configurable retry policies for both preemptions and failures.
        Particularly useful for long-running jobs on preemptible resources.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute.
                Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources.
            max_retries_preemption (int): Maximum number of retries on
                preemption. Defaults to 1,000,000 (effectively unlimited).
            max_retries_failure (int): Maximum number of retries on failure.
                Defaults to 10.
            **kwargs: Additional keyword arguments passed to the remote function.

        Returns:
            Any: The result from successful execution of the remote function.

        Raises:
            RuntimeError: If the job is preempted more than max_retries_preemption
                times or fails more than max_retries_failure times.
            Exception: The last encountered exception if all retries are exhausted.

        Example:
            >>> @ray.remote
            >>> def long_running_task(data):
            ...     # Task that might be preempted
            ...     return process(data)
            >>>
            >>> config = TpuAcceleratorConfig(type="v4-8", preemptible=True)
            >>> result = RayExecutor.execute_resumable(
            ...     long_running_task,
            ...     config,
            ...     max_retries_preemption=100,
            ...     max_retries_failure=5,
            ...     data=my_data
            ... )
        """
        num_failures = 0
        num_preemptions = 0
        attempt = 0
        problem: Exception | None = None

        while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
            logger.info(f"Running on Attempt {attempt}")
            attempt += 1
            problem = None
            try:
                out = ray.get(
                    cls.execute(
                        remote_fn=remote_fn,
                        accelerator_config=accelerator_config,
                        **kwargs,
                    )
                )
            except ray.exceptions.RayTaskError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(f"Preempted {num_preemptions} times, {e}")
                else:
                    num_failures += 1
                    logger.warning(f"Failed {num_failures} times (RayTaskError)", exc_info=e)
                continue
            except Exception as e:
                problem = e
                num_failures += 1
                if num_failures >= max_retries_failure:
                    logger.exception("Failed too many times", exc_info=e)
                    raise e
                else:
                    logger.warning(f"Failed {num_failures} times", exc_info=e)
                    continue

            if isinstance(out, JobSucceeded):
                result = out.result
                logger.info("Success")
                return result
            elif isinstance(out, JobPreempted):
                problem = out.error
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
            elif isinstance(out, JobFailed):
                problem = out.error
                num_failures += 1
                logger.warning(
                    f"JobFailed reported. Incrementing failure count to {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
            elif isinstance(out, JobError):
                problem = out.error
                num_failures += 1
                logger.warning(f"Failed {num_failures} times", exc_info=problem)
            else:
                raise RuntimeError(f"Unexpected result: {out}")

        if num_preemptions >= max_retries_preemption:
            raise RuntimeError("Preempted too many times") from problem
        elif num_failures >= max_retries_failure:
            raise RuntimeError("Failed too many times") from problem

    @classmethod
    def execute_multislice_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """Execute a multi-slice function with automatic retry on failures.

        Provides fault-tolerant execution of Ray remote functions across
        multiple TPU slices with coordinated retry mechanisms. All slices
        must succeed for the execution to be considered successful.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute
                on each slice. Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources with multi-slice support (pod_count > 1).
            max_retries_preemption (int): Maximum number of retries when
                any slice is preempted. Defaults to 1,000,000.
            max_retries_failure (int): Maximum number of retries when any
                slice fails. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the remote
                function on each slice.

        Returns:
            list[Any]: List of results from successful execution on all slices.
                The order corresponds to slice IDs (0-indexed). All slices
                must complete successfully.

        Raises:
            RuntimeError: If any slice is preempted more than max_retries_preemption
                times or fails more than max_retries_failure times.
            RayError: If execute_multislice fails during setup or coordination.
            Exception: The last encountered exception if retries are exhausted.

        Note:
            This method implements an all-or-nothing retry policy. If any
            slice fails or is preempted, the entire multi-slice execution
            is retried.

        Example:
            >>> @ray.remote
            >>> def distributed_training(data_shard):
            ...     # Training logic for each slice
            ...     return trained_weights
            >>>
            >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
            >>> results = RayExecutor.execute_multislice_resumable(
            ...     distributed_training,
            ...     tpu_config,
            ...     max_retries_preemption=50,
            ...     max_retries_failure=3,
            ...     data_shard=sharded_data
            ... )
            >>> # results contains 4 trained_weights, one from each slice
        """
        num_failures = 0
        num_preemptions = 0
        attempt = 0
        problem: Exception | None = None

        while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
            logger.info(f"Running multislice on Attempt {attempt}")
            attempt += 1
            problem = None
            all_slice_job_statuses: list[JobStatus] = []

            try:
                slice_futures: list[ray.ObjectRef] = cls.execute_multislice(
                    remote_fn=remote_fn,
                    accelerator_config=accelerator_config,
                    **kwargs,
                )
                all_slice_job_statuses = ray.get(slice_futures)

            except ray.exceptions.RayTaskError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(
                        f"A slice was preempted (RayTaskError). Preemption count: {num_preemptions}. Error: {e}"
                    )
                else:
                    num_failures += 1
                    logger.warning(f"A slice failed (RayTaskError). Failure count: {num_failures}.", exc_info=e)
                continue
            except RayError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(
                        f"Multislice operation preempted during setup/coordination (RayError). "
                        f"Preemption count: {num_preemptions}. Error: {e}"
                    )
                else:
                    num_failures += 1
                    logger.warning(
                        f"Multislice operation failed during setup/coordination (RayError)."
                        f" Failure count: {num_failures}.",
                        exc_info=e,
                    )
                continue

            except Exception as e:
                problem = e
                num_failures += 1
                if num_failures >= max_retries_failure:
                    logger.exception(
                        "Multislice operation failed too many times (non-Ray/RayTaskError).",
                        exc_info=e,
                    )
                    raise e
                else:
                    logger.warning(
                        f"Multislice operation failed (non-Ray/RayTaskError). Failure count: {num_failures}.",
                        exc_info=e,
                    )
                    continue

            current_attempt_overall_failed = False
            current_attempt_overall_preempted = False
            aggregated_successful_slice_results = [None] * len(all_slice_job_statuses)

            if not all_slice_job_statuses:
                logger.warning(
                    "execute_multislice returned or resulted in empty job statuses after ray.get(). Treating as failure."
                )
                num_failures += 1
                problem = problem or RuntimeError("Empty job statuses from execute_multislice after ray.get()")
                continue

            for slice_idx, single_slice_status in enumerate(all_slice_job_statuses):
                if isinstance(single_slice_status, JobSucceeded):
                    aggregated_successful_slice_results[slice_idx] = single_slice_status.result
                elif isinstance(single_slice_status, JobPreempted):
                    if not problem:
                        problem = single_slice_status.error
                    current_attempt_overall_preempted = True
                    logger.warning(
                        f"Slice {slice_idx} preempted. Error: {single_slice_status.error}",
                        exc_info=single_slice_status.error,
                    )
                elif isinstance(single_slice_status, JobFailed):
                    if not problem:
                        problem = single_slice_status.error
                    current_attempt_overall_failed = True
                    logger.warning(
                        f"Slice {slice_idx} failed (JobFailed). Error: {single_slice_status.error}",
                        exc_info=single_slice_status.error,
                    )
                elif isinstance(single_slice_status, JobError):
                    if not problem:
                        problem = single_slice_status.error
                    current_attempt_overall_failed = True
                    logger.warning(
                        f"Slice {slice_idx} reported JobError. Error: {single_slice_status.error}",
                        exc_info=single_slice_status.error,
                    )
                else:
                    err_msg = (
                        f"Unexpected result type {type(single_slice_status)} "
                        f"from slice {slice_idx}: {single_slice_status}"
                    )
                    if not problem:
                        problem = RuntimeError(err_msg)
                    current_attempt_overall_failed = True
                    logger.error(err_msg)

            if current_attempt_overall_failed:
                num_failures += 1
                logger.warning(
                    f"At least one slice failed or reported an error this attempt. Overall failure "
                    f"count: {num_failures}. Last/first error: {problem}",
                )
                continue

            if current_attempt_overall_preempted:
                num_preemptions += 1
                logger.warning(
                    f"At least one slice was preempted (and no failures) this attempt. Overall preemption count:"
                    f" {num_preemptions}. Last/first error: {problem}",
                )
                continue
            if (
                not current_attempt_overall_failed
                and not current_attempt_overall_preempted
                and all(res is not None for res in aggregated_successful_slice_results)
            ):
                logger.info("All slices succeeded in this attempt.")
                return aggregated_successful_slice_results
            else:
                logger.error(
                    "Inconsistent state in multislice resumable logic after "
                    "processing slice results. Treating as failure."
                )
                num_failures += 1
                problem = problem or RuntimeError("Inconsistent state in multislice resumable after processing results")
                continue

        if num_preemptions >= max_retries_preemption:
            logger.error(f"Multislice job preempted too many times ({num_preemptions} >= {max_retries_preemption}).")
            raise RuntimeError(f"Preempted too many times ({num_preemptions})") from problem
        elif num_failures >= max_retries_failure:
            logger.error(f"Multislice job failed too many times ({num_failures} >= {max_retries_failure}).")
            raise RuntimeError(f"Failed too many times ({num_failures})") from problem

        raise RuntimeError(
            "Exhausted retries for multislice execution without explicit success or reaching failure/preemption limits."
        ) from problem


def execute_resumable(accelerator_config: AcceleratorConfigType):
    """Decorator for fault-tolerant single-pod execution.

    Wraps a Ray remote function to automatically use RayExecutor.execute_resumable
    with the specified accelerator configuration.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources to use for execution.

    Returns:
        Callable: Decorator function that wraps the remote function.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
        >>>
        >>> @execute_resumable(tpu_config)
        >>> @ray.remote
        >>> def my_task(data):
        ...     return process(data)
        >>>
        >>> result = my_task(input_data)  # Automatically retries on failure
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute_resumable(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def execute(accelerator_config: AcceleratorConfigType):
    """Decorator for single-pod execution without retry.

    Wraps a Ray remote function to automatically use RayExecutor.execute
    with the specified accelerator configuration. Results are automatically
    retrieved with ray.get().

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources to use for execution.

    Returns:
        Callable: Decorator function that wraps the remote function.

    Example:
        >>> gpu_config = GpuAcceleratorConfig(count=2, type="a100")
        >>>
        >>> @execute(gpu_config)
        >>> @ray.remote
        >>> def gpu_task(tensor):
        ...     return tensor.cuda() * 2
        >>>
        >>> result = gpu_task(my_tensor)  # Executes on GPU, no retry
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return ray.get(
                RayExecutor.execute(
                    remote_fn=remote_fn,
                    accelerator_config=accelerator_config,
                    **kwargs,
                )
            )

        return wrapper

    return decorator


def execute_multislice(accelerator_config: AcceleratorConfigType):
    """Decorator for multi-slice execution without retry.

    Wraps a Ray remote function to automatically use RayExecutor.execute_multislice
    with the specified accelerator configuration. Results from all slices are
    automatically retrieved with ray.get().

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources with multi-slice support (pod_count > 1).

    Returns:
        Callable: Decorator function that wraps the remote function.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
        >>>
        >>> @execute_multislice(tpu_config)
        >>> @ray.remote
        >>> def parallel_compute(data_shard):
        ...     return compute_result(data_shard)
        >>>
        >>> results = parallel_compute(sharded_data)  # Returns list from 4 slices
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return ray.get(
                RayExecutor.execute_multislice(
                    remote_fn=remote_fn,
                    accelerator_config=accelerator_config,
                    **kwargs,
                )
            )

        return wrapper

    return decorator


def execute_multislice_resumable(accelerator_config: AcceleratorConfigType):
    """Decorator for fault-tolerant multi-slice execution.

    Wraps a Ray remote function to automatically use RayExecutor.execute_multislice_resumable
    with the specified accelerator configuration. Provides automatic retry on
    preemption or failure of any slice.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources with multi-slice support (pod_count > 1).

    Returns:
        Callable: Decorator function that wraps the remote function.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4, preemptible=True)
        >>>
        >>> @execute_multislice_resumable(tpu_config)
        >>> @ray.remote
        >>> def resilient_training(data_batch):
        ...     # Long-running training that might be preempted
        ...     return train_model(data_batch)
        >>>
        >>> results = resilient_training(training_data)  # Auto-retries on failure
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute_multislice_resumable(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator
