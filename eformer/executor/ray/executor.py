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
import functools
import logging

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

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
    """
    Provides methods to execute Ray remote functions, particularly for
    workloads on TPUs/GPUs, including multi-slice and resumable executions.
    """

    @staticmethod
    def execute(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        **kwargs,
    ):
        """
        Executes a Ray remote function on a single slice or non-TPU setup.

        Args:
                remote_fn: The Ray RemoteFunction to execute.
                accelerator_config: Configuration for the accelerator resources.

        Returns:
                A Ray ObjectRef representing the future result of the execution.

        Raises:
                AssertionError: If `pod_count` in `accelerator_config` is not 1,
                                          suggesting `execute_multislice` should be used.
        """
        assert getattr(accelerator_config, "pod_count", 1) == 1, (
            "Multi-slice workloads on TPUs should use 'execute_multislice'."
        )

        def do_run(
            remote_fn,
            accelerator_config: AcceleratorConfigType,
            kwargs,
        ) -> JobStatus:
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
        """
        Executes a Ray remote function across multiple TPU slices.

        Args:
                remote_fn: The Ray RemoteFunction to execute on each slice.
                accelerator_config: Configuration for the accelerator resources,
                                                        including multi-slice details.

        Returns:
                A list of Ray ObjectRefs, each representing the future result
                of `do_run` on a corresponding actor/slice.

        Raises:
                RayError: If fetching slice information fails, if the coordinator IP
                                  cannot be determined, or if actor creation/remote calls fail.
        """

        class MultisliceActor:
            def __init__(self, accelerator_config: AcceleratorConfigType):
                import socket

                runtime_name = accelerator_config.runtime_name
                num_workers = accelerator_config.worker_count

                if isinstance(accelerator_config, TpuAcceleratorConfig):  # refresh on TPUs
                    runtime_name = ray.util.accelerators.tpu.get_current_pod_name()
                    num_workers = ray.util.accelerators.tpu.get_current_pod_worker_count()

                self.runtime_name = runtime_name
                self.num_workers = num_workers
                self.num_slices = getattr(accelerator_config, "pod_count", 1)
                self.accelerator_config = accelerator_config
                self.ip = socket.gethostbyname(socket.gethostname())

            def get_slice_info(self):
                """Return pod name, host count, and IP address."""
                return self.runtime_name, self.num_workers, self.ip

            def do_run(
                self,
                remote_fn: RemoteFunction,
                coordinator_ip,
                slice_id,
                kwargs,
            ) -> JobStatus:
                mxla_env = {
                    "MEGASCALE_COORDINATOR_ADDRESS": f"{coordinator_ip}:{MEGASCALE_DEFAULT_PORT}",
                    "MEGASCALE_NUM_SLICES": str(self.num_slices),
                    "MEGASCALE_PORT": str(MEGASCALE_DEFAULT_PORT),
                    "MEGASCALE_SLICE_ID": str(slice_id),
                    ENV_CALL_SLICE: str(slice_id),
                }

                info = JobInfo(
                    self.accelerator_config.runtime_name,
                    "running",
                    self.accelerator_config.resource_name,
                )
                futures = []
                for idx in range(self.num_workers):
                    base_env_vars = {ENV_CALL_INDEX: str(idx)}
                    final_env_vars = {**base_env_vars, **mxla_env}
                    _call = accelerator_config.redecorate_remote_fn_for_call(
                        remote_fn=remote_fn,
                        env_vars=final_env_vars,
                    )
                    futures.append(_call.remote(**kwargs))
                try:
                    out = ray.get(futures)
                    logger.info("Job finished")
                    return JobSucceeded(info, out)
                except RayError as e:
                    logger.exception(f"Ray error {e}. Killing futures for this slice")
                    RayResources.cancel_all_futures(futures)
                    return handle_ray_error(info, e)
                except Exception as e:
                    logger.exception(f"Exception {e}")
                    RayResources.cancel_all_futures(futures)
                    return JobFailed(info, e)

        if accelerator_config.head_name is None and not isinstance(
            accelerator_config,
            TpuAcceleratorConfig,
        ):
            MultisliceActor = ray.remote(MultisliceActor)
        else:
            default_name = f"TPU-{accelerator_config.tpu_version}-head"
            resources = {accelerator_config.head_name or default_name: accelerator_config.head_workers}
            MultisliceActor = ray.remote(resources=resources)(MultisliceActor)

        actors = [MultisliceActor.remote(accelerator_config) for _ in range(getattr(accelerator_config, "pod_count", 1))]
        futures = [actor.get_slice_info.remote() for actor in actors]
        try:
            logger.info("Getting slice infos...")
            slice_infos = ray.get(futures)
            logger.info(f"slice infos {slice_infos}")
        except RayError as e:
            logger.exception(f"RayError while getting slice_infos: {e}")
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    logger.exception("Failed to kill actor after primary failure in get_slice_info")
            raise e

        if (
            not slice_infos
            or not isinstance(slice_infos, list)
            or not slice_infos[0]
            or not isinstance(slice_infos[0], tuple)
            or len(slice_infos[0]) < 3
        ):
            logger.error(
                "Failed to get valid slice_info to determine coordinator IP. Slice_infos: %s",
                slice_infos,
            )
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    logger.exception("Failed to kill actor after slice_info retrieval failure")
            raise RayError(f"Could not determine coordinator IP from malformed or empty slice_infos: {slice_infos}")
        coordinator_ip = slice_infos[0][2]
        return [
            actor.do_run.remote(
                remote_fn=remote_fn,
                coordinator_ip=coordinator_ip,
                slice_id=slice_id,
                kwargs=kwargs,
            )
            for slice_id, actor in enumerate(actors)
        ]

    @classmethod
    def execute_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """
        Executes a Ray remote function with retries for preemptions and failures.

        Args:
                remote_fn: The Ray RemoteFunction to execute.
                accelerator_config: Configuration for the accelerator resources.
                max_retries_preemption: Maximum number of retries on preemption.
                max_retries_failure: Maximum number of retries on failure.

        Returns:
                The result of the successful execution.

        Raises:
                RuntimeError: If the job is preempted or fails too many times.
                Exception: Propagates the last encountered exception if retries are exhausted.
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
        """
        Executes a multi-slice Ray remote function with retries for preemptions and failures.

        Args:
                remote_fn: The Ray RemoteFunction to execute.
                accelerator_config: Configuration for the accelerator resources.
                max_retries_preemption: Maximum number of retries on preemption.
                max_retries_failure: Maximum number of retries on failure.

        Returns:
                A list of results from the successful execution on all slices.
                The order of results corresponds to the slice IDs.

        Raises:
                RuntimeError: If the job is preempted or fails too many times across slices.
                RayError: If `execute_multislice` itself raises an error during setup.
                Exception: Propagates the last encountered exception if retries are exhausted.
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
    """
    Decorator to wrap Ray remote functions for execution via RayExecutor.execute_resumable.
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
    """
    Decorator to wrap Ray remote functions for execution via RayExecutor.execute.
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
    """
    Decorator to wrap Ray remote functions for execution via RayExecutor.execute_multislice.
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
    """
    Decorator to wrap Ray remote functions for execution via RayExecutor.execute_multislice_resumable.
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
