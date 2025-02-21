import abc
import logging
import socket
import typing as tp

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from .._statics import (
	TpuFailed,
	TpuInfo,
	TpuPreempted,
	TpuRunError,
	TpuRunResult,
	TpuSuccess,
)
from ._manager import (
	cancel_all_futures,
	handle_ray_error,
	redecorate_remote_fn_for_tpu,
)

logger = logging.getLogger("ray")
RemoteFuncType = RemoteFunction | tp.Callable
TPUType = str

# fmt:off

class TPUBaseExecutor(abc.ABC):
  """
  Base class for TPU executors with abstract execution methods.

  Attributes:
      execute (classmethod): Method to submit a job to the TPU.
      execute_resumable (classmethod): Resilient method to handle preemptions/failures.
  """

  @classmethod
  @abc.abstractmethod
  def execute(*arg, **kwargs):
    """
    Submit a TPU job for execution.

    Returns:
        ray.ObjectRef: Reference to the TPU job result.
    """

  @classmethod
  @abc.abstractmethod
  def execute_resumable(*arg, **kwargs):
    """Submit a TPU job with automatic retry on preemption/failure."""

# fmt:on


class TPUExecutor(TPUBaseExecutor):
	"""
	Executor for single TPU pod with preemption/failure handling.

	Methods:
	    execute: Submit a TPU job.
	    execute_resumable: Retry-based execution with fault-tolerance.
	"""

	@classmethod
	def execute(
		cls,
		remote_fn: RemoteFuncType,
		tpu_type: TPUType,
		runner_resources: tp.Optional[dict] = None,
		num_hosts: tp.Optional[int] = None,
		verbose: bool = True,
		do_remotecall: bool = True,
		num_slice_calls: int = 1,
	) -> ray.ObjectRef:
		"""
		Submit a job to the TPU pod.

		Args:
		    remote_fn (RemoteFuncType): Ray remote function or callable.
		    tpu_type (str): TPU type (e.g., 'v4-8').
				runner_resources (dict, optional): customized resources to pass to ray_fn.remote.
				num_hosts (int, optional): number of hosts to execute each row call on.
				verbose (bool): whenever to log some information which are not really usefull.
				do_remotecall (bool): whenever to call remote fucntion automatically or not.
				num_slice_calls (int): how many slices of code should be executed.

		Returns:
		    ray.ObjectRef: Reference to the TPU job's result.

		Raises:
		    RayError: If job encounters an unrecoverable error.
		"""
		if runner_resources is None:
			runner_resources = {f"TPU-{tpu_type}-head": 1}

		def do_run(
			remote_fn,
			local_num_hosts,
			local_verbose,
			slice_idx,
		) -> TpuRunResult:
			"""
			Execute the remote function on the TPU.

			Returns:
			    TpuRunResult: Result of the TPU execution.
			"""
			logging.basicConfig(level=logging.INFO)
			num_hosts = (
				local_num_hosts or ray.util.accelerators.tpu.get_current_pod_worker_count()
			)
			remote_fn, tpu_name = redecorate_remote_fn_for_tpu(
				remote_fn=remote_fn,
				num_hosts=num_hosts,
				verbose=local_verbose,
			)
			info = TpuInfo(tpu_name, "ACTIVE", "TPU")

			try:
				futures = [remote_fn.remote(slice_idx) for _ in range(num_hosts)]
			except TypeError as e:
				if str(e) != "too many positional arguments":
					raise e
				futures = [remote_fn.remote() for _ in range(num_hosts)]
			try:
				out = ray.get(futures)
				if local_verbose:
					logger.info("TPU job finished")
				return TpuSuccess(info, out)
			except RayError as e:
				cancel_all_futures(futures)
				return handle_ray_error(info, e)
			except Exception as e:
				cancel_all_futures(futures)
				return TpuFailed(info, e)

		if runner_resources == Ellipsis:
			do_run = ray.remote(do_run)
		else:
			do_run = ray.remote(resources=runner_resources)(do_run)
		if do_remotecall:
			if num_slice_calls > 1:
				return [
					do_run.remote(remote_fn, num_hosts, verbose, slice_idx)
					for slice_idx in range(num_slice_calls)
				]
			return do_run.remote(remote_fn, num_hosts, verbose, 0)
		return do_run

	@classmethod
	def execute_resumable(
		cls,
		remote_fn: RemoteFuncType,
		tpu_type: TPUType,
		runner_resources: tp.Optional[dict] = None,
		num_hosts: tp.Optional[int] = None,
		verbose: bool = True,
		num_slice_calls: int = 1,
		max_retries_preemption: int = int(1e6),
		max_retries_failure: int = 10,
	):
		"""
		Run a TPU job with automatic preemption/failure retries.

		Args:
		    remote_fn (RemoteFuncType): Function to execute.
		    tpu_type (str): TPU type (e.g., 'v4-8').
				runner_resources (dict, optional): customized resources to pass to ray_fn.remote.
				num_hosts (int, optional): number of hosts to execute each row call on.
				verbose (bool): whenever to log some information which are not really usefull.
				num_slice_calls (int): how many slices of code should be executed.
		    max_retries_preemption (int): Maximum preemption retries.
		    max_retries_failure (int): Maximum failure retries.

		Raises:
		    RuntimeError: If retries are exhausted without success.
		"""
		num_failures = 0
		num_preemptions = 0
		attempt = 0
		problem: Exception | None = None

		while (
			num_failures < max_retries_failure and num_preemptions < max_retries_preemption
		):
			logger.info(f"Running on TPU {tpu_type}. Attempt {attempt}")
			attempt += 1
			problem = None
			try:
				out = ray.get(
					cls.execute(
						remote_fn=remote_fn,
						tpu_type=tpu_type,
						runner_resources=runner_resources,
						do_remotecall=True,
						num_hosts=num_hosts,
						num_slice_calls=num_slice_calls,
						verbose=verbose,
					)
				)
			except ray.exceptions.RayTaskError as e:
				problem = e
				if "preempted" in str(e).lower():
					num_preemptions += 1
					logger.warning(f"Preempted {num_preemptions} times, {e}")
				else:
					num_failures += 1
					logger.warning(f"Failed {num_failures} times", exc_info=e)
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

			if isinstance(out, TpuSuccess):
				result = out.result
				logger.info("Success")
				return result
			elif isinstance(out, TpuPreempted):
				problem = out.error
				num_preemptions += 1
				logger.warning(
					f"Preempted {num_preemptions} times. {problem}", exc_info=problem
				)
			elif isinstance(out, TpuFailed):
				num_preemptions += 1
				logger.warning(
					f"TPU node failure. Treating as preempted: {num_preemptions} times"
				)
			elif isinstance(out, TpuRunError):
				problem = out.error
				num_failures += 1
				logger.warning(f"Failed {num_failures} times", exc_info=problem)
			else:
				raise RuntimeError(f"Unexpected result: {out}")

		if num_preemptions >= max_retries_preemption:
			raise RuntimeError("Preempted too many times") from problem
		elif num_failures >= max_retries_failure:
			raise RuntimeError("Failed too many times") from problem


class TPUMultiSliceExecutor(TPUBaseExecutor):
	"""
	Executor for multiple TPU slices with coordination and fault tolerance.

	Methods:
	    execute: Submit jobs to multiple TPU slices.
	    execute_resumable: Retry-based execution with cross-slice resilience.
	"""

	@staticmethod
	def execute(
		remote_fn: RemoteFuncType,
		tpu_type: TPUType,
		num_slices: int,
		runner_resources: tp.Optional[dict] = None,
		num_hosts: tp.Optional[int] = None,
		verbose: bool = True,
	) -> list[ray.ObjectRef]:
		"""
		Submit jobs across multiple TPU slices.

		Args:
		    remote_fn (RemoteFuncType): Function to execute on each slice.
		    tpu_type (str): TPU type (e.g., 'v4-8').
		    num_slices (int): Number of TPU slices.
				runner_resources (dict, optional): customized resources to pass to ray_fn.remote.
				num_hosts (int, optional): number of hosts to execute each row call on.
				verbose (bool): whenever to log some information which are not really usefull.

		Returns:
		    list[ray.ObjectRef]: References to each slice's job result.
		"""
		if runner_resources is None:
			runner_resources = {f"TPU-{tpu_type}-head": 1}

		class MultisliceActor:
			def __init__(self, local_num_hosts, local_verbose):
				self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
				self.num_hosts = (
					local_num_hosts or ray.util.accelerators.tpu.get_current_pod_worker_count()
				)
				self.ip = socket.gethostbyname(socket.gethostname())
				self.local_verbose = local_verbose

			def get_slice_info(self):
				"""Return pod name, host count, and IP address."""
				return self.pod_name, self.num_hosts, self.ip

			def do_run(self, remote_fn, coordinator_ip, slice_id, num_slices) -> TpuRunResult:
				"""
				Execute the remote function on this TPU slice.

				Args:
				    remote_fn (RemoteFuncType): Function to run.
				    coordinator_ip (str): Coordinator node IP address.
				    slice_id (int): Unique identifier for this slice.
				    num_slices (int): Total number of slices.

				Returns:
				    TpuRunResult: Result from executing on this slice.
				"""

				port = 8081
				mxla_env = {
					"MEGASCALE_COORDINATOR_ADDRESS": f"{coordinator_ip}:{port}",
					"MEGASCALE_NUM_SLICES": str(num_slices),
					"MEGASCALE_PORT": f"{port}",
					"MEGASCALE_SLICE_ID": str(slice_id),
				}

				remote_fn, tpu_name = redecorate_remote_fn_for_tpu(
					remote_fn,
					self.num_hosts,
					verbose=self.local_verbose,
					env_vars=mxla_env,
				)

				info = TpuInfo(tpu_name, "ACTIVE", "TPU")
				futures = [remote_fn.remote() for _ in range(self.num_hosts)]
				try:
					out = ray.get(futures)
					logger.info("TPU job finished")
					return TpuSuccess(info, out)
				except RayError as e:
					logger.exception(f"Ray error {e}. Killing futures for this slice")
					cancel_all_futures(futures)
					return handle_ray_error(info, e)
				except Exception as e:
					logger.exception(f"Exception {e}")
					cancel_all_futures(futures)
					return TpuFailed(info, e)

		if runner_resources == Ellipsis:
			MultisliceActor = ray.remote(MultisliceActor)
		else:
			MultisliceActor = ray.remote(resources=runner_resources)(MultisliceActor)
		actors = [
			MultisliceActor.remote(
				local_num_hosts=num_hosts,
				local_verbose=verbose,
			)
			for _ in range(num_slices)
		]
		futures = [actor.get_slice_info.remote() for actor in actors]
		try:
			logger.info("Getting slice infos...")
			slice_infos = ray.get(futures)
			logger.info(f"TPU slice infos {slice_infos}")
		except RayError as e:
			logger.exception(e)
			for actor in actors:
				try:
					ray.kill(actor)
				except Exception:
					logger.exception("Failed to kill actor after primary failure")
			return futures

		coordinator_ip = slice_infos[0][2]
		return [
			actor.do_run.remote(
				remote_fn=remote_fn,
				coordinator_ip=coordinator_ip,
				slice_id=slice_id,
				num_slices=num_slices,
			)
			for slice_id, actor in enumerate(actors)
		]

	@classmethod
	def execute_resumable(
		cls,
		remote_fn: RemoteFuncType,
		tpu_type: TPUType,
		num_slices: int,
		runner_resources: tp.Optional[dict] = None,
		num_hosts: tp.Optional[int] = None,
		verbose: bool = True,
		max_retries_preemption: int = int(1e6),
		max_retries_failure: int = 10,
	):
		"""
		Run jobs across TPU slices with automatic retries.

		Args:
		    remote_fn (RemoteFuncType): Function to execute.
		    tpu_type (str): TPU type (e.g., 'v4-8').
		    num_slices (int): Number of TPU slices.
				runner_resources (dict, optional): customized resources to pass to ray_fn.remote.
				num_hosts (int, optional): number of hosts to execute each row call on.
				verbose (bool): whenever to log some information which are not really usefull.
		    max_retries_preemption (int): Preemption retry limit.
		    max_retries_failure (int): Failure retry limit.

		Returns:
		    list[object]: Results from all TPU slices.

		Raises:
		    RuntimeError: If retries are exhausted across all slices.
		"""
		num_failures = 0
		num_preemptions = 0
		attempt = 0
		problem: Exception | None = None

		while (
			num_failures < max_retries_failure and num_preemptions < max_retries_preemption
		):
			logger.info(f"Running on TPU {tpu_type}. Attempt {attempt}")
			attempt += 1
			problem = None
			futures = cls.execute(
				remote_fn=remote_fn,
				tpu_type=tpu_type,
				num_slices=num_slices,
				runner_resources=runner_resources,
				verbose=verbose,
				num_hosts=num_hosts,
			)
			try:
				outs = ray.get(futures)
			except ray.exceptions.ActorUnavailableError as e:
				problem = e
				num_preemptions += 1
				logger.warning(f"Preempted {num_preemptions} times, {e}")
				continue
			except ray.exceptions.ActorDiedError as e:
				problem = e
				num_preemptions += 1
				logger.warning(f"Preempted {num_preemptions} times, {e}")
				continue
			except ray.exceptions.RayTaskError as e:
				for f in futures:
					try:
						ray.cancel(f)
					except Exception:
						logger.exception("Failed to kill job after primary failure")
				problem = e
				if "preempted" in str(e).lower():
					num_preemptions += 1
					logger.warning(f"Preempted {num_preemptions} times, {e}")
				else:
					num_failures += 1
					logger.warning(f"Failed {num_failures} times", exc_info=e)
				continue
			except Exception as e:
				for f in futures:
					try:
						ray.cancel(f)
					except Exception:
						logger.exception("Failed to kill job after primary failure")
				problem = e
				num_failures += 1
				if num_failures >= max_retries_failure:
					logger.exception("Failed too many times", exc_info=e)
					raise e
				else:
					logger.warning(f"Failed {num_failures} times", exc_info=e)
					continue

			if all(isinstance(out, TpuSuccess) for out in outs):
				results = [out.result for out in outs]
				logger.info("Success")
				return results
			elif any(isinstance(out, TpuPreempted) for out in outs):
				out = None
				for o in outs:
					if isinstance(o, TpuPreempted):
						out = o
				assert out is not None
				problem = out.error
				num_preemptions += 1
				logger.warning(
					f"Preempted {num_preemptions} times. {problem}",
					exc_info=problem,
				)
			elif any(isinstance(out, TpuFailed) for out in outs):
				num_preemptions += 1
				logger.warning(
					f"TPU node failure. Treating as preempted: {num_preemptions} times"
				)
			elif any(isinstance(out, TpuRunError) for out in outs):
				out = None
				for o in outs:
					if isinstance(o, TpuRunError):
						out = o
				assert out is not None
				problem = out.error
				num_preemptions += 1
				problem = out.error
				num_failures += 1
				logger.warning(f"Failed {num_failures} times", exc_info=problem)
			else:
				raise RuntimeError(f"Unexpected result: {out}")

		if num_preemptions >= max_retries_preemption:
			raise RuntimeError("Preempted too many times") from problem
		elif num_failures >= max_retries_failure:
			raise RuntimeError("Failed too many times") from problem
