import abc
import logging
import socket
import typing as tp

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from .._statics import (
	RunError,
	RunFailed,
	RunInfo,
	RunPreempted,
	RunResult,
	RunSuccess,
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
		runtime_env: tp.Optional[tp.Dict[str, tp.Any]] = None,
		kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
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
		if kwargs is None:
			kwargs = {}
		if runner_resources is None:
			runner_resources = {f"TPU-{tpu_type}-head": 1}
		if runtime_env is None:
			runtime_env = {}

		def do_run(
			remote_fn,
			local_num_hosts,
			local_verbose,
			slice_idx,
		) -> RunResult:
			"""
			Execute the remote function on the TPU.

			Returns:
			    RunResult: Result of the TPU execution.
			"""
			logging.basicConfig(level=logging.INFO)
			num_hosts = (
				local_num_hosts or ray.util.accelerators.tpu.get_current_pod_worker_count()
			)
			runtime_env.update({"TPX_SLICE_INDEX": str(slice_idx)})

			tpu_name = ray.util.accelerators.tpu.get_current_pod_name()

			static_inputs = dict(
				remote_fn=remote_fn,
				num_hosts=num_hosts,
				verbose=local_verbose,
			)

			info = RunInfo(tpu_name, "ACTIVE", "TPU")
			futures = []
			for idx in range(num_hosts):
				runtime_env.update({"TPX_INDEX": str(idx)})
				_call = redecorate_remote_fn_for_tpu(
					env_vars=runtime_env,
					**static_inputs,
				)
				futures.append(_call.remote(**kwargs))
			try:
				out = ray.get(futures)
				if local_verbose:
					logger.info("TPU job finished")
				return RunSuccess(info, out)
			except RayError as e:
				cancel_all_futures(futures)
				return handle_ray_error(info, e)
			except Exception as e:
				cancel_all_futures(futures)
				return RunFailed(info, e)

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
		runtime_env: tp.Optional[tp.Dict[str, tp.Any]] = None,
		kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
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
						kwargs=kwargs,
						runtime_env=runtime_env,
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

			if isinstance(out, RunSuccess):
				result = out.result
				logger.info("Success")
				return result
			elif isinstance(out, RunPreempted):
				problem = out.error
				num_preemptions += 1
				logger.warning(
					f"Preempted {num_preemptions} times. {problem}", exc_info=problem
				)
			elif isinstance(out, RunFailed):
				num_preemptions += 1
				logger.warning(
					f"TPU node failure. Treating as preempted: {num_preemptions} times"
				)
			elif isinstance(out, RunError):
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
		runtime_env: tp.Optional[tp.Dict[str, tp.Any]] = None,
		kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
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
		if kwargs is None:
			kwargs = {}
		if runtime_env is None:
			runtime_env = {}

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

			def do_run(
				self,
				remote_fn,
				coordinator_ip,
				slice_id,
				num_slices,
				call_kwargs,
				call_runtime_env,
			) -> RunResult:
				"""
				Execute the remote function on this TPU slice.

				Args:
				    remote_fn (RemoteFuncType): Function to run.
				    coordinator_ip (str): Coordinator node IP address.
				    slice_id (int): Unique identifier for this slice.
				    num_slices (int): Total number of slices.

				Returns:
				    RunResult: Result from executing on this slice.
				"""

				port = 8081

				mxla_env = {
					"MEGASCALE_COORDINATOR_ADDRESS": f"{coordinator_ip}:{port}",
					"MEGASCALE_NUM_SLICES": str(num_slices),
					"MEGASCALE_PORT": f"{port}",
					"MEGASCALE_SLICE_ID": str(slice_id),
					"TPX_SLICE_INDEX": str(slice_id),
				}
				mxla_env.update(**call_runtime_env)
				tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
				call_inputs = dict(
					remote_fn=remote_fn,
					num_hosts=self.num_hosts,
					verbose=self.local_verbose,
				)

				info = RunInfo(tpu_name, "ACTIVE", "TPU")
				futures = []
				for idx in range(self.num_hosts):
					mxla_env.update({"TPX_INDEX": str(idx)})
					remote_fn = redecorate_remote_fn_for_tpu(env_vars=mxla_env, **call_inputs)
					futures.append(remote_fn.remote(**call_kwargs))
				try:
					out = ray.get(futures)
					logger.info("TPU job finished")
					return RunSuccess(info, out)
				except RayError as e:
					logger.exception(f"Ray error {e}. Killing futures for this slice")
					cancel_all_futures(futures)
					return handle_ray_error(info, e)
				except Exception as e:
					logger.exception(f"Exception {e}")
					cancel_all_futures(futures)
					return RunFailed(info, e)

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
				call_kwargs=kwargs,
				call_runtime_env=runtime_env,
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
		runtime_env: tp.Optional[tp.Dict[str, tp.Any]] = None,
		kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
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
				runtime_env=runtime_env,
				kwargs=kwargs,
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

			if all(isinstance(out, RunSuccess) for out in outs):
				results = [out.result for out in outs]
				logger.info("Success")
				return results
			elif any(isinstance(out, RunPreempted) for out in outs):
				out = None
				for o in outs:
					if isinstance(o, RunPreempted):
						out = o
				assert out is not None
				problem = out.error
				num_preemptions += 1
				logger.warning(
					f"Preempted {num_preemptions} times. {problem}",
					exc_info=problem,
				)
			elif any(isinstance(out, RunFailed) for out in outs):
				num_preemptions += 1
				logger.warning(
					f"TPU node failure. Treating as preempted: {num_preemptions} times"
				)
			elif any(isinstance(out, RunError) for out in outs):
				out = None
				for o in outs:
					if isinstance(o, RunError):
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
