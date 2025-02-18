import logging
import socket
import typing as tp

import ray
from ray.exceptions import (
	ActorDiedError,
	ActorUnavailableError,
	NodeDiedError,
	RayError,
	RaySystemError,
	RayTaskError,
	WorkerCrashedError,
)
from ray.remote_function import RemoteFunction

from .._statics import (
	TPUFailed,
	TPUInfo,
	TPUPreempted,
	TPURunError,
	TPURunResult,
	TPUSuccess,
)
from ._manager import TPUFunctionDecorator

logger = logging.getLogger(__name__)

# Type aliases
RemoteFunctionType = tp.Union[RemoteFunction, tp.Callable]
RayObjectRefType = ray.ObjectRef


class TPUExecutor:
	"""Handles TPU job execution and error management."""

	@staticmethod
	def execute(remote_fn: RemoteFunctionType, tpu_type: str) -> RayObjectRefType:
		"""Run a remote function on a TPU pod."""

		@ray.remote(resources={f"TPU-{tpu_type}-head": 1})
		def do_run(remote_fn) -> TPURunResult:
			logging.basicConfig(level=logging.INFO)
			num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
			remote_fn, tpu_name = TPUFunctionDecorator.configure_for_tpu(remote_fn, num_hosts)

			info = TPUInfo(tpu_name, "ACTIVE", "TPU")
			futures = [remote_fn.remote() for _ in range(num_hosts)]

			try:
				result = ray.get(futures)
				logger.info("TPU job completed successfully")
				return TPUSuccess(info, result)
			except RayError as e:
				TPUExecutor._cancel_futures(futures)
				return TPUExecutor._handle_ray_error(info, e)
			except Exception as e:
				TPUExecutor._cancel_futures(futures)
				return TPUFailed(info, e)

		return do_run.remote(remote_fn)

	@staticmethod
	def execute_resumable(
		remote_fn: RemoteFunctionType,
		tpu_type: str,
		max_retries_preemption: int = int(1e6),
		max_retries_failure: int = 10,
	) -> tp.Any:
		"""Run a function on a TPU pod with automatic retry on failure."""
		num_failures = num_preemptions = 0
		attempt = 0

		while (
			num_failures < max_retries_failure and num_preemptions < max_retries_preemption
		):
			logger.info(f"TPU Execution Attempt {attempt + 1} on {tpu_type}")
			attempt += 1

			try:
				result = ray.get(TPUExecutor.run_on_pod(remote_fn, tpu_type))

				if isinstance(result, TPUSuccess):
					logger.info("TPU execution succeeded")
					return result.result
				elif isinstance(result, (TPUPreempted, TPUFailed)):
					num_preemptions += 1
					logger.warning(f"TPU preempted/failed: {num_preemptions} times")
				elif isinstance(result, TPURunError):
					num_failures += 1
					logger.warning(f"TPU execution failed: {num_failures} times")

			except ray.exceptions.RayTaskError as e:
				if "preempted" in str(e).lower():
					num_preemptions += 1
					logger.warning(f"TPU preempted: {num_preemptions} times")
				else:
					num_failures += 1
					logger.warning(f"TPU task error: {num_failures} times", exc_info=e)
			except Exception as e:
				num_failures += 1
				if num_failures >= max_retries_failure:
					logger.error("Maximum failure retries exceeded", exc_info=e)
					raise
				logger.warning(f"TPU execution failed: {num_failures} times", exc_info=e)

		raise RuntimeError(
			f"TPU execution failed after {attempt} attempts: "
			f"{num_failures} failures, {num_preemptions} preemptions"
		)

	@staticmethod
	def _cancel_futures(futures: tp.List[RayObjectRefType]) -> None:
		"""Cancel all pending futures."""
		for future in futures:
			try:
				ray.cancel(future)
			except Exception as e:
				logger.error(f"Failed to cancel future: {e}")

	@staticmethod
	def _handle_ray_error(tpu_info: TPUInfo, error: RayError) -> TPURunResult:
		"""Handle various types of Ray errors."""
		if isinstance(
			error,
			(
				NodeDiedError,
				ActorUnavailableError,
				ActorDiedError,
				WorkerCrashedError,
			),
		):
			logger.exception("TPU node/worker error", exc_info=error)
			return TPUPreempted(tpu_info, error)
		elif isinstance(error, RaySystemError):
			logger.exception("Ray system error", exc_info=error)
			return TPURunError(tpu_info, error)
		elif isinstance(error, RayTaskError):
			return TPURunError(tpu_info, error)
		else:
			logger.exception("Unhandled Ray error", exc_info=error)
			return TPURunError(tpu_info, error)


class TPUMultiSliceExecutor:
	"""Handles execution across multiple TPU slices."""

	@ray.remote
	class MultisliceActor:
		"""Actor for managing multi-slice TPU execution."""

		def __init__(self):
			self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
			self.num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
			self.ip = socket.gethostbyname(socket.gethostname())

		def get_slice_info(self) -> tp.Tuple[str, int, str]:
			return self.pod_name, self.num_hosts, self.ip

		def _execute(
			self,
			remote_fn: RemoteFunctionType,
			coordinator_ip: str,
			slice_id: int,
			num_slices: int,
		) -> TPURunResult:
			"""Execute the function on this slice."""
			mxla_env = {
				"MEGASCALE_COORDINATOR_ADDRESS": f"{coordinator_ip}:8081",
				"MEGASCALE_NUM_SLICES": str(num_slices),
				"MEGASCALE_PORT": "8081",
				"MEGASCALE_SLICE_ID": str(slice_id),
			}
			remote_fn, tpu_name = TPUFunctionDecorator.configure_for_tpu(
				remote_fn,
				self.num_hosts,
				env_vars=mxla_env,
			)
			info = TPUInfo(tpu_name, "ACTIVE", "TPU")
			futures = [remote_fn.remote() for _ in range(self.num_hosts)]
			try:
				result = ray.get(futures)
				logger.info(f"Slice {slice_id} completed successfully")
				return TPUSuccess(info, result)
			except RayError as e:
				TPUExecutor._cancel_futures(futures)
				return TPUExecutor._handle_ray_error(info, e)
			except Exception as e:
				TPUExecutor._cancel_futures(futures)
				return TPUFailed(info, e)

	@staticmethod
	def execute(
		remote_fn: RemoteFunctionType,
		num_slices: int,
	) -> tp.List[RayObjectRefType]:
		"""Execute a function across multiple TPU slices."""
		actors = [TPUMultiSliceExecutor.MultisliceActor.remote() for _ in range(num_slices)]

		try:
			slice_infos = ray.get([actor.get_slice_info.remote() for actor in actors])
			logger.info(f"TPU slice configuration: {slice_infos}")
		except RayError as e:
			logger.exception("Failed to initialize TPU slices")
			for actor in actors:
				try:
					ray.kill(actor)
				except Exception:
					logger.exception("Failed to clean up actor")
			raise e

		coordinator_ip = slice_infos[0][2]
		return [
			actor._execute.remote(
				remote_fn,
				coordinator_ip,
				i,
				num_slices,
			)
			for i, actor in enumerate(actors)
		]
