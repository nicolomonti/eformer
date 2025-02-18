import functools
import logging
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
import typing as tp

import mergedeep
import ray
from ray._private.accelerators import TPUAcceleratorManager
from ray.remote_function import RemoteFunction

logger = logging.getLogger(__name__)

# Type aliases
RemoteFunctionType = tp.Union[RemoteFunction, tp.Callable]
RayObjectRefType = ray.ObjectRef


class TPUProcessManager:
	"""Manages TPU processes and their execution."""

	@staticmethod
	def run_in_process(fn: tp.Callable, args: tuple, kwargs: dict) -> tp.Any:
		"""Execute a function in a separate process."""

		def target_fn(queue: Queue, args: tuple, kwargs: dict) -> None:
			try:
				result = fn(*args, **kwargs)
				queue.put((True, result))
			except Exception as e:
				queue.put((False, e))

		queue = Queue()
		process = Process(target=target_fn, args=(queue, args, kwargs))
		process.start()
		process.join()

		try:
			success, value = queue.get(timeout=10)
			if success:
				return value
			raise value
		except QueueEmpty as e:
			process.terminate()
			raise RuntimeError("Process execution timed out") from e


class TPUFunctionDecorator:
	"""Handles decoration and configuration of TPU functions."""

	@staticmethod
	def forkify_remote_fn(remote_fn: RemoteFunctionType) -> RemoteFunctionType:
		"""Force a remote function to run in its own process."""
		if isinstance(remote_fn, RemoteFunction):
			fn = remote_fn._function
			wrapped_fn = functools.wraps(fn)(
				lambda *args, **kwargs: TPUProcessManager.run_in_process(fn, args, kwargs)
			)
			return RemoteFunction(
				language=remote_fn._language,
				function=wrapped_fn,
				function_descriptor=remote_fn._function_descriptor,
				task_options=remote_fn._default_options,
			)
		return functools.partial(TPUProcessManager.run_in_process, remote_fn)

	@staticmethod
	def configure_for_tpu(
		remote_fn: RemoteFunctionType,
		num_hosts: int,
		**runtime_env,
	) -> tp.Tuple[RemoteFunction, str]:
		"""Configure a remote function for TPU execution."""
		remote_fn = TPUFunctionDecorator.forkify_remote_fn(remote_fn)
		if not isinstance(remote_fn, RemoteFunction):
			remote_fn = ray.remote(remote_fn)

		tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
		num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()

		runtime_env = TPUFunctionDecorator._merge_runtime_envs(
			remote_fn._runtime_env,
			runtime_env,
		)

		remote_fn = remote_fn.options(
			runtime_env=runtime_env,
			resources={tpu_name: 1, "TPU": num_tpus_per_host},
		)

		logger.info(
			f"TPU Configuration: {tpu_name}, Hosts: {num_hosts}, TPUs per host: {num_tpus_per_host}"
		)
		return remote_fn, tpu_name

	@staticmethod
	def _merge_runtime_envs(*envs: tp.Optional[dict]) -> dict:
		"""Merge multiple runtime environments."""
		sources = [env for env in envs if env is not None]
		return mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)
