import functools
import logging
import multiprocessing
import sys
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from typing import Callable, Optional

import mergedeep
import ray
import tblib
from ray._private.accelerators import TPUAcceleratorManager
from ray.exceptions import (
	NodeDiedError,
	RayError,
	RaySystemError,
	RayTaskError,
	WorkerCrashedError,
)
from ray.remote_function import RemoteFunction

from .._statics import (
	TpuInfo,
	TpuPreempted,
	TpuRunError,
)

logger = logging.getLogger("ray")


def redecorate_remote_fn_for_tpu(
	remote_fn,
	num_hosts,
	verbose,
	**runtime_env,
):
	remote_fn = forkify_remote_fn(remote_fn)
	if not isinstance(remote_fn, RemoteFunction):
		remote_fn = ray.remote(remote_fn)

	tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
	num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
	sources = [e for e in [remote_fn._runtime_env, runtime_env] if e is not None]
	runtime_env = mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)

	remote_fn = remote_fn.options(
		runtime_env=runtime_env,
		resources={tpu_name: 1, "TPU": num_tpus_per_host},
	)
	if verbose:
		logger.info(
			f"Running on TPU {tpu_name} with {num_hosts} hosts "
			f"and {num_tpus_per_host} TPUs per host"
		)
	return remote_fn, tpu_name


def handle_ray_error(tpu_info: TpuInfo, e: RayError):
	if isinstance(e, NodeDiedError):
		logger.exception("Node died", exc_info=e)
		return TpuPreempted(tpu_info, e)
	elif isinstance(
		e, ray.exceptions.ActorUnavailableError | ray.exceptions.ActorDiedError
	):
		logger.exception("Actor died", exc_info=e)
		return TpuPreempted(tpu_info, e)
	elif isinstance(e, WorkerCrashedError):
		logger.exception("Worker crashed", exc_info=e)
		return TpuPreempted(tpu_info, e)
	elif isinstance(e, RaySystemError):
		logger.exception("System error", exc_info=e)
		return TpuRunError(tpu_info, e)
	elif isinstance(e, RayTaskError):
		logger.exception(f"Task error {e}", exc_info=e)
		return TpuRunError(tpu_info, e)

	else:
		logger.exception("Unknown error", exc_info=e)
		return TpuRunError(tpu_info, e)


@dataclass
class ExceptionInfo:
	ex: Optional[BaseException]
	tb: tblib.Traceback

	def restore(self):
		if self.ex is not None:
			exc_value = self.ex.with_traceback(self.tb.as_traceback())
			return (self.ex.__class__, exc_value, self.tb.as_traceback())
		else:
			return (
				Exception,
				Exception("Process failed with no exception"),
				self.tb.as_traceback(),
			)

	def reraise(self):
		if self.ex is not None:
			raise self.ex.with_traceback(self.tb.as_traceback())
		else:
			raise Exception("Process failed with no exception").with_traceback(
				self.tb.as_traceback()
			)


def ser_exc_info(exception=None) -> ExceptionInfo:
	if exception is None:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		tb = tblib.Traceback(exc_traceback)
		return ExceptionInfo(exc_value, tb)
	else:
		tb = exception.__traceback__
		tb = tblib.Traceback(tb)
		return ExceptionInfo(exception, tb)


def forkify_remote_fn(remote_fn: RemoteFunction | Callable):
	if isinstance(remote_fn, RemoteFunction):
		fn = remote_fn._function

		@functools.wraps(fn)
		def wrapped_fn(*args, **kwargs):
			return separate_process_fn(fn, args, kwargs)

		remote_fn = RemoteFunction(
			language=remote_fn._language,
			function=wrapped_fn,
			function_descriptor=remote_fn._function_descriptor,
			task_options=remote_fn._default_options,
		)
		return remote_fn
	else:
		return functools.partial(separate_process_fn, remote_fn)


def separate_process_fn(underlying_function, args, kwargs):
	def target_fn(queue, args, kwargs):
		try:
			result = underlying_function(*args, **kwargs)
			queue.put((True, result))
		except Exception as e:
			info = ser_exc_info(e)
			queue.put((False, info))

	queue = multiprocessing.Queue()
	process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
	process.start()
	process.join()

	# Retrieve the result or error from the queue
	logger.info("Process finished")
	try:
		success, value = queue.get(timeout=int(1e6))
	except QueueEmpty as e:
		logger.error("Process timed out")
		process.terminate()
		raise RuntimeError("Process timed out") from e

	if success:
		return value
	else:
		raise ValueError(value)


def cancel_all_futures(futures):
	for f in futures:
		try:
			ray.cancel(f)
		except Exception:
			logger.exception("Failed to kill job after primary failure")
