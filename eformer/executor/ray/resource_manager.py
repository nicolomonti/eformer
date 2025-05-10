# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
import multiprocessing
import os
import typing as tp
import uuid
from dataclasses import dataclass, field, replace
from queue import Empty as QueueEmpty
from typing import Any, Callable, Dict, Optional, Protocol

import mergedeep
import ray
from ray._private.accelerators import NvidiaGPUAcceleratorManager, TPUAcceleratorManager
from ray.remote_function import RemoteFunction
from ray.runtime_env import RuntimeEnv

from .types import ExceptionInfo

logger = logging.getLogger("ray")


@dataclass
class RayResources:
	"""
	A representation of resource requirements for Ray tasks and actors.

	This dataclass encapsulates all resource specifications needed when creating
	Ray tasks or actors, allowing for easy conversion between different resource
	representation formats used by Ray.
	"""

	cpu_allocation: int = 1
	gpu_allocation: int = 0
	custom_resources: Dict[str, float] = field(default_factory=dict)
	execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
	hardware_type: Optional[str] = None

	def to_kwargs(self) -> Dict[str, Any]:
		"""
		Convert resource specifications to kwargs for ray.remote() decorator.

		Returns:
		    Dictionary of keyword arguments compatible with ray.remote().
		"""
		remote_kwargs = {
			"num_cpus": self.cpu_allocation,
			"num_gpus": self.gpu_allocation,
			"resources": self.custom_resources,
			"runtime_env": self.execution_env,
		}

		if self.hardware_type is not None:
			remote_kwargs["accelerator_type"] = self.hardware_type

		return remote_kwargs

	def to_resource_dict(self) -> Dict[str, float]:
		"""
		Converts resource specifications to a dictionary format for resource reporting.

		Note: This is primarily for resource visualization and reporting, not for
		direct use with ray.remote(). For ray.remote(), use to_kwargs() instead.

		Returns:
		    Dictionary mapping resource names to quantities.
		"""
		resource_dict = {"CPU": self.cpu_allocation, "GPU": self.gpu_allocation}
		resource_dict.update(self.custom_resources)

		if self.hardware_type is not None:
			resource_dict[f"accelerator_type:{self.hardware_type}"] = 0.001

		return resource_dict

	@staticmethod
	def from_resource_dict(resource_spec: Dict[str, float]) -> "RayResources":
		"""
		Create a RayResources instance from a resource dictionary.

		Args:
		    resource_spec: Dictionary mapping resource names to quantities.

		Returns:
		    A new RayResources instance representing the specified resources.
		"""

		cpu_count = resource_spec.pop("CPU", 0)
		gpu_count = resource_spec.pop("GPU", 0)

		hardware_type = None
		accelerator_keys = [
			k for k in resource_spec.keys() if k.startswith("accelerator_type:")
		]
		if accelerator_keys:
			hardware_type = accelerator_keys[0].split(":", 1)[1]
			for key in accelerator_keys:
				resource_spec.pop(key)

		return RayResources(
			cpu_allocation=cpu_count,
			gpu_allocation=gpu_count,
			custom_resources=resource_spec,
			hardware_type=hardware_type,
		)

	def forkify_remote_fn(remote_fn: RemoteFunction | Callable):
		if isinstance(remote_fn, RemoteFunction):
			fn = remote_fn._function

			@functools.wraps(fn)
			def wrapped_fn(*args, **kwargs):
				return RayResources.separate_process_fn(fn, args, kwargs)

			remote_fn = RemoteFunction(
				language=remote_fn._language,
				function=wrapped_fn,
				function_descriptor=remote_fn._function_descriptor,
				task_options=remote_fn._default_options,
			)
			return remote_fn
		else:
			return functools.partial(RayResources.separate_process_fn, remote_fn)

	@staticmethod
	def separate_process_fn(underlying_function, args, kwargs):
		def target_fn(queue, args, kwargs):
			try:
				result = underlying_function(*args, **kwargs)
				queue.put((True, result))
			except Exception as e:
				info = ExceptionInfo.ser_exc_info(e)
				queue.put((False, info))

		queue = multiprocessing.Queue()
		process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
		process.start()
		process.join()

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

	@staticmethod
	def update_fn_resource_env(
		remote_fn: RemoteFunction | tp.Callable,
		runtime_env: tp.Union[tp.Dict[str, str], tp.Dict[str, tp.Dict[str, str]]],
		**extra_env,
	):
		sources = [
			e for e in [remote_fn._runtime_env, runtime_env, extra_env] if e is not None
		]
		return mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)

	@staticmethod
	def cancel_all_futures(futures):
		for future in futures:
			try:
				ray.cancel(future)
			except Exception:
				logger.exception("Failed to kill job after primary failure")


class HardwareType:
	"""
	Constants representing known accelerator and hardware types that can be requested.

	These constants provide standardized identifiers for common hardware accelerators
	to ensure consistent naming across the application.
	"""

	NVIDIA_TESLA_V100 = "V100"
	NVIDIA_TESLA_P100 = "P100"
	NVIDIA_TESLA_T4 = "T4"
	NVIDIA_TESLA_P4 = "P4"
	NVIDIA_TESLA_K80 = "K80"
	NVIDIA_TESLA_A10G = "A10G"
	NVIDIA_L4 = "L4"
	NVIDIA_L40S = "L40S"
	NVIDIA_A100 = "A100"
	NVIDIA_H100 = "H100"
	NVIDIA_H200 = "H200"
	NVIDIA_H20 = "H20"
	INTEL_MAX_1550 = "Intel-GPU-Max-1550"
	INTEL_MAX_1100 = "Intel-GPU-Max-1100"
	INTEL_GAUDI = "Intel-GAUDI"
	AMD_INSTINCT_MI100 = "AMD-Instinct-MI100"
	AMD_INSTINCT_MI250x = "AMD-Instinct-MI250X"
	AMD_INSTINCT_MI250 = "AMD-Instinct-MI250X-MI250"
	AMD_INSTINCT_MI210 = "AMD-Instinct-MI210"
	AMD_INSTINCT_MI300x = "AMD-Instinct-MI300X-OAM"
	AMD_RADEON_R9_200_HD_7900 = "AMD-Radeon-R9-200-HD-7900"
	AMD_RADEON_HD_7900 = "AMD-Radeon-HD-7900"
	AWS_NEURON_CORE = "aws-neuron-core"
	GOOGLE_TPU_V2 = "TPU-V2"
	GOOGLE_TPU_V3 = "TPU-V3"
	GOOGLE_TPU_V4 = "TPU-V4"
	GOOGLE_TPU_V5P = "TPU-V5P"
	GOOGLE_TPU_V5LITEPOD = "TPU-V5LITEPOD"
	GOOGLE_TPU_V6E = "TPU-V6E"
	HUAWEI_NPU_910B = "Ascend910B"
	HUAWEI_NPU_910B4 = "Ascend910B4"
	NVIDIA_A100_40G = "A100-40G"
	NVIDIA_A100_80G = "A100-80G"


def available_cpu_cores() -> int:
	"""
	Determine the number of logical CPU cores available on the current system.

	Returns:
	    Integer count of available logical CPU cores.
	"""
	num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
	if num_cpus is not None:
		return int(num_cpus)

	try:
		return os.cpu_count()
	except NotImplementedError:
		return 1


class ComputeResourceConfig(Protocol):
	"""
	A protocol defining the interface for hardware resource configurations.

	This protocol establishes a contract for all resource configuration classes,
	ensuring they provide the necessary methods for Ray task and actor deployment.
	The implementations are primarily used for training and inference workloads.
	"""

	execution_env: RuntimeEnv
	head_name: tp.Union[str, None] = None
	head_workers: int = 1

	def hardware_identifier(self) -> Optional[str]:
		"""
		Get the identifier for the hardware accelerator being used.

		Returns:
		    String identifier for the hardware accelerator or None if no accelerator is used.
		"""
		return None

	def get_remote_options(self) -> Dict[str, Any]:
		"""
		Get keyword arguments for ray.remote() based on this resource configuration.

		Returns:
		    Dictionary of arguments suitable for passing to ray.remote().
		"""
		return self.to_ray_resources().to_kwargs()

	def to_ray_resources(self) -> RayResources:
		"""
		Convert this configuration to a RayResources object.

		Returns:
		    RayResources instance representing the hardware resources.
		"""
		...

	def create_remote_decorator(self) -> Callable[[Any], Any]:
		"""
		Create a ray.remote decorator with this resource configuration.

		Returns:
		    A ray.remote decorator that can be applied to functions or classes.
		"""
		return ray.remote(**self.get_remote_options())

	def with_environment_variables(
		self,
		env_vars: Optional[Dict[str, str]] = None,
		/,
		**kwargs,
	) -> "ComputeResourceConfig":
		"""
		Create a new resource configuration with additional environment variables.

		This method allows for adding or overriding environment variables without
		modifying other aspects of the resource configuration.

		Args:
		    env_vars: Dictionary of environment variables to add or override
		    **kwargs: Additional environment variables as keyword arguments

		Returns:
		    A new ComputeResourceConfig with the combined environment variables.
		"""
		current_env_vars = self.execution_env.get("env_vars", {})
		new_env_vars = {**current_env_vars, **(env_vars or {}), **kwargs}
		updated_env = RuntimeEnv(**{**self.execution_env, "env_vars": new_env_vars})
		return replace(self, execution_env=updated_env)


@dataclass(frozen=True)
class CpuAcceleratorConfig(ComputeResourceConfig):
	"""
	Resource configuration for CPU-only workloads.

	This configuration is suitable for local development, batch processing,
	or any tasks that don't require hardware acceleration.
	"""

	core_count: int = field(default_factory=available_cpu_cores)
	execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
	resource_name: str = field(default="GPU")
	runtime_name: str = field(default_factory=uuid.uuid4)
	worker_count: int = 1

	def hardware_identifier(self) -> Optional[str]:
		"""
		Get the hardware identifier (none for CPU-only configuration).

		Returns:
		    None since no specialized hardware accelerator is used.
		"""
		return None

	def get_remote_options(self) -> Dict[str, Any]:
		"""
		Get Ray remote options for CPU-only execution.

		Returns:
		    Dictionary of options for ray.remote().
		"""
		return {"num_cpus": self.core_count, "runtime_env": self.execution_env}

	def to_ray_resources(self) -> RayResources:
		"""
		Convert to Ray resource specifications.

		Returns:
		    RayResources object representing CPU-only allocation.
		"""
		return RayResources(
			cpu_allocation=self.core_count,
			gpu_allocation=0,
			execution_env=self.execution_env,
		)


@dataclass(frozen=True)
class GpuAcceleratorConfig(ComputeResourceConfig):
	"""
	Resource configuration for GPU-accelerated workloads.

	This configuration specifies GPU requirements for computationally intensive
	tasks such as neural network training and inference.
	"""

	device_count: int = 1
	execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
	gpu_model: Optional[str] = None
	cpu_count: int = 1
	chips_per_host: int = field(
		default_factory=NvidiaGPUAcceleratorManager.get_current_node_num_accelerators
	)
	runtime_name: str = field(default_factory=uuid.uuid4)
	worker_count: int = 1
	resource_name: str = field(default="GPU")

	def hardware_identifier(self) -> Optional[str]:
		"""
		Get the hardware identifier for the GPU model.

		Returns:
		    String identifier for the GPU model or None if any GPU is acceptable.
		"""
		return self.gpu_model

	def get_remote_options(self) -> Dict[str, Any]:
		"""
		Get Ray remote options for GPU accelerated execution.

		Returns:
		    Dictionary of options for ray.remote().
		"""
		remote_options = {
			"num_cpus": self.cpu_count,
			"num_gpus": self.device_count,
			"runtime_env": self.execution_env,
		}

		if self.gpu_model is not None:
			remote_options["accelerator_type"] = self.gpu_model

		return remote_options

	def to_ray_resources(self) -> RayResources:
		"""
		Convert to Ray resource specifications.

		Returns:
		    RayResources object representing GPU resource allocation.
		"""
		return RayResources(
			cpu_allocation=self.cpu_count,
			gpu_allocation=self.device_count,
			hardware_type=self.gpu_model,
			execution_env=self.execution_env,
		)


@dataclass(frozen=True)
class TpuAcceleratorConfig(ComputeResourceConfig):
	"""
	Resource configuration for TPU-accelerated workloads.

	This configuration is suitable for large-scale machine learning tasks using
	Google's Tensor Processing Units (TPUs), particularly for transformer models
	and other matrix-heavy computation.
	"""

	tpu_version: str
	pod_count: int = 1
	execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
	cpu_count: int = 2
	chips_per_host: int = field(
		default_factory=TPUAcceleratorManager.get_current_node_num_accelerators
	)
	worker_count: int = field(
		default_factory=ray.util.accelerators.tpu.get_current_pod_worker_count
	)
	runtime_name: str = field(
		default_factory=ray.util.accelerators.tpu.get_current_pod_name
	)
	resource_name: str = field(default="TPU")

	def hardware_identifier(self) -> str:
		"""
		Get the hardware identifier for the TPU configuration.

		Returns:
		    String identifier for the TPU version and size.
		"""
		return self.tpu_version

	def get_remote_options(self) -> Dict[str, Any]:
		"""
		Get Ray remote options for TPU-accelerated execution.

		Returns:
		    Dictionary of options for ray.remote().
		"""
		return {
			"num_cpus": self.cpu_count,
			"resources": {self.tpu_version: self.pod_count},
			"runtime_env": self.execution_env,
		}

	def to_ray_resources(self) -> RayResources:
		"""
		Convert to Ray resource specifications for TPU resources.

		Returns:
		    RayResources object representing TPU resource allocation.
		"""
		return RayResources(
			cpu_allocation=self.cpu_count,
			custom_resources={self.tpu_version: float(self.pod_count)},
			execution_env=self.execution_env,
		)

	def redecorate_remote_fn_for_call(
		self,
		remote_fn: RemoteFunction | tp.Callable,
		**extra_envs,
	):
		remote_fn = RayResources.forkify_remote_fn(remote_fn)
		if not isinstance(remote_fn, RemoteFunction):
			remote_fn = ray.remote(remote_fn)

		tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
		runtime_env = RayResources.update_fn_resource_env(
			remote_fn=remote_fn,
			runtime_env=self.execution_env,
			**extra_envs,
		)
		remote_fn = remote_fn.options(
			runtime_env=runtime_env,
			resources={tpu_name: 1, self.resource_name: self.chips_per_host},
		)
		return remote_fn


AcceleratorConfigType: tp.TypeAlias = tp.Union[
	TpuAcceleratorConfig,
	GpuAcceleratorConfig,
	CpuAcceleratorConfig,
]
