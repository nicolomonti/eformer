# all of this part is a copy-paste from Stanford/Tpexec
# https://github.com/stanford-crfm/levanter/blob/main/src/levanter/distributed.py

import atexit
import itertools
import logging
import os
import re
import socket
import typing as tp
from dataclasses import dataclass

import jax
import ray
from jax._src import clusters, distributed

logger = logging.getLogger("tpexec")


_JOBID_PARAM = "SLURM_JOB_ID"
_NODE_LIST_CHOICES = ["SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"]
_PROCESS_COUNT = "SLURM_NTASKS"
_PROCESS_ID = "SLURM_PROCID"
_LOCAL_PROCESS_ID = "SLURM_LOCALID"
_NUM_NODES = "SLURM_STEP_NUM_NODES"
_TASKS_PER_NODE = "SLURM_STEP_TASKS_PER_NODE"
_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_NODE_NAME = "SLURMD_NODENAME"


class TpexecSlurmCluster(clusters.SlurmCluster):
	@classmethod
	def get_coordinator_address(cls) -> str:
		id = os.environ[_JOBID_PARAM]
		port = _choose_port(id)
		node_list = TpexecSlurmCluster._node_list()
		if node_list is None:
			raise ValueError(
				"Could not find node list in environment variables. You must set coordinator_address manually."
			)
		delims = {",", "["}
		ind = next((i for i, ch in enumerate(node_list) if ch in delims), len(node_list))
		if ind == len(node_list) or node_list[ind] == ",":
			return f"{node_list[:ind]}:{port}"
		else:
			prefix = node_list[:ind]
			suffix = node_list[ind + 1 :]
			delims2 = {",", "-"}
			ind2 = next((i for i, ch in enumerate(suffix) if ch in delims2), None)
			return f"{prefix}{suffix[:ind2]}:{port}"

	@classmethod
	def _node_list(cls):
		return next((os.environ[o] for o in _NODE_LIST_CHOICES if o in os.environ), None)

	@classmethod
	def get_local_device_ids_for_process(cls) -> tp.Optional[tp.List[int]]:
		local_process_id = cls.get_local_process_id()

		if local_process_id is None:
			return None

		if _VISIBLE_DEVICES not in os.environ:
			return None

		local_process_count = cls._infer_local_process_count()

		all_visible_devices = [int(x) for x in os.environ[_VISIBLE_DEVICES].split(",")]

		if len(all_visible_devices) % local_process_count != 0:
			raise ValueError(
				f"Number of visible devices ({len(all_visible_devices)}) is not divisible by the number "
				f"of local tasks ({local_process_count})"
			)
			return None

		num_devices_per_local_process = len(all_visible_devices) // local_process_count

		begin = local_process_id * num_devices_per_local_process
		return all_visible_devices[begin : begin + num_devices_per_local_process]

	@classmethod
	def _infer_local_process_count(cls):
		node_list = TpexecSlurmCluster._node_list()
		if node_list is None:
			raise ValueError(
				"Could not find node list in environment variables. You must set coordinator_address manually."
			)

		node_list = _square_brace_expand(node_list)
		local_node = os.environ[_NODE_NAME]
		local_node_index = node_list.index(local_node)
		unrolled_tasks_per_node = []
		multi_match = re.compile(r"(\d+)\(x(\d+)\)")
		for x in os.environ[_TASKS_PER_NODE].split(","):
			match = multi_match.match(x)
			if match:
				unrolled_tasks_per_node.extend([int(match.group(1))] * int(match.group(2)))
			else:
				unrolled_tasks_per_node.append(int(x))

		tasks_on_local_node = unrolled_tasks_per_node[local_node_index]
		return tasks_on_local_node


def _square_brace_expand(node_list):
	parts = re.findall(r"(\[.*?\]|[^\[\]]+)", node_list)

	def generate_numbers(number_string):
		if "-" in number_string:
			start, end = map(int, number_string.split("-"))
			return [
				str(i).zfill(len(number_string.split("-")[0])) for i in range(start, end + 1)
			]
		else:
			return [number_string]

	processed_parts = []
	for part in parts:
		if part.startswith("[") and part.endswith("]"):
			number_sequences = part.strip("[]").split(",")
			processed_parts.append(
				list(
					itertools.chain.from_iterable(
						generate_numbers(seq) for seq in number_sequences
					)
				)
			)
		else:
			processed_parts.append([part])

	expanded_nodes = [
		"".join(combination) for combination in itertools.product(*processed_parts)
	]

	return expanded_nodes


def logical_cpu_core_count():
	"""Returns the number of logical CPU cores available to the process."""
	num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
	if num_cpus is not None:
		return int(num_cpus)

	try:
		return os.cpu_count()
	except NotImplementedError:
		return 1


def _remove_if_possible(path):
	try:
		os.remove(path)
	except OSError:
		pass


def _touch(file_path):
	with open(file_path, "a"):
		os.utime(file_path, None)


def _choose_port(id):
	port = int(id) % 2**12 + (65535 - 2**12 + 1)
	return port


def _is_this_machine(host):
	"""
	Checks if the given host identifies this machine.
	"""
	if host == "localhost" or host == "0.0.0.0":
		return True
	try:
		machine_ips = [
			addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)
		]
		host_ip = socket.gethostbyname(host)
	except socket.gaierror:
		return False
	return any(host_ip == machine_ip for machine_ip in machine_ips)


def _is_local_leader():
	import atexit

	import filelock
	from jax.experimental.multihost_utils import broadcast_one_to_all

	if jax.process_count() == 1:
		return True

	import random

	random_id = random.randint(0, 1000000)
	random_id = broadcast_one_to_all(random_id)

	lock = filelock.FileLock(f"/tmp/tpexec_local_process_zero_lock.{random_id}")
	action_performed_file = f"/tmp/tpexec_local_process_zero_action_performed.{random_id}"

	try:
		with lock.acquire(timeout=0.1):
			if not os.path.exists(action_performed_file):
				_touch(action_performed_file)
				return True
			else:
				return False
			atexit.register(_remove_if_possible, lock.lock_file)
			atexit.register(_remove_if_possible, action_performed_file)
	except filelock.Timeout:
		return False


_already_initialized = False


def auto_ray_cluster(
	address: tp.Optional[str] = None,
	namespace: tp.Optional[str] = "tpexec",
	start_workers: bool = True,
	fail_if_cluster_already_initialized: bool = False,
	**kwargs,
):
	global _already_initialized

	if _already_initialized:
		logger.warning(
			"auto_ray_cluster has already been called. Ignoring subsequent calls."
		)
		return

	def _munge_address_port(address: str):
		host, port_str = address.split(":")
		port = int(port_str)
		return host, port

	if address is None:
		if os.getenv("RAY_ADDRESS") is not None:
			address = os.getenv("RAY_ADDRESS")
			logger.info("Auto-discovered ray address using RAY_ADDRESS: %s", address)
		else:
			coord_address = getattr(distributed.global_state, "coordinator_address", None)
			if coord_address is None:
				logger.info("No auto-discovered ray address found. Using ray.init('local').")
				address = "local"
			else:
				logger.info(
					f"Auto-discovered ray address using JAX coordinator address: {coord_address}"
				)
				host, port = _munge_address_port(coord_address)

				ray_port = _choose_port(port + 240)
				address = f"{host}:{ray_port}"
				num_cpus = logical_cpu_core_count()

				if _is_local_leader():
					if _is_this_machine(host):
						logger.info(
							f"Starting ray head on port {ray_port}. We are process the coordinator {host}."
						)
						logger.info(f"Starting ray head with num_cpus set to {num_cpus}.")
						ret = os.system(
							f"ray start --head --port {ray_port} --num-cpus {num_cpus} --dashboard-host=0.0.0.0"
						)
						if ret != 0:
							if not fail_if_cluster_already_initialized:
								logger.warning(
									f"Failed to start ray head with exit code {ret}. Checking if we can connect to"
									" the head..."
								)
								ret = os.system("ray status")
								if ret != 0:
									raise RuntimeError(f"Failed to start ray head with exit code {ret}")
								else:
									logger.info(
										f"Ray head already running on port {ray_port}. Connecting to it."
									)
							else:
								raise RuntimeError(f"Failed to start ray head with exit code {ret}")
						else:
							logger.info(f"Successfully started ray head on port {ray_port}.")

						atexit.register(lambda: os.system("ray stop -g 10 --force &> /dev/null"))
					elif start_workers:
						logger.info(
							f"Starting ray worker and connecting to {address}. We are process {jax.process_index()}."
						)
						logger.info(f"Starting ray worker with num_cpus set to {num_cpus}.")
						ret = os.system(f"ray start --address {address} --num-cpus {num_cpus}")
						if ret != 0:
							raise RuntimeError(f"Failed to start ray head with exit code {ret}")
						else:
							logger.info(
								f"Successfully started ray worker and connected to {address}."
							)

	logger.info(
		f"ray.init(address={repr(address)}, namespace={repr(namespace)}, **{repr(kwargs)})"
	)

	for i in range(0, 5):
		try:
			ray.init(address=address, namespace=namespace, **kwargs)
			break
		except Exception as e:
			if i == 4:
				raise e
			else:
				logger.warning(f"Failed to initialize ray with address {address}. Retrying...")
				continue

	def do_shutdown():
		logger.info("Shutting down ray...")
		ray.shutdown()

	atexit.register(do_shutdown)
	_already_initialized = True


@dataclass(frozen=True)
class DistributedConfig:
	coordinator_address: tp.Optional[str] = None
	num_processes: tp.Optional[int] = None
	process_id: tp.Optional[int] = None
	local_device_ids: tp.Optional[tp.Union[int, tp.List[int]]] = None

	def _is_distributed(self):
		if (
			(self.coordinator_address is not None)
			or (self.num_processes is not None)
			or (self.process_id is not None)
			or (self.local_device_ids is not None)
		):
			return True

		if any(env.is_env_present() for env in clusters.ClusterEnv._cluster_types):
			return True

		return False

	def initialize(self):
		if self._is_distributed():
			device_ids = self.local_device_ids
			coordinator_address = self.coordinator_address

			if TpexecSlurmCluster.is_env_present():
				if device_ids is None:
					device_ids = TpexecSlurmCluster.get_local_device_ids_for_process()

				if coordinator_address is None:
					coordinator_address = TpexecSlurmCluster.get_coordinator_address()

			jax.distributed.initialize(
				coordinator_address,
				self.num_processes,
				self.process_id,
				device_ids,
				initialization_timeout=30 * 60,
			)
			logger.info(
				f"Initialized jax.distributed with {jax.device_count()} devices, {jax.process_count()} processes,"
				f" coordinator_address={coordinator_address}, process_id={self.process_id}, my"
				f" device_ids={device_ids}."
			)
		else:
			logger.info(
				"Not initializing jax.distributed because no distributed config "
				"was provided, and no cluster was detected."
			)


@dataclass
class RayConfig:
	address: tp.Optional[str] = None
	start_workers: bool = True
	auto_start_cluster: bool = True

	def initialize(self):
		if self.auto_start_cluster:
			auto_ray_cluster(address=self.address, start_workers=self.start_workers)
