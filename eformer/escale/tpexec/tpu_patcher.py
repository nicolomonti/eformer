#!/usr/bin/env python3
# eformer.escale.tpexec.tpu_patcher - Ray TPU Cluster Setup for TPU Slices
# This script properly configures Ray to recognize all TPU cores across hosts.
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


import argparse
import json
import os
import subprocess
import sys
import time
import pathlib
import requests
import yaml

TPU_VERSION = os.getenv("TPU_VERSION", "v4")
TPU_SLICE_SIZE = int(os.getenv("TPU_SLICE_SIZE", "8"))
TPU_CORES_PER_HOST = int(os.getenv("TPU_CORES_PER_HOST", "8"))
SSH_USER = os.getenv("PATCHER_USER")
INTERNAL_IPS = ["0.0.0.0"]
EXTERNAL_IPS = ["0.0.0.0"]

RAY_PATH = f"{pathlib.Path.home()}/.local/bin/ray"


def get_external_ip():
	"""Get the external IP address of the machine."""
	try:
		# Note
		# this method may fail if users using vpn, but why should someone use vpn on TPUs

		# Using a reliable service that returns just the IP
		response = requests.get("https://api.ipify.org", timeout=5)
		if response.status_code == 200:
			return response.text

		# Fallback services if the first one fails
		response = requests.get("https://ipinfo.io/ip", timeout=5)
		if response.status_code == 200:
			return response.text

		response = requests.get("https://checkip.amazonaws.com", timeout=5)
		if response.status_code == 200:
			return response.text.strip()

		return "Could not determine external IP"
	except Exception as e:
		return f"Error getting external IP: {e}"


def get_local_ip():
	"""Get the local IP address of the machine."""
	try:
		hostname_output = (
			subprocess.check_output("hostname -I", shell=True).decode().strip()
		)
		return hostname_output.split()[0]
	except Exception as e:
		print(f"Error getting local IP: {e}")
		return "127.0.0.1"


def run_ssh_command(
	target_ip,
	command,
	use_sudo=False,
):
	"""Run a command on a remote host via SSH."""
	print(f"Running SSH command on {target_ip}: {command}")

	if use_sudo:
		full_command = f"sudo {command}"
	else:
		full_command = command

	ssh_command = [
		"ssh",
		"-o",
		"StrictHostKeyChecking=no",
		"-o",
		"ConnectTimeout=5",
		f"{SSH_USER}@{target_ip}",
		full_command,
	]

	try:
		subprocess.run(ssh_command, check=True)
		return True
	except subprocess.CalledProcessError:
		print(f"SSH command failed on {target_ip}")
		return False


def run_local_command(
	command,
	use_sudo=False,
	check=True,
	capture_output=False,
):
	"""Run a command locally with optional output capture."""
	if isinstance(command, list):
		cmd_str = " ".join(command)
	else:
		cmd_str = command

	print(f"Running local command: {cmd_str}")

	if use_sudo and not isinstance(command, list):
		full_command = f"sudo {command}"
	elif use_sudo and isinstance(command, list):
		full_command = ["sudo"] + command
		full_command = " ".join(full_command)
	else:
		full_command = command
		if isinstance(full_command, list):
			full_command = " ".join(full_command)
	try:
		if capture_output:
			if isinstance(full_command, list):
				result = subprocess.run(
					[full_command],
					shell=True,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE,
				)
			else:
				result = subprocess.run(
					[full_command],
					shell=True,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE,
				)
			return result.stdout
		else:
			if isinstance(full_command, list):
				subprocess.run(full_command, shell=True, check=check)
			else:
				subprocess.run(full_command, shell=True, check=check)
			return True
	except subprocess.CalledProcessError:
		print(f"Local command failed: {cmd_str}")
		return False if not capture_output else ""
	except FileNotFoundError:
		print(f"Command not found: {cmd_str}")
		return False if not capture_output else ""


def create_verification_script():
	"""Create a Python script to verify the Ray cluster setup."""
	script_content = """
import ray
import time
import sys
import os

# Try to connect multiple times
for i in range(5):
    try:
        ray.init(address='auto')
        break
    except Exception as e:
        print(f"Connection attempt {i+1} failed: {e}")
        time.sleep(5)
        if i == 4:  # Last attempt failed
            print("Could not connect to Ray cluster")
            sys.exit(1)

# Print cluster resources
try:
    resources = ray.cluster_resources()
    print("\\nCLUSTER RESOURCES:\\n==================")
    for k, v in sorted(resources.items()):
        print(f"{k}: {v}")
    
    # Check if we have the expected TPUs
    tpu_count = resources.get('TPU', 0)
    print(f"\\nTOTAL TPU COUNT: {tpu_count}")
    
    # Get expected TPU count from environment variable
    expected_tpu = int(os.environ.get('EXPECTED_TPU_COUNT', 64))
    if tpu_count < expected_tpu * 0.9:  # Allow for 10% missing
        print(f"WARNING: Not all TPU cores are detected! Expected ~{expected_tpu}")
        sys.exit(2)
    else:
        print(f"SUCCESS: TPU cores detected ({tpu_count}/{expected_tpu})")
except Exception as e:
    print(f"Error getting cluster resources: {e}")
    sys.exit(3)

ray.shutdown()
"""

	with open("/tmp/verify_ray_tpu.py", "w") as f:
		f.write(script_content)


def check_ray_running(ip):
	"""Check if Ray is already running on a node."""
	local_ip = get_local_ip()

	if local_ip == ip:
		try:
			subprocess.check_output("ps aux | grep -v grep | grep 'ray::IDLE'", shell=True)
			return True
		except subprocess.CalledProcessError:
			return False
	else:
		return run_ssh_command(ip, "ps aux | grep -v grep | grep 'ray::IDLE'", False)


def start_ray_head(head_ip, use_external):
	"""Start Ray head node."""
	global TPU_CORES_PER_HOST, TPU_VERSION, TPU_SLICE_SIZE

	print(f"Starting Ray head node on {head_ip}")
	local_ip = get_local_ip()

	if check_ray_running(head_ip):
		print(f"Ray is already running on head node {head_ip}. Stopping it first...")
		if local_ip == head_ip:
			run_local_command([RAY_PATH, "stop"], False)
		else:
			run_ssh_command(head_ip, f"{RAY_PATH} stop", False)
		time.sleep(3)

	# Prepare the resources JSON string
	resources = json.dumps(
		{
			"TPU": TPU_CORES_PER_HOST,
			f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE}-head": 1,
			f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
		}
	)

	# Run locally or via SSH
	if local_ip == head_ip:
		print("Starting Ray head node locally")
		cmd = (
			RAY_PATH,
			"start",
			"--head",
			"--port=6379",
			f"--resources='{resources}'",
			f"--node-ip-address={head_ip}",
			"--dashboard-host=0.0.0.0",
		)
		success = run_local_command(cmd, False)
	else:
		print("Starting Ray head node remotely")
		cmd = f"{RAY_PATH} start --head --port=6379 --resources='{resources}' --node-ip-address={head_ip} --dashboard-host=0.0.0.0"
		success = run_ssh_command(head_ip, cmd, False)

	if not success:
		print("Failed to start Ray head node")
		return False

	return True


def start_ray_worker(head_ip, worker_ip, use_external, worker_count):
	"""Start Ray worker node."""
	global TPU_CORES_PER_HOST, TPU_VERSION, INTERNAL_IPS
	i = 0
	for idx, ip in enumerate(INTERNAL_IPS):
		if ip == worker_ip:
			i = idx - 1
			break

	print(f"Starting Ray worker on {worker_ip} (index {i})")
	local_ip = get_local_ip()
	if check_ray_running(worker_ip):
		print(f"Ray is already running on worker node {worker_ip}. Stopping it first...")
		if local_ip == worker_ip:
			run_local_command([RAY_PATH, "stop"], False)
		else:
			run_ssh_command(worker_ip, f"{RAY_PATH} stop", False)
		time.sleep(3)

	# Prepare the resources JSON string
	resources = f'{{"TPU":{TPU_CORES_PER_HOST},"TPU-{TPU_VERSION}-worker-{i}":1,"accelerator_type:TPU-{TPU_VERSION.upper()}":1}}'

	# Run locally or via SSH
	if local_ip == worker_ip:
		print("Starting Ray worker locally")
		cmd = f"{RAY_PATH} start --address={head_ip}:6379 --resources='{resources}' --node-ip-address={worker_ip}"
		success = run_local_command(cmd, False)
	else:
		print("Starting Ray worker remotely")
		cmd = f"{RAY_PATH} start --address={head_ip}:6379 --resources='{resources}' --node-ip-address={worker_ip}"
		success = run_ssh_command(worker_ip, cmd, False)

	if success:
		print(f"Successfully started Ray worker on {worker_ip}")
		return True
	else:
		print(f"Failed to start Ray worker on {worker_ip}")
		return False


def verify_ray_cluster(head_ip, expected_tpu_count):
	"""Verify Ray cluster setup."""
	print("Verifying Ray cluster setup...")
	create_verification_script()

	# Run the verification script with expected TPU count
	env = os.environ.copy()
	env["EXPECTED_TPU_COUNT"] = str(expected_tpu_count)

	try:
		subprocess.run(["python", "/tmp/verify_ray_tpu.py"], env=env, check=True)
		print("Verification successful!")
		status = 0
	except subprocess.CalledProcessError as e:
		print(f"Verification failed with status {e.returncode}")
		status = e.returncode

	# Clean up
	os.remove("/tmp/verify_ray_tpu.py")
	return status == 0


def stop_cluster(use_external):
	"""Stop Ray cluster on all nodes."""
	global EXTERNAL_IPS, INTERNAL_IPS

	ips = EXTERNAL_IPS if use_external else INTERNAL_IPS

	print("Stopping Ray on all nodes...")
	local_ip = get_local_ip()

	for ip in ips:
		print(f"Stopping Ray on {ip}...")

		if local_ip == ip:
			print("Stopping Ray locally")
			run_local_command(f"{RAY_PATH} stop", False)
		else:
			run_ssh_command(ip, f"{RAY_PATH} stop", False)

	print("Ray cluster stopped on all nodes.")


def setup_cluster(use_external):
	"""Set up Ray cluster."""
	global TPU_CORES_PER_HOST, TPU_VERSION, TPU_SLICE_SIZE, EXTERNAL_IPS, INTERNAL_IPS

	ips = EXTERNAL_IPS if use_external else INTERNAL_IPS

	head_ip = ips[0]
	worker_ips = ips[1:]

	# Calculate TPU cores per host based on slice size and host count
	TPU_CORES_PER_HOST = TPU_SLICE_SIZE // len(ips)

	print(f"Setting up Ray cluster with {'external' if use_external else 'internal'} IPs")
	print(f"TPU Version: {TPU_VERSION}")
	print(f"TPU Slice Size: {TPU_SLICE_SIZE}")
	print(f"TPU Cores per Host: {TPU_CORES_PER_HOST}")
	print(f"SSH User: {SSH_USER}")
	print(f"Head node: {head_ip}")
	print(f"Worker nodes: {', '.join(worker_ips)}")

	# Stop any existing Ray processes first
	stop_cluster(use_external)
	print("Waiting for processes to fully stop...")
	time.sleep(5)

	# Start the head node
	if not start_ray_head(head_ip, use_external):
		print("Failed to start Ray head node. Exiting.")
		return False

	print("Waiting for head node to initialize...")
	time.sleep(10)

	# Start worker nodes
	for worker_ip in worker_ips:
		print(f"Starting worker node at {worker_ip}")
		start_ray_worker(head_ip, worker_ip, use_external, len(worker_ips))

	print("Ray cluster setup complete!")
	print(f"Total expected TPU cores: {TPU_SLICE_SIZE}")

	# Allow some time for workers to fully register
	print("Waiting for workers to register resources...")
	time.sleep(15)

	# Verify the cluster
	verify_ray_cluster(head_ip, TPU_SLICE_SIZE)

	print("")
	print("IMPORTANT: To use Ray in your applications, initialize with:")
	print(f"ray.init(address='{head_ip}:6379')")

	return True


def test_ssh_connectivity(use_external):
	"""Test SSH connectivity to all nodes."""
	global EXTERNAL_IPS, INTERNAL_IPS, SSH_USER

	ips = EXTERNAL_IPS if use_external else INTERNAL_IPS

	print("Testing SSH connectivity to all nodes...")
	all_good = True

	for ip in ips:
		print(f"Testing connection to {ip}... ", end="")

		ssh_command = [
			"ssh",
			"-o",
			"StrictHostKeyChecking=no",
			"-o",
			"ConnectTimeout=5",
			"-o",
			"BatchMode=yes",
			f"{SSH_USER}@{ip}",
			"echo OK",
		]

		try:
			subprocess.run(
				ssh_command,
				check=True,
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
			)
			print("SUCCESS")
		except subprocess.CalledProcessError:
			print("FAILED")
			all_good = False

	if all_good:
		print("All SSH connections successful!")
		return True
	else:
		print("Some SSH connections failed. Please check your SSH setup.")
		print("Make sure your SSH key is added to the authorized_keys file on all nodes.")
		print(f"You may need to run: ssh-copy-id {SSH_USER}@<node-ip>")
		return False


def read_ips_from_yaml(yaml_file):
	"""Read IPs from YAML file."""
	global INTERNAL_IPS, EXTERNAL_IPS

	try:
		with open(yaml_file, "r") as file:
			data = yaml.safe_load(file)

		# Extract internal IPs
		INTERNAL_IPS = data.get("internal_ips", [])

		# Extract external IPs
		EXTERNAL_IPS = data.get("external_ips", [])

		# Validate that we have at least one IP
		if not INTERNAL_IPS:
			print("ERROR: No internal IPs found in the YAML file", file=sys.stderr)
			return False

		print(
			f"Read {len(INTERNAL_IPS)} internal IPs and {len(EXTERNAL_IPS)} external IPs from {yaml_file}"
		)
		return True

	except Exception as e:
		print(f"ERROR: {str(e)}", file=sys.stderr)
		return False


def main():
	global TPU_VERSION, TPU_SLICE_SIZE, SSH_USER, INTERNAL_IPS, EXTERNAL_IPS

	parser = argparse.ArgumentParser(description="Ray TPU Cluster Setup for TPU Slices")
	parser.add_argument(
		"--external",
		action="store_true",
		help="Use external IPs instead of internal IPs",
	)
	parser.add_argument(
		"--stop",
		action="store_true",
		help="Stop the Ray cluster",
	)
	parser.add_argument(
		"--verify",
		action="store_true",
		help="Verify the Ray cluster setup",
	)
	parser.add_argument(
		"--tpu-version",
		default=TPU_VERSION,
		help=f"Set TPU version (default: {TPU_VERSION})",
	)
	parser.add_argument(
		"--tpu-slice",
		type=int,
		default=TPU_SLICE_SIZE,
		help=f"Set TPU slice size (default: {TPU_SLICE_SIZE})",
	)
	parser.add_argument(
		"--num-slices",
		type=int,
		default=1,
		help="Number of TPU slices to combine (default: 1)",
	)
	parser.add_argument(
		"--ssh-user",
		default=SSH_USER,
		help=f"SSH username to use (default: {SSH_USER})",
	)
	parser.add_argument(
		"--config",
		help="Path to YAML config file with IP addresses",
	)
	parser.add_argument(
		"--test-ssh",
		action="store_true",
		help="Test SSH connectivity to all nodes",
	)
	# New arguments
	parser.add_argument(
		"--internal-ips",
		help="Comma-separated list of internal IPs for all slices (e.g. 10.164.0.3,10.164.0.7,...)",
	)
	parser.add_argument(
		"--external-ips",
		help="Comma-separated list of external IPs for all slices (e.g. 35.224.1.1,35.224.1.2,...)",
	)
	parser.add_argument(
		"--self-job",
		action="store_true",
		help="Run only on the current machine (no SSH to other machines)",
	)
	parser.add_argument(
		"--slice-config",
		help="Path to YAML config file with slice configurations",
	)

	args = parser.parse_args()
	using_external = args.external or args.internal_ips is None
	# Update global variables based on arguments
	TPU_VERSION = args.tpu_version
	TPU_SLICE_SIZE = args.tpu_slice
	num_slices = args.num_slices
	total_tpu_cores = TPU_SLICE_SIZE * num_slices

	if args.ssh_user:
		SSH_USER = args.ssh_user

	# Read IPs from command line if provided
	if args.internal_ips:
		INTERNAL_IPS = args.internal_ips.split(",")
		print(f"Using internal IPs from command line: {INTERNAL_IPS}")

	if args.external_ips:
		EXTERNAL_IPS = args.external_ips.split(",")
		print(f"Using external IPs from command line: {EXTERNAL_IPS}")

	# Read config file if provided (this will override command line IPs)
	if args.config:
		if not read_ips_from_yaml(args.config):
			print("Failed to read configuration from YAML file.")
			return 1

	# Read slice config if provided
	slice_configs = []
	if args.slice_config:
		try:
			with open(args.slice_config, "r") as file:
				slice_data = yaml.safe_load(file)
				slice_configs = slice_data.get("slices", [])
				print(f"Loaded {len(slice_configs)} slice configurations")
		except Exception as e:
			print(f"Error reading slice config: {e}")
			return 1
	else:
		# Create default slice config based on IPs and num_slices
		ips_to_use = EXTERNAL_IPS if using_external else INTERNAL_IPS

		# Calculate hosts per slice
		total_hosts = len(ips_to_use)
		if total_hosts % num_slices != 0:
			print(
				f"Warning: {total_hosts} hosts cannot be evenly divided into {num_slices} slices"
			)

		hosts_per_slice = total_hosts // num_slices

		# Create slice configs
		for i in range(num_slices):
			start_idx = i * hosts_per_slice
			end_idx = start_idx + hosts_per_slice
			slice_ips = ips_to_use[start_idx:end_idx]

			slice_configs.append(
				{
					"name": f"slice-{i + 1}",
					"ips": slice_ips,
					"tpu_cores": TPU_SLICE_SIZE,
				}
			)

	# Print slice configuration
	print("\nSlice Configuration:")
	print("====================")
	for i, slice_config in enumerate(slice_configs):
		print(f"Slice {i + 1}: {slice_config['name']}")
		print(f"  IPs: {slice_config['ips']}")
		print(f"  TPU Cores: {slice_config['tpu_cores']}")
	print(f"Total TPU Cores: {total_tpu_cores}")
	print("====================\n")

	if args.self_job:
		local_ip = get_external_ip() if using_external else get_local_ip()
		print(f"Running in self-job mode on {local_ip}")
		print(f"Using Ray command: {RAY_PATH}")
		try:
			subprocess.run(
				[RAY_PATH, "--version"],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				check=True,
			)
		except (subprocess.CalledProcessError, FileNotFoundError) as e:
			print("ERROR: Ray command not found or not working.")
			print("Please install Ray with: pip install -U ray")
			print("Make sure ray is in your PATH or in ~/.local/bin/")
			print("Exec", str(e))
			return 1

		print("Stopping any existing Ray processes...")
		try:
			subprocess.run([RAY_PATH, "stop"], stderr=subprocess.DEVNULL)
		except Exception:
			print("No Ray process was running or could not stop Ray")

		time.sleep(3)
		if not ips_to_use and EXTERNAL_IPS:
			print("Warning: No internal IPs provided but external IPs exist.")
			print("For self-job mode, internal IPs are required.")
			print("Attempting to use local IP as internal IP...")
			ips_to_use = [local_ip]  # Use local IP as fallback

		# Find which slice this machine belongs to
		slice_idx = -1
		host_idx_in_slice = -1

		for i, slice_config in enumerate(slice_configs):
			# Update slice_config IPs to use internal IPs if available
			if args.self_job:
				# If slice config has external IPs but we need internal, try to match by position
				if local_ip in slice_config["ips"]:
					slice_idx = i
					host_idx_in_slice = slice_config["ips"].index(local_ip)
					break

		# If not found in slice configs, try to find in global IP list
		if slice_idx == -1:
			if local_ip in ips_to_use:
				# Find which slice this would belong to based on distribution
				host_idx = ips_to_use.index(local_ip)
				hosts_per_slice = len(ips_to_use) // num_slices
				slice_idx = host_idx // hosts_per_slice
				host_idx_in_slice = host_idx % hosts_per_slice
			else:
				print(
					f"Error: Local IP {local_ip} not found in any slice configuration or IP list"
				)
				return 1

		# If we still have slice configs from earlier, use them
		# Otherwise reconstruct the slice config for this node
		if slice_configs:
			slice_config = slice_configs[slice_idx]
		else:
			# Create a basic slice config based on position
			hosts_per_slice = len(ips_to_use) // num_slices
			start_idx = slice_idx * hosts_per_slice
			end_idx = start_idx + hosts_per_slice
			slice_ips = ips_to_use[start_idx:end_idx]
			slice_config = {
				"name": f"slice-{slice_idx + 1}",
				"ips": slice_ips,
				"tpu_cores": TPU_SLICE_SIZE,
			}

		# Always use internal IPs for head nodes in self-job mode
		head_ip = ips_to_use[0]  # Global head is first IP of internal IPs
		slice_head_ip = slice_config["ips"][0]  # Slice head is first IP of this slice

		# Determine TPU cores per host for this slice
		slice_tpu_cores = slice_config.get("tpu_cores", TPU_SLICE_SIZE)
		hosts_in_slice = len(slice_config["ips"])
		tpu_cores_per_host = slice_tpu_cores // hosts_in_slice

		if args.stop:
			run_local_command(f"{RAY_PATH} stop", False)
			return 0

		# Determine if this is the global head, a slice head, or a worker
		if local_ip == head_ip:
			print("This machine is the global head node")
			# Start head node with specific resources
			resources = {
				"TPU": tpu_cores_per_host,
				f"TPU-{TPU_VERSION}-{slice_tpu_cores}-head": 1,  # Standard head marker
				f"TPU-{TPU_VERSION}-{total_tpu_cores}-global-head": 1,  # Global head marker
				f"slice-{slice_idx}": 1,
				f"slice-{slice_idx}-host-{host_idx_in_slice}": 1,
				f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
			}

			resources_str = str(resources).replace("'", '"')
			cmd = [
				RAY_PATH,
				"start",
				"--head",
				"--port=6379",
				f"--resources='{resources_str}'",
				f"--node-ip-address={local_ip}",
				"--dashboard-host=0.0.0.0",
			]
			run_local_command(cmd, False)

		elif local_ip == slice_head_ip and local_ip != head_ip:
			print(
				f"This machine is a slice head node for slice {slice_idx + 1}, connecting to global head at {head_ip}"
			)
			# Start slice head as a worker connecting to global head
			# Include the standard head marker for this slice
			resources = json.dumps(
				{
					"TPU": tpu_cores_per_host,
					f"TPU-{TPU_VERSION}-{slice_tpu_cores}-head": 1,
					f"TPU-{TPU_VERSION}-{slice_tpu_cores}-slice-{slice_idx}-head": 1,
					f"slice-{slice_idx}": 1,
					f"slice-{slice_idx}-host-{host_idx_in_slice}": 1,
					f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
				}
			)
			cmd = [
				RAY_PATH,
				"start",
				"--address",
				f"{head_ip}:6379",
				f"--resources='{resources}'",
				f"--node-ip-address={local_ip}",
			]
			run_local_command(cmd, False)

		else:
			print(
				f"This machine is a worker node in slice {slice_idx + 1}, connecting to global head at {head_ip}"
			)
			# Start worker node
			resources = json.dumps(
				{
					"TPU": tpu_cores_per_host,
					f"TPU-{TPU_VERSION}-worker": 1,
					f"slice-{slice_idx}": 1,
					f"slice-{slice_idx}-host-{host_idx_in_slice}": 1,
					f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
				}
			)

			cmd = f"{RAY_PATH} start --address={head_ip}:6379 --resources='{resources}' --node-ip-address={local_ip}"
			run_local_command(cmd, False)

		return 0

	# Test SSH connectivity if requested
	if args.test_ssh:
		if not test_ssh_connectivity(using_external):
			return 1
		return 0

	if args.stop:
		stop_cluster(using_external)
		return 0
	elif args.verify:
		head_ip = EXTERNAL_IPS[0] if using_external else INTERNAL_IPS[0]
		if verify_ray_cluster(head_ip, total_tpu_cores):
			return 0
		else:
			return 1
	else:
		if setup_cluster(using_external):
			return 0
		else:
			return 1


if __name__ == "__main__":
	sys.exit(main())
