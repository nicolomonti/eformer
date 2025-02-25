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
import os
import subprocess
import sys
import time

import yaml

TPU_VERSION = os.getenv("TPU_VERSION", "v4")
TPU_SLICE_SIZE = int(os.getenv("TPU_SLICE_SIZE", "8"))
TPU_CORES_PER_HOST = int(os.getenv("TPU_CORES_PER_HOST", "8"))
SSH_USER = os.getenv("PATCHER_USER")
INTERNAL_IPS = ["0.0.0.0"]
EXTERNAL_IPS = ["0.0.0.0"]


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


def run_ssh_command(target_ip, command, use_sudo=False):
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


def run_local_command(command, use_sudo=False):
	"""Run a command locally."""
	print(f"Running local command: {command}")

	if use_sudo:
		full_command = f"sudo {command}"
	else:
		full_command = command

	try:
		subprocess.run(full_command, shell=True, check=True)
		return True
	except subprocess.CalledProcessError:
		print(f"Local command failed: {command}")
		return False


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

	# Check if Ray is already running
	if check_ray_running(head_ip):
		print(f"Ray is already running on head node {head_ip}. Stopping it first...")
		if local_ip == head_ip:
			run_local_command("ray stop", False)
		else:
			run_ssh_command(head_ip, "~/.local/bin/ray stop", False)
		time.sleep(3)

	# Prepare the resources JSON string
	resources = f'{{"TPU":{TPU_CORES_PER_HOST},"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE}-head":1,"accelerator_type:TPU-{TPU_VERSION.upper()}":1}}'

	# Run locally or via SSH
	if local_ip == head_ip:
		print("Starting Ray head node locally")
		cmd = f"ray start --head --port=6379 --resources='{resources}' --node-ip-address={head_ip} --dashboard-host=0.0.0.0"
		success = run_local_command(cmd, False)
	else:
		print("Starting Ray head node remotely")
		cmd = f"~/.local/bin/ray start --head --port=6379 --resources='{resources}' --node-ip-address={head_ip} --dashboard-host=0.0.0.0"
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
			run_local_command("ray stop", False)
		else:
			run_ssh_command(worker_ip, "~/.local/bin/ray stop", False)
		time.sleep(3)

	# Prepare the resources JSON string
	resources = f'{{"TPU":{TPU_CORES_PER_HOST},"TPU-{TPU_VERSION}-worker-{i}":1,"accelerator_type:TPU-{TPU_VERSION.upper()}":1}}'

	# Run locally or via SSH
	if local_ip == worker_ip:
		print("Starting Ray worker locally")
		cmd = f"ray start --address={head_ip}:6379 --resources='{resources}' --node-ip-address={worker_ip}"
		success = run_local_command(cmd, False)
	else:
		print("Starting Ray worker remotely")
		cmd = f"~/.local/bin/ray start --address={head_ip}:6379 --resources='{resources}' --node-ip-address={worker_ip}"
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
			run_local_command("ray stop", False)
		else:
			run_ssh_command(ip, "~/.local/bin/ray stop", False)

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
	global TPU_VERSION, TPU_SLICE_SIZE, SSH_USER

	parser = argparse.ArgumentParser(description="Ray TPU Cluster Setup for TPU Slices")
	parser.add_argument(
		"--external",
		action="store_true",
		help="Use external IPs instead of internal IPs",
	)
	parser.add_argument("--stop", action="store_true", help="Stop the Ray cluster")
	parser.add_argument(
		"--verify", action="store_true", help="Verify the Ray cluster setup"
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
		"--ssh-user",
		default=SSH_USER,
		help=f"SSH username to use (default: {SSH_USER})",
	)
	parser.add_argument("--config", help="Path to YAML config file with IP addresses")
	parser.add_argument(
		"--test-ssh", action="store_true", help="Test SSH connectivity to all nodes"
	)

	args = parser.parse_args()

	# Update global variables based on arguments
	TPU_VERSION = args.tpu_version
	TPU_SLICE_SIZE = args.tpu_slice
	if args.ssh_user:
		SSH_USER = args.ssh_user

	# Read config file if provided
	if args.config:
		if not read_ips_from_yaml(args.config):
			print("Failed to read configuration from YAML file.")
			return 1

	# Test SSH connectivity if requested
	if args.test_ssh:
		if not test_ssh_connectivity(args.external):
			return 1
		return 0

	# Determine action based on arguments
	if args.stop:
		stop_cluster(args.external)
		return 0
	elif args.verify:
		head_ip = EXTERNAL_IPS[0] if args.external else INTERNAL_IPS[0]
		if verify_ray_cluster(head_ip, TPU_SLICE_SIZE):
			return 0
		else:
			return 1
	else:
		if setup_cluster(args.external):
			return 0
		else:
			return 1


if __name__ == "__main__":
	sys.exit(main())
