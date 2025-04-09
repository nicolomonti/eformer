"""
Direct suppression of Triton kernel autotuning logs.

This module provides an alternative approach to suppress Triton kernel autotuning logs
by directly redirecting stderr in C using ctypes.
"""

import ctypes
import logging
import os
import sys
from contextlib import contextmanager

# Load the C library
try:
	if sys.platform == "win32":
		libc = ctypes.CDLL("msvcrt")
	else:
		libc = ctypes.CDLL(None)

	# Get file descriptor for stderr (usually 2)
	STDERR_FILENO = 2

	# Check if we can access the functions we need
	if hasattr(libc, "dup") and hasattr(libc, "dup2"):
		SUPPRESSION_AVAILABLE = True
	else:
		SUPPRESSION_AVAILABLE = False
except (OSError, AttributeError):
	SUPPRESSION_AVAILABLE = False


class OutputSuppressor:
	"""Class to suppress C/C++ stderr output (including XLA/JAX/Triton logs)."""

	def __init__(self):
		self.null_fd = None
		self.old_stderr_fd = None
		self.suppressing = False

	def start(self):
		"""Start suppressing C-level stderr output."""
		if not SUPPRESSION_AVAILABLE or self.suppressing:
			return False

		try:
			# Save original stderr fd
			self.old_stderr_fd = libc.dup(STDERR_FILENO)

			# Open a null file to redirect stderr to
			if sys.platform == "win32":
				self.null_fd = os.open("NUL", os.O_WRONLY)
			else:
				self.null_fd = os.open("/dev/null", os.O_WRONLY)

			# Redirect stderr to the null file
			libc.dup2(self.null_fd, STDERR_FILENO)
			self.suppressing = True
			return True
		except (OSError, AttributeError):
			self.cleanup()
			return False

	def stop(self):
		"""Stop suppressing C-level stderr output and restore original stderr."""
		if not self.suppressing:
			return

		try:
			# Restore original stderr
			if self.old_stderr_fd is not None:
				libc.dup2(self.old_stderr_fd, STDERR_FILENO)
		finally:
			self.cleanup()

	def cleanup(self):
		"""Clean up resources."""
		if self.old_stderr_fd is not None:
			try:
				os.close(self.old_stderr_fd)
			except OSError:
				pass
			self.old_stderr_fd = None

		if self.null_fd is not None:
			try:
				os.close(self.null_fd)
			except OSError:
				pass
			self.null_fd = None

		self.suppressing = False

	def __del__(self):
		"""Ensure resources are cleaned up."""
		self.stop()


# Global suppressor instance
_suppressor = OutputSuppressor()


def suppress_triton_logs():
	"""Start suppressing Triton kernel autotuning logs at the C++ level."""
	return _suppressor.start()


def restore_triton_logs():
	"""Restore normal logging behavior."""
	_suppressor.stop()


def silence_all_triton_output():
	"""Apply all available methods to silence Triton autotuning output.

	This is the most comprehensive approach that combines multiple methods:
	1. Sets environment variables to control XLA/JAX logging
	2. Redirects C-level stderr output
	3. Sets Python logging to ERROR level

	Returns:
		True if suppression was fully successful, False otherwise
	"""
	logging.getLogger("triton").setLevel(logging.ERROR)
	logging.getLogger("jax").setLevel(logging.ERROR)
	logging.getLogger("xla_bridge").setLevel(logging.ERROR)
	c_suppression = suppress_triton_logs()
	return c_suppression


def enable_all_triton_output():
	"""Reverse all suppression methods and restore normal logging.

	This function:
	1. Restores environment variables to default values
	2. Restores C-level stderr output
	3. Sets Python logging back to INFO level
	"""
	logging.getLogger("triton").setLevel(logging.INFO)
	logging.getLogger("jax").setLevel(logging.INFO)
	logging.getLogger("xla_bridge").setLevel(logging.INFO)

	restore_triton_logs()


@contextmanager
def no_triton_logs():
	"""Context manager to temporarily suppress Triton kernel autotuning logs."""
	started = suppress_triton_logs()
	try:
		yield started
	finally:
		if started:
			restore_triton_logs()


if __name__ == "__main__":
	# Example usage
	with no_triton_logs():
		print("Inside the context manager - C/C++ stderr is suppressed")

	# Or manual control
	suppress_triton_logs()
	print("C/C++ stderr is now suppressed")
	restore_triton_logs()
	print("C/C++ stderr is restored")
