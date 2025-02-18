from dataclasses import dataclass
import typing as tp


@dataclass
class TPUInfo:
	"""Information about a TPU pod."""

	name: str
	state: str
	kind: str


@dataclass
class TPURunResult:
	"""Base class for TPU job results."""

	info: TPUInfo


@dataclass
class TPUSuccess(TPURunResult):
	"""Successful TPU job execution."""

	result: tp.Any


@dataclass
class TPUPreempted(TPURunResult):
	"""TPU job was preempted."""

	error: Exception


@dataclass
class TPUFailed(TPURunResult):
	"""TPU job failed due to an error."""

	error: Exception


@dataclass
class TPURunError(TPURunResult):
	"""TPU job encountered a runtime error."""

	error: Exception
