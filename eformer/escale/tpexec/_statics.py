from dataclasses import dataclass


@dataclass
class RunInfo:
	"""Internal class to hold information about a TPU pod."""

	name: str
	state: str
	kind: str


@dataclass
class RunResult:
	"""Internal class to hold the result of a TPU job."""

	info: RunInfo


@dataclass
class RunSuccess(RunResult):
	result: object


@dataclass
class RunPreempted(RunResult):
	error: Exception


@dataclass
class RunFailed(RunResult):
	error: Exception


@dataclass
class RunError(RunResult):
	error: Exception
