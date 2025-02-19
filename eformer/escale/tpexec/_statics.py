from dataclasses import dataclass


@dataclass
class TpuInfo:
	"""Internal class to hold information about a TPU pod."""

	name: str
	state: str
	kind: str


@dataclass
class TpuRunResult:
	"""Internal class to hold the result of a TPU job."""

	info: TpuInfo


@dataclass
class TpuSuccess(TpuRunResult):
	result: object


@dataclass
class TpuPreempted(TpuRunResult):
	error: Exception


@dataclass
class TpuFailed(TpuRunResult):
	error: Exception


@dataclass
class TpuRunError(TpuRunResult):
	error: Exception
