# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass

import ray
import tblib
from ray.exceptions import NodeDiedError, RayError, RaySystemError, RayTaskError, WorkerCrashedError
from tblib import Traceback

logger = logging.getLogger("ray")


def handle_ray_error(job_info: JobInfo, e: RayError):
    if isinstance(e, NodeDiedError):
        logger.exception("Node died", exc_info=e)
        return JobPreempted(job_info, e)
    elif isinstance(e, ray.exceptions.ActorUnavailableError | ray.exceptions.ActorDiedError):
        logger.exception("Actor died", exc_info=e)
        return JobPreempted(job_info, e)
    elif isinstance(e, WorkerCrashedError):
        logger.exception("Worker crashed", exc_info=e)
        return JobPreempted(job_info, e)
    elif isinstance(e, RaySystemError):
        logger.exception("System error", exc_info=e)
        return JobError(job_info, e)
    elif isinstance(e, RayTaskError):
        logger.exception(f"Task error {e}", exc_info=e)
        return JobError(job_info, e)
    else:
        logger.exception("Unknown error", exc_info=e)
        return JobError(job_info, e)


@dataclass
class ExceptionInfo:
    ex: BaseException | None
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
            raise Exception("Process failed with no exception").with_traceback(self.tb.as_traceback())

    @classmethod
    def ser_exc_info(cls, exception=None):
        if exception is None:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = tblib.Traceback(exc_traceback)
            return ExceptionInfo(exc_value, tb)
        else:
            tb = exception.__traceback__
            tb = tblib.Traceback(tb)
            return ExceptionInfo(exception, tb)


@dataclass
class JobInfo:
    """
    Metadata describing a TPU/GPU/CPU job managed via Ray.

    Attributes:
        name (str): A human-readable identifier for the job.
        state (str): The current state of the job (e.g., "pending", "running", "succeeded", "failed").
        kind (str): The type or classification of the job (e.g., "training", "inference").
    """

    name: str
    state: str
    kind: str


@dataclass
class JobStatus:
    """
    Base class representing the final status of a job after a Ray call.

    This class wraps job metadata and serves as a common interface for
    distinguishing between successful and failed executions.

    Attributes:
        info (JobInfo): Metadata about the job.
    """

    info: JobInfo


@dataclass
class JobSucceeded(JobStatus):
    """
    Indicates that the job completed successfully and returned a result.

    Attributes:
        result (object): The output produced by the job.
    """

    result: object


@dataclass
class JobPreempted(JobStatus):
    """
    Indicates that the job was interrupted or preempted, likely by external factors
    such as TPU quota eviction or infrastructure scaling events.

    Attributes:
        error (Exception): The exception raised due to preemption.
    """

    error: Exception


@dataclass
class JobFailed(JobStatus):
    """
    Indicates that the job ran to completion but failed due to an expected runtime issue.

    This could include errors such as invalid input, failed assertions, or handled exceptions.

    Attributes:
        error (Exception): The exception describing why the job failed.
    """

    error: Exception


@dataclass
class JobError(JobStatus):
    """
    Indicates that the job encountered an internal or unexpected error.

    This is typically reserved for unexpected exceptions, infrastructure issues,
    or serialization problems in the Ray runtime.

    Attributes:
        error (Exception): The exception or error message from the failure.
    """

    error: Exception


def print_remote_raise(ray_error):
    """
    Args:
        ray_error: The .error attribute from a Ray task output,
                   containing a pickled exception with tblib.Traceback.
    """
    tb: Traceback = ray_error.cause.args[0].tb
    traceback.print_tb(tb.as_traceback())
