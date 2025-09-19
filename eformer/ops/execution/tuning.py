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

"""Automatic hyperparameter optimization for JAX functions.

This module provides comprehensive autotune functionality that automatically
optimizes function hyperparameters by testing different combinations and
selecting the best performing configuration based on execution timing.

Key Components:
    autotune: Main decorator/function for automatic hyperparameter optimization
    AutotuneConfig: Configuration class for structured autotune parameters
    Autotuner: Core optimization engine with timing and profiling capabilities
    AutotuneData: Data container for optimization results and measurements

The autotune system supports:
    - Multiple hyperparameter optimization strategies
    - JAX compilation with automatic sharding
    - Performance profiling and statistical timing analysis
    - Caching of optimal configurations
    - Thread-safe execution with configurable worker pools
    - Decorator patterns: @autotune, @autotune(config), @autotune(hyperparams={})

Example Usage:
    >>> @autotune(hyperparams={'batch_size': [32, 64, 128]})
    >>> def my_function(x, batch_size=32):
    ...     return jax.vmap(lambda xi: xi * 2, in_axes=0)(x)

    >>> # Or with configuration object
    >>> config = AutotuneConfig(hyperparams={'lr': [0.01, 0.001]})
    >>> @autotune(config)
    >>> def train_step(params, lr=0.01):
    ...     return params * lr

The optimization process:
    1. Generates all hyperparameter combinations
    2. Compiles functions with JAX for each combination
    3. Measures execution performance using profiling or timing
    4. Selects optimal configuration based on speed and stability
    5. Caches results for future use
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import logging
import os
import random as pyrandom
import re
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from pprint import pformat
from typing import Any, Generic, TypeVar

import jax
import jax.core
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.interpreters import pxla
from jax.sharding import PartitionSpec, Sharding, SingleDeviceSharding
from tqdm import tqdm

from ..config.cache import overlay_cache
from ..utils.fingerprint import device_fingerprint
from . import profiler

Cfg = TypeVar("Cfg")


@dataclass
class Measurement:
    """Container for a single performance measurement.

    Stores the configuration and corresponding execution time
    for a single hyperparameter combination during optimization.

    Attributes:
        cfg: The hyperparameter configuration that was tested
        seconds: Execution time in seconds for this configuration
    """

    cfg: Any
    seconds: float


@dataclass
class AutotuneData(Generic[Cfg]):
    """Container for all optimization measurements and results.

    Stores performance measurements for all tested hyperparameter
    configurations and provides utilities to analyze the results.

    Type Parameters:
        Cfg: Configuration type (e.g., dict, dataclass, etc.)

    Attributes:
        measurements: List of all performance measurements taken
    """

    measurements: list[Measurement]

    @property
    def fastest_config(self) -> Cfg:
        """Get the configuration with the fastest execution time.

        Returns:
            The configuration that achieved the lowest execution time

        Raises:
            ValueError: If no measurements are available
        """
        if not self.measurements:
            raise ValueError("No measurements available to determine fastest config")
        return min(self.measurements, key=lambda m: m.seconds).cfg


class Autotuner(Generic[Cfg]):
    """Core autotuning engine for hyperparameter optimization.

    This class provides the fundamental optimization algorithm that tests
    different configurations and measures their performance to find the
    optimal hyperparameter settings.

    Type Parameters:
        Cfg: Configuration type for hyperparameters

    Attributes:
        warmup: Number of warmup iterations before timing
        iters: Number of timing iterations for measurement accuracy
    """

    def __init__(self, warmup=1, iters=3):
        """Initialize the autotuner with timing parameters.

        Args:
            warmup: Number of warmup calls to stabilize performance
            iters: Number of timing iterations for statistical accuracy
        """
        self.warmup, self.iters = warmup, iters

    def autotune(self, make_fn, args, kwargs, candidates: Iterable[Cfg]) -> AutotuneData[Cfg]:
        """Optimize hyperparameters by testing candidate configurations.

        Tests each candidate configuration by compiling and timing the
        function execution, then returns all measurements for analysis.

        Args:
            make_fn: Factory function that creates a function given a config
            args: Positional arguments for the function being optimized
            kwargs: Keyword arguments for the function being optimized
            candidates: Iterable of candidate configurations to test

        Returns:
            AutotuneData containing all performance measurements

        Raises:
            RuntimeError: If compilation or execution fails for all candidates
        """
        measures = []
        for cfg in candidates:
            try:
                fn = make_fn(cfg)
                c = jax.jit(fn).lower(*args, **kwargs).compile()

                # Warmup phase
                for _ in range(self.warmup):
                    _ = c(*args, **kwargs).block_until_ready()

                # Timing phase
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _ = c(*args, **kwargs).block_until_ready()
                dt = (time.perf_counter() - t0) / self.iters
                measures.append(Measurement(cfg, dt))
            except Exception as e:
                autotune_logger.warning(f"Configuration {cfg} failed: {e}")
                measures.append(Measurement(cfg, float("inf")))

        if not measures or all(m.seconds == float("inf") for m in measures):
            raise RuntimeError("All candidate configurations failed to execute")

        return AutotuneData(measures)


@dataclass(frozen=True)
class Entry:
    """Cache entry for storing optimal configurations.

    Represents a single cached optimization result with the operation
    identifier, call signature, and optimal configuration.

    Attributes:
        op_id_v: Operation identifier for the optimized function
        call_key: Hash key representing the function call signature
        cfg: The optimal configuration found for this operation
    """

    op_id_v: str
    call_key: str
    cfg: Any


@dataclass(frozen=True)
class AutotuningResult:
    """Result container for device-specific optimization results.

    Stores all optimized configurations for a specific device and provides
    context manager functionality for temporary cache overlays.

    Attributes:
        device: Device identifier these results apply to
        entries: Tuple of optimization entries (operation -> config mappings)
    """

    device: str
    entries: tuple[Entry, ...]

    def as_overlay(self):
        """Convert results to cache overlay mapping format.

        Returns:
            Dictionary mapping (device, op_id, call_key) -> configuration
        """
        mapping = {(self.device, e.op_id_v, e.call_key): e.cfg for e in self.entries}
        return mapping

    def __enter__(self):
        """Enter context manager to apply optimization results as cache overlay.

        Returns:
            Self for use in with statements
        """
        self._ctx = overlay_cache(self.as_overlay())
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager and restore previous cache state.

        Args:
            exc_type: Exception type (if any)
            exc: Exception instance (if any)
            tb: Exception traceback (if any)
        """
        self._ctx.__exit__(exc_type, exc, tb)
        delattr(self, "_ctx")


def autotune_recorded(hyperparameter_selector, *, show_progress=False, repetition_count=1):
    """Record and replay optimal hyperparameters for functions.

    This function provides caching and replay functionality for previously
    optimized hyperparameters, avoiding redundant tuning operations.

    Args:
        hyperparameter_selector: Function to select hyperparameters from recorded data
        show_progress: Whether to display progress bars during optimization
        repetition_count: Number of times to repeat the optimization process

    Returns:
        Decorated function with recorded hyperparameter optimization
    """
    """Autotune all recorded invocations for the current device."""
    from ..registry import get_invocations

    dev = device_fingerprint()
    invs = get_invocations(dev)
    entries = []
    for op_id_v, d in invs.items():
        for call_key, (kernel, args, kwargs) in d.items():
            inv_args, inv_kwargs = kernel.prepare(*args, **kwargs)

            static_fun_kwargs = {k: v for k, v in inv_kwargs.items() if callable(v)}
            dyn_kwargs = {k: v for k, v in inv_kwargs.items() if k not in static_fun_kwargs}

            tmp_inv = type(
                "Tmp",
                (),
                dict(
                    op_id=kernel.op_id, args=inv_args, kwargs=dyn_kwargs, batch_axes=None, override_cfg=None, stamp=False
                ),
            )()
            candidates = tuple(kernel.candidate_cfgs(tmp_inv))

            def mk(c, _run=kernel.run, _static=static_fun_kwargs):
                return partial(_run, cfg=c, **_static)

            best_cfg, best_t = None, float("inf")
            for c in candidates:
                t = benchmark(mk(c), *inv_args, **dyn_kwargs)
                if t < best_t:
                    best_cfg, best_t = c, t

            hyperparameter_selector.cache.put(dev, op_id_v, call_key, best_cfg)
            if hyperparameter_selector.persistent and hyperparameter_selector.persist_autotune:
                hyperparameter_selector.persistent.put(dev, op_id_v, call_key, best_cfg)
            entries.append(Entry(op_id_v, call_key, best_cfg))
    return AutotuningResult(dev, tuple(entries))


def _split_static_callable_kwargs(kwargs):
    """Split keyword arguments into static and dynamic components.

    Separates callable arguments (static) from regular arguments (dynamic)
    for proper JAX compilation and execution.

    Args:
        kwargs: Dictionary of keyword arguments to split

    Returns:
        Tuple of (static_kwargs, dynamic_kwargs)
    """
    static = {k: v for k, v in kwargs.items() if callable(v)}
    return static, {k: v for k, v in kwargs.items() if k not in static}


def benchmark(fn, *args, warmup=1, iters=5, **kwargs) -> float:
    """Benchmark function execution time with JAX compilation.

    Compiles the function with JAX and measures its execution time
    over multiple iterations, handling both static and dynamic arguments.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for the function
        warmup: Number of warmup iterations before timing
        iters: Number of timing iterations for measurement
        **kwargs: Keyword arguments for the function

    Returns:
        Average execution time per iteration in seconds
    """
    static, dyn = _split_static_callable_kwargs(kwargs)

    if static:

        def fn_wrapped(*a, _fn=fn, _static=static, **k):
            return _fn(*a, **(k | _static))

        c = jax.jit(fn_wrapped).lower(*args, **dyn).compile()
        for _ in range(warmup):
            _ = c(*args, **dyn).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = c(*args, **dyn).block_until_ready()
        return (time.perf_counter() - t0) / iters
    else:
        c = jax.jit(fn).lower(*args, **kwargs).compile()
        for _ in range(warmup):
            _ = c(*args, **kwargs).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = c(*args, **kwargs).block_until_ready()
        return (time.perf_counter() - t0) / iters


PREFIX_FN = "autotune_fn_{}"


@dataclasses.dataclass
class _Config:
    """Internal configuration for autotune behavior.

    This class defines global settings that control how the autotune
    system operates, including timing parameters, caching behavior,
    and optimization strategies.

    Attributes:
        allow_fallback_timing: Allow Python-level timing if profiling fails
        must_find_at_least_profiler_result_fraction: Minimum fraction of configs that must be profiled
        profiling_samples: Number of profiling samples to collect
        find_optimal_layouts_automatically: Enable automatic layout optimization
        max_compilation_time_seconds: Maximum time allowed for compilation
        min_timing_iterations: Minimum number of timing iterations
        max_timing_iterations: Maximum number of timing iterations
        timing_warmup_iterations: Number of warmup iterations before timing
        cache_size_limit: Maximum number of cached optimization results
        enable_detailed_logging: Enable verbose logging output
    """

    allow_fallback_timing: bool = True
    must_find_at_least_profiler_result_fraction: float = 0.5
    profiling_samples: int = 5
    find_optimal_layouts_automatically: bool = False
    max_compilation_time_seconds: float = 300.0
    min_timing_iterations: int = 3
    max_timing_iterations: int = 10
    timing_warmup_iterations: int = 2
    cache_size_limit: int = 1000
    enable_detailed_logging: bool = False


class _UnspecifiedT:
    """Sentinel type for unspecified parameters.

    Used to distinguish between None values and truly unspecified
    parameters in the autotune configuration system.
    """

    pass


UNSPECIFIED = _UnspecifiedT()


@dataclasses.dataclass
class AutotuneConfig:
    """Configuration class for autotune decorator and function.

    This class provides a structured way to configure autotune behavior
    and can be used with the @autotune(config) decorator pattern.
    """

    hyperparams: dict[Any, Any] | None = None
    max_workers: int = 32
    in_shardings: Any = UNSPECIFIED
    out_shardings: Any = UNSPECIFIED
    device: jax.Device | _UnspecifiedT = UNSPECIFIED  # type:ignore
    example_args: tuple[Any] | None = None
    example_kws: dict[Any, Any] | None = None
    sample_num: int = 2**63 - 1
    event_filter_regex: str | None = None
    warmup_iters: int | None = None
    timing_iters: int | None = None
    timeout: float | None = None
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for passing to autotune function."""
        result = {}
        for k, v in dataclasses.asdict(self).items():
            if not isinstance(v, _UnspecifiedT):
                result[k] = v
        return result


CONFIG = _Config()


autotune_logger = logging.getLogger("eformer.autotune")
if not autotune_logger.handlers:
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
    log_handler.setFormatter(log_formatter)
    autotune_logger.addHandler(log_handler)
autotune_logger.setLevel(logging.WARNING)


def set_autotune_log_level(level):
    """Set the logging level for autotune operations.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    autotune_logger.setLevel(level)


context_escape_pool_executor = ThreadPoolExecutor(max_workers=8)
_global_tuning_lock = threading.Lock()


@dataclasses.dataclass
class CompileResult:
    """Result of function compilation attempt.

    Contains information about whether compilation succeeded,
    any error messages, and optimal input formats if computed.

    Attributes:
        status: Whether compilation was successful
        error_msg: Error message if compilation failed
        optimal_formats: Computed optimal input formats (if enabled)
    """

    status: bool
    error_msg: str | None = None
    optimal_formats: Any = None


@dataclasses.dataclass
class TimingResult:
    """Performance measurement result for a hyperparameter configuration.

    Contains the tested hyperparameters and statistical timing information
    including mean execution time and standard deviation.

    Attributes:
        hyperparams: Dictionary of hyperparameter names to values
        t_mean: Mean execution time in seconds
        t_std: Standard deviation of execution times
    """

    hyperparams: dict[Any, Any]
    t_mean: float
    t_std: float


def _get_global_mesh():
    """Get the current global JAX mesh for distributed computation.

    Returns:
        The global mesh if available, None otherwise
    """
    env = pxla.thread_resources.env
    mesh = env.physical_mesh
    return None if mesh.empty else mesh


def _get_default_device():
    """Get the default JAX device for computation.

    Returns:
        The configured default device or the first available device
    """
    if jax.config.values["jax_default_device"] is not None:
        return jax.config.values["jax_default_device"]
    return jax.devices()[0]


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output.

    Temporarily redirects both stdout and stderr to devnull,
    useful for suppressing verbose output during profiling.

    Yields:
        None
    """
    devnull, stdout, stderr = open(os.devnull, "w+"), os.dup(1), os.dup(2)
    os.dup2(devnull.fileno(), 1), os.dup2(devnull.fileno(), 2)
    yield
    os.dup2(stdout, 1), os.dup2(stderr, 2)


def _try_call(
    fn: Callable[[], None],
    resolved_args,
    resolved_kwargs,
    compile_only: bool = False,
    compute_layouts: bool = False,
    optimal_formats: Any | None = None,
    timeout: float | None = None,
) -> CompileResult:
    """Attempt to call the function and return whether it compiles and runs.

    Args:
        fn: Function to test
        resolved_args: Arguments for the function
        resolved_kwargs: Keyword arguments for the function
        compile_only: If True, only test compilation without execution
        compute_layouts: If True, compute optimal input formats
        optimal_formats: Pre-computed optimal formats to use
        timeout: Compilation timeout in seconds

    Returns:
        CompileResult with status, error message, and optimal formats
    """

    if timeout is not None and CONFIG.enable_detailed_logging:
        autotune_logger.info(
            f"Compilation timeout configured: {timeout}s "
            "(Note: Thread-based timeout not yet implemented for compilation phase)"
        )

    # Initialize optimal_input_formats to avoid UnboundLocalError
    optimal_input_formats = None

    try:
        if compile_only:
            if compute_layouts:

                def to_shape(x):
                    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if isinstance(x, jax.Array) else x

                (argument_shapes, keyword_shapes) = jax.tree.map(to_shape, (resolved_args, resolved_kwargs))
                try:
                    compiled_function = jax.jit(fn).lower(*argument_shapes, **keyword_shapes).compile()
                    optimal_input_formats = compiled_function.input_formats
                    if CONFIG.enable_detailed_logging:
                        autotune_logger.info(
                            f"Successfully computed optimal input formats for function compilation:\n"
                            f"{pformat(optimal_input_formats, width=120)}"
                        )
                except Exception as compilation_error:
                    autotune_logger.warning(
                        f"Layout optimization failed during compilation: "
                        f"{compilation_error.__class__.__name__}: {compilation_error}"
                    )
                    optimal_input_formats = None
            else:
                _ = jax.jit(fn).lower(*resolved_args, **resolved_kwargs).compile()
        else:
            if optimal_input_formats is not None:

                def place_array_on_optimal_device(array_data, target_format):
                    """Place JAX arrays on optimal devices according to computed formats."""
                    return jax.device_put(array_data, target_format) if isinstance(array_data, jax.Array) else array_data

                try:
                    (optimally_placed_args, optimally_placed_kwargs) = jax.tree.map(
                        place_array_on_optimal_device, (resolved_args, resolved_kwargs), optimal_input_formats
                    )
                    _ = jax.block_until_ready(fn(*optimally_placed_args, **optimally_placed_kwargs))
                except Exception:
                    autotune_logger.warning(
                        "Failed to place arrays on optimal devices - falling back to original argument placement"
                    )
                    _ = jax.block_until_ready(fn(*resolved_args, **resolved_kwargs))
            else:
                _ = jax.block_until_ready(fn(*resolved_args, **resolved_kwargs))
        return CompileResult(True, None, optimal_input_formats)
    except Exception as e:
        msg = f"{type(e).__name__}: {e!s}"
        if CONFIG.enable_detailed_logging:
            msg = traceback.format_exc()
        return CompileResult(False, msg, optimal_input_formats)


def _time_fn(
    target_function: Callable[[], None], measurement_rounds: int = 5, calls_per_round: int = 3, warmup_calls: int = 2
) -> tuple[float, float]:
    """Precisely time a function execution with statistical analysis.

    This function provides thread-safe, accurate timing measurements by:
    - Using a global lock to prevent system interference
    - Performing warmup calls to stabilize performance
    - Taking multiple measurements for statistical reliability
    - Returning both mean and standard deviation for analysis

    Args:
        target_function: The function to measure execution time for
        measurement_rounds: Number of independent timing rounds for statistics
        calls_per_round: Number of function calls to average per round
        warmup_calls: Number of initial calls to stabilize performance

    Returns:
        Tuple of (mean_execution_time, time_std_deviation) in seconds
        Returns (inf, inf) if any timing measurement fails
    """
    with _global_tuning_lock:

        def _execute_and_block():
            """Execute function and ensure computation completes."""
            return jax.block_until_ready(target_function())

        for warmup_iteration in range(warmup_calls):
            try:
                _execute_and_block()
            except Exception as warmup_error:
                autotune_logger.warning(
                    f"Warmup iteration {warmup_iteration + 1}/{warmup_calls} failed: "
                    f"{warmup_error.__class__.__name__}: {warmup_error}"
                )
                return float("inf"), float("inf")

        execution_times = []

        for round_number in range(measurement_rounds):
            round_start_time = time.perf_counter()
            try:
                for _ in range(calls_per_round):
                    _execute_and_block()
                round_total_time = time.perf_counter() - round_start_time
                execution_times.append(round_total_time)
            except Exception as timing_error:
                autotune_logger.warning(
                    f"Timing round {round_number + 1}/{measurement_rounds} failed: "
                    f"{timing_error.__class__.__name__}: {timing_error}"
                )
                execution_times.append(float("inf"))

        valid_execution_times = [time_value for time_value in execution_times if not np.isinf(time_value)]
        if not valid_execution_times:
            autotune_logger.warning("All timing measurements failed - returning infinite time")
            return float("inf"), float("inf")

        normalized_times = np.array(valid_execution_times) / calls_per_round

        if len(normalized_times) > 2:
            outlier_filtered_times = np.sort(normalized_times)[1:-1]
        else:
            outlier_filtered_times = normalized_times

        mean_execution_time = np.mean(outlier_filtered_times)
        time_standard_deviation = np.std(outlier_filtered_times)

        if CONFIG.enable_detailed_logging:
            autotune_logger.debug(
                f"Timing analysis complete: {len(valid_execution_times)} successful rounds, "
                f"mean={mean_execution_time:.2e}s, std={time_standard_deviation:.2e}s"
            )

        return float(mean_execution_time), float(time_standard_deviation)


def _calculate_timing_score(timing_result: TimingResult) -> float:
    """Calculate optimization score for hyperparameter selection.

    Combines mean execution time with standard deviation penalty to prefer
    both fast and stable performance characteristics.

    Args:
        timing_result: Timing measurements with mean and standard deviation

    Returns:
        Combined timing score (lower is better) where:
        score = mean_time + 0.1 * std_deviation
        The 0.1 weighting penalizes high variance while prioritizing speed
    """
    return timing_result.t_mean + 0.1 * timing_result.t_std


def _create_parameterized_function(
    target_function: Callable[..., Any],
    hyperparameter_values: dict[str, Any],
    output_shardings: Sharding | _UnspecifiedT = UNSPECIFIED,
    function_id: int = 0,
) -> Callable[..., Any]:
    """Create a JAX-compiled function with embedded hyperparameters for timing.

    This function wraps the target function with specific hyperparameter values,
    applies JAX compilation with optional output sharding, and creates a callable
    suitable for performance measurement.

    Args:
        target_function: The function to be optimized and timed
        hyperparameter_values: Dictionary of hyperparameter names to values
        output_shardings: JAX sharding specification for outputs (optional)
        function_id: Unique identifier for function naming in profiling

    Returns:
        JAX-compiled function with embedded hyperparameters ready for timing
    """

    jax_compiler = partial(jax.jit, out_shardings=output_shardings if output_shardings is not UNSPECIFIED else None)

    def parameterized_function(*function_args, **function_kwargs):
        """Execute target function with embedded hyperparameters."""

        combined_kwargs = dict(function_kwargs, **hyperparameter_values)
        return target_function(*function_args, **combined_kwargs)

    function_name = PREFIX_FN.format(function_id)
    parameterized_function.__name__ = function_name
    parameterized_function.__qualname__ = function_name

    return jax_compiler(parameterized_function)


def _normalize_sharding(
    arg: jax.Array | np.ndarray | Any,
    sharding_or_spec: PartitionSpec | Sharding | None,
    default_device: jax.Device,  # type:ignore
):
    if not isinstance(arg, jax.Array | np.ndarray):
        return None
    if isinstance(sharding_or_spec, Sharding):
        return sharding_or_spec
    global_mesh = _get_global_mesh()
    if isinstance(sharding_or_spec, PartitionSpec) and global_mesh is not None:
        return jax.NamedSharding(global_mesh, sharding_or_spec)
    elif isinstance(sharding_or_spec, PartitionSpec) and global_mesh is None:
        raise ValueError("If specifying shardings via ParitionSpec, a global mesh must be defined")
    else:
        return SingleDeviceSharding(default_device)


def _experimental_time_with_profiler(
    _timing_closure: Callable[[], None],
    platform: str,
    total_calls_number: int,
    event_filter_regex: str | None = None,
    min_duration_ns: float = 1000.0,
    max_events_per_profile: int | None = None,
) -> dict[int, tuple[float, float]]:
    function_timings = {}
    pbar = tqdm(range(total_calls_number), desc=f"Profiling {platform}", disable=autotune_logger.level > logging.INFO)
    for it in pbar:
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        profile_path = Path(tempfile.mkdtemp(prefix=f"tuning_profile_{now}_")).absolute()
        profile_path.mkdir(exist_ok=True)
        with suppress_stdout_stderr():
            with jax.profiler.trace(str(profile_path)):
                _timing_closure()
        profile_files = sorted(profile_path.glob("**/*.xplane.pb"), key=lambda f: f.stat().st_mtime)
        if len(profile_files) == 0:
            raise RuntimeError("No profile was created.")
        latest_profile = profile_files[-1]
        profile_proto = profiler.parse_profile_from_bytes(latest_profile.read_bytes())
        device_plane_id = profiler.find_device_plane_ids(profile_proto, platform)[0]

        try:
            profile_events = profiler.get_events_from_plane(
                profile_proto,
                device_plane_id,
                prefix_filter="jit_",
                event_filter_regex=event_filter_regex,
                min_duration_ns=min_duration_ns,
                max_events=max_events_per_profile,
            )
        except profiler.ProfilingError as e:
            autotune_logger.warning(f"Profiling failed for iteration {it}: {e}")
            continue
        fn_format = f"jit_{PREFIX_FN.format('([0-9]+)')}.*"
        for k, durations in profile_events.items():
            if not re.match(fn_format, k):
                continue
            key = int(re.match(fn_format, k)[1])
            function_timings.setdefault(key, []).append(durations)

    for key, durations in function_timings.items():
        if len(durations) > 2:
            durations = sorted(durations)[1:-1]
        function_timings[key] = (float(np.mean(durations)), float(np.std(durations)))

    return function_timings


@partial(jax.jit, static_argnames=("sds", "sharding"))
def _get_random_value(sds, sharding=None):
    """Random values based on the tracer shape and dtype, and the sharding."""

    if hasattr(sds, "shape") and hasattr(sds, "dtype"):
        if jnp.issubdtype(sds.dtype, jnp.floating):
            return jax.jit(lambda key: random.normal(key, sds.shape, sds.dtype), out_shardings=sharding)(random.key(0))
        elif jnp.issubdtype(sds.dtype, jnp.integer):
            return jax.jit(lambda: jnp.zeros(sds.shape, sds.dtype), out_shardings=sharding)()
        else:
            raise ValueError(f"Unsupported dtype {sds.dtype}")
    else:
        return sds


def _try_hash_input(args, kws, must_be_concrete: bool = True):
    """For eager mode tunable, hash the shape, dtype and sharding of the inputs."""

    flat_vals, struct = jax.tree.flatten((args, kws))
    all_concrete = all(jax.core.is_concrete(x) for x in flat_vals if isinstance(x, jax.Array))
    if not all_concrete and must_be_concrete:
        return None

    def _get_sharding(x):
        try:
            return x.sharding
        except AttributeError:
            return jax.typeof(x).sharding

    def array_to_hashable(x):
        return x if not isinstance(x, jax.Array) else hash((jax.typeof(x), _get_sharding(x)))

    try:
        return hash((struct, tuple(array_to_hashable(x) for x in flat_vals)))
    except:  # noqa: E722
        return None


def autotune(
    fn: Callable[..., Any] | AutotuneConfig | None = None,
    hyperparams: dict[Any, Any] | None = None,
    max_workers: int = 32,
    in_shardings: Any = UNSPECIFIED,
    out_shardings: Any = UNSPECIFIED,
    device: jax.Device | _UnspecifiedT = UNSPECIFIED,  # type:ignore
    example_args: tuple[Any] | None = None,
    example_kws: dict[Any, Any] | None = None,
    sample_num: int = 2**63 - 1,
    event_filter_regex: str | None = None,
    warmup_iters: int | None = None,
    timing_iters: int | None = None,
    timeout: float | None = None,
    cache_key: str | None = None,
):
    """Automatically optimize function hyperparameters for best performance.

    This decorator/function provides comprehensive hyperparameter optimization
    by testing different configurations and selecting the fastest one.

    Supports multiple usage patterns:
        @autotune  # Use defaults
        @autotune(hyperparams={'batch_size': [32, 64]})  # Direct parameters
        @autotune(config)  # Using AutotuneConfig object

    Args:
        fn: Function to optimize, AutotuneConfig object, or None for decorator
        hyperparams: Dictionary mapping parameter names to candidate values
        max_workers: Maximum number of parallel worker threads
        in_shardings: Input sharding specifications for distributed computation
        out_shardings: Output sharding specifications for distributed computation
        device: Target device for computation (GPU, TPU, CPU)
        example_args: Example arguments for shape/sharding inference
        example_kws: Example keyword arguments for shape/sharding inference
        sample_num: Maximum number of hyperparameter combinations to test
        event_filter_regex: Regex filter for profiling events
        warmup_iters: Number of warmup iterations before timing
        timing_iters: Number of timing iterations for statistical accuracy
        timeout: Maximum compilation time per configuration
        cache_key: Custom cache key for optimization results

    Returns:
        Decorated function with optimal hyperparameters, or decorator if fn is None

    Raises:
        TypeError: If fn is not callable when expected
        ValueError: If hyperparameter configuration is invalid
        RuntimeError: If no hyperparameter combinations compile successfully

    Examples:
        >>> @autotune(hyperparams={'lr': [0.01, 0.001, 0.0001]})
        >>> def train_step(params, gradients, lr=0.01):
        ...     return params - lr * gradients

        >>> # The function will automatically use the optimal lr value
        >>> optimized_params = train_step(params, grads)

        >>> # Access optimization results
        >>> print(train_step.optimal_hyperparams)  # {'lr': 0.001}
        >>> print(train_step.timing_results)       # Timing data for all configs
    """
    if fn is None or isinstance(fn, AutotuneConfig):
        config = fn if isinstance(fn, AutotuneConfig) else None

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            final_params = {
                "hyperparams": hyperparams,
                "max_workers": max_workers,
                "in_shardings": in_shardings,
                "out_shardings": out_shardings,
                "device": device,
                "example_args": example_args,
                "example_kws": example_kws,
                "sample_num": sample_num,
                "event_filter_regex": event_filter_regex,
                "warmup_iters": warmup_iters,
                "timing_iters": timing_iters,
                "timeout": timeout,
                "cache_key": cache_key,
            }

            if config is not None:
                config_dict = config.to_dict()
                for key, value in config_dict.items():
                    if key in final_params and final_params[key] == globals().get(f"{key}", None):
                        final_params[key] = value

            return autotune(func, **final_params)

        return decorator

    if not callable(fn):
        raise TypeError("fn must be callable")

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    if sample_num < 0:
        raise ValueError("sample_num must be non-negative")

    hyperparams_ = hyperparams if hyperparams is not None else dict()

    warmup_iters = warmup_iters if warmup_iters is not None else CONFIG.timing_warmup_iterations
    timing_iters = timing_iters if timing_iters is not None else CONFIG.min_timing_iterations
    timeout = timeout if timeout is not None else CONFIG.max_compilation_time_seconds

    def _get_best_hyperparams(function_args, function_kwargs):
        """Execute comprehensive hyperparameter optimization.

        This is the core optimization method that:
        1. Resolves argument sharding and device placement
        2. Compiles functions with different hyperparameter combinations
        3. Profiles execution performance for each combination
        4. Selects optimal hyperparameters based on timing analysis

        Args:
            function_args: Positional arguments for the target function
            function_kwargs: Keyword arguments for the target function

        Returns:
            Tuple of (optimal_function, optimal_hyperparams, all_results)
        """

        def _extract_array_type(data):
            """Extract JAX array type information for sharding analysis."""
            return data if not isinstance(data, jax.Array) else jax.typeof(data)

        if len(function_args) == 0 or all(x is None or jax.core.is_concrete(x) for x in jax.tree.leaves(function_args)):
            autotune_logger.debug("All function arguments are concrete values - using provided arguments directly")
            resolved_args = function_args
        elif example_args is not None:
            autotune_logger.debug(
                "Using provided example arguments for optimization "
                f"(shape signature: {jax.tree.map(lambda x: getattr(x, 'shape', type(x).__name__), example_args)})"
            )
            resolved_args = example_args
            if in_shardings is not UNSPECIFIED or device is not UNSPECIFIED:
                raise ValueError(
                    "Cannot combine example_args with explicit in_shardings or device configuration. "
                    "Example arguments should already be properly sharded and placed."
                )
        else:
            autotune_logger.debug("Selecting random input arguments.")
            resolved_device = device if isinstance(device, jax.Device) else _get_default_device()
            if isinstance(resolved_device, str):
                resolved_device = jax.devices(resolved_device)[0]
            input_shardings = (
                in_shardings if in_shardings is not UNSPECIFIED else jax.tree.map(lambda _: None, function_args)
            )
            input_shardings = (input_shardings,) if len(function_args) == 1 else input_shardings
            normalized_shardings = jax.tree.map(
                partial(_normalize_sharding, default_device=resolved_device),
                tuple(function_args),
                tuple(input_shardings),
            )
            resolved_args = jax.tree.map(
                lambda x, s: _get_random_value(_extract_array_type(x), s), function_args, normalized_shardings
            )

        if len(function_kwargs) == 0 or all(v is None or jax.core.is_concrete(v) for v in function_kwargs.values()):
            autotune_logger.debug("All keyword arguments are concrete - using provided values directly")
            resolved_kwargs = function_kwargs
        elif example_kws is not None:
            autotune_logger.debug(
                f"Using provided example keyword arguments for optimization (keys: {list(example_kws.keys())})"
            )
            resolved_kwargs = example_kws
        else:
            autotune_logger.debug("Generating random keyword argument values for abstract shapes")
            resolved_kwargs = jax.tree.map(lambda x: _get_random_value(_extract_array_type(x)), function_kwargs)

        hyperparams_norm = {}
        for k, v in hyperparams_.items():
            if isinstance(v, tuple | list):
                if len(v) == 0:
                    raise ValueError(f"Hyperparameter '{k}' has empty list of values")
                hyperparams_norm[k] = tuple(v)
            else:
                hyperparams_norm[k] = (v,)

        executor = ThreadPoolExecutor(max_workers=max_workers)

        fns = dict()

        if hyperparams_norm:
            hyperparam_settings = dict(enumerate(itertools.product(*hyperparams_norm.values())))
            total_combinations = len(hyperparam_settings)

            if sample_num < total_combinations:
                if sample_num == 0:
                    hyperparam_settings = {0: tuple()}
                else:
                    sample_idx = sorted(
                        pyrandom.sample(list(range(total_combinations)), k=min(sample_num, total_combinations))
                    )
                    hyperparam_settings_ = list(hyperparam_settings.items())
                    hyperparam_settings = dict([hyperparam_settings_[idx] for idx in sample_idx])

            autotune_logger.info(
                f"Testing {len(hyperparam_settings)} hyperparameter combinations out of {total_combinations} possible"
            )
        else:
            hyperparam_settings = {0: tuple()}

        with _global_tuning_lock:
            optimal_formats = {}
            for it in range(2):
                compile_only, find_optimal_layouts = (it == 0), CONFIG.find_optimal_layouts_automatically
                compiles: dict[Future[CompileResult], int] = dict()
                for i, vals in hyperparam_settings.items():
                    hs = dict(zip(hyperparams_norm.keys(), vals, strict=True))
                    fns[i] = _create_parameterized_function(fn, hs, output_shardings=out_shardings, function_id=i)

                    opts = dict(
                        optimal_formats=optimal_formats.get(i, None),
                        compute_layouts=find_optimal_layouts,
                        timeout=timeout,
                    )
                    compiles[
                        executor.submit(
                            _try_call, fns[i], resolved_args, resolved_kwargs, compile_only=compile_only, **opts
                        )
                    ] = i

                future_pbar = tqdm(
                    total=len(compiles), disable=autotune_logger.level > logging.INFO, desc="Compiling..."
                )
                successful_compiles = {}
                for fut in as_completed(compiles):
                    result = fut.result()
                    if result.status:
                        successful_compiles[compiles[fut]] = result
                    future_pbar.update(1)
                future_pbar.close()

                if compile_only and find_optimal_layouts:
                    for k, x in successful_compiles.items():
                        optimal_formats[k] = x.optimal_formats
                if len(successful_compiles) == 0:
                    for compile_result, i in compiles.items():
                        autotune_logger.error(
                            f"Hyperparameters {hyperparam_settings[i]} failed to compile with message:"
                            f"\n{compile_result.result().error_msg}"
                        )
                    raise ValueError("No hyperparameters compiled successfully")
                autotune_logger.debug("Down to %d hyperparameters", len(successful_compiles))

                hyperparam_settings = {i: hyperparam_settings[i] for i in successful_compiles.keys()}
                fns = {i: fns[i] for i in successful_compiles.keys()}

        results = dict()
        try:
            args_with_device = [
                next(iter(args.devices())) for args in jax.tree.leaves(resolved_args) if hasattr(args, "devices")
            ]
            if len(args_with_device) > 0:
                platform = args_with_device[0].platform
            else:
                platform = _get_default_device().platform

            def _timing_closure():
                hs = list(hyperparam_settings.items())
                pyrandom.shuffle(hs)
                for i, _ in hs:
                    _try_call(
                        fns[i],
                        resolved_args,
                        resolved_kwargs,
                        compile_only=False,
                        optimal_formats=optimal_formats.get(i, None),
                    )

            profiler_timings = _experimental_time_with_profiler(
                _timing_closure,
                platform,
                CONFIG.profiling_samples,
                event_filter_regex=event_filter_regex,
                min_duration_ns=1000.0,
                max_events_per_profile=10000,
            )
            fraction_measured = sum(1 for i in hyperparam_settings.keys() if i in profiler_timings) / len(
                hyperparam_settings
            )
            if fraction_measured < CONFIG.must_find_at_least_profiler_result_fraction:
                msg = "Could not find profiler results for some hyperparameter settings:"
                for i in [i for i in hyperparam_settings.keys() if i not in profiler_timings]:
                    msg += f"\n  - {i}: {hyperparam_settings[i]}"
                raise RuntimeError(msg)
            else:
                for i in hyperparam_settings.keys():
                    if i not in profiler_timings:
                        autotune_logger.warning(
                            f"Could not find profiler results for hyperparameter settings: {hyperparam_settings[i]}"
                        )
                        profiler_timings[i] = (
                            float("inf"),
                            float("inf"),
                        )
            for i, hs in hyperparam_settings.items():
                hs = dict(zip(hyperparams_norm.keys(), hs, strict=True))
                results[i] = TimingResult(hs, *profiler_timings[i])
        except Exception:
            if not CONFIG.allow_fallback_timing:
                print(traceback.format_exc())
                raise RuntimeError(
                    f"Need to fall back to the python-level timing, but {CONFIG=} prohibits it."
                ) from None

            autotune_logger.warning(traceback.format_exc())
            autotune_logger.warning("Could not time with the profiler, falling back to Python-level timing")
            _opts = dict(total=len(hyperparam_settings), disable=autotune_logger.level > logging.INFO, desc="Timing...")
            hs_pbar = tqdm(hyperparam_settings.items(), **_opts)
            for i, hs in hs_pbar:
                hs = dict(zip(hyperparams_norm.keys(), hs, strict=True))
                results[i] = TimingResult(
                    hs,
                    *_time_fn(partial(lambda fn: fn(*resolved_args, **resolved_kwargs), fns[i]), measurement_rounds=10),
                )

        results = sorted(results.items(), key=lambda x: _calculate_timing_score(x[1]))
        idx, optimal_hyperparams = results[0][0], results[0][1].hyperparams
        autotune_logger.debug("\n" + pformat(dict(results), width=300))
        autotune_logger.debug(f"optimal hyperparams: {optimal_hyperparams}")
        return fns[idx], optimal_hyperparams, results

    if hasattr(fn, "timing_result"):
        raise ValueError("Wrapping a `tune`d function in the `tune` decorator the second time is not supported.")

    @wraps(fn)
    def wrapped_fn(*args, **kws):
        input_hash = _try_hash_input(args, kws)
        cache_lookup_key = f"{cache_key}:{input_hash}" if cache_key and input_hash else input_hash

        if cache_lookup_key is not None and cache_lookup_key in wrapped_fn.hyperparams_cache:
            optimal_hyperparameters, results = wrapped_fn.hyperparams_cache[cache_lookup_key]
            if CONFIG.enable_detailed_logging:
                autotune_logger.info(f"Using cached hyperparameters for key: {cache_lookup_key}")
        else:
            with jax.core.eval_context():
                _, optimal_hyperparameters, results = _get_best_hyperparams(args, kws)

            if cache_lookup_key is not None:
                if len(wrapped_fn.hyperparams_cache) >= CONFIG.cache_size_limit:
                    oldest_key = next(iter(wrapped_fn.hyperparams_cache))
                    del wrapped_fn.hyperparams_cache[oldest_key]

                wrapped_fn.hyperparams_cache[cache_lookup_key] = (optimal_hyperparameters, results)
                if CONFIG.enable_detailed_logging:
                    autotune_logger.info(f"Cached hyperparameters for key: {cache_lookup_key}")

        wrapped_fn.timing_results.clear()
        wrapped_fn.timing_results.update(results)
        wrapped_fn.optimal_hyperparams.clear()
        wrapped_fn.optimal_hyperparams.update(optimal_hyperparameters)

        try:
            return fn(*args, **dict(kws, **optimal_hyperparameters))
        except Exception as e:
            autotune_logger.error(f"Execution failed with optimal hyperparameters {optimal_hyperparameters}: {e}")
            raise

    wrapped_fn.timing_results = {}
    wrapped_fn.hyperparams_cache = {}
    wrapped_fn.optimal_hyperparams = {}
    return wrapped_fn
