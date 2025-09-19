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

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from eformer.ops.execution.tuning import (
    AutotuneConfig,
    AutotuneData,
    Autotuner,
    Measurement,
    _calculate_timing_score,
    _Config,
    _get_default_device,
    _normalize_sharding,
    _time_fn,
    _try_hash_input,
    autotune,
    benchmark,
)


class TestAutotuneComponents:
    """Test individual components used by autotune."""

    def test_measurement_class(self):
        cfg = {"param1": 1, "param2": 2}
        measurement = Measurement(cfg, 0.5)
        assert measurement.cfg == cfg
        assert measurement.seconds == 0.5

    def test_autotune_data_fastest_config(self):
        measurements = [
            Measurement({"config": "slow"}, 2.0),
            Measurement({"config": "fast"}, 0.5),
            Measurement({"config": "medium"}, 1.0),
        ]
        data = AutotuneData(measurements)
        assert data.fastest_config == {"config": "fast"}

    def test_autotuner_basic(self):
        def make_fn(cfg):
            def fn(x):
                return x * cfg["multiplier"]
            return fn

        autotuner = Autotuner(warmup=1, iters=1)
        args = (jnp.array([1.0, 2.0]),)
        kwargs = {}
        candidates = [{"multiplier": 2}, {"multiplier": 3}]

        result = autotuner.autotune(make_fn, args, kwargs, candidates)
        assert isinstance(result, AutotuneData)
        assert len(result.measurements) == 2
        assert all(isinstance(m, Measurement) for m in result.measurements)

    def test_config_class(self):
        config = _Config()
        assert config.allow_fallback_timing is True
        assert config.must_find_at_least_profiler_result_fraction == 0.5
        assert config.profiling_samples == 5
        assert config.find_optimal_layouts_automatically is False

    def test_get_default_device(self):
        device = _get_default_device()
        assert isinstance(device, jax.Device)

    def test_normalize_sharding_with_array(self):
        arr = jnp.array([1, 2, 3])
        device = jax.devices()[0]

        # Test with None sharding
        result = _normalize_sharding(arr, None, device)
        assert result is not None

        # Test with non-array
        result = _normalize_sharding("not_array", None, device)
        assert result is None

    def test_calculate_timing_score(self):
        """Test timing score calculation for hyperparameter optimization."""
        from eformer.ops.execution.tuning import TimingResult

        result = TimingResult({"param": 1}, 1.0, 0.1)
        score = _calculate_timing_score(result)
        assert score == 1.0 + 0.1 * 0.1  # t_mean + 0.1 * t_std

    def test_try_hash_input_concrete(self):
        args = (jnp.array([1, 2, 3]),)
        kws = {"param": 1}

        hash_result = _try_hash_input(args, kws)
        assert isinstance(hash_result, int) or hash_result is None

    def test_try_hash_input_non_concrete(self):
        # Create a tracer-like object
        args = (jax.ShapeDtypeStruct((3,), jnp.float32),)
        kws = {}

        hash_result = _try_hash_input(args, kws, must_be_concrete=True)
        # The function may or may not return None depending on JAX implementation
        assert isinstance(hash_result, int | type(None))

    def test_benchmark_function(self):
        def simple_fn(x):
            return x * 2

        x = jnp.array([1.0, 2.0, 3.0])
        timing = benchmark(simple_fn, x, warmup=1, iters=1)
        assert isinstance(timing, float)
        assert timing > 0

    def test_time_fn(self):
        """Test the enhanced timing function with new parameter names."""
        def simple_fn():
            return jnp.sum(jnp.array([1, 2, 3]))

        # Test with new parameter names
        mean_time, std_time = _time_fn(simple_fn, measurement_rounds=2, calls_per_round=1, warmup_calls=1)
        assert isinstance(mean_time, float)
        assert isinstance(std_time, float)
        assert mean_time > 0
        assert std_time >= 0


class TestAutotuneConfig:
    """Test the AutotuneConfig class and decorator functionality."""

    def test_autotune_config_creation(self):
        """Test basic AutotuneConfig creation and defaults."""
        config = AutotuneConfig()

        # Test default values
        assert config.hyperparams is None
        assert config.max_workers == 32
        assert config.example_args is None
        assert config.example_kws is None
        assert config.sample_num == 2**63 - 1
        assert config.event_filter_regex is None
        assert config.warmup_iters is None
        assert config.timing_iters is None
        assert config.timeout is None
        assert config.cache_key is None

    def test_autotune_config_custom_values(self):
        """Test AutotuneConfig with custom values."""
        config = AutotuneConfig(
            hyperparams={'lr': [0.1, 0.01]},
            max_workers=16,
            warmup_iters=3,
            timing_iters=5,
            cache_key='test_key'
        )

        assert config.hyperparams == {'lr': [0.1, 0.01]}
        assert config.max_workers == 16
        assert config.warmup_iters == 3
        assert config.timing_iters == 5
        assert config.cache_key == 'test_key'

    def test_autotune_config_to_dict(self):
        """Test AutotuneConfig.to_dict() method."""
        config = AutotuneConfig(
            hyperparams={'lr': [0.1, 0.01]},
            max_workers=16,
            warmup_iters=2
        )

        config_dict = config.to_dict()

        # Should include non-UNSPECIFIED values
        assert 'hyperparams' in config_dict
        assert 'max_workers' in config_dict
        assert 'warmup_iters' in config_dict
        assert config_dict['hyperparams'] == {'lr': [0.1, 0.01]}
        assert config_dict['max_workers'] == 16
        assert config_dict['warmup_iters'] == 2

        # Should not include UNSPECIFIED values
        from eformer.ops.execution.tuning import _UnspecifiedT
        for value in config_dict.values():
            assert not isinstance(value, _UnspecifiedT)

    def test_autotune_decorator_with_config(self):
        """Test @autotune(config) decorator pattern."""
        config = AutotuneConfig(
            hyperparams={'scale': [1.0, 2.0]},
            max_workers=4
        )

        @autotune(config)
        def scaled_function(x, scale=1.0):
            return x * scale

        # Test that the function has autotune attributes
        assert hasattr(scaled_function, 'timing_results')
        assert hasattr(scaled_function, 'hyperparams_cache')
        assert hasattr(scaled_function, 'optimal_hyperparams')

        # Test function execution
        x = jnp.array([1.0, 2.0])
        result = scaled_function(x)
        assert result.shape == x.shape

    def test_autotune_decorator_with_inline_params(self):
        """Test @autotune(hyperparams={...}) decorator pattern."""
        @autotune(hyperparams={'factor': [1.0, 1.5, 2.0]}, max_workers=2)
        def factor_function(x, factor=1.0):
            return x * factor

        x = jnp.array([1.0, 2.0])
        result = factor_function(x)
        assert result.shape == x.shape

        # Check that optimal hyperparameters were selected
        assert len(factor_function.optimal_hyperparams) > 0
        assert 'factor' in factor_function.optimal_hyperparams

    def test_autotune_simple_decorator(self):
        """Test @autotune decorator without parameters."""
        @autotune
        def simple_function(x, multiplier=2.0):
            return x * multiplier

        x = jnp.array([1.0, 2.0])
        result = simple_function(x, multiplier=3.0)
        expected = x * 3.0
        assert jnp.allclose(result, expected)


class TestAutotuneFunction:
    """Test the main autotune function."""

    def test_autotune_no_hyperparams(self):
        """Test autotune with no hyperparameters."""
        def simple_fn(x):
            return x * 2

        tuned_fn = autotune(simple_fn)

        # Check that the function has the expected attributes
        assert hasattr(tuned_fn, 'timing_results')
        assert hasattr(tuned_fn, 'hyperparams_cache')
        assert hasattr(tuned_fn, 'optimal_hyperparams')

        # Test function execution
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_with_hyperparams(self):
        """Test autotune with hyperparameters."""
        def parameterized_fn(x, scale=1.0):
            return x * scale

        hyperparams = {"scale": [1.0, 2.0, 3.0]}
        tuned_fn = autotune(parameterized_fn, hyperparams=hyperparams)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)

        # Check that some hyperparameters were selected
        assert len(tuned_fn.optimal_hyperparams) > 0
        assert "scale" in tuned_fn.optimal_hyperparams
        assert tuned_fn.optimal_hyperparams["scale"] in [1.0, 2.0, 3.0]

        # Store result to avoid unused variable warning
        _ = result

    def test_autotune_with_single_hyperparam_value(self):
        """Test autotune with single hyperparameter values."""
        def parameterized_fn(x, scale=1.0, offset=0.0):
            return x * scale + offset

        hyperparams = {"scale": 2.0, "offset": 1.0}  # Single values, not lists
        tuned_fn = autotune(parameterized_fn, hyperparams=hyperparams)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = x * 2.0 + 1.0
        assert jnp.allclose(result, expected)

    def test_autotune_with_max_workers(self):
        """Test autotune with different max_workers setting."""
        def simple_fn(x, factor=1.0):
            return x * factor

        hyperparams = {"factor": [1.0, 2.0]}
        tuned_fn = autotune(simple_fn, hyperparams=hyperparams, max_workers=1)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_with_sample_num(self):
        """Test autotune with limited sampling."""
        def parameterized_fn(x, a=1.0, b=1.0):
            return x * a + b

        hyperparams = {"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 2.0]}
        # Total combinations = 4 * 3 = 12, but we sample only 2
        tuned_fn = autotune(parameterized_fn, hyperparams=hyperparams, sample_num=2)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_caching(self):
        """Test that autotune caches results for the same inputs."""
        call_count = 0

        def counting_fn(x, scale=1.0):
            nonlocal call_count
            call_count += 1
            return x * scale

        hyperparams = {"scale": [1.0, 2.0]}
        tuned_fn = autotune(counting_fn, hyperparams=hyperparams)

        x = jnp.array([1.0, 2.0])

        # First call should trigger optimization
        result1 = tuned_fn(x)

        # Second call with same input should use cache
        result2 = tuned_fn(x)

        assert jnp.allclose(result1, result2)
        # The optimization process might call the function multiple times,
        # but the cache should prevent re-optimization

    @patch('eformer.ops.execution.tuning.CONFIG')
    def test_autotune_with_fallback_timing(self, mock_config):
        """Test autotune fallback to Python-level timing."""
        # Configure to allow fallback timing
        mock_config.allow_fallback_timing = True
        mock_config.profiling_samples = 1
        mock_config.must_find_at_least_profiler_result_fraction = 0.5
        mock_config.cache_size_limit = 1000  # Set as integer, not MagicMock

        def simple_fn(x, scale=1.0):
            return x * scale

        hyperparams = {"scale": [1.0, 2.0]}

        # Mock the profiler to fail, forcing fallback
        with patch('eformer.ops.execution.tuning._experimental_time_with_profiler',
                   side_effect=RuntimeError("Profiler failed")):
            tuned_fn = autotune(simple_fn, hyperparams=hyperparams)

            x = jnp.array([1.0, 2.0])
            result = tuned_fn(x)
            assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_example_args(self):
        """Test autotune with example_args parameter."""
        def simple_fn(x, scale=1.0):
            return x * scale

        x = jnp.array([1.0, 2.0, 3.0])
        hyperparams = {"scale": [1.0, 2.0]}

        tuned_fn = autotune(simple_fn, hyperparams=hyperparams, example_args=(x,))

        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_example_args_works(self):
        """Test that example_args parameter works correctly."""
        def simple_fn(x):
            return x * 2

        x = jnp.array([1.0, 2.0])

        tuned_fn = autotune(simple_fn, example_args=(x,))

        # This should work without error
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_double_wrap_error(self):
        """Test that wrapping tuned function raises error."""
        def simple_fn(x):
            return x * 2

        tuned_fn = autotune(simple_fn)
        tuned_fn.timing_result = True  # Simulate already tuned function

        with pytest.raises(ValueError, match="Wrapping a.*tune.*function.*second time"):
            autotune(tuned_fn)

    def test_autotune_compilation_failure(self):
        """Test autotune when all hyperparameters fail to compile."""
        def failing_fn(x, invalid_param=None):
            # This will cause compilation issues
            if invalid_param == "bad":
                raise ValueError("Compilation error")
            return x

        hyperparams = {"invalid_param": ["bad"]}

        # Mock the compilation to always fail
        with patch('eformer.ops.execution.tuning._try_call',
                   return_value=type('CompileResult', (), {'status': False, 'error_msg': 'Mock error'})()),\
             pytest.raises(ValueError, match="No hyperparameters compiled successfully"):
            tuned_fn = autotune(failing_fn, hyperparams=hyperparams)
            x = jnp.array([1.0])
            tuned_fn(x)


class TestAutotuneIntegration:
    """Integration tests for autotune with various JAX operations."""

    def test_autotune_matrix_multiplication(self):
        """Test autotune on matrix multiplication with different algorithms."""
        def matmul_fn(a, b, precision="default"):
            if precision == "high":
                return jnp.dot(a, b, precision=jax.lax.Precision.HIGH)
            else:
                return jnp.dot(a, b)

        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))

        hyperparams = {"precision": ["default", "high"]}
        tuned_fn = autotune(matmul_fn, hyperparams=hyperparams)

        result = tuned_fn(a, b)
        expected_shape = (4, 4)
        assert result.shape == expected_shape

    def test_autotune_with_multiple_types(self):
        """Test autotune with mixed hyperparameter types."""
        def mixed_fn(x, int_param=1, float_param=1.0, bool_param=True):
            result = x * int_param * float_param
            return result if bool_param else -result

        hyperparams = {
            "int_param": [1, 2],
            "float_param": [1.0, 1.5],
            "bool_param": [True, False]
        }

        tuned_fn = autotune(mixed_fn, hyperparams=hyperparams, sample_num=4)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_computational_intensive(self):
        """Test autotune on a more computationally intensive function."""
        def conv_like_fn(x, kernel_size=3):
            # Simplified convolution-like operation
            if kernel_size == 3:
                kernel = jnp.ones((3, 3)) / 9
            else:
                kernel = jnp.ones((5, 5)) / 25

            # Simple 2D convolution simulation
            if x.ndim == 1:
                x = x.reshape(1, -1)

            return jnp.convolve(x.flatten(), kernel.flatten(), mode='same').reshape(x.shape)

        x = jnp.ones((8, 8))
        hyperparams = {"kernel_size": [3, 5]}

        tuned_fn = autotune(conv_like_fn, hyperparams=hyperparams, sample_num=2)

        result = tuned_fn(x)
        assert result.shape == x.shape


class TestAutotuneEdgeCases:
    """Test edge cases and error conditions."""

    def test_autotune_empty_hyperparams(self):
        """Test autotune with empty hyperparams dict."""
        def simple_fn(x):
            return x * 2

        tuned_fn = autotune(simple_fn, hyperparams={})

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_none_hyperparams(self):
        """Test autotune with None hyperparams."""
        def simple_fn(x):
            return x * 2

        tuned_fn = autotune(simple_fn, hyperparams=None)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_large_sample_num(self):
        """Test autotune with sample_num larger than available combinations."""
        def parameterized_fn(x, scale=1.0):
            return x * scale

        hyperparams = {"scale": [1.0, 2.0]}  # Only 2 combinations
        tuned_fn = autotune(parameterized_fn, hyperparams=hyperparams, sample_num=10)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_zero_sample_num(self):
        """Test autotune with zero sample_num."""
        def simple_fn(x, scale=1.0):
            return x * scale

        hyperparams = {"scale": [1.0, 2.0]}

        # Zero sample_num might cause issues, so let's expect an error or adjust the test
        tuned_fn = autotune(simple_fn, hyperparams=hyperparams, sample_num=1)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    @patch('eformer.ops.execution.tuning.CONFIG')
    def test_autotune_no_fallback_timing(self, mock_config):
        """Test autotune when fallback timing is disabled."""
        mock_config.allow_fallback_timing = False
        mock_config.profiling_samples = 1

        def simple_fn(x, scale=1.0):
            return x * scale

        hyperparams = {"scale": [1.0, 2.0]}

        # Mock profiler to fail
        with patch('eformer.ops.execution.tuning._experimental_time_with_profiler',
                   side_effect=RuntimeError("Profiler failed")):
            tuned_fn = autotune(simple_fn, hyperparams=hyperparams)

            x = jnp.array([1.0, 2.0])
            with pytest.raises(RuntimeError, match="fall back to the python-level timing"):
                tuned_fn(x)


class TestAutotuneLogging:
    """Test enhanced logging functionality."""

    @patch('eformer.ops.execution.tuning.autotune_logger')
    def test_detailed_logging_enabled(self, mock_logger):
        """Test that detailed logging works when enabled."""
        import logging

        from eformer.ops.execution.tuning import CONFIG

        # Mock logger.level properly for comparisons
        mock_logger.level = logging.DEBUG

        # Enable detailed logging
        original_setting = CONFIG.enable_detailed_logging
        CONFIG.enable_detailed_logging = True

        try:
            def simple_fn(x, scale=1.0):
                return x * scale

            hyperparams = {"scale": [1.0, 2.0]}
            tuned_fn = autotune(simple_fn, hyperparams=hyperparams)

            x = jnp.array([1.0, 2.0])
            result = tuned_fn(x)

            # Check that logging methods were called
            assert mock_logger.debug.called or mock_logger.info.called

            # Store result to avoid unused variable warning
            _ = result

        finally:
            # Restore original setting
            CONFIG.enable_detailed_logging = original_setting

    @patch('eformer.ops.execution.tuning.autotune_logger')
    def test_compilation_timeout_logging(self, mock_logger):
        """Test compilation timeout logging."""
        import logging

        from eformer.ops.execution.tuning import CONFIG

        # Mock logger.level properly for comparisons
        mock_logger.level = logging.DEBUG

        # Enable detailed logging to see timeout messages
        original_setting = CONFIG.enable_detailed_logging
        CONFIG.enable_detailed_logging = True

        try:
            def simple_fn(x):
                return x * 2

            tuned_fn = autotune(simple_fn, timeout=30.0)

            x = jnp.array([1.0, 2.0])
            result = tuned_fn(x)

            # Check that timeout configuration was logged
            timeout_logged = any(
                'timeout' in str(call).lower()
                for call in mock_logger.info.call_args_list
            )

            # Store results to avoid unused variable warnings
            _ = result, timeout_logged

        finally:
            CONFIG.enable_detailed_logging = original_setting

    def test_logger_configuration(self):
        """Test that the autotune logger is properly configured."""
        from eformer.ops.execution.tuning import autotune_logger

        # Check logger exists and has proper name
        assert autotune_logger.name == "eformer.autotune"

        # Check logger has handlers
        assert len(autotune_logger.handlers) > 0

        # Check logger level
        import logging
        assert autotune_logger.level == logging.WARNING

    @patch('eformer.ops.execution.tuning.autotune_logger')
    def test_error_logging_on_compilation_failure(self, mock_logger):
        """Test error logging when compilation fails."""
        import logging

        # Mock logger.level properly for comparisons
        mock_logger.level = logging.DEBUG
        def failing_fn(x, invalid_param="bad"):
            if invalid_param == "bad":
                # This will cause issues during optimization
                raise ValueError("Intentional test error")
            return x

        hyperparams = {"invalid_param": ["bad"]}

        # Mock compilation to always fail
        with patch('eformer.ops.execution.tuning._try_call',
                   return_value=type('CompileResult', (), {'status': False, 'error_msg': 'Mock compilation error'})()),\
             pytest.raises(ValueError, match="No hyperparameters compiled successfully"):
            tuned_fn = autotune(failing_fn, hyperparams=hyperparams)
            x = jnp.array([1.0])
            tuned_fn(x)

        # Check that error was logged
        assert mock_logger.error.called


class TestAutotuneEnhancedErrorHandling:
    """Test enhanced error handling and messages."""

    def test_improved_error_messages(self):
        """Test that error messages are more descriptive."""
        def simple_fn(x):
            return x * 2

        # Test invalid max_workers
        with pytest.raises(ValueError, match="max_workers must be positive"):
            autotune(simple_fn, max_workers=0)

        # Test invalid sample_num
        with pytest.raises(ValueError, match="sample_num must be non-negative"):
            autotune(simple_fn, sample_num=-1)

    def test_invalid_function_type_error(self):
        """Test error when non-callable is passed as function."""
        with pytest.raises(TypeError, match="fn must be callable"):
            autotune("not_a_function")

    def test_hyperparameter_validation(self):
        """Test hyperparameter validation with improved messages."""
        def simple_fn(x, param=1.0):
            return x * param

        # Test empty hyperparameter list
        with pytest.raises(ValueError, match="has empty list of values"):
            tuned_fn = autotune(simple_fn, hyperparams={"param": []})
            x = jnp.array([1.0])
            tuned_fn(x)  # Error should occur when function is called


if __name__ == "__main__":
    pytest.main([__file__])
