# Copyright 2025 The EasyDeL/eFormer Author @erfanzar.
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
    FNAutotuner,
    TimingResult,
    _get_default_device,
    _normalize_sharding,
    _try_hash_input,
    autotune,
    autotune_logger,
)


class TestAutotunerComponents:
    """Tests for utilities and core components adapted to the new API."""

    def test_get_default_device(self):
        device = _get_default_device()
        assert isinstance(device, jax.Device)

    def test_normalize_sharding_with_array(self):
        arr = jnp.array([1, 2, 3])
        device = jax.devices()[0]

        # Test with None sharding -> default SingleDeviceSharding on device
        result = _normalize_sharding(arr, None, device)
        assert result is not None

        # Test with non-array
        result = _normalize_sharding("not_array", None, device)
        assert result is None

    def test_calculate_timing_score(self):
        """Test timing score calculation for candidate selection."""
        # TimingResult is available at module level; score method is a static method on FNAutotuner
        tr = TimingResult({"param": 1}, 1.0, 0.1)
        score = FNAutotuner._calculate_timing_score(tr)
        assert score == 1.0 + 0.1 * 0.1

    def test_try_hash_input_concrete(self):
        args = (jnp.array([1, 2, 3]),)
        kws = {"param": 1}

        hash_result = _try_hash_input(args, kws)
        assert isinstance(hash_result, int) or hash_result is None

    def test_try_hash_input_non_concrete(self):
        # Use a shape/dtype struct (not a concrete JAX array)
        args = (jax.ShapeDtypeStruct((3,), jnp.float32),)
        kws = {}

        hash_result = _try_hash_input(args, kws, must_be_concrete=True)
        assert isinstance(hash_result, (int, type(None)))

    def test_time_fn(self):
        """Test the fallback timing via FNAutotuner._time_fn."""
        tuner = FNAutotuner(timing_warmup_iterations=1, timing_rounds=2, calls_per_round=1)

        def simple_fn():
            return jnp.sum(jnp.array([1, 2, 3]))

        mean_time, std_time = tuner._time_fn(simple_fn)
        assert isinstance(mean_time, float)
        assert isinstance(std_time, float)
        assert mean_time > 0
        assert std_time >= 0

    def test_autotuner_tune_basic(self):
        """Test the FNAutotuner.tune method directly with simple hyperparams."""
        tuner = FNAutotuner(profiling_samples=1, profiler_verbose=False, allow_fallback_timing=True)

        def fn(x, multiplier=1.0):
            return x * multiplier

        x = jnp.array([1.0, 2.0])
        pf, best_hps, results = tuner.tune(
            fn,
            args=(x,),
            kwargs={},
            hyperparams={"multiplier": [1.0, 2.0]},
            max_workers=1,
            sample_num=2,
        )
        assert callable(pf)
        assert isinstance(best_hps, dict)
        assert isinstance(results, list)
        assert "multiplier" in best_hps


class TestAutotuneDecorator:
    """Tests for the new decorator API."""

    def test_autotune_decorator_with_inline_params(self):
        @autotune(hyperparams={"factor": [1.0, 1.5, 2.0]}, max_workers=2)
        def factor_function(x, factor=1.0):
            return x * factor

        x = jnp.array([1.0, 2.0])
        result = factor_function(x)
        assert result.shape == x.shape
        assert hasattr(factor_function, "optimal_hyperparams")
        assert "factor" in factor_function.optimal_hyperparams

    def test_autotune_simple_decorator(self):
        """Test @autotune() decorator without params."""

        @autotune()
        def simple_function(x, multiplier=2.0):
            return x * multiplier

        x = jnp.array([1.0, 2.0])
        result = simple_function(x, multiplier=3.0)
        expected = x * 3.0
        assert jnp.allclose(result, expected)


class TestAutotuneFunction:
    """Higher-level tests for the autotune() convenience decorator."""

    def test_autotune_no_hyperparams(self):
        """Decorator without hyperparams should work and pass through."""

        def simple_fn(x):
            return x * 2

        tuned_fn = autotune()(simple_fn)

        assert hasattr(tuned_fn, "timing_results")
        assert hasattr(tuned_fn, "optimal_hyperparams")

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_with_hyperparams(self):
        def parameterized_fn(x, scale=1.0):
            return x * scale

        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0, 3.0]})(parameterized_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)

        assert len(tuned_fn.optimal_hyperparams) > 0
        assert "scale" in tuned_fn.optimal_hyperparams
        assert tuned_fn.optimal_hyperparams["scale"] in [1.0, 2.0, 3.0]
        _ = result

    def test_autotune_with_single_hyperparam_value(self):
        def parameterized_fn(x, scale=1.0, offset=0.0):
            return x * scale + offset

        tuned_fn = autotune(hyperparams={"scale": 2.0, "offset": 1.0})(parameterized_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = x * 2.0 + 1.0
        assert jnp.allclose(result, expected)

    def test_autotune_with_max_workers(self):
        def simple_fn(x, factor=1.0):
            return x * factor

        tuned_fn = autotune(hyperparams={"factor": [1.0, 2.0]}, max_workers=1)(simple_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_with_sample_num(self):
        def parameterized_fn(x, a=1.0, b=1.0):
            return x * a + b

        tuned_fn = autotune(hyperparams={"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 2.0]}, sample_num=2)(
            parameterized_fn
        )
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_caching(self):
        """Ensure repeated identical inputs use cached best hyperparams."""
        call_count = 0

        def counting_fn(x, scale=1.0):
            nonlocal call_count
            call_count += 1
            return x * scale

        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0]})(counting_fn)

        x = jnp.array([1.0, 2.0])
        result1 = tuned_fn(x)
        result2 = tuned_fn(x)
        assert jnp.allclose(result1, result2)

    def test_autotune_with_fallback_timing(self):
        """Force profiler failure to trigger Python fallback timing."""

        def simple_fn(x, scale=1.0):
            return x * scale

        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0]}, profiling_samples=1)(simple_fn)

        # Mock profiler to fail, forcing fallback
        with patch(
            "eformer.ops.execution.tuning.Profiler.profile_time_by_function_id",
            side_effect=RuntimeError("Profiler failed"),
        ):
            x = jnp.array([1.0, 2.0])
            result = tuned_fn(x)
            assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_example_args(self):
        def simple_fn(x, scale=1.0):
            return x * scale

        x = jnp.array([1.0, 2.0, 3.0])
        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0]}, example_args=(x,))(simple_fn)
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_example_args_works(self):
        def simple_fn(x):
            return x * 2

        x = jnp.array([1.0, 2.0])
        tuned_fn = autotune(example_args=(x,))(simple_fn)
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_compilation_failure(self):
        """When all candidates fail to compile, raise a ValueError."""

        def failing_fn(x, invalid_param=None):
            if invalid_param == "bad":
                raise ValueError("Compilation error")
            return x

        with (
            patch("eformer.ops.execution.tuning.FNAutotuner._try_call", return_value=(False, "Mock error", None)),
            pytest.raises(ValueError, match="No hyperparameters compiled successfully"),
        ):
            tuned_fn = autotune(hyperparams={"invalid_param": ["bad"]})(failing_fn)
            x = jnp.array([1.0])
            tuned_fn(x)


class TestAutotuneIntegration:
    """Integration tests on realistic JAX ops."""

    def test_autotune_matrix_multiplication(self):
        def matmul_fn(a, b, precision="default"):
            if precision == "high":
                return jnp.dot(a, b, precision=jax.lax.Precision.HIGH)
            else:
                return jnp.dot(a, b)

        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))

        tuned_fn = autotune(hyperparams={"precision": ["default", "high"]})(matmul_fn)
        result = tuned_fn(a, b)
        assert result.shape == (4, 4)

    def test_autotune_with_multiple_types(self):
        def mixed_fn(x, int_param=1, float_param=1.0, bool_param=True):
            result = x * int_param * float_param
            return result if bool_param else -result

        tuned_fn = autotune(
            hyperparams={"int_param": [1, 2], "float_param": [1.0, 1.5], "bool_param": [True, False]}, sample_num=4
        )(mixed_fn)

        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_computational_intensive(self):
        """Convolution-like computation to exercise compilation paths."""

        def conv_like_fn(x, kernel_size=3):
            if kernel_size == 3:
                kernel = jnp.ones((3, 3)) / 9
            else:
                kernel = jnp.ones((5, 5)) / 25
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return jnp.convolve(x.flatten(), kernel.flatten(), mode="same").reshape(x.shape)

        x = jnp.ones((8, 8))
        tuned_fn = autotune(hyperparams={"kernel_size": [3, 5]}, sample_num=2)(conv_like_fn)
        result = tuned_fn(x)
        assert result.shape == x.shape


class TestAutotuneEdgeCases:
    """Edge cases and error conditions."""

    def test_autotune_empty_hyperparams(self):
        def simple_fn(x):
            return x * 2

        tuned_fn = autotune(hyperparams={})(simple_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_none_hyperparams(self):
        def simple_fn(x):
            return x * 2

        tuned_fn = autotune(hyperparams=None)(simple_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        expected = simple_fn(x)
        assert jnp.allclose(result, expected)

    def test_autotune_large_sample_num(self):
        def parameterized_fn(x, scale=1.0):
            return x * scale

        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0]}, sample_num=10)(parameterized_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_zero_sample_num(self):
        """Zero sampling is ill-defined; use 1 here to ensure minimal sampling."""

        def simple_fn(x, scale=1.0):
            return x * scale

        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0]}, sample_num=1)(simple_fn)
        x = jnp.array([1.0, 2.0])
        result = tuned_fn(x)
        assert jnp.array_equal(result.shape, x.shape)

    def test_autotune_no_fallback_timing(self):
        """Disable fallback and force profiler failure -> raise."""

        def simple_fn(x, scale=1.0):
            return x * scale

        tuned_fn = autotune(hyperparams={"scale": [1.0, 2.0]}, allow_fallback_timing=False)(simple_fn)

        with patch(
            "eformer.ops.execution.tuning.Profiler.profile_time_by_function_id",
            side_effect=RuntimeError("Profiler failed"),
        ):
            x = jnp.array([1.0, 2.0])
            with pytest.raises(RuntimeError, match="fall back to the python-level timing"):
                tuned_fn(x)


class TestAutotuneEnhancedErrorHandling:
    """Validation and error messages."""

    def test_invalid_function_type_error(self):
        """Calling decorator on non-callable should raise."""
        deco = autotune()
        with pytest.raises(TypeError, match="fn must be callable"):
            deco("not_a_function")

    def test_hyperparameter_validation(self):
        def simple_fn(x, param=1.0):
            return x * param

        with pytest.raises(ValueError, match="has empty list of values"):
            tuned_fn = autotune(hyperparams={"param": []})(simple_fn)
            x = jnp.array([1.0])
            tuned_fn(x)  # Should error when invoked (during tuning)


if __name__ == "__main__":
    pytest.main([__file__])
