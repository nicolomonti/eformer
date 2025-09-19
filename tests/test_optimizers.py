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

import json

import jax
import jax.numpy as jnp
import optax
import pytest

from eformer.optimizers import (
    AdafactorConfig,
    AdamWConfig,
    LionConfig,
    MarsConfig,
    MuonConfig,
    OptimizerFactory,
    RMSPropConfig,
    SchedulerConfig,
    SchedulerFactory,
    WhiteKronConfig,
)


class TestSerializationMixin:
    """Test the SerializationMixin functionality."""

    def test_to_dict_filters_private_fields(self):
        config = AdamWConfig()
        config._private_field = "should_not_appear"
        result = config.to_dict()
        assert "_private_field" not in result
        assert "b1" in result
        assert "b2" in result

    def test_from_dict_with_valid_data(self):
        data = {"b1": 0.95, "b2": 0.998, "eps": 1e-7}
        config = AdamWConfig.from_dict(data)
        assert config.b1 == 0.95
        assert config.b2 == 0.998
        assert config.eps == 1e-7

    def test_from_dict_with_extra_keys(self):
        data = {"b1": 0.95, "unknown_key": "value"}
        with pytest.warns(UserWarning, match="Ignoring unexpected keys"):
            config = AdamWConfig.from_dict(data)
        assert config.b1 == 0.95

    def test_to_json_serialization(self):
        config = AdamWConfig(b1=0.95, b2=0.998)
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["b1"] == 0.95
        assert data["b2"] == 0.998

    def test_from_json_deserialization(self):
        json_str = '{"b1": 0.95, "b2": 0.998, "eps": 1e-7}'
        config = AdamWConfig.from_json(json_str)
        assert config.b1 == 0.95
        assert config.b2 == 0.998
        assert config.eps == 1e-7


class TestSchedulerConfig:
    """Test SchedulerConfig functionality."""

    def test_default_initialization(self):
        config = SchedulerConfig()
        assert config.scheduler_type is None
        assert config.learning_rate == 5e-5
        assert config.exponent == 1.0

    def test_linear_scheduler_validation(self):
        with pytest.raises(ValueError, match="Linear scheduler requires learning_rate_end"):
            SchedulerConfig(scheduler_type="linear", steps=1000)

    def test_scheduler_requires_steps(self):
        with pytest.raises(ValueError, match="Steps must be specified for non-constant schedulers"):
            SchedulerConfig(scheduler_type="cosine")

    def test_warmup_validation(self):
        with pytest.raises(ValueError, match="Steps required when using warmup"):
            SchedulerConfig(warmup_steps=100)

        with pytest.raises(ValueError, match="Warmup steps must be less than total steps"):
            SchedulerConfig(warmup_steps=1000, steps=500, scheduler_type="cosine")

    def test_valid_linear_config(self):
        config = SchedulerConfig(scheduler_type="linear", steps=1000, learning_rate_end=1e-6, warmup_steps=100)
        assert config.scheduler_type == "linear"
        assert config.steps == 1000
        assert config.learning_rate_end == 1e-6
        assert config.warmup_steps == 100

    def test_valid_cosine_config(self):
        config = SchedulerConfig(scheduler_type="cosine", steps=1000, warmup_steps=100)
        assert config.scheduler_type == "cosine"
        assert config.steps == 1000
        assert config.warmup_steps == 100


class TestOptimizerConfigs:
    """Test individual optimizer configuration classes."""

    def test_adafactor_config_defaults(self):
        config = AdafactorConfig()
        assert config.min_dim_size_to_factor == 128
        assert config.decay_rate == 0.8
        assert config.multiply_by_parameter_scale is True
        assert config.factored is True

    def test_adamw_config_defaults(self):
        config = AdamWConfig()
        assert config.b1 == 0.9
        assert config.b2 == 0.999
        assert config.eps == 1e-8
        assert config.eps_root == 0.0

    def test_lion_config_defaults(self):
        config = LionConfig()
        assert config.b1 == 0.9
        assert config.b2 == 0.99
        assert config.mu_dtype is None

    def test_rmsprop_config_defaults(self):
        config = RMSPropConfig()
        assert config.decay == 0.9
        assert config.initial_scale == 0.0
        assert config.momentum is None
        assert config.nesterov is False

    def test_muon_config_defaults(self):
        config = MuonConfig()
        assert config.ns_coeffs == (3.4445, -4.775, 2.0315)
        assert config.ns_steps == 5
        assert config.beta == 0.95
        assert config.nesterov is True

    def test_mars_config_defaults(self):
        config = MarsConfig()
        assert config.weight_decay == 0.1
        assert config.beta1 == 0.95
        assert config.beta2 == 0.99
        assert config.gamma == 0.025

    def test_white_kron_config_defaults(self):
        config = WhiteKronConfig()
        assert config.lr_style == "adam"
        assert config.b1 == 0.95
        assert config.normalize_grads is False
        assert config.max_size_dense == 16384
        assert config.preconditioner_lr == 0.7
        assert config.preconditioner_init_scale == 1.0
        assert config.dtype == jnp.bfloat16
        assert config.block_size == 256
        assert config.pipeline_axis_size == 1
        assert config.noise_scale == 1e-9
        assert config.weight_decay == 0.1


class TestSchedulerFactory:
    """Test SchedulerFactory functionality."""

    def test_constant_scheduler(self):
        config = SchedulerConfig(learning_rate=0.001)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == 0.001
        assert scheduler(1000) == 0.001

    def test_linear_scheduler_without_warmup(self):
        config = SchedulerConfig(scheduler_type="linear", learning_rate=0.001, learning_rate_end=0.0001, steps=1000)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == pytest.approx(0.001, rel=1e-3)
        assert scheduler(1000) == pytest.approx(0.0001, rel=1e-3)

    def test_linear_scheduler_with_warmup(self):
        config = SchedulerConfig(
            scheduler_type="linear", learning_rate=0.001, learning_rate_end=0.0001, steps=1000, warmup_steps=100
        )
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == pytest.approx(1e-8, rel=1e-2)
        assert scheduler(100) == pytest.approx(0.001, rel=1e-3)
        # Linear scheduler with warmup has slight precision differences
        assert scheduler(1000) == pytest.approx(0.00019, rel=1e-1)

    def test_cosine_scheduler_without_warmup(self):
        config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.001, steps=1000)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == 0.001
        assert scheduler(1000) < 0.001

    def test_cosine_scheduler_with_warmup(self):
        config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.001, steps=1000, warmup_steps=100)
        scheduler = SchedulerFactory.create_scheduler(config)
        assert scheduler(0) == pytest.approx(1e-8, rel=1e-2)
        assert scheduler(100) == pytest.approx(0.001, rel=1e-3)

    def test_custom_scheduler(self):
        config = SchedulerConfig(steps=1000)

        def custom_fn(steps):
            return optax.constant_schedule(0.005)

        scheduler = SchedulerFactory.create_scheduler(config, custom_scheduler=custom_fn)
        assert scheduler(0) == 0.005

    def test_custom_scheduler_requires_steps(self):
        config = SchedulerConfig()

        def custom_fn(steps):
            return optax.constant_schedule(0.005)

        with pytest.raises(ValueError, match="Custom schedulers require steps configuration"):
            SchedulerFactory.create_scheduler(config, custom_scheduler=custom_fn)

    def test_unsupported_scheduler_type(self):
        config = SchedulerConfig(scheduler_type="unknown", steps=1000)
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            SchedulerFactory.create_scheduler(config)


class TestOptimizerFactory:
    """Test OptimizerFactory functionality."""

    def test_optimizer_registry(self):
        assert "adafactor" in OptimizerFactory._OPTIMIZER_REGISTRY
        assert "adamw" in OptimizerFactory._OPTIMIZER_REGISTRY
        assert "lion" in OptimizerFactory._OPTIMIZER_REGISTRY
        assert "rmsprop" in OptimizerFactory._OPTIMIZER_REGISTRY
        assert "skew" in OptimizerFactory._OPTIMIZER_REGISTRY
        assert "quad" in OptimizerFactory._OPTIMIZER_REGISTRY

    def test_register_optimizer(self):
        class DummyOptimizer:
            pass

        class DummyConfig:
            pass

        OptimizerFactory.register_optimizer("dummy", DummyOptimizer, DummyConfig)
        assert "dummy" in OptimizerFactory._OPTIMIZER_REGISTRY
        assert OptimizerFactory._OPTIMIZER_REGISTRY["dummy"] == (DummyOptimizer, DummyConfig)

    def test_unsupported_optimizer_type(self):
        scheduler_config = SchedulerConfig()
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            OptimizerFactory.create("unknown", scheduler_config)

    def test_create_adamw_optimizer(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig(b1=0.95, b2=0.998)

        optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_optimizer_with_defaults(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)

        optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_optimizer_with_weight_decay(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config, weight_decay=0.01)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_with_gradient_clipping(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config, clip_grad=1.0)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_with_gradient_accumulation(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, scheduler = OptimizerFactory.create(
            "adamw", scheduler_config, optimizer_config, gradient_accumulation_steps=4
        )

        assert isinstance(optimizer, optax.MultiSteps)

    def test_create_skew_optimizer(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(b1=0.9, preconditioner_lr=0.5)

        optimizer, scheduler = OptimizerFactory.create("skew", scheduler_config, optimizer_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_quad_optimizer(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(normalize_grads=True, block_size=128)

        optimizer, scheduler = OptimizerFactory.create("quad", scheduler_config, optimizer_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_create_white_kron_with_defaults(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)

        optimizer, scheduler = OptimizerFactory.create("skew", scheduler_config)

        assert isinstance(optimizer, optax.GradientTransformation)
        assert scheduler(0) == 0.001

    def test_invalid_config_type(self):
        scheduler_config = SchedulerConfig(learning_rate=0.001)
        wrong_config = LionConfig()  # Wrong config for AdamW

        with pytest.raises(TypeError, match="Invalid config type"):
            OptimizerFactory.create("adamw", scheduler_config, wrong_config)

    def test_warmup_without_scheduler_type(self):
        with pytest.raises(ValueError, match="Steps required when using warmup"):
            SchedulerConfig(learning_rate=0.001, warmup_steps=100)

    def test_dtype_conversion(self):
        config = AdamWConfig()
        config.mu_dtype = "float16"
        OptimizerFactory._convert_dtypes(config)
        assert config.mu_dtype == jnp.float16

    def test_invalid_dtype_conversion(self):
        config = AdamWConfig()
        config.mu_dtype = "invalid_dtype"
        with pytest.raises(ValueError, match="Invalid dtype specified"):
            OptimizerFactory._convert_dtypes(config)

    def test_generate_template(self):
        template = OptimizerFactory.generate_template("adamw")
        assert "AdamWConfig" in template
        assert "b1" in template
        assert "b2" in template

    def test_generate_template_unknown_optimizer(self):
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            OptimizerFactory.generate_template("unknown")

    def test_serialize_config_dict(self):
        config = AdamWConfig(b1=0.95, b2=0.998)
        result = OptimizerFactory.serialize_config(config, format="dict")
        assert isinstance(result, dict)
        assert result["b1"] == 0.95

    def test_serialize_config_json(self):
        config = AdamWConfig(b1=0.95, b2=0.998)
        result = OptimizerFactory.serialize_config(config, format="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["b1"] == 0.95

    def test_serialize_config_invalid_format(self):
        config = AdamWConfig()
        with pytest.raises(ValueError, match="Supported formats"):
            OptimizerFactory.serialize_config(config, format="xml")

    def test_deserialize_config_dict(self):
        data = {"b1": 0.95, "b2": 0.998}
        config = OptimizerFactory.deserialize_config("adamw", data, format="dict")
        assert isinstance(config, AdamWConfig)
        assert config.b1 == 0.95

    def test_deserialize_config_json(self):
        json_str = '{"b1": 0.95, "b2": 0.998}'
        config = OptimizerFactory.deserialize_config("adamw", json_str, format="json")
        assert isinstance(config, AdamWConfig)
        assert config.b1 == 0.95

    def test_deserialize_config_unknown_optimizer(self):
        data = {"b1": 0.95}
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            OptimizerFactory.deserialize_config("unknown", data)

    def test_deserialize_config_invalid_data_type_json(self):
        with pytest.raises(TypeError, match="Expected string input for JSON format"):
            OptimizerFactory.deserialize_config("adamw", {"b1": 0.95}, format="json")

    def test_deserialize_config_invalid_data_type_dict(self):
        with pytest.raises(TypeError, match="Expected dictionary input for dict format"):
            OptimizerFactory.deserialize_config("adamw", "invalid", format="dict")

    def test_deserialize_config_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            OptimizerFactory.deserialize_config("adamw", {}, format="xml")


class TestOptimizerIntegration:
    """Integration tests for optimizers with actual JAX operations."""

    def test_adamw_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (10, 10))}
        grads = {"w": jax.random.normal(key, (10, 10))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = AdamWConfig()

        optimizer, _ = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (10, 10)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_lion_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (5, 5))}
        grads = {"w": jax.random.normal(key, (5, 5))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = LionConfig()

        optimizer, _ = OptimizerFactory.create("lion", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (5, 5)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_cosine_schedule_progression(self):
        scheduler_config = SchedulerConfig(scheduler_type="cosine", learning_rate=0.01, steps=1000)
        optimizer_config = AdamWConfig()

        _, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)

        lr_start = scheduler(0)
        lr_mid = scheduler(500)
        lr_end = scheduler(1000)

        assert lr_start == 0.01
        assert lr_mid < lr_start
        assert lr_end < lr_mid

    def test_linear_schedule_progression(self):
        scheduler_config = SchedulerConfig(
            scheduler_type="linear", learning_rate=0.01, learning_rate_end=0.001, steps=1000
        )
        optimizer_config = AdamWConfig()

        _, scheduler = OptimizerFactory.create("adamw", scheduler_config, optimizer_config)

        lr_start = scheduler(0)
        lr_mid = scheduler(500)
        lr_end = scheduler(1000)

        assert lr_start == 0.01
        assert lr_mid == pytest.approx(0.0055, rel=1e-3)
        assert lr_end == 0.001

    def test_skew_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (8, 8))}
        grads = {"w": jax.random.normal(key, (8, 8))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(b1=0.9, block_size=4)

        optimizer, _ = OptimizerFactory.create("skew", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (8, 8)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_quad_step_update(self):
        key = jax.random.PRNGKey(42)
        params = {"w": jax.random.normal(key, (6, 6))}
        grads = {"w": jax.random.normal(key, (6, 6))}

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(preconditioner_lr=0.5, max_size_dense=1000)

        optimizer, _ = OptimizerFactory.create("quad", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert "w" in new_params
        assert new_params["w"].shape == (6, 6)
        assert not jnp.allclose(params["w"], new_params["w"])

    def test_white_kron_with_different_shapes(self):
        """Test white_kron optimizers with different parameter shapes."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        params = {
            "dense": jax.random.normal(keys[0], (10, 20)),  # 2D matrix
            "bias": jax.random.normal(keys[1], (20,)),      # 1D vector
            "small": jax.random.normal(keys[2], (3, 3)),    # Small matrix
            "large": jax.random.normal(keys[3], (50, 100)), # Larger matrix
        }
        grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(123), x.shape), params)

        scheduler_config = SchedulerConfig(learning_rate=0.001)
        optimizer_config = WhiteKronConfig(max_size_dense=1000, block_size=8)

        optimizer, _ = OptimizerFactory.create("skew", scheduler_config, optimizer_config)
        opt_state = optimizer.init(params)

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Check all parameters were updated
        for key in params:
            assert key in new_params
            assert new_params[key].shape == params[key].shape
            assert not jnp.allclose(params[key], new_params[key])


if __name__ == "__main__":
    pytest.main([__file__])

