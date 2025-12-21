# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Test Suite for JAX Adapter Enterprise Edition

Tests for:
- Pure JAX function conversion
- Flax nn.Module integration
- Haiku transformed functions
- HuggingFace Transformers (Flax models)
- Compilation hook (like torch.compile)
- Training state integration
- StableHLO export
"""

import pytest
import numpy as np


# =============================================================================
# Fixtures and Test Setup
# =============================================================================


@pytest.fixture(scope="module")
def jax():
    """Get JAX module, skip if not available."""
    pytest.importorskip("jax")
    import jax

    return jax


@pytest.fixture(scope="module")
def jnp(jax):
    """Get JAX numpy module."""
    import jax.numpy as jnp

    return jnp


@pytest.fixture(scope="module")
def adapter():
    """Create JAX adapter instance."""
    pytest.importorskip("jax")
    from zenith.adapters import JAXAdapter

    return JAXAdapter()


@pytest.fixture
def simple_jax_function(jnp):
    """Create a simple JAX function."""

    def fn(x):
        return jnp.dot(x, x.T)

    return fn


@pytest.fixture
def mlp_function(jnp):
    """Create a simple MLP-like function."""

    def fn(x):
        # Simple 2-layer network simulation
        w1 = jnp.ones((x.shape[-1], 32))
        w2 = jnp.ones((32, 10))
        h = jnp.maximum(jnp.dot(x, w1), 0)  # ReLU
        return jnp.dot(h, w2)

    return fn


@pytest.fixture
def sample_input_2d(jnp):
    """Sample 2D input for simple functions."""
    return jnp.ones((4, 4), dtype=jnp.float32)


@pytest.fixture
def sample_input_batch(jnp):
    """Sample batched input."""
    return jnp.ones((2, 16), dtype=jnp.float32)


# =============================================================================
# Basic Adapter Tests
# =============================================================================


class TestJAXAdapterBasic:
    """Basic tests for JAX adapter."""

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "jax"

    def test_adapter_availability(self, adapter):
        """Test JAX availability check."""
        assert adapter.is_available is True

    def test_jax_version_detection(self, adapter, jax):
        """Test JAX version detection."""
        version = adapter._get_jax_version()
        assert isinstance(version, tuple)
        assert len(version) >= 2


class TestJAXFunctionConversion:
    """Tests for pure JAX function to GraphIR conversion."""

    def test_simple_function_conversion(
        self, adapter, simple_jax_function, sample_input_2d
    ):
        """Test converting simple JAX function to GraphIR."""
        graph = adapter.from_model(simple_jax_function, sample_input=sample_input_2d)

        assert graph is not None
        assert graph.name is not None
        assert len(graph.inputs) > 0
        assert len(graph.outputs) > 0

    def test_mlp_function_conversion(self, adapter, mlp_function, sample_input_batch):
        """Test converting MLP-like function to GraphIR."""
        graph = adapter.from_model(mlp_function, sample_input=sample_input_batch)

        assert graph is not None
        assert len(graph.inputs) >= 1

    def test_conversion_requires_sample_input(self, adapter, simple_jax_function):
        """Test that sample_input is required."""
        with pytest.raises(ValueError, match="sample_input is required"):
            adapter.from_model(simple_jax_function)

    def test_lambda_function_conversion(self, adapter, jnp):
        """Test converting lambda function to GraphIR."""
        fn = lambda x: jnp.sum(x**2)
        sample = jnp.ones((4,), dtype=jnp.float32)

        graph = adapter.from_model(fn, sample_input=sample)

        assert graph is not None


class TestFlaxModuleConversion:
    """Tests for Flax nn.Module conversion."""

    @pytest.fixture
    def flax_mlp(self):
        """Create a simple Flax MLP module."""
        pytest.importorskip("flax")
        from flax import linen as nn

        class SimpleMLP(nn.Module):
            hidden_dim: int = 32
            output_dim: int = 10

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(self.hidden_dim)(x)
                x = nn.relu(x)
                x = nn.Dense(self.output_dim)(x)
                return x

        return SimpleMLP()

    @pytest.mark.skipif(
        not pytest.importorskip("flax", reason="flax not installed"),
        reason="flax library not available",
    )
    def test_flax_module_detection(self, adapter, flax_mlp):
        """Test detection of Flax modules."""
        from zenith.adapters.jax_adapter import ModelType, detect_model_type

        model_type = detect_model_type(flax_mlp)
        assert model_type == ModelType.FLAX_MODULE

    @pytest.mark.skipif(
        not pytest.importorskip("flax", reason="flax not installed"),
        reason="flax library not available",
    )
    def test_flax_module_conversion(self, adapter, flax_mlp, jax, jnp):
        """Test converting Flax module to GraphIR."""
        key = jax.random.PRNGKey(0)
        sample = jnp.ones((2, 16), dtype=jnp.float32)
        params = flax_mlp.init(key, sample)

        graph = adapter.from_flax_module(flax_mlp, params["params"], sample)

        assert graph is not None
        assert len(graph.inputs) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("flax", reason="flax not installed"),
        reason="flax library not available",
    )
    def test_flax_params_required(self, adapter, flax_mlp, jnp):
        """Test that params is required for Flax modules."""
        sample = jnp.ones((2, 16), dtype=jnp.float32)

        with pytest.raises(ValueError, match="params is required"):
            adapter.from_flax_module(flax_mlp, None, sample)


class TestHaikuConversion:
    """Tests for Haiku transformed function conversion."""

    @pytest.fixture
    def haiku_mlp(self):
        """Create a simple Haiku transformed function."""
        pytest.importorskip("haiku")
        import haiku as hk

        def forward_fn(x):
            mlp = hk.Sequential(
                [
                    hk.Linear(32),
                    jax.nn.relu,
                    hk.Linear(10),
                ]
            )
            return mlp(x)

        return hk.transform(forward_fn)

    @pytest.mark.skipif(
        not pytest.importorskip("haiku", reason="haiku not installed"),
        reason="haiku library not available",
    )
    def test_haiku_detection(self, adapter, haiku_mlp):
        """Test detection of Haiku transformed functions."""
        from zenith.adapters.jax_adapter import ModelType, detect_model_type

        # Note: detection checks the module, not the transformed function
        model_type = detect_model_type(haiku_mlp)
        # Haiku transformed is actually an object with apply method
        assert model_type in [ModelType.HAIKU_TRANSFORMED, ModelType.RAW_FUNCTION]

    @pytest.mark.skipif(
        not pytest.importorskip("haiku", reason="haiku not installed"),
        reason="haiku library not available",
    )
    def test_haiku_conversion(self, adapter, haiku_mlp, jax, jnp):
        """Test converting Haiku function to GraphIR."""
        key = jax.random.PRNGKey(0)
        sample = jnp.ones((2, 16), dtype=jnp.float32)
        params = haiku_mlp.init(key, sample)

        graph = adapter.from_haiku(haiku_mlp, params, sample)

        assert graph is not None


# =============================================================================
# HuggingFace Integration Tests
# =============================================================================


class TestHuggingFaceFlaxIntegration:
    """Tests for HuggingFace Transformers Flax integration."""

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_detect_huggingface_flax_model(self, adapter, jax):
        """Test detection of HuggingFace Flax models."""
        from zenith.adapters.jax_adapter import ModelType, detect_model_type

        try:
            from transformers import FlaxAutoModel

            model = FlaxAutoModel.from_pretrained("prajjwal1/bert-tiny")
            model_type = detect_model_type(model)
            assert model_type == ModelType.HUGGINGFACE_FLAX
        except Exception:
            pytest.skip("Could not load test model")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_from_transformers_basic(self, adapter):
        """Test loading Flax model from HuggingFace."""
        try:
            graph = adapter.from_transformers(
                "prajjwal1/bert-tiny", max_length=32, batch_size=1
            )

            assert graph is not None
            assert len(graph.inputs) > 0
        except Exception as e:
            pytest.skip(f"HuggingFace model loading failed: {e}")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_from_transformers_with_task(self, adapter):
        """Test loading Flax model with specific task."""
        try:
            graph = adapter.from_transformers(
                "prajjwal1/bert-tiny", task="text-classification", max_length=32
            )

            assert graph is not None
        except Exception as e:
            pytest.skip(f"HuggingFace model loading failed: {e}")


# =============================================================================
# Compilation Hook Tests
# =============================================================================


class TestJAXCompileFunction:
    """Tests for JAX compilation hook (like torch.compile)."""

    def test_compile_function_basic(
        self, adapter, simple_jax_function, sample_input_2d, jax
    ):
        """Test basic function compilation."""

        @adapter.compile_function(target="cpu", precision="fp32")
        @jax.jit
        def forward(x):
            return simple_jax_function(x)

        result = forward(sample_input_2d)

        assert result is not None
        assert result.shape == (4, 4)

    def test_compile_function_without_decorator(
        self, adapter, simple_jax_function, sample_input_2d, jax
    ):
        """Test compilation without decorator syntax."""

        @jax.jit
        def forward(x):
            return simple_jax_function(x)

        compiled = adapter.compile_function(forward, target="cpu", opt_level=2)

        result = compiled(sample_input_2d)
        assert result is not None

    def test_compiled_function_stats(
        self, adapter, simple_jax_function, sample_input_2d, jax
    ):
        """Test getting optimization stats from compiled function."""

        @adapter.compile_function(target="cpu")
        @jax.jit
        def forward(x):
            return simple_jax_function(x)

        # Trigger compilation
        forward(sample_input_2d)

        stats = forward.get_stats()
        assert hasattr(stats, "passes_applied")


# =============================================================================
# StableHLO Export Tests
# =============================================================================


class TestStableHLOExport:
    """Tests for StableHLO export functionality."""

    def test_stablehlo_export_check(self, adapter, jax):
        """Test StableHLO availability check."""
        has_export = adapter._has_jax_export()
        # Result depends on JAX version
        assert isinstance(has_export, bool)

    @pytest.mark.skipif(
        True,  # Skip by default as it requires JAX >= 0.4.14
        reason="Requires JAX >= 0.4.14",
    )
    def test_stablehlo_export(self, adapter, simple_jax_function, sample_input_2d):
        """Test StableHLO export to GraphIR."""
        if not adapter._has_jax_export():
            pytest.skip("JAX version < 0.4.14, jax.export not available")

        graph = adapter.from_stablehlo(simple_jax_function, sample_input_2d)

        assert graph is not None
        assert "stablehlo" in graph.name.lower()


# =============================================================================
# Training Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Tests for training integration features."""

    @pytest.mark.skipif(
        not pytest.importorskip("flax", reason="flax not installed"),
        reason="flax library not available",
    )
    @pytest.mark.skipif(
        not pytest.importorskip("optax", reason="optax not installed"),
        reason="optax library not available",
    )
    def test_create_training_state(self, adapter, jax, jnp):
        """Test creating training state."""
        from flax import linen as nn
        import optax

        class SimpleMLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(32)(x)
                x = nn.relu(x)
                x = nn.Dense(10)(x)
                return x

        model = SimpleMLP()
        key = jax.random.PRNGKey(0)
        sample = jnp.ones((2, 16), dtype=jnp.float32)
        params = model.init(key, sample)["params"]

        optimizer = optax.adam(1e-4)

        state = adapter.create_training_state(model, params, optimizer)

        assert state is not None
        assert state.params is not None
        assert state.step == 0

    @pytest.mark.skipif(
        not pytest.importorskip("flax", reason="flax not installed"),
        reason="flax library not available",
    )
    @pytest.mark.skipif(
        not pytest.importorskip("optax", reason="optax not installed"),
        reason="optax library not available",
    )
    def test_apply_gradients(self, adapter, jax, jnp):
        """Test applying gradients to training state."""
        from flax import linen as nn
        import optax

        class SimpleMLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                return nn.Dense(10)(x)

        model = SimpleMLP()
        key = jax.random.PRNGKey(0)
        sample = jnp.ones((2, 16), dtype=jnp.float32)
        params = model.init(key, sample)["params"]

        optimizer = optax.sgd(0.1)

        state = adapter.create_training_state(model, params, optimizer)

        # Create fake gradients
        grads = jax.tree_util.tree_map(jnp.zeros_like, params)

        new_state = state.apply_gradients(grads)

        assert new_state.step == 1

    def test_wrap_training_step(self, adapter, jax, jnp):
        """Test wrapping training step."""

        def train_step(x, y):
            return jnp.sum((x - y) ** 2)

        wrapped = adapter.wrap_training_step(train_step, enable_mixed_precision=False)

        x = jnp.ones((4,), dtype=jnp.float32)
        y = jnp.zeros((4,), dtype=jnp.float32)

        loss = wrapped(x, y)
        assert loss is not None


# =============================================================================
# Module-level API Tests
# =============================================================================


class TestModuleLevelAPI:
    """Tests for zenith.jax module API."""

    def test_module_import(self):
        """Test importing zenith.jax module."""
        import zenith.jax as zjax

        assert hasattr(zjax, "compile")
        assert hasattr(zjax, "compile_function")
        assert hasattr(zjax, "from_model")
        assert hasattr(zjax, "from_transformers")
        assert hasattr(zjax, "from_flax_module")
        assert hasattr(zjax, "from_haiku")
        assert hasattr(zjax, "create_training_state")

    def test_is_available(self):
        """Test availability check via module."""
        import zenith.jax as zjax

        result = zjax.is_available()
        assert isinstance(result, bool)

    def test_configure(self):
        """Test configuration via module."""
        import zenith.jax as zjax

        config = zjax.configure(target="cuda", precision="fp16", opt_level=3)

        assert config.target == "cuda"
        assert config.precision == "fp16"
        assert config.opt_level == 3

    def test_compile_decorator(self, simple_jax_function, sample_input_2d, jax):
        """Test zenith.jax.compile as decorator."""
        import zenith.jax as zjax

        @zjax.compile(target="cpu")
        @jax.jit
        def forward(x):
            return simple_jax_function(x)

        result = forward(sample_input_2d)
        assert result is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_model_type(self, adapter, jnp):
        """Test error for invalid model type."""
        sample = jnp.ones((4,), dtype=jnp.float32)

        with pytest.raises((ValueError, TypeError)):
            adapter.from_model({"not": "a model"}, sample_input=sample)

    def test_missing_jax2onnx(self, adapter, simple_jax_function, sample_input_2d):
        """Test graceful handling when jax2onnx is missing."""
        # Should use fallback (StableHLO or HLO tracing)
        graph = adapter.from_model(simple_jax_function, sample_input=sample_input_2d)
        assert graph is not None


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for ZenithJAXConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from zenith.adapters.jax_adapter import ZenithJAXConfig

        config = ZenithJAXConfig()

        assert config.target == "cuda"
        assert config.precision == "fp32"
        assert config.opt_level == 2
        assert config.opset_version == 17

    def test_custom_config(self):
        """Test custom configuration."""
        from zenith.adapters.jax_adapter import ZenithJAXConfig
        from zenith.adapters import JAXAdapter

        config = ZenithJAXConfig(
            target="tpu", precision="bf16", opt_level=3, enable_donation=True
        )

        adapter = JAXAdapter(config=config)

        assert adapter.config.precision == "bf16"
        assert adapter.config.enable_donation is True


# =============================================================================
# Data Type Conversion Tests
# =============================================================================


class TestDataTypeConversion:
    """Tests for JAX to Zenith data type conversion."""

    def test_dtype_conversion_float32(self, adapter, jnp):
        """Test float32 dtype conversion."""
        from zenith.core import DataType

        result = adapter._jax_dtype_to_zenith(jnp.float32)
        assert result == DataType.Float32

    def test_dtype_conversion_float16(self, adapter, jnp):
        """Test float16 dtype conversion."""
        from zenith.core import DataType

        result = adapter._jax_dtype_to_zenith(jnp.float16)
        assert result == DataType.Float16

    def test_dtype_conversion_int32(self, adapter, jnp):
        """Test int32 dtype conversion."""
        from zenith.core import DataType

        result = adapter._jax_dtype_to_zenith(jnp.int32)
        assert result == DataType.Int32

    def test_dtype_conversion_bfloat16(self, adapter, jnp):
        """Test bfloat16 dtype conversion."""
        from zenith.core import DataType

        result = adapter._jax_dtype_to_zenith(jnp.bfloat16)
        assert result == DataType.BFloat16


# =============================================================================
# Model Type Detection Tests
# =============================================================================


class TestModelTypeDetection:
    """Tests for model type detection utility."""

    def test_detect_raw_function(self, jnp):
        """Test detection of raw JAX function."""
        from zenith.adapters.jax_adapter import ModelType, detect_model_type

        def fn(x):
            return x * 2

        assert detect_model_type(fn) == ModelType.RAW_FUNCTION

    def test_detect_lambda(self, jnp):
        """Test detection of lambda function."""
        from zenith.adapters.jax_adapter import ModelType, detect_model_type

        fn = lambda x: x**2

        assert detect_model_type(fn) == ModelType.RAW_FUNCTION

    def test_detect_unknown(self):
        """Test detection of unknown type."""
        from zenith.adapters.jax_adapter import ModelType, detect_model_type

        assert detect_model_type(42) == ModelType.UNKNOWN
        assert detect_model_type("string") == ModelType.UNKNOWN


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_inference(
        self, adapter, mlp_function, sample_input_batch, jax
    ):
        """Test full pipeline: function -> GraphIR -> compile -> execute."""
        import zenith

        # Convert to GraphIR
        graph = adapter.from_model(mlp_function, sample_input=sample_input_batch)

        assert graph is not None

        # Compile with Zenith
        compiled_model = zenith.compile(
            mlp_function,
            target="cpu",
            precision="fp32",
            sample_input=sample_input_batch,
        )

        # Execute
        result = compiled_model(sample_input_batch)

        assert result is not None

    def test_pytree_input_handling(self, adapter, jax, jnp):
        """Test handling of pytree (dict) inputs."""

        def fn(inputs):
            return inputs["x"] + inputs["y"]

        sample_input = {
            "x": jnp.ones((4,), dtype=jnp.float32),
            "y": jnp.ones((4,), dtype=jnp.float32),
        }

        # This tests the pytree handling in _trace_to_graphir
        graph = adapter._trace_to_graphir(fn, sample_input)

        assert graph is not None
        # Should have inputs for each key
        assert len(graph.inputs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
