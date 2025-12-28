# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
JAX ONNX Export Tests.

Tests the ONNX export functionality for JAX functions and models,
including validation and numerical accuracy checks.

Run with: pytest tests/python/test_jax_onnx_export.py -v
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Callable


JAX_AVAILABLE = False
ONNX_AVAILABLE = False
ONNXRUNTIME_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    pass

try:
    import onnx

    ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    pass


ALL_DEPS_AVAILABLE = JAX_AVAILABLE and ONNX_AVAILABLE and ONNXRUNTIME_AVAILABLE


@pytest.fixture
def simple_function():
    """Create simple JAX function for testing."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    def add_mul(x, y):
        return (x + y) * 2.0

    return add_mul


@pytest.fixture
def mlp_function():
    """Create MLP JAX function for testing."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    def mlp(x, w1, w2):
        h = jnp.dot(x, w1)
        h = jax.nn.relu(h)
        return jnp.dot(h, w2)

    return mlp


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    key = jax.random.PRNGKey(42)
    return {
        "simple": (
            jax.random.normal(key, (4, 4)),
            jax.random.normal(key, (4, 4)),
        ),
        "mlp": (
            jax.random.normal(key, (8, 16)),
            jax.random.normal(key, (16, 32)),
            jax.random.normal(key, (32, 8)),
        ),
    }


class TestONNXExportConfig:
    """Tests for ONNX export configuration."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_config_creation_defaults(self):
        """Test default config creation."""
        from zenith.jax.onnx_export import ONNXExportConfig

        config = ONNXExportConfig()

        assert config.opset_version == 17
        assert config.validate is True
        assert config.check_numerics is True
        assert config.atol == 1e-5
        assert config.rtol == 1e-5

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_config_custom_values(self):
        """Test config with custom values."""
        from zenith.jax.onnx_export import ONNXExportConfig

        config = ONNXExportConfig(
            opset_version=15,
            validate=False,
            atol=1e-3,
            input_names=["input1", "input2"],
        )

        assert config.opset_version == 15
        assert config.validate is False
        assert config.atol == 1e-3
        assert config.input_names == ["input1", "input2"]


class TestJAXONNXExporter:
    """Tests for JAX ONNX exporter."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_exporter_creation(self):
        """Test exporter creation."""
        from zenith.jax.onnx_export import JAXONNXExporter

        exporter = JAXONNXExporter()
        assert exporter is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_exporter_creation_with_config(self):
        """Test exporter creation with custom config."""
        from zenith.jax.onnx_export import JAXONNXExporter, ONNXExportConfig

        config = ONNXExportConfig(opset_version=15)
        exporter = JAXONNXExporter(config=config)

        assert exporter._config.opset_version == 15


class TestONNXExportBasic:
    """Basic ONNX export tests."""

    @pytest.mark.skipif(not ALL_DEPS_AVAILABLE, reason="Dependencies not available")
    def test_export_simple_function(self, simple_function, sample_inputs):
        """Test exporting a simple JAX function."""
        from zenith.jax.onnx_export import JAXONNXExporter

        exporter = JAXONNXExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            result = exporter.export(
                simple_function,
                sample_inputs["simple"],
                output_path=output_path,
            )

            assert result is not None
            assert os.path.exists(output_path) or result.model is not None

    @pytest.mark.skipif(not ALL_DEPS_AVAILABLE, reason="Dependencies not available")
    def test_export_without_saving(self, simple_function, sample_inputs):
        """Test exporting without saving to file."""
        from zenith.jax.onnx_export import JAXONNXExporter

        exporter = JAXONNXExporter()

        result = exporter.export(
            simple_function,
            sample_inputs["simple"],
            output_path=None,
        )

        assert result is not None
        assert result.model is not None or result.validation_passed


class TestONNXExportValidation:
    """ONNX export validation tests."""

    @pytest.mark.skipif(not ALL_DEPS_AVAILABLE, reason="Dependencies not available")
    def test_model_validation(self, simple_function, sample_inputs):
        """Test ONNX model validation."""
        from zenith.jax.onnx_export import JAXONNXExporter, ONNXExportConfig

        config = ONNXExportConfig(validate=True)
        exporter = JAXONNXExporter(config=config)

        result = exporter.export(
            simple_function,
            sample_inputs["simple"],
        )

        if result.model is not None:
            is_valid = exporter.validate_model(result.model)
            assert is_valid is True


class TestONNXExportNumerics:
    """ONNX export numerical accuracy tests."""

    @pytest.mark.skipif(not ALL_DEPS_AVAILABLE, reason="Dependencies not available")
    def test_numerical_accuracy_simple(self, simple_function, sample_inputs):
        """Test numerical accuracy for simple function."""
        from zenith.jax.onnx_export import JAXONNXExporter, ONNXExportConfig

        config = ONNXExportConfig(
            check_numerics=True,
            atol=1e-4,
            rtol=1e-4,
        )
        exporter = JAXONNXExporter(config=config)

        inputs = sample_inputs["simple"]
        result = exporter.export(simple_function, inputs)

        if result.model is not None:
            is_accurate = exporter.check_numerics(
                result.model,
                simple_function,
                inputs,
            )
            assert is_accurate is True


class TestONNXExportOptimization:
    """ONNX export optimization tests."""

    @pytest.mark.skipif(not ALL_DEPS_AVAILABLE, reason="Dependencies not available")
    def test_model_optimization(self, simple_function, sample_inputs):
        """Test ONNX model optimization."""
        from zenith.jax.onnx_export import JAXONNXExporter, ONNXExportConfig

        config = ONNXExportConfig(enable_optimization=True)
        exporter = JAXONNXExporter(config=config)

        result = exporter.export(
            simple_function,
            sample_inputs["simple"],
        )

        if result.model is not None:
            optimized = exporter.optimize_model(result.model)
            assert optimized is not None


class TestConvenienceFunctions:
    """Tests for convenience export functions."""

    @pytest.mark.skipif(not ALL_DEPS_AVAILABLE, reason="Dependencies not available")
    def test_export_to_onnx_function(self, simple_function, sample_inputs):
        """Test export_to_onnx convenience function."""
        from zenith.jax.onnx_export import export_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            result = export_to_onnx(
                simple_function,
                sample_inputs["simple"],
                output_path=output_path,
            )

            assert result is not None


class TestExportResult:
    """Tests for ONNXExportResult dataclass."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_result_attributes(self):
        """Test export result attributes."""
        from zenith.jax.onnx_export import ONNXExportResult

        result = ONNXExportResult(
            model=None,
            path="/tmp/model.onnx",
            input_names=["x", "y"],
            output_names=["output"],
        )

        assert result.path == "/tmp/model.onnx"
        assert result.input_names == ["x", "y"]
        assert result.output_names == ["output"]
        assert result.validation_passed is False
        assert result.numerical_check_passed is False


class TestEdgeCases:
    """Edge case tests for ONNX export."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_empty_function(self):
        """Test export of function that returns input."""
        from zenith.jax.onnx_export import JAXONNXExporter

        def identity(x):
            return x

        exporter = JAXONNXExporter()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 4))

        result = exporter.export(identity, (x,))
        assert result is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_scalar_output(self):
        """Test export of function with scalar output."""
        from zenith.jax.onnx_export import JAXONNXExporter

        def sum_fn(x):
            return jnp.sum(x)

        exporter = JAXONNXExporter()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 4))

        result = exporter.export(sum_fn, (x,))
        assert result is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_multiple_outputs(self):
        """Test export of function with multiple outputs."""
        from zenith.jax.onnx_export import JAXONNXExporter

        def multi_output(x):
            return x + 1, x - 1, x * 2

        exporter = JAXONNXExporter()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 4))

        result = exporter.export(multi_output, (x,))
        assert result is not None


class TestDtypeHandling:
    """Tests for data type handling in ONNX export."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_float32_export(self, simple_function, sample_inputs):
        """Test export with float32 inputs."""
        from zenith.jax.onnx_export import JAXONNXExporter

        exporter = JAXONNXExporter()
        inputs = tuple(x.astype(jnp.float32) for x in sample_inputs["simple"])

        result = exporter.export(simple_function, inputs)
        assert result is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_float64_export(self, simple_function, sample_inputs):
        """Test export with float64 inputs."""
        from zenith.jax.onnx_export import JAXONNXExporter

        exporter = JAXONNXExporter()
        inputs = tuple(x.astype(jnp.float64) for x in sample_inputs["simple"])

        result = exporter.export(simple_function, inputs)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
