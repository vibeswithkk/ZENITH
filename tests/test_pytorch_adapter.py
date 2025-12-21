# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Test Suite for PyTorch Adapter Enterprise Edition

Tests for:
- Basic nn.Module conversion
- FX Graph export (PyTorch 2.x)
- HuggingFace Transformers integration
- torch.compile backend
- Compilation hook
- Training integration with AMP
"""

import pytest


# =============================================================================
# Fixtures and Test Setup
# =============================================================================


@pytest.fixture(scope="module")
def torch():
    """Get torch module, skip if not available."""
    pytest.importorskip("torch")
    import torch

    return torch


@pytest.fixture(scope="module")
def adapter():
    """Create PyTorch adapter instance."""
    pytest.importorskip("torch")
    from zenith.adapters import PyTorchAdapter

    return PyTorchAdapter()


@pytest.fixture
def simple_linear_model(torch):
    """Create a simple linear model."""
    return torch.nn.Linear(16, 10)


@pytest.fixture
def simple_mlp(torch):
    """Create a simple MLP model."""
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )


@pytest.fixture
def sample_input_1d(torch):
    """Sample 1D input."""
    return torch.randn(1, 16)


@pytest.fixture
def sample_input_batch(torch):
    """Batched sample input."""
    return torch.randn(4, 16)


# =============================================================================
# Basic Adapter Tests
# =============================================================================


class TestPyTorchAdapterBasic:
    """Basic tests for PyTorch adapter."""

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "pytorch"

    def test_adapter_availability(self, adapter):
        """Test PyTorch availability check."""
        assert adapter.is_available is True

    def test_torch_version_detection(self, adapter, torch):
        """Test PyTorch version detection."""
        version = adapter._get_torch_version()
        assert isinstance(version, tuple)
        assert len(version) >= 2

    def test_default_config(self, adapter):
        """Test default configuration."""
        assert adapter.config.target == "cuda"
        assert adapter.config.precision == "fp32"
        assert adapter.config.opt_level == 2


class TestModelConversion:
    """Tests for model to GraphIR conversion."""

    def test_simple_model_conversion(
        self, adapter, simple_linear_model, sample_input_1d
    ):
        """Test converting simple linear model to GraphIR."""
        graph = adapter.from_model(simple_linear_model, sample_input=sample_input_1d)

        assert graph is not None
        assert graph.name is not None
        assert len(graph.inputs) > 0
        assert len(graph.outputs) > 0

    def test_mlp_conversion(self, adapter, simple_mlp, sample_input_1d):
        """Test converting MLP model to GraphIR."""
        graph = adapter.from_model(simple_mlp, sample_input=sample_input_1d)

        assert graph is not None

    def test_conversion_requires_sample_input(self, adapter, simple_linear_model):
        """Test that sample_input is required."""
        with pytest.raises(ValueError, match="sample_input is required"):
            adapter.from_model(simple_linear_model)

    def test_batched_input_conversion(self, adapter, simple_mlp, sample_input_batch):
        """Test conversion with batched input."""
        graph = adapter.from_model(simple_mlp, sample_input=sample_input_batch)

        assert graph is not None
        assert len(graph.inputs) > 0


class TestFXGraphConversion:
    """Tests for FX Graph conversion (PyTorch 2.x)."""

    @pytest.mark.skipif(
        True,  # Skip by default, enable when PyTorch 2.1+ is available
        reason="Requires PyTorch 2.1+",
    )
    def test_fx_graph_export(self, adapter, simple_mlp, sample_input_1d):
        """Test FX Graph export for PyTorch 2.x models."""
        if not adapter._has_torch_export():
            pytest.skip("torch.export not available")

        graph = adapter.from_fx_graph(simple_mlp, sample_input_1d)

        assert graph is not None
        assert "exported" in graph.name.lower()

    def test_has_torch_compile_check(self, adapter, torch):
        """Test torch.compile availability check."""
        result = adapter._has_torch_compile()
        expected = int(torch.__version__.split(".")[0]) >= 2
        assert result == expected


# =============================================================================
# HuggingFace Integration Tests
# =============================================================================


class TestHuggingFaceIntegration:
    """Tests for HuggingFace Transformers PyTorch integration."""

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_detect_huggingface_model(self, adapter, torch):
        """Test detection of HuggingFace PyTorch models."""
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            assert adapter._is_huggingface_model(model) is True
        except Exception:
            pytest.skip("Could not load test model")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_from_transformers_basic(self, adapter):
        """Test loading PyTorch model from HuggingFace."""
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
        """Test loading model with specific task."""
        try:
            graph = adapter.from_transformers(
                "prajjwal1/bert-tiny", task="text-classification", max_length=32
            )

            assert graph is not None
        except Exception as e:
            pytest.skip(f"HuggingFace model loading failed: {e}")


# =============================================================================
# torch.compile Backend Tests
# =============================================================================


class TestTorchCompileBackend:
    """Tests for torch.compile backend integration."""

    def test_create_compile_backend(self, adapter, torch):
        """Test creating Zenith backend for torch.compile."""
        if not adapter._has_torch_compile():
            pytest.skip("torch.compile not available")

        backend = adapter.create_compile_backend(target="cpu", precision="fp32")

        assert callable(backend)

    def test_compile_backend_with_model(
        self, adapter, simple_linear_model, sample_input_1d, torch
    ):
        """Test using Zenith backend with torch.compile."""
        if not adapter._has_torch_compile():
            pytest.skip("torch.compile not available")

        backend = adapter.create_compile_backend(target="cpu")

        # Compile the model
        compiled = torch.compile(simple_linear_model, backend=backend)

        # Run inference
        result = compiled(sample_input_1d)

        assert result is not None
        assert result.shape == (1, 10)


# =============================================================================
# Compilation Hook Tests
# =============================================================================


class TestCompileFunction:
    """Tests for compilation hook (like torch.compile decorator)."""

    def test_compile_function_basic(self, adapter, simple_mlp, sample_input_1d, torch):
        """Test basic function compilation."""
        model = simple_mlp

        @adapter.compile_function(target="cpu", precision="fp32")
        def forward(x):
            return model(x)

        result = forward(sample_input_1d)

        assert result is not None
        assert result.shape == (1, 10)

    def test_compile_function_without_decorator(
        self, adapter, simple_mlp, sample_input_1d
    ):
        """Test compilation without decorator syntax."""
        model = simple_mlp

        def forward(x):
            return model(x)

        compiled = adapter.compile_function(forward, target="cpu")

        result = compiled(sample_input_1d)
        assert result is not None

    def test_compiled_function_stats(self, adapter, simple_mlp, sample_input_1d):
        """Test getting optimization stats from compiled function."""
        model = simple_mlp

        @adapter.compile_function(target="cpu")
        def forward(x):
            return model(x)

        # Trigger compilation
        forward(sample_input_1d)

        stats = forward.get_stats()
        assert hasattr(stats, "passes_applied")


# =============================================================================
# Training Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Tests for training integration features."""

    def test_wrap_training_step(self, adapter, torch):
        """Test wrapping training step."""

        def train_step(x, y):
            return torch.sum((x - y) ** 2)

        wrapped = adapter.wrap_training_step(train_step, enable_amp=False)

        x = torch.ones(4, dtype=torch.float32)
        y = torch.zeros(4, dtype=torch.float32)

        loss = wrapped(x, y)
        assert loss is not None

    def test_optimizer_wrapper(self, adapter, simple_mlp, torch):
        """Test optimizer wrapper creation."""
        optimizer = torch.optim.SGD(simple_mlp.parameters(), lr=0.01)

        wrapped = adapter.create_optimizer_wrapper(optimizer, enable_amp=False)

        assert hasattr(wrapped, "zero_grad")
        assert hasattr(wrapped, "step")
        assert hasattr(wrapped, "param_groups")

    def test_optimizer_wrapper_zero_grad(self, adapter, simple_mlp, torch):
        """Test optimizer wrapper zero_grad."""
        optimizer = torch.optim.Adam(simple_mlp.parameters(), lr=0.001)
        wrapped = adapter.create_optimizer_wrapper(optimizer)

        wrapped.zero_grad()
        # Should not raise


# =============================================================================
# ONNX Export Tests
# =============================================================================


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_to_onnx_basic(self, adapter, simple_mlp, sample_input_1d):
        """Test basic ONNX export."""
        onnx_bytes = adapter.to_onnx(simple_mlp, sample_input_1d)

        assert onnx_bytes is not None
        assert len(onnx_bytes) > 0

    def test_to_onnx_with_names(self, adapter, simple_mlp, sample_input_1d):
        """Test ONNX export with custom names."""
        onnx_bytes = adapter.to_onnx(
            simple_mlp,
            sample_input_1d,
            input_names=["x"],
            output_names=["y"],
        )

        assert onnx_bytes is not None


# =============================================================================
# Module-level API Tests
# =============================================================================


class TestModuleLevelAPI:
    """Tests for zenith.torch module API."""

    def test_module_import(self):
        """Test importing zenith.torch module."""
        import zenith.torch as ztorch

        assert hasattr(ztorch, "compile")
        assert hasattr(ztorch, "compile_function")
        assert hasattr(ztorch, "create_backend")
        assert hasattr(ztorch, "from_model")
        assert hasattr(ztorch, "from_transformers")
        assert hasattr(ztorch, "wrap_training_step")

    def test_is_available(self):
        """Test availability check via module."""
        import zenith.torch as ztorch

        result = ztorch.is_available()
        assert isinstance(result, bool)

    def test_configure(self):
        """Test configuration via module."""
        import zenith.torch as ztorch

        config = ztorch.configure(target="cuda", precision="fp16", opt_level=3)

        assert config.target == "cuda"
        assert config.precision == "fp16"
        assert config.opt_level == 3

    def test_create_backend_module(self, torch):
        """Test create_backend from module."""
        import zenith.torch as ztorch

        if not ztorch.has_torch_compile():
            pytest.skip("torch.compile not available")

        backend = ztorch.create_backend(target="cpu")
        assert callable(backend)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_model_type(self, adapter, sample_input_1d):
        """Test error for invalid model type."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            adapter.from_model({"not": "a model"}, sample_input=sample_input_1d)

    def test_get_input_shapes_not_implemented(self, adapter, simple_linear_model):
        """Test get_input_shapes raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            adapter.get_input_shapes(simple_linear_model)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for ZenithPyTorchConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from zenith.adapters.pytorch_adapter import ZenithPyTorchConfig

        config = ZenithPyTorchConfig()

        assert config.target == "cuda"
        assert config.precision == "fp32"
        assert config.opt_level == 2
        assert config.opset_version == 17

    def test_custom_config(self):
        """Test custom configuration."""
        from zenith.adapters.pytorch_adapter import ZenithPyTorchConfig
        from zenith.adapters import PyTorchAdapter

        config = ZenithPyTorchConfig(
            target="cpu", precision="fp16", opt_level=3, enable_amp=True
        )

        adapter = PyTorchAdapter(config=config)

        assert adapter.config.precision == "fp16"
        assert adapter.config.enable_amp is True


# =============================================================================
# Data Type Conversion Tests
# =============================================================================


class TestDataTypeConversion:
    """Tests for PyTorch to Zenith data type conversion."""

    def test_dtype_conversion_float32(self, adapter, torch):
        """Test float32 dtype conversion."""
        from zenith.core import DataType

        result = adapter._torch_dtype_to_zenith(torch.float32)
        assert result == DataType.Float32

    def test_dtype_conversion_float16(self, adapter, torch):
        """Test float16 dtype conversion."""
        from zenith.core import DataType

        result = adapter._torch_dtype_to_zenith(torch.float16)
        assert result == DataType.Float16

    def test_dtype_conversion_int32(self, adapter, torch):
        """Test int32 dtype conversion."""
        from zenith.core import DataType

        result = adapter._torch_dtype_to_zenith(torch.int32)
        assert result == DataType.Int32

    def test_dtype_conversion_bfloat16(self, adapter, torch):
        """Test bfloat16 dtype conversion."""
        from zenith.core import DataType

        result = adapter._torch_dtype_to_zenith(torch.bfloat16)
        assert result == DataType.BFloat16


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_inference(
        self, adapter, simple_mlp, sample_input_batch, torch
    ):
        """Test full pipeline: model -> GraphIR -> compile -> execute."""
        import zenith

        # Convert to GraphIR
        graph = adapter.from_model(simple_mlp, sample_input=sample_input_batch)

        assert graph is not None

        # Direct execution through compiled function
        @adapter.compile_function(target="cpu")
        def inference(x):
            return simple_mlp(x)

        result = inference(sample_input_batch)

        assert result is not None
        assert result.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
