# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Integration Tests for Zenith Unified API

Tests for Phase 2: API Unification
- zenith.compile() integration
- ztorch integration with torch.compile
- API consistency across entry points
"""

import pytest
import numpy as np


class TestZenithCompileAPI:
    """Tests for zenith.compile() function."""

    def test_compile_import(self):
        """Test that compile can be imported."""
        from zenith import compile

        assert compile is not None

    def test_compile_function_signature(self):
        """Test compile function has correct signature."""
        from zenith import compile
        import inspect

        sig = inspect.signature(compile)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "target" in params
        assert "precision" in params
        assert "sample_input" in params


class TestZenithRuntimeIntegration:
    """Tests for ZenithEngine integration."""

    def test_engine_from_runtime(self):
        """Test importing engine from runtime module."""
        from zenith.runtime import ZenithEngine, CompileConfig

        engine = ZenithEngine(backend="cuda")
        assert engine.backend == "cuda"

        config = CompileConfig(precision="fp16")
        assert config.precision == "fp16"

    def test_list_operations(self):
        """Test listing supported operations."""
        from zenith.runtime import ZenithEngine

        engine = ZenithEngine()
        ops = engine.list_supported_ops()

        assert isinstance(ops, list)
        assert len(ops) >= 30  # We registered 35 ops


class TestPyTorchIntegration:
    """Tests for PyTorch/ztorch integration."""

    @pytest.fixture
    def torch(self):
        """Get PyTorch if available."""
        try:
            import torch

            return torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_ztorch_import(self):
        """Test importing ztorch module."""
        try:
            import zenith.torch as ztorch

            assert ztorch is not None
            assert hasattr(ztorch, "create_backend")
            assert hasattr(ztorch, "compile")
        except ImportError:
            pytest.skip("ztorch requires PyTorch")

    def test_create_backend(self, torch):
        """Test creating torch.compile backend."""
        import zenith.torch as ztorch

        backend = ztorch.create_backend(target="cpu", precision="fp32")
        assert callable(backend)

    def test_configure(self, torch):
        """Test ztorch configuration."""
        import zenith.torch as ztorch

        config = ztorch.configure(target="cuda", precision="fp16")
        assert config.target == "cuda"
        assert config.precision == "fp16"


class TestTensorFlowIntegration:
    """Tests for TensorFlow/ztf integration."""

    def test_ztf_import(self):
        """Test importing ztf module."""
        try:
            import zenith.tensorflow as ztf

            assert ztf is not None
            assert hasattr(ztf, "compile")
            assert hasattr(ztf, "compile_function")
            assert hasattr(ztf, "from_model")
            assert hasattr(ztf, "is_available")
        except ImportError:
            pytest.skip("ztf requires TensorFlow")

    def test_ztf_configure(self):
        """Test ztf configuration."""
        try:
            import zenith.tensorflow as ztf

            config = ztf.configure(target="cuda", precision="fp16")
            assert config.target == "cuda"
            assert config.precision == "fp16"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_ztf_adapter_exists(self):
        """Test TensorFlow adapter class exists."""
        from zenith.adapters.tensorflow_adapter import TensorFlowAdapter
        from zenith.adapters.tensorflow_adapter import ZenithCompiledFunction

        adapter = TensorFlowAdapter()
        assert adapter.name == "tensorflow"


class TestJAXIntegration:
    """Tests for JAX/zjax integration."""

    def test_zjax_import(self):
        """Test importing zjax module."""
        try:
            import zenith.jax as zjax

            assert zjax is not None
            assert hasattr(zjax, "compile")
            assert hasattr(zjax, "compile_function")
            assert hasattr(zjax, "from_model")
            assert hasattr(zjax, "is_available")
        except ImportError:
            pytest.skip("zjax requires JAX")

    def test_zjax_configure(self):
        """Test zjax configuration."""
        try:
            import zenith.jax as zjax

            config = zjax.configure(target="cuda", precision="fp16")
            assert config.target == "cuda"
            assert config.precision == "fp16"
        except ImportError:
            pytest.skip("JAX not available")

    def test_zjax_adapter_exists(self):
        """Test JAX adapter class exists."""
        from zenith.adapters.jax_adapter import JAXAdapter
        from zenith.adapters.jax_adapter import ZenithCompiledJAXFunction

        adapter = JAXAdapter()
        assert adapter.name == "jax"


class TestAPIConsistency:
    """Tests for API consistency across entry points."""

    def test_main_api_exports(self):
        """Test main zenith module exports."""
        import zenith

        # Core functions
        assert hasattr(zenith, "compile")
        assert hasattr(zenith, "optimize")

        # Core types
        assert hasattr(zenith, "GraphIR")

    def test_runtime_exports(self):
        """Test runtime module exports."""
        from zenith import runtime

        assert hasattr(runtime, "ZenithEngine")
        assert hasattr(runtime, "CompileConfig")
        assert hasattr(runtime, "GraphExecutor")
        assert hasattr(runtime, "KernelDispatcher")
        assert hasattr(runtime, "KernelRegistry")
        assert hasattr(runtime, "ExecutionContext")
        assert hasattr(runtime, "MemoryManager")


class TestCompileConfig:
    """Tests for compilation configuration."""

    def test_config_defaults(self):
        """Test config default values."""
        from zenith.runtime.engine import CompileConfig

        config = CompileConfig()
        assert config.precision == "fp32"
        assert config.mode == "default"
        assert config.verbose == 2

    def test_config_precision_enum(self):
        """Test precision enum conversion."""
        from zenith.runtime.engine import CompileConfig
        from zenith.runtime.kernel_registry import Precision

        config = CompileConfig(precision="fp16")
        assert config.get_precision() == Precision.FP16

        config = CompileConfig(precision="fp32")
        assert config.get_precision() == Precision.FP32


class TestEndToEndSimple:
    """Simple end-to-end tests (without actual GPU)."""

    def test_kernel_registry_integration(self):
        """Test kernel registry works with runtime."""
        from zenith.runtime import ZenithEngine
        from zenith.runtime.kernel_registry import Precision, get_registry

        registry = get_registry()
        engine = ZenithEngine()

        # Both should use same registry
        engine_ops = engine.list_supported_ops()
        registry_ops = registry.list_supported_ops()

        assert set(engine_ops) == set(registry_ops)

    def test_cpu_kernel_execution(self):
        """Test CPU kernel fallback execution."""
        from zenith.runtime.kernel_registry import get_registry, Precision

        registry = get_registry()

        # Get CPU add kernel
        add_kernel = registry.get_kernel("Add", Precision.FP32)
        assert add_kernel is not None

        # Execute it
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = add_kernel.kernel_fn(a, b)
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_cpu_relu_execution(self):
        """Test CPU ReLU kernel execution."""
        from zenith.runtime.kernel_registry import get_registry, Precision

        registry = get_registry()

        relu_kernel = registry.get_kernel("ReLU", Precision.FP32)
        assert relu_kernel is not None

        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = relu_kernel.kernel_fn(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_cpu_softmax_execution(self):
        """Test CPU softmax kernel execution."""
        from zenith.runtime.kernel_registry import get_registry, Precision

        registry = get_registry()

        softmax_kernel = registry.get_kernel("Softmax", Precision.FP32)
        assert softmax_kernel is not None

        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = softmax_kernel.kernel_fn(x)

        # Check output sums to 1
        assert np.abs(result.sum() - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
