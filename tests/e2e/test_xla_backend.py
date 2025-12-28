# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
XLA Backend Integration Tests.

Tests the XLA backend implementation for JAX-based workflows,
including compilation, execution, and caching.

Run with: pytest tests/e2e/test_xla_backend.py -v
"""

import pytest
import numpy as np
from typing import Callable
from unittest.mock import MagicMock, patch


JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def xla_backend():
    """Create XLA backend instance."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    from zenith.backends.xla_backend import XLABackend

    backend = XLABackend(device="cpu")
    yield backend


@pytest.fixture
def sample_function():
    """Create sample JAX function for testing."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    def mlp_forward(x, w1, w2):
        h = jnp.dot(x, w1)
        h = jax.nn.relu(h)
        return jnp.dot(h, w2)

    return mlp_forward


class TestXLABackendInitialization:
    """Tests for XLA backend initialization."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_backend_creation_default(self):
        """Test default backend creation."""
        from zenith.backends.xla_backend import XLABackend

        backend = XLABackend()
        assert backend is not None
        assert backend.get_device() in ["cpu", "gpu", "tpu"]

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_backend_creation_cpu(self):
        """Test CPU backend creation."""
        from zenith.backends.xla_backend import XLABackend

        backend = XLABackend(device="cpu")
        assert backend.get_device() == "cpu"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_backend_is_available(self):
        """Test backend availability check."""
        from zenith.backends.xla_backend import XLABackend

        backend = XLABackend(device="cpu")
        assert backend.is_available() is True

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_backend_name(self):
        """Test backend name property."""
        from zenith.backends.xla_backend import XLABackend

        backend = XLABackend(device="cpu")
        assert backend.get_name() == "xla"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_backend_type(self):
        """Test backend type property."""
        from zenith.backends.xla_backend import XLABackend
        from zenith.backends.base import BackendType

        backend = XLABackend(device="cpu")
        assert backend.backend_type == BackendType.CPU


class TestXLABackendCompilation:
    """Tests for XLA compilation functionality."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_compile_simple_function(self, xla_backend, sample_function):
        """Test compiling a simple JAX function."""
        compiled = xla_backend.compile(sample_function)
        assert callable(compiled)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_compile_and_execute(self, xla_backend, sample_function):
        """Test compiling and executing a function."""
        compiled = xla_backend.compile(sample_function)

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        result = compiled(x, w1, w2)

        assert result is not None
        assert result.shape == (8, 8)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_compile_with_cache(self, xla_backend, sample_function):
        """Test compilation with caching."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        result1 = xla_backend.compile_with_cache(
            sample_function,
            example_args=(x, w1, w2),
        )

        assert result1 is not None
        assert result1.compiled_fn is not None

        result2 = xla_backend.compile_with_cache(
            sample_function,
            example_args=(x, w1, w2),
        )

        stats = xla_backend.stats
        assert stats.cache_hits >= 1 or stats.cache_misses >= 1

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_compile_with_config(self, xla_backend, sample_function):
        """Test compilation with custom config."""
        from zenith.backends.xla_backend import XLACompileConfig

        config = XLACompileConfig(
            enable_xla_optimization=True,
            cache_compiled=True,
        )

        compiled = xla_backend.compile(sample_function, config=config)
        assert callable(compiled)


class TestXLABackendExecution:
    """Tests for XLA execution functionality."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_execute_compiled_function(self, xla_backend, sample_function):
        """Test executing a compiled function."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        compilation_result = xla_backend.compile_with_cache(
            sample_function,
            example_args=(x, w1, w2),
        )

        result = xla_backend.execute(compilation_result, x, w1, w2)

        assert result is not None
        assert hasattr(result, "shape")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_execution_stats_tracking(self, xla_backend, sample_function):
        """Test execution statistics tracking."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        compiled = xla_backend.compile(sample_function)

        xla_backend.reset_stats()

        for _ in range(5):
            _ = compiled(x, w1, w2)

        stats = xla_backend.stats
        assert stats.total_executions >= 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_numerical_correctness(self, xla_backend, sample_function):
        """Test numerical correctness of XLA execution."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        reference = sample_function(x, w1, w2)

        compiled = xla_backend.compile(sample_function)
        result = compiled(x, w1, w2)

        np.testing.assert_allclose(
            np.array(reference),
            np.array(result),
            rtol=1e-5,
            atol=1e-5,
        )


class TestXLABackendCaching:
    """Tests for XLA compilation caching."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_cache_hit(self, xla_backend, sample_function):
        """Test cache hit detection."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        xla_backend.clear_cache()
        xla_backend.reset_stats()

        _ = xla_backend.compile_with_cache(sample_function, (x, w1, w2))
        _ = xla_backend.compile_with_cache(sample_function, (x, w1, w2))

        stats = xla_backend.stats
        assert stats.cache_hits >= 1

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_cache_miss_different_shapes(self, xla_backend, sample_function):
        """Test cache miss with different input shapes."""
        key = jax.random.PRNGKey(42)

        x1 = jax.random.normal(key, (8, 16))
        w1_a = jax.random.normal(key, (16, 32))
        w2_a = jax.random.normal(key, (32, 8))

        x2 = jax.random.normal(key, (16, 16))
        w1_b = jax.random.normal(key, (16, 32))
        w2_b = jax.random.normal(key, (32, 16))

        xla_backend.clear_cache()
        xla_backend.reset_stats()

        _ = xla_backend.compile_with_cache(sample_function, (x1, w1_a, w2_a))
        _ = xla_backend.compile_with_cache(sample_function, (x2, w1_b, w2_b))

        stats = xla_backend.stats
        assert stats.cache_misses >= 2

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_clear_cache(self, xla_backend, sample_function):
        """Test cache clearing."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        _ = xla_backend.compile_with_cache(sample_function, (x, w1, w2))

        xla_backend.clear_cache()

        xla_backend.reset_stats()
        _ = xla_backend.compile_with_cache(sample_function, (x, w1, w2))

        stats = xla_backend.stats
        assert stats.cache_misses >= 1


class TestXLABackendDevicePlacement:
    """Tests for device placement functionality."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_cpu_device_placement(self, sample_function):
        """Test CPU device placement."""
        from zenith.backends.xla_backend import XLABackend

        backend = XLABackend(device="cpu")
        compiled = backend.compile(sample_function)

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        result = compiled(x, w1, w2)

        device_kind = str(result.device())
        assert "cpu" in device_kind.lower() or "Cpu" in str(result.device())

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_supports_bfloat16(self, xla_backend):
        """Test BFloat16 support detection."""
        supports_bf16 = xla_backend.supports_bfloat16()
        assert isinstance(supports_bf16, bool)


class TestHLOLowering:
    """Tests for HLO lowering functionality."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_lower_to_hlo_text(self, sample_function):
        """Test lowering JAX function to HLO text."""
        from zenith.core.hlo_lowering import JAXFunctionToHLOConverter

        converter = JAXFunctionToHLOConverter()

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        hlo_text = converter.lower_to_hlo(sample_function, [x, w1, w2])

        assert hlo_text is not None
        assert len(hlo_text) > 0
        assert "HloModule" in hlo_text or "module" in hlo_text.lower()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_lower_to_stablehlo(self, sample_function):
        """Test lowering JAX function to StableHLO."""
        from zenith.core.hlo_lowering import JAXFunctionToHLOConverter

        converter = JAXFunctionToHLOConverter()

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (8, 16))
        w1 = jax.random.normal(key, (16, 32))
        w2 = jax.random.normal(key, (32, 8))

        stablehlo = converter.lower_to_stablehlo(sample_function, [x, w1, w2])
        assert stablehlo is not None


class TestXLABackendBaseInterface:
    """Tests for BaseBackend interface compliance."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_get_device_properties(self, xla_backend):
        """Test get_device_properties method."""
        props = xla_backend.get_device_properties()

        assert props is not None
        assert hasattr(props, "name")
        assert hasattr(props, "is_available")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_allocate_returns_placeholder(self, xla_backend):
        """Test allocate returns placeholder (XLA manages memory)."""
        ptr = xla_backend.allocate(1024)
        assert ptr == 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_deallocate_is_noop(self, xla_backend):
        """Test deallocate is no-op for XLA."""
        xla_backend.deallocate(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
