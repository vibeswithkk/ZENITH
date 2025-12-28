# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
JAX Primitives Unit Tests.

Tests custom JAX primitives for correctness, gradient computation,
JIT compilation, and vmap support.

Run with: pytest tests/python/test_jax_primitives.py -v
"""

import pytest
import numpy as np
from typing import Callable

JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def sample_attention_inputs():
    """Create sample inputs for attention testing."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    key = jax.random.PRNGKey(42)
    batch, heads, seq, dim = 2, 4, 32, 64

    q = jax.random.normal(key, (batch, heads, seq, dim))
    k = jax.random.normal(key, (batch, heads, seq, dim))
    v = jax.random.normal(key, (batch, heads, seq, dim))

    return q, k, v


@pytest.fixture
def sample_layernorm_inputs():
    """Create sample inputs for layernorm testing."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    key = jax.random.PRNGKey(42)
    batch, seq, dim = 2, 32, 512

    x = jax.random.normal(key, (batch, seq, dim))
    weight = jax.random.normal(key, (dim,))
    bias = jax.random.normal(key, (dim,))

    return x, weight, bias


class TestPrimitiveRegistry:
    """Tests for primitive registry functionality."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_registry_singleton(self):
        """Test registry is singleton."""
        from zenith.jax.primitives import get_primitive_registry

        reg1 = get_primitive_registry()
        reg2 = get_primitive_registry()

        assert reg1 is reg2

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_list_primitives(self):
        """Test listing registered primitives."""
        from zenith.jax.primitives import list_primitives

        primitives = list_primitives()

        assert isinstance(primitives, list)
        assert "zenith_fused_attention" in primitives
        assert "zenith_fused_layernorm" in primitives
        assert "zenith_fused_gelu" in primitives
        assert "zenith_fused_softmax" in primitives


class TestFusedAttention:
    """Tests for fused attention primitive."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_attention_basic(self, sample_attention_inputs):
        """Test basic fused attention computation."""
        from zenith.jax.primitives import fused_attention

        q, k, v = sample_attention_inputs
        output = fused_attention(q, k, v)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_attention_numerical_correctness(self, sample_attention_inputs):
        """Test numerical correctness against reference implementation."""
        from zenith.jax.primitives import fused_attention

        q, k, v = sample_attention_inputs

        # Reference implementation
        scale = 1.0 / np.sqrt(q.shape[-1])
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        reference = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Zenith implementation
        output = fused_attention(q, k, v)

        np.testing.assert_allclose(
            np.array(output),
            np.array(reference),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_attention_with_mask(self, sample_attention_inputs):
        """Test attention with causal mask."""
        from zenith.jax.primitives import fused_attention

        q, k, v = sample_attention_inputs
        batch, heads, seq, dim = q.shape

        # Create causal mask
        mask = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        mask = mask[None, None, :, :]  # (1, 1, seq, seq)

        output = fused_attention(q, k, v, mask=mask)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_attention_jit(self, sample_attention_inputs):
        """Test JIT compilation of fused attention."""
        from zenith.jax.primitives import fused_attention

        q, k, v = sample_attention_inputs

        # JIT compile
        jit_attention = jit(fused_attention)

        output = jit_attention(q, k, v)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_attention_gradient(self, sample_attention_inputs):
        """Test gradient computation for fused attention."""
        from zenith.jax.primitives import fused_attention

        q, k, v = sample_attention_inputs

        def loss_fn(q, k, v):
            out = fused_attention(q, k, v)
            return jnp.sum(out)

        # Compute gradients
        grads = grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        dq, dk, dv = grads

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

        # All gradients should be finite
        assert jnp.all(jnp.isfinite(dq))
        assert jnp.all(jnp.isfinite(dk))
        assert jnp.all(jnp.isfinite(dv))


class TestFusedLayerNorm:
    """Tests for fused layer normalization primitive."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_layernorm_basic(self, sample_layernorm_inputs):
        """Test basic layer normalization."""
        from zenith.jax.primitives import fused_layernorm

        x, weight, bias = sample_layernorm_inputs
        output = fused_layernorm(x, weight, bias)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_layernorm_numerical_correctness(self, sample_layernorm_inputs):
        """Test numerical correctness against reference."""
        from zenith.jax.primitives import fused_layernorm

        x, weight, bias = sample_layernorm_inputs
        eps = 1e-5

        # Reference implementation
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        reference = (x - mean) / jnp.sqrt(var + eps) * weight + bias

        # Zenith implementation
        output = fused_layernorm(x, weight, bias, eps=eps)

        np.testing.assert_allclose(
            np.array(output),
            np.array(reference),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_layernorm_jit(self, sample_layernorm_inputs):
        """Test JIT compilation."""
        from zenith.jax.primitives import fused_layernorm

        x, weight, bias = sample_layernorm_inputs

        jit_layernorm = jit(fused_layernorm)
        output = jit_layernorm(x, weight, bias)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_layernorm_gradient(self, sample_layernorm_inputs):
        """Test gradient computation."""
        from zenith.jax.primitives import fused_layernorm

        x, weight, bias = sample_layernorm_inputs

        def loss_fn(x, weight, bias):
            out = fused_layernorm(x, weight, bias)
            return jnp.sum(out)

        grads = grad(loss_fn, argnums=(0, 1, 2))(x, weight, bias)

        dx, dweight, dbias = grads

        assert dx.shape == x.shape
        assert dweight.shape == weight.shape
        assert dbias.shape == bias.shape


class TestFusedGELU:
    """Tests for fused GELU activation primitive."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_gelu_basic(self):
        """Test basic GELU computation."""
        from zenith.jax.primitives import fused_gelu

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 32, 256))
        output = fused_gelu(x)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_gelu_approximate_vs_exact(self):
        """Test approximate vs exact GELU."""
        from zenith.jax.primitives import fused_gelu

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 32))

        approx = fused_gelu(x, approximate=True)
        exact = fused_gelu(x, approximate=False)

        # Should be close but not identical
        assert not jnp.allclose(approx, exact)
        assert jnp.allclose(approx, exact, atol=0.1)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_gelu_gradient(self):
        """Test GELU gradient."""
        from zenith.jax.primitives import fused_gelu

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 32))

        def loss_fn(x):
            return jnp.sum(fused_gelu(x))

        dx = grad(loss_fn)(x)

        assert dx.shape == x.shape
        assert jnp.all(jnp.isfinite(dx))


class TestFusedSoftmax:
    """Tests for fused softmax primitive."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_softmax_basic(self):
        """Test basic softmax computation."""
        from zenith.jax.primitives import fused_softmax

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 16, 32))
        output = fused_softmax(x)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(output >= 0)
        assert jnp.all(output <= 1)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_softmax_sums_to_one(self):
        """Test softmax sums to 1 along axis."""
        from zenith.jax.primitives import fused_softmax

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 16, 32))
        output = fused_softmax(x, axis=-1)

        sums = jnp.sum(output, axis=-1)
        np.testing.assert_allclose(np.array(sums), 1.0, rtol=1e-5)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_softmax_numerical_stability(self):
        """Test numerical stability with large values."""
        from zenith.jax.primitives import fused_softmax

        # Create tensor with large values that would overflow naive softmax
        x = jnp.array([1000.0, 1001.0, 1002.0])
        output = fused_softmax(x)

        assert jnp.all(jnp.isfinite(output))
        assert jnp.sum(output) > 0.99

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_fused_softmax_gradient(self):
        """Test softmax gradient."""
        from zenith.jax.primitives import fused_softmax

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 32))

        def loss_fn(x):
            return jnp.sum(fused_softmax(x))

        dx = grad(loss_fn)(x)

        assert dx.shape == x.shape
        assert jnp.all(jnp.isfinite(dx))


class TestXLAKernels:
    """Tests for XLA custom kernels."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_kernel_registry_singleton(self):
        """Test kernel registry is singleton."""
        from zenith.runtime.xla_kernels import get_kernel_registry

        reg1 = get_kernel_registry()
        reg2 = get_kernel_registry()

        assert reg1 is reg2

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_list_kernels(self):
        """Test listing registered kernels."""
        from zenith.runtime.xla_kernels import list_kernels

        kernels = list_kernels()

        assert isinstance(kernels, list)
        assert len(kernels) >= 3

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_xla_fused_attention(self, sample_attention_inputs):
        """Test XLA fused attention kernel."""
        from zenith.runtime.xla_kernels import xla_fused_attention

        q, k, v = sample_attention_inputs
        output = xla_fused_attention(q, k, v)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_xla_fused_layernorm(self, sample_layernorm_inputs):
        """Test XLA fused layernorm kernel."""
        from zenith.runtime.xla_kernels import xla_fused_layernorm

        x, weight, bias = sample_layernorm_inputs
        output = xla_fused_layernorm(x, weight, bias)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_xla_fused_softmax(self):
        """Test XLA fused softmax kernel."""
        from zenith.runtime.xla_kernels import xla_fused_softmax

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 16, 32))
        output = xla_fused_softmax(x)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
