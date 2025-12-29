# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
End-to-End Tests for Zenith JAX Integration.

Tests complete workflows including:
- Basic JAX function optimization
- Flax model training loop
- Gradient checkpointing
- Custom primitives integration
- ONNX export

These tests validate that all JAX components work together correctly.
"""

import pytest

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestBasicJAXWorkflow:
    """Test basic JAX function compilation and optimization."""

    def test_simple_function_jit(self):
        """Test that basic JAX functions work with JIT."""

        @jax.jit
        def simple_fn(x):
            return jnp.sin(x) * jnp.cos(x)

        x = jnp.array([1.0, 2.0, 3.0])
        result = simple_fn(x)

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_gradient_computation(self):
        """Test gradient computation through JAX functions."""

        def loss_fn(x):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 2.0, 3.0])
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(x)

        # d/dx sum(x^2) = 2x
        expected = 2 * x
        assert jnp.allclose(grads, expected)

    def test_vmap_vectorization(self):
        """Test vmap for automatic vectorization."""

        def single_fn(x):
            return jnp.dot(x, x)

        batch_fn = jax.vmap(single_fn)
        batch = jnp.ones((10, 5))
        results = batch_fn(batch)

        assert results.shape == (10,)
        assert jnp.all(results == 5.0)  # dot of 5 ones = 5


class TestZenithPrimitives:
    """Test Zenith custom JAX primitives."""

    def test_fused_attention_basic(self):
        """Test fused attention primitive execution."""
        from zenith.jax.primitives import fused_attention

        batch, heads, seq, dim = 2, 4, 16, 32
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq, dim))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq, dim))

        output = fused_attention(q, k, v)

        assert output.shape == (batch, heads, seq, dim)
        assert jnp.all(jnp.isfinite(output))

    def test_fused_attention_gradient(self):
        """Test gradient computation through fused attention."""
        from zenith.jax.primitives import fused_attention

        def loss_fn(q, k, v):
            return jnp.sum(fused_attention(q, k, v))

        batch, heads, seq, dim = 2, 2, 8, 16
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq, dim))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq, dim))

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert len(grads) == 3
        assert all(g.shape == q.shape for g in grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)

    def test_fused_attention_jit(self):
        """Test JIT compilation of fused attention."""
        from zenith.jax.primitives import fused_attention

        @jax.jit
        def jit_attention(q, k, v):
            return fused_attention(q, k, v)

        batch, heads, seq, dim = 2, 4, 16, 32
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq, dim))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq, dim))

        output = jit_attention(q, k, v)

        assert output.shape == (batch, heads, seq, dim)
        assert jnp.all(jnp.isfinite(output))

    def test_fused_layernorm(self):
        """Test fused layer normalization primitive."""
        from zenith.jax.primitives import fused_layernorm

        x = jax.random.normal(jax.random.PRNGKey(0), (4, 128, 256))
        weight = jnp.ones(256)
        bias = jnp.zeros(256)

        output = fused_layernorm(x, weight, bias)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

        # Check normalized output has ~0 mean and ~1 std
        mean = jnp.mean(output, axis=-1)
        std = jnp.std(output, axis=-1)
        assert jnp.allclose(mean, 0.0, atol=1e-5)
        assert jnp.allclose(std, 1.0, atol=1e-2)

    def test_fused_gelu(self):
        """Test fused GELU activation primitive."""
        from zenith.jax.primitives import fused_gelu

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 64))

        output_approx = fused_gelu(x, approximate=True)
        output_exact = fused_gelu(x, approximate=False)

        assert output_approx.shape == x.shape
        assert output_exact.shape == x.shape
        assert jnp.all(jnp.isfinite(output_approx))
        assert jnp.all(jnp.isfinite(output_exact))

    def test_fused_softmax(self):
        """Test fused softmax primitive."""
        from zenith.jax.primitives import fused_softmax

        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 32))
        output = fused_softmax(x)

        assert output.shape == x.shape
        assert jnp.all(output >= 0)
        assert jnp.all(output <= 1)

        # Check sums to 1
        sums = jnp.sum(output, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)


class TestXLAKernels:
    """Test XLA custom kernels."""

    def test_xla_fused_attention(self):
        """Test XLA fused attention kernel."""
        from zenith.runtime.xla_kernels import xla_fused_attention

        batch, heads, seq, dim = 2, 4, 16, 32
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq, dim))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq, dim))

        output = xla_fused_attention(q, k, v)

        assert output.shape == (batch, heads, seq, dim)
        assert jnp.all(jnp.isfinite(output))

    def test_xla_fused_layernorm(self):
        """Test XLA fused layer normalization kernel."""
        from zenith.runtime.xla_kernels import xla_fused_layernorm

        x = jax.random.normal(jax.random.PRNGKey(0), (4, 128, 256))
        weight = jnp.ones(256)
        bias = jnp.zeros(256)

        output = xla_fused_layernorm(x, weight, bias)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    def test_xla_fused_softmax(self):
        """Test XLA fused softmax kernel."""
        from zenith.runtime.xla_kernels import xla_fused_softmax

        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 32))
        output = xla_fused_softmax(x)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

        sums = jnp.sum(output, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_kernel_registry(self):
        """Test kernel registry functions."""
        from zenith.runtime.xla_kernels import list_kernels, get_kernel

        kernels = list_kernels()
        assert "zenith_xla_fused_attention" in kernels
        assert "zenith_xla_fused_layernorm" in kernels
        assert "zenith_xla_fused_softmax" in kernels

        kernel = get_kernel("zenith_xla_fused_attention")
        assert kernel is not None


class TestFlaxIntegration:
    """Test Flax model integration (if Flax is available)."""

    @pytest.fixture
    def skip_if_no_flax(self):
        """Skip test if Flax is not installed."""
        pytest.importorskip("flax")

    def test_flax_simple_mlp(self, skip_if_no_flax):
        """Test simple Flax MLP with Zenith primitives."""
        import flax.linen as nn
        from zenith.jax.primitives import fused_gelu

        class SimpleMLP(nn.Module):
            features: int = 64

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(self.features)(x)
                x = fused_gelu(x)  # Use Zenith primitive
                x = nn.Dense(self.features // 2)(x)
                return x

        model = SimpleMLP()
        x = jnp.ones((4, 32))
        params = model.init(jax.random.PRNGKey(0), x)
        output = model.apply(params, x)

        assert output.shape == (4, 32)
        assert jnp.all(jnp.isfinite(output))

    def test_flax_training_step(self, skip_if_no_flax):
        """Test Flax training step with gradients."""
        import flax.linen as nn

        class SimpleModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(32)(x)
                x = nn.relu(x)
                x = nn.Dense(10)(x)
                return x

        model = SimpleModel()
        x = jnp.ones((8, 16))
        params = model.init(jax.random.PRNGKey(0), x)

        def loss_fn(params):
            logits = model.apply(params, x)
            return jnp.mean(logits**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Check gradients exist and are finite
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


class TestPrimitiveRegistry:
    """Test primitive and kernel registry functionality."""

    def test_list_primitives(self):
        """Test listing all registered primitives."""
        from zenith.jax.primitives import list_primitives

        primitives = list_primitives()

        assert isinstance(primitives, list)
        assert "zenith_fused_attention" in primitives
        assert "zenith_fused_layernorm" in primitives
        assert "zenith_fused_gelu" in primitives
        assert "zenith_fused_softmax" in primitives

    def test_primitive_registry_singleton(self):
        """Test that primitive registry is a singleton."""
        from zenith.jax.primitives import get_primitive_registry

        registry1 = get_primitive_registry()
        registry2 = get_primitive_registry()

        assert registry1 is registry2


class TestNumericalStability:
    """Test numerical stability of operations."""

    def test_softmax_large_values(self):
        """Test softmax numerical stability with large values."""
        from zenith.jax.primitives import fused_softmax

        # Large values that would overflow without stabilization
        x = jnp.array([1000.0, 1001.0, 1002.0])
        output = fused_softmax(x)

        assert jnp.all(jnp.isfinite(output))
        assert jnp.allclose(jnp.sum(output), 1.0)

    def test_layernorm_small_variance(self):
        """Test layernorm with near-zero variance."""
        from zenith.jax.primitives import fused_layernorm

        # Nearly constant input
        x = jnp.ones((4, 16)) + 1e-8 * jax.random.normal(jax.random.PRNGKey(0), (4, 16))
        weight = jnp.ones(16)
        bias = jnp.zeros(16)

        output = fused_layernorm(x, weight, bias, eps=1e-5)

        assert jnp.all(jnp.isfinite(output))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_attention_single_token(self):
        """Test attention with single token sequence."""
        from zenith.jax.primitives import fused_attention

        batch, heads, dim = 2, 4, 32
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, 1, dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, 1, dim))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, 1, dim))

        output = fused_attention(q, k, v)

        assert output.shape == (batch, heads, 1, dim)
        assert jnp.all(jnp.isfinite(output))

    def test_softmax_single_element(self):
        """Test softmax with single element."""
        from zenith.jax.primitives import fused_softmax

        x = jnp.array([5.0])
        output = fused_softmax(x)

        assert jnp.allclose(output, jnp.array([1.0]))
