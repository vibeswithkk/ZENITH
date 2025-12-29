# Zenith JAX Integration Guide

Zenith provides comprehensive JAX support including custom primitives, XLA kernels, gradient checkpointing, mixed precision training, and ONNX export.

## Installation

```bash
pip install pyzenith jax jaxlib
```

For GPU support:
```bash
pip install pyzenith "jax[cuda12]"
```

## Quick Start

### Using Zenith Primitives

```python
import jax
import jax.numpy as jnp
from zenith.jax.primitives import fused_attention, fused_layernorm, fused_gelu

# Fused multi-head attention
batch, heads, seq, dim = 2, 8, 512, 64
q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq, dim))
k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq, dim))
v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq, dim))

# Fully JIT-compatible and differentiable
output = fused_attention(q, k, v)

# Works with jax.grad
def loss_fn(q, k, v):
    return jnp.sum(fused_attention(q, k, v))

grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
```

### XLA Custom Kernels

```python
from zenith.runtime.xla_kernels import (
    xla_fused_attention,
    xla_fused_layernorm,
    xla_fused_softmax,
)

# Memory-efficient attention with tiled computation
output = xla_fused_attention(q, k, v)

# Fused layer normalization
x = jax.random.normal(jax.random.PRNGKey(0), (4, 128, 512))
weight = jnp.ones(512)
bias = jnp.zeros(512)
normalized = xla_fused_layernorm(x, weight, bias)
```

### With Flax Models

```python
import flax.linen as nn
from zenith.jax.primitives import fused_gelu, fused_layernorm

class TransformerBlock(nn.Module):
    dim: int = 512
    heads: int = 8

    @nn.compact
    def __call__(self, x):
        # Fused attention (use primitives for custom ops)
        q = nn.Dense(self.dim)(x)
        k = nn.Dense(self.dim)(x)
        v = nn.Dense(self.dim)(x)
        
        # Reshape for multi-head attention
        B, L, D = x.shape
        q = q.reshape(B, L, self.heads, D // self.heads).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.heads, D // self.heads).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.heads, D // self.heads).transpose(0, 2, 1, 3)
        
        attn_out = fused_attention(q, k, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)
        
        # Fused layer norm
        x = x + attn_out
        weight = self.param('ln_weight', nn.initializers.ones, (D,))
        bias = self.param('ln_bias', nn.initializers.zeros, (D,))
        x = fused_layernorm(x, weight, bias)
        
        # FFN with fused GELU
        x = x + fused_gelu(nn.Dense(self.dim * 4)(x))
        return nn.Dense(self.dim)(x)
```

## Available Primitives

| Primitive | Description | JIT | Grad |
|-----------|-------------|-----|------|
| `fused_attention` | Scaled dot-product attention | Yes | Yes |
| `fused_layernorm` | Layer normalization | Yes | Yes |
| `fused_gelu` | GELU activation | Yes | Yes |
| `fused_softmax` | Numerically stable softmax | Yes | Yes |

## ONNX Export

```python
from zenith.jax.onnx_export import export_to_onnx

def model_fn(x):
    return jax.nn.relu(x) * 2

onnx_model = export_to_onnx(
    model_fn,
    input_shapes={"x": (1, 256)},
    output_path="model.onnx"
)
```

## Performance Tips

1. **Use JIT compilation** for all performance-critical code
2. **Batch operations** to maximize GPU utilization
3. **Use fused primitives** to reduce memory bandwidth
4. **Enable mixed precision** for faster training on modern GPUs

## Compatibility

- JAX 0.4.x and 0.6.x+ (automatic API detection)
- CPU, GPU (CUDA), and TPU
- Flax, Haiku, and pure JAX models
