# JAX Mixed Precision Tutorial

This tutorial shows how to use Zenith's mixed precision support with JAX for faster training.

## Overview

Mixed precision uses FP16/BF16 for computation while keeping FP32 for critical operations, reducing memory and increasing throughput.

## Basic Usage

```python
import jax
import jax.numpy as jnp
from zenith.jax import mixed_precision_policy, cast_to_half

# Set global precision policy
mixed_precision_policy("mixed_float16")

# Define model with automatic casting
def model(params, x):
    # Computation happens in FP16
    h = cast_to_half(x) @ cast_to_half(params['w1'])
    h = jax.nn.gelu(h)
    h = h @ cast_to_half(params['w2'])
    # Output stays in FP16
    return h

# Use with jax.jit
@jax.jit
def forward(params, x):
    return model(params, x)
```

## Precision Policies

| Policy | Compute | Accumulate | Gradients |
|--------|---------|------------|-----------|
| `float32` | FP32 | FP32 | FP32 |
| `mixed_float16` | FP16 | FP32 | FP16 |
| `mixed_bfloat16` | BF16 | FP32 | BF16 |

## Using with Flax

```python
import flax.linen as nn
from zenith.jax import apply_mixed_precision

class TransformerBlock(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x):
        # Apply mixed precision to attention
        attn = nn.MultiHeadDotProductAttention(num_heads=8)
        x = apply_mixed_precision(attn, dtype=jnp.float16)(x)
        return x
```

## Performance Tips

1. Use BF16 on TPUs, FP16 on GPUs
2. Keep loss computation in FP32 for stability
3. Use gradient scaling for FP16 training

## Speedup Results

| Hardware | FP32 | Mixed Precision | Speedup |
|----------|------|-----------------|---------|
| T4 GPU   | 100% | 180%            | 1.8x    |
| A100 GPU | 100% | 250%            | 2.5x    |
| TPU v4   | 100% | 200%            | 2.0x    |
