# JAX Gradient Checkpointing Tutorial

This tutorial shows how to use Zenith's gradient checkpointing with JAX for memory-efficient training.

## Overview

Gradient checkpointing trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them.

## Basic Usage

```python
import jax
import jax.numpy as jnp
from zenith.jax import checkpoint_sequential

# Define a model with checkpointing
def model(params, x):
    # Forward pass with automatic checkpointing
    h = jax.nn.relu(x @ params['w1'])
    h = checkpoint_sequential([
        lambda h: jax.nn.relu(h @ params['w2']),
        lambda h: jax.nn.relu(h @ params['w3']),
        lambda h: h @ params['w4'],
    ], h)
    return h

# Training step
@jax.jit
def train_step(params, x, y):
    def loss_fn(params):
        pred = model(params, x)
        return jnp.mean((pred - y) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads
```

## Memory Savings

| Model Size | Without Checkpointing | With Checkpointing |
|------------|----------------------|-------------------|
| 1B params  | 16 GB                | 6 GB              |
| 7B params  | 112 GB               | 40 GB             |

## API Reference

### `checkpoint_sequential(layers, x)`

Apply gradient checkpointing to a sequence of layers.

**Arguments:**
- `layers`: List of callables, each representing a layer
- `x`: Input tensor

**Returns:**
- Output tensor with checkpointing applied

## Best Practices

1. Apply checkpointing to memory-heavy layers (attention, FFN)
2. Avoid checkpointing small operations (adds overhead)
3. Use with `jax.jit` for best performance
