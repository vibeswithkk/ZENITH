# Zenith + JAX (Hybrid Mode)

Zenith mendukung integrasi dengan JAX untuk high-performance computing.

## Installation

```bash
pip install pyzenith[jax]
```

## Quick Start

```python
import jax
import jax.numpy as jnp

# JAX function
def mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b

# Initialize params
key = jax.random.PRNGKey(0)
params = [
    (jax.random.normal(key, (784, 256)) * 0.01, jnp.zeros(256)),
    (jax.random.normal(key, (256, 128)) * 0.01, jnp.zeros(128)),
    (jax.random.normal(key, (128, 10)) * 0.01, jnp.zeros(10)),
]

# JIT compile for speed
mlp_jit = jax.jit(mlp)

# Run
x = jax.random.normal(key, (32, 784))
output = mlp_jit(params, x)
print(f"Output shape: {output.shape}")
```

## Flax Integration

```python
from flax import linen as nn
import jax
import jax.numpy as jnp

class MLP(nn.Module):
    hidden_dim: int = 256
    output_dim: int = 10
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# Initialize
model = MLP()
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 784)))

# JIT compile
@jax.jit
def forward(params, x):
    return model.apply(params, x)

# Run
x = jax.random.normal(key, (32, 784))
output = forward(params, x)
```

## Quantization dengan Zenith

```python
import numpy as np
from zenith.optimization.quantization import quantize

# Convert JAX params to numpy
def quantize_jax_params(params):
    quantized_params = {}
    for name, param in params.items():
        if isinstance(param, dict):
            quantized_params[name] = quantize_jax_params(param)
        else:
            arr = np.array(param)
            if arr.ndim >= 2:  # Weight matrix
                q, scale, zp = quantize(arr)
                quantized_params[name] = {'data': q, 'scale': scale, 'zp': zp}
            else:  # Bias
                quantized_params[name] = arr
    return quantized_params

# Quantize
q_params = quantize_jax_params(params['params'])
```

## GPU Acceleration

```python
import jax

# Check GPU
print(jax.devices())  # [gpu(id=0)]

# Move to GPU
x_gpu = jax.device_put(x, jax.devices('gpu')[0])
output = mlp_jit(params, x_gpu)
```

## Benchmark

```python
import time

def benchmark_jax(fn, *args, warmup=10, runs=50):
    # Warmup
    for _ in range(warmup):
        fn(*args).block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args).block_until_ready()
        times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)

t = benchmark_jax(mlp_jit, params, x)
print(f"JAX JIT: {t:.2f} ms")
```

## Best Practices

1. **Always use `jax.jit`** for compiled functions
2. **Use `.block_until_ready()`** for accurate benchmarking
3. **Batch operations** for GPU efficiency
4. **Use `jax.tree_util`** for parameter manipulation

---

[← TensorFlow](09_tensorflow.md) | [Next: Full Performance →](11_full_performance.md)
