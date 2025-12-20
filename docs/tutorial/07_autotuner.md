# 7. Autotuner

Auto-tuning untuk optimasi kernel parameters.

## Search Space

Define parameter space to search:

```python
from zenith.optimization.autotuner import SearchSpace

space = SearchSpace("matmul_space")
space.define("block_size", [16, 32, 64, 128])
space.define("num_warps", [2, 4, 8])
```

## Benchmark Function

Create function yang return execution time:

```python
def benchmark(config):
    """Return time in ms - lower is better."""
    # Simulate: bigger block_size = faster
    return 1000 / (config["block_size"] * config["num_warps"])
```

## Grid Search

```python
from zenith.optimization.autotuner import KernelAutotuner, GridSearch

autotuner = KernelAutotuner(strategy=GridSearch())
```

## Tuning Config

```python
from zenith.optimization.autotuner import TuningConfig

config = TuningConfig(
    op_name="matmul",
    input_shapes=[(512, 512)]
)
```

## Run Tuning

```python
best_params, best_time = autotuner.tune(
    config, 
    space, 
    benchmark, 
    max_trials=12
)

print(f"Best Config: {best_params}")
print(f"Best Time: {best_time:.3f} ms")
```

**Output:**
```
Best Config: {'block_size': 128, 'num_warps': 8}
Best Time: 0.000 ms
```

## Complete Example

```python
from zenith.optimization.autotuner import (
    KernelAutotuner, 
    SearchSpace, 
    TuningConfig, 
    GridSearch
)

# Define space
space = SearchSpace("matmul_space")
space.define("block_size", [16, 32, 64])
space.define("num_warps", [2, 4])

# Benchmark
def benchmark(config):
    return 1000 / (config["block_size"] * config["num_warps"])

# Tune
autotuner = KernelAutotuner(strategy=GridSearch())
config = TuningConfig(op_name="matmul", input_shapes=[(512, 512)])
best_params, best_time = autotuner.tune(config, space, benchmark, max_trials=6)

print(f"Result: {best_params}")
```

---

[← Triton](06_triton.md) | [Back to Index →](index.md)
