# Quick Start Guide

Get started with Zenith in 5 minutes.

## Basic Usage

### 1. Import Zenith

```python
import zenith
import torch
```

### 2. Define Your Model

```python
model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.ReLU(),
    torch.nn.Linear(3072, 768),
)
```

### 3. Compile with Zenith

```python
optimized = zenith.compile(
    model,
    target="cuda",    # or "cpu"
    precision="fp16", # or "fp32", "int8"
)
```

### 4. Run Inference

```python
input_tensor = torch.randn(1, 768).cuda()
output = optimized(input_tensor)
```

## Complete Example

```python
import zenith
import torch
import time

# Define model
model = torch.nn.Linear(1024, 1024).cuda()
x = torch.randn(32, 1024).cuda()

# Baseline timing
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    y = model(x)
torch.cuda.synchronize()
baseline = (time.perf_counter() - start) * 1000

# Compile with Zenith
optimized = zenith.compile(model, target="cuda", precision="fp16")

# Optimized timing
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    y = optimized(x)
torch.cuda.synchronize()
optimized_time = (time.perf_counter() - start) * 1000

print(f"Baseline: {baseline:.2f}ms")
print(f"Optimized: {optimized_time:.2f}ms")
print(f"Speedup: {baseline/optimized_time:.2f}x")
```

## Verbosity Control

Control logging output:

```python
# Silent mode
zenith.set_verbosity(0)

# Debug mode
zenith.set_verbosity(4)
```

## Next Steps

- [PyTorch Integration](03_pytorch_integration.md)
- [Optimization Options](04_optimization_options.md)
