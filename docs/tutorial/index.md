# Zenith Tutorial

Tutorial lengkap untuk Zenith ML Framework.

## Installation Options

| Command | Use Case |
|---------|----------|
| `pip install pyzenith` | Core only |
| `pip install pyzenith[pytorch]` | PyTorch users |
| `pip install pyzenith[tensorflow]` | TensorFlow users |
| `pip install pyzenith[jax]` | JAX users |
| `pip install pyzenith[all]` | All frameworks |

## Tutorial Contents

### Getting Started
1. [Getting Started](01_getting_started.md) - Installation and first steps
2. [Basics](02_basics.md) - Core modules and imports

### Core Features
3. [Quantization](03_quantization.md) - INT8 quantization
4. [QAT](04_qat.md) - Quantization-Aware Training

### Framework Integration
5. [Zenith + PyTorch](05_pytorch.md) - Hybrid mode with PyTorch
6. [Triton Deployment](06_triton.md) - Production serving
7. [Auto-Tuner](07_autotuner.md) - Hardware optimization
8. [PyTorch Training](08_pytorch_training.md) - Training workflows

## Quick Reference

### Zenith Core API

```python
import zenith
from zenith import backends
from zenith.core import GraphIR, DataType
from zenith.optimization import PassManager

# Check version
print(zenith.__version__)  # 0.2.1

# Check backends
print(backends.get_available_backends())  # ['cpu'] or ['cpu', 'cuda']
```

### zenith.compile() - Main Entry Point

```python
import zenith
import torch

# Any PyTorch model
model = torch.nn.Linear(1024, 512).cuda()

# Compile with Zenith
optimized = zenith.compile(
    model,
    target="cuda",     # "cpu" or "cuda"
    precision="fp32",  # "fp32", "fp16", or "int8"
    opt_level=2,       # 0=none, 1=basic, 2=full
)

# Use normally
output = optimized(input)
```

### Quantization

```python
from zenith.optimization.quantization import quantize, dequantize

# Quantize weights to INT8
quantized, scale, zero_point = quantize(
    weights,
    num_bits=8,
    method=CalibrationMethod.MINMAX
)

# Size reduction: 4x
```

### FP16 Mode (Tensor Core)

```python
# Best for inference on GPU
model = model.half()  # Convert to FP16
input = input.half()

with torch.no_grad():
    output = model(input)  # Uses Tensor Cores
```

## Interactive Tutorial

Run the complete tutorial in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/notebooks/zenith_complete_tutorial.ipynb)

## Performance Comparison

| Mode | Speed | Memory | Use Case |
|------|-------|--------|----------|
| FP32 | 1x | 4 bytes/param | Training |
| FP16 | 2-6x | 2 bytes/param | GPU Inference |
| INT8 | 2-4x | 1 byte/param | Edge Deployment |

## Next Steps

- [API Reference](../API.md)
- [Architecture](../ARCHITECTURE.md)
- [Benchmark Report](../../BENCHMARK_REPORT.md)
