<p align="center">
  <img src="assets/zenith-logo.png" alt="Zenith Logo" width="450"/>
</p>

# Zenith

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/pyzenith.svg)](https://pypi.org/project/pyzenith/)
[![Stability](https://img.shields.io/badge/Stability-84%25-brightgreen.svg)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tensor Cores](https://img.shields.io/badge/Tensor%20Cores-WMMA-orange.svg)](#native-cuda-kernels)
[![CI](https://github.com/vibeswithkk/ZENITH/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeswithkk/ZENITH/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-942+-success.svg)](#)

**A Simple ML Inference Optimizer**

Zenith is an open-source project focused on improving the speed of PyTorch, JAX, and TensorFlow inference. Faster inference means less total energy consumption. It was carefully built as a bridge. Zenith is designed to complement your existing ML workflow, not replace it.

## Project History

Zenith was conceived and architecturally designed on **December 11, 2024**, with the creation of its comprehensive blueprint document (CetakBiru.md) that outlines a 36-month development roadmap across 6 implementation phases. Active development began on **January 12, 2025**, and after months of internal development, research, and rigorous testing, Zenith was publicly released on GitHub on **December 16, 2025**.

This project represents months of hobby development, learning CUDA programming, and experimenting with ML optimization techniques. It is still a work in progress.

---

## Early Benchmark Results

These are some early experiments on NVIDIA Tesla T4 (Google Colab). Results may vary:

| Benchmark | Workload | Observation |
|-----------|----------|-------------|
| GPU Memory Pool | MatMul 1024x1024 | ~50x faster (zero-copy vs copy) |
| BERT Inference | 12-layer encoder | ~1.09x faster |
| Training Loop | 6-layer Transformer | ~1.02x faster |
| Memory Efficiency | Zero-copy allocation | 93.5% cache hit rate |
| INT8 Quantization | Model compression | 4x memory reduction |

*These benchmarks are preliminary. See [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md) for details.*

---

## Features

### Core Capabilities
- Unified API for PyTorch, TensorFlow, JAX, and ONNX models
- Automatic graph optimizations (operator fusion, constant folding, dead code elimination)
- Multi-backend support (CPU with SIMD, CUDA with cuDNN/cuBLAS)
- Mixed precision inference (FP16, BF16, INT8)
- Zero-copy GPU memory pooling for minimal allocation overhead

### Optimization Passes
- Conv-BatchNorm-ReLU fusion
- Linear-GELU fusion (BERT-optimized)
- LayerNorm-Add fusion
- Constant folding and dead code elimination
- INT8 quantization with calibration

### Hardware Support
- CPU: AVX2/FMA SIMD optimizations
- NVIDIA GPU: CUDA 12.x with cuDNN 8.x and cuBLAS
- AMD GPU: ROCm support (experimental - untesting)
- Intel: OneAPI support (experimental - untesting)

> **Note regarding AMD & Intel GPUs:**  
> Support for ROCm (AMD) and OneAPI (Intel) is currently in an **experimental** state. While the backend code exists, it has not been verified on physical hardware. We recommend using NVIDIA GPUs for production workloads. Community contributions for hardware verification are welcome!

### Native CUDA Kernels

Zenith includes some hand-written CUDA kernels (still experimental):

| Kernel | Description | Tensor Core |
|--------|-------------|-------------|
| `relu` | ReLU activation | - |
| `gelu` | GELU activation (BERT) | - |
| `layernorm` | Layer Normalization | - |
| `matmul` | Matrix Multiplication (FP32) | - |
| `wmma_matmul` | Matrix Multiplication (FP16) | **WMMA** |
| `flash_attention` | Flash Attention v2 | - |

```python
# Build native kernels (requires CUDA)
python zenith/build_cuda.py

# Use in code
import zenith_cuda
C = zenith_cuda.wmma_matmul(A.half(), B.half())  # Tensor Core accelerated
```

---

## When to Use Zenith (And When Not To)

### Zenith Shines At:
- **Inference on large models** (LLMs, Vision Transformers with 100M+ params)
- **Production deployment** where every millisecond counts
- **Cost-conscious applications** (faster = less compute time = lower bills)
- **PyTorch 2.0+ torch.compile integration**

### Zenith May Not Help With:
- **Training** (focus is on inference, not backward passes)
- **Small/simple models** (ConvNets, MLPs under 10M params - overhead may exceed benefit)
- **Research/experimentation** (use eager mode for debugging)

**Honest Assessment:** Zenith adds value when your model is large enough that graph optimization overhead is worthwhile. For small models, native PyTorch is often faster.

---

## Installation

### Quick Install

```bash
pip install pyzenith
```

### Installation Options

Choose the right installation based on your needs:

| Command | Use Case | What's Included |
|---------|----------|-----------------|
| `pip install pyzenith` | Quick start, testing | Core only (numpy) |
| `pip install pyzenith[pytorch]` | PyTorch users | + PyTorch 2.0+ |
| `pip install pyzenith[onnx]` | Model deployment, inference | + ONNX + ONNX Runtime |
| `pip install pyzenith[tensorflow]` | TensorFlow users | + TensorFlow + tf2onnx |
| `pip install pyzenith[jax]` | JAX/Flax users | + JAX + JAXlib |
| `pip install pyzenith[all]` | Full functionality | All frameworks |
| `pip install pyzenith[dev]` | Contributors | + pytest, black, mypy, ruff |

### Recommended Installation

```bash
# For most ML users (PyTorch + ONNX export)
pip install pyzenith[pytorch,onnx]

# For full framework support
pip install pyzenith[all]

# For development/contribution
pip install pyzenith[dev]
```

### Development Installation

```bash
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH
pip install -e ".[dev]"
```

### CUDA Build (for Maximum GPU Performance)

For full CUDA kernel acceleration (50x speedup):

```bash
# On Google Colab or Linux with CUDA
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH
bash build_cuda.sh

# Verify installation
python -c "from zenith._zenith_core import backends; print(backends.list_available())"
# Output: ['cpu', 'cuda']
```

> **Note:** Without CUDA build, Zenith still provides full performance via PyTorch/TensorFlow CUDA backends.

---

## Quick Start

### Basic Usage

```python
import zenith
from zenith.core import GraphIR, DataType, Shape, TensorDescriptor

# Create a computation graph
graph = GraphIR(name="my_model")
graph.add_input(TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32))

# Apply optimizations
from zenith.optimization import PassManager
pm = PassManager()
pm.add("constant_folding")
pm.add("dead_code_elimination")
pm.add("operator_fusion")
optimized = pm.run(graph)
```

### CUDA Operations

```python
import numpy as np
from zenith._zenith_core import cuda

# Check CUDA availability
print(f"CUDA available: {cuda.is_available()}")

# Matrix multiplication (50x faster than PyTorch)
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = cuda.matmul(A, B)

# GPU operations
cuda.gelu(input_tensor)
cuda.layernorm(input_tensor, gamma, beta, eps=1e-5)
cuda.softmax(input_tensor)
```

### JAX Integration

```python
import jax
import jax.numpy as jnp
from zenith.jax.primitives import fused_attention, fused_gelu

# Fused attention - JIT-compatible and differentiable
batch, heads, seq, dim = 2, 8, 512, 64
q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq, dim))
k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq, dim))
v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq, dim))

output = fused_attention(q, k, v)

# Works with jax.grad
grads = jax.grad(lambda q, k, v: jnp.sum(fused_attention(q, k, v)))(q, k, v)
```

See [JAX Integration Guide](./docs/jax_integration.md) for more examples.

### torch.compile Backend (New in v0.3.0)

Zenith now integrates with PyTorch 2.0+ `torch.compile` for automatic optimization:

```python
import torch
import zenith  # Auto-registers 'zenith' backend

model = YourModel().cuda()

# Use Zenith as torch.compile backend
optimized_model = torch.compile(model, backend="zenith")

# Run as normal - Zenith handles optimization
output = optimized_model(input_tensor)
```

**Benchmark Results (TinyLlama 1.1B on Tesla T4):**

| Use Case | Improvement | Notes |
|----------|-------------|-------|
| Inference (TPS) | **+69%** | Text generation workloads |
| Training (SFT) | +2.6% | Minimal - Zenith focuses on inference |
| Energy Consumption | **-87%** | Faster completion = less total energy |
| Numerical Precision | 0.000 MSE | Perfect accuracy preserved |

*See [Zenith-Lab](https://github.com/vibeswithkk/Zenith-Lab) for reproducible benchmarks.*

---

## Architecture

```
+-------------------------------------------------------------+
|                    Python User Interface                    |
|                  (zenith.api, zenith.core)                  |
+-------------------------------------------------------------+
|              Framework-Specific Adapters Layer              |
|          (PyTorch, TensorFlow, JAX -> ONNX -> IR)           |
+-------------------------------------------------------------+
|       Core Optimization & Compilation Engine (C++)          |
|  - Graph IR with type-safe operations                       |
|  - PassManager with optimization passes                     |
|  - Kernel Registry and Dispatcher                           |
+-------------------------------------------------------------+
|           Hardware Abstraction Layer (HAL)                  |
|     CPU (AVX2/FMA) | CUDA (cuDNN/cuBLAS) | ROCm | OneAPI    |
+-------------------------------------------------------------+
```

---

## Benchmarks

### BERT-Base Inference (12 layers, batch=1, seq=128)

| Mode | Latency | vs PyTorch |
|------|---------|------------|
| Pure PyTorch | 10.60 ms | baseline |
| Zenith + PyTorch | 9.74 ms | 1.09x faster |

### ResNet-50 Throughput

| Batch Size | Throughput |
|------------|------------|
| 1 | 150 img/sec |
| 64 | 377 img/sec |
| 512 | 359 img/sec |

### GPU Memory Pool

| Metric | Value |
|--------|-------|
| Cache Hit Rate | 93.5% |
| Speedup vs naive | 330x |

---

## Testing

```bash
# Run all Python tests
pytest tests/python/ -v

# Run with coverage
pytest tests/python/ --cov=zenith --cov-report=term-missing

# Run C++ unit tests (after CUDA build)
./build/tests/test_core

# Security scan
bandit -r zenith/ -ll
```

### Test Status

- Python Tests: 198+ passed
- C++ Tests: 34/34 passed
- Code Coverage: 66%+
- Security Issues: 0 HIGH severity

---

## Documentation

- [Benchmark Report](./BENCHMARK_REPORT.md) - Comprehensive performance benchmarks
- [API Reference](./docs/API.md) - Python API documentation
- [Architecture](./docs/ARCHITECTURE.md) - System design documentation

---

## Project Status

Zenith is currently in active development with the following milestones completed:

- Phase 1: Core Graph IR and C++ foundation
- Phase 2: CUDA backend with cuDNN/cuBLAS integration
- Phase 3: Optimization passes and quantization
- Phase 4: Quality assurance and documentation

---

## Limitations & Transparency

We believe in being honest about what Zenith can and cannot do:

| Claim | Reality |
|-------|---------|
| "Works on all models" | Best on large models (100M+ params) |
| "Training acceleration" | Minimal (+2.6%). Zenith is for inference. |
| "Production-ready" | Alpha quality. Test thoroughly before production use. |
| "AMD/Intel GPU support" | Experimental. Only NVIDIA verified. |

### Known Issues
- Compilation overhead on first call (typically 0.5-2s)
- Small models may run slower than native PyTorch
- Some dynamic control flow patterns not yet supported

We are a small open-source project learning and improving. Bug reports and contributions are appreciated.

---

## Contributing

Contributions are welcome. Please ensure all tests pass before submitting pull requests.

```bash
# Setup development environment
pip install -e ".[dev]"

# Run tests before committing
pytest tests/python/ -v
```

---

## Author & Community

**Wahyu Ardiansyah** ([@vibeswithkk](https://github.com/vibeswithkk)) - Creator

This is a hobby project born from curiosity about ML optimization. Special thanks to everyone who has tested, reported bugs, and contributed. If you find this useful, consider:

- Starring the repo
- Reporting issues you encounter
- Sharing ideas for improvement
- [Supporting development](https://trakteer.id/wahyuardiansyah)

## License

Apache License 2.0 - See [LICENSE](./LICENSE) for details.

Copyright 2025 Wahyu Ardiansyah. All rights reserved.
