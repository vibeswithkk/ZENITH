# Zenith

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/pyzenith.svg)](https://pypi.org/project/pyzenith/)
[![CI](https://github.com/vibeswithkk/ZENITH/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeswithkk/ZENITH/actions/workflows/ci.yml)
[![CodeQL](https://github.com/vibeswithkk/ZENITH/actions/workflows/codeql.yml/badge.svg)](https://github.com/vibeswithkk/ZENITH/actions/workflows/codeql.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**Cross-Platform ML Optimization Framework**

Zenith is a model-agnostic and hardware-agnostic unification and optimization framework for Machine Learning. It provides enterprise-grade performance optimizations that consistently outperform PyTorch in both inference and training workloads.

## Project History

Zenith was conceived and architecturally designed on **December 11, 2024**, with the creation of its comprehensive blueprint document (CetakBiru.md) that outlines a 36-month development roadmap across 6 implementation phases. Active development began on **January 12, 2025**, and after 11 months of internal development, research, and rigorous testing, Zenith was publicly released on GitHub on **December 16, 2025**.

The project represents nearly a year of dedicated work in building a production-ready ML optimization framework from the ground up, implementing CUDA backends with cuDNN/cuBLAS integration, graph optimization passes, mixed precision support, and comprehensive testing infrastructure.

---

## Performance Highlights

| Benchmark | Workload | Result |
|-----------|----------|--------|
| GPU Memory Pool | MatMul 1024x1024 | **50x faster** than PyTorch |
| BERT Inference | 12-layer encoder | **1.09x faster** than PyTorch |
| Training Loop | 6-layer Transformer | **1.02x faster** than PyTorch |
| Memory Efficiency | Zero-copy allocation | **93.5% cache hit rate** |
| INT8 Quantization | Model compression | **4x memory reduction** |

*Benchmarked on NVIDIA Tesla T4 (Google Colab). See [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md) for full results.*

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
- AMD GPU: ROCm support (planned)
- Intel: OneAPI support (planned)

---

## Installation

```bash
# Install from PyPI
pip install pyzenith

# Install with optional dependencies
pip install pyzenith[onnx,pytorch]

# Development installation
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH
pip install -e ".[dev]"
```

### CUDA Build (for GPU acceleration)

```bash
# On Google Colab or Linux with CUDA
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH
bash build_cuda.sh

# Verify installation
python -c "from zenith._zenith_core import backends; print(backends.list_available())"
# Output: ['cpu', 'cuda']
```

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

## Contributing

Contributions are welcome. Please ensure all tests pass before submitting pull requests.

```bash
# Setup development environment
pip install -e ".[dev]"

# Run tests before committing
pytest tests/python/ -v
```

---

## Author

**Wahyu Ardiansyah** - Lead Architect and Developer

## License

Apache License 2.0 - See [LICENSE](./LICENSE) for details.

Copyright 2025 Wahyu Ardiansyah. All rights reserved.
