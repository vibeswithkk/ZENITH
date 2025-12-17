# Zenith

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![Coverage](https://img.shields.io/badge/Coverage-66%25-yellow.svg)]()
[![Tests](https://img.shields.io/badge/Tests-198%20passed-brightgreen.svg)]()

**Cross-Platform ML Optimization Framework**

A model-agnostic and hardware-agnostic unification and optimization framework for Machine Learning.

## Features

- Unified API for PyTorch, TensorFlow, JAX, and ONNX models
- Automatic graph optimizations (fusion, constant folding, dead code elimination)
- Multi-backend support (CPU, CUDA, ROCm, TPU)
- Mixed precision training and inference (FP16, BF16, INT8)
- Property-based testing with mathematical guarantees

## Installation

```bash
# Basic installation
pip install zenith-ml

# With framework support
pip install zenith-ml[onnx,pytorch,tensorflow,jax]

# Development installation
pip install -e ".[dev]"
```

## Quick Start

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
optimized = pm.run(graph)
```

## Architecture

```
+-------------------------------------------------------------+
|                    Python User Interface                    |
+-------------------------------------------------------------+
|              Framework-Specific Adapters Layer              |
|          (PyTorch, TensorFlow, JAX -> ONNX -> IR)           |
+-------------------------------------------------------------+
|       Core Optimization & Compilation Engine (C++)          |
|  - High-Level Graph Optimizer & IR                          |
|  - Kernel Scheduler & Auto-Tuner                            |
|  - Mathematical Kernel Library                              |
+-------------------------------------------------------------+
|           Hardware Abstraction Layer (HAL)                  |
|              CPU (SIMD) | CUDA | ROCm | TPU                 |
+-------------------------------------------------------------+
```

## Documentation

- [API Reference](./docs/API.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Blueprint](./CetakBiru.md)

## Development

```bash
# Run tests
pytest tests/python/ -v

# Run with coverage
pytest tests/python/ --cov=zenith --cov-report=term-missing

# Security scan
bandit -r zenith/ -ll
```

## Current Status

- Phase 4: Quality Assurance & Documentation
- 198 tests passing
- 66% code coverage
- 0 HIGH severity security issues

## Author

**Wahyu Ardiansyah** - Lead Architect

## License

Apache License 2.0 - See [LICENSE](./LICENSE)
