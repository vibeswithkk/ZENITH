# Changelog

All notable changes to Zenith will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-12-25

### Added
- **WMMA Tensor Core MatMul** - FP16 matrix multiplication with Tensor Core acceleration
- Native CUDA kernel JIT compilation via `build_cuda.py`
- Runtime Tensor Core detection using `cudaGetDeviceProperties`
- `zenith_cuda.wmma_matmul()` for FP16->FP32 matmul

### Changed
- Kernel registry now prioritizes JIT-compiled kernels (priority=25)
- Updated stability score to 84%

### Fixed
- Fixed `__CUDA_ARCH__` compilation issue with runtime detection
- Fixed JIT module path discovery for cross-process imports

## [0.2.0] - 2025-12-24

### Added
- Native CUDA kernel bindings via pybind11
- JIT-compiled kernels: relu, gelu, layernorm, matmul, flash_attention
- Kernel registry auto-dispatch system
- Priority-based kernel selection (native > PyTorch > CPU)

### Changed
- Improved kernel fallback chain
- Enhanced test coverage (942+ tests)

## [0.1.0] - 2025-12-16

### Added
- Initial public release
- PyTorch, TensorFlow, JAX adapters
- ONNX model support
- Graph optimization passes (fusion, folding, DCE)
- INT8 quantization with calibration
- Triton Inference Server backend
- Mixed precision support (FP16, BF16, INT8)
- Comprehensive test suite
- Documentation and tutorials

### Infrastructure
- PyPI package: `pyzenith`
- GitHub Actions CI/CD
- CodeQL security scanning
