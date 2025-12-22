# Installation Guide

This guide covers how to install Zenith on your system.

## Requirements

- Python 3.9 or higher
- NumPy 1.20 or higher
- PyTorch, TensorFlow, or JAX (depending on your use case)

### Optional Dependencies

- CUDA 11.0+ for GPU acceleration
- cuDNN 8.0+ for optimized deep learning operations
- matplotlib for benchmark visualization

## Installation

### From PyPI (Recommended)

```bash
pip install pyzenith
```

### From Source

```bash
git clone https://github.com/vibeswithkk/zenith.git
cd zenith
pip install -e .
```

### With GPU Support

```bash
pip install pyzenith[cuda]
```

## Verification

Verify your installation:

```python
import zenith

print(f"Zenith version: {zenith.__version__}")
print(f"Native bindings: {zenith.is_native()}")
print(f"CUDA available: {zenith.backends.is_cuda_available()}")
```

Expected output:

```
Zenith version: 0.1.4
Native bindings: True
CUDA available: True  # or False if no GPU
```

## Troubleshooting

### ModuleNotFoundError: No module named 'zenith'

Ensure you installed the package correctly:

```bash
pip install --upgrade pyzenith
```

### CUDA not detected

1. Verify CUDA is installed: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall with CUDA support: `pip install pyzenith[cuda]`

## Next Steps

- [Quick Start Guide](02_quick_start.md)
- [PyTorch Integration](03_pytorch_integration.md)
