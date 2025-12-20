# Zenith Tutorials

Tutorial penggunaan Zenith ML Optimization Framework.

## Available Tutorials

| Tutorial | Description |
|----------|-------------|
| [zenith_tutorial.ipynb](zenith_tutorial.ipynb) | Complete guide dengan PyTorch, JAX, QAT, dan Triton |

## Quick Start

```python
# Install
!git clone https://github.com/vibeswithkk/ZENITH.git
%cd ZENITH
!pip install -e .

# Use
from zenith.optimization.qat import FakeQuantize, QATConfig
from zenith.serving.triton_client import MockTritonClient
```

## Topics Covered

1. **Getting Started** - Install, import, backend check
2. **PyTorch + Zenith** - QAT untuk model PyTorch
3. **JAX + Zenith** - Optimasi fungsi JAX
4. **Triton Deployment** - Mock server demo
5. **Auto-tuning** - Kernel optimization
6. **Load Testing** - Performance testing

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/tutorials/zenith_tutorial.ipynb)
