# Zenith Tutorials

Tutorial penggunaan Zenith ML Optimization Framework.

## Tutorial W3Schools-Style

Tutorial lengkap ada di **[docs/tutorial/](../docs/tutorial/)**:

| Chapter | Topic |
|---------|-------|
| [1. Getting Started](../docs/tutorial/01_getting_started.md) | Install & Setup |
| [2. Basics](../docs/tutorial/02_basics.md) | Imports & Modules |
| [3. Quantization](../docs/tutorial/03_quantization.md) | FakeQuantize |
| [4. QAT](../docs/tutorial/04_qat.md) | QAT Training |
| [5. PyTorch](../docs/tutorial/05_pytorch.md) | PyTorch Integration |
| [6. Triton](../docs/tutorial/06_triton.md) | Deployment |
| [7. Autotuner](../docs/tutorial/07_autotuner.md) | Auto-tuning |

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/docs/tutorial/zenith_tutorial_colab.ipynb)

## Quick Start

```python
# Install
!git clone https://github.com/vibeswithkk/ZENITH.git
%cd ZENITH
!pip install -e .

# Use
from zenith.optimization.qat import FakeQuantize
from zenith.serving.triton_client import MockTritonClient
```
