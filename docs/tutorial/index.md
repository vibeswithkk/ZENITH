# Zenith Tutorial

Welcome to the Zenith Tutorial!

Zenith adalah framework optimasi ML yang bekerja sebagai **pelengkap** (bukan pengganti) framework seperti PyTorch dan JAX.

## Table of Contents

| Chapter | Topic |
|---------|-------|
| [1. Getting Started](01_getting_started.md) | Install & Setup |
| [2. Basics](02_basics.md) | Import, Version, Backends |
| [3. Quantization](03_quantization.md) | FakeQuantize |
| [4. QAT Training](04_qat.md) | Quantization-Aware Training |
| [5. PyTorch](05_pytorch.md) | Zenith + PyTorch |
| [6. Triton](06_triton.md) | Deployment with Triton |
| [7. Autotuner](07_autotuner.md) | Kernel Auto-tuning |
| [8. PyTorch Training](08_pytorch_training.md) | Training dengan Data Besar |

## Quick Start

```python
# Install
pip install pyzenith

# Or from source
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH && pip install -e .

# Use
from zenith.optimization.qat import FakeQuantize
```

---

[Next: Getting Started →](01_getting_started.md)
