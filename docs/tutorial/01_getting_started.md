# 1. Getting Started

## Install Zenith

Install dari PyPI:
```bash
pip install pyzenith
```

Atau dari source (untuk development):
```bash
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH
pip install -e .
```

## Verify Installation

```python
import zenith
print(zenith.__version__)
```

**Output:**
```
0.1.4
```

## Check Backends

```python
from zenith import backends

print(f"CPU Available: {backends.is_cpu_available()}")
print(f"CUDA Available: {backends.is_cuda_available()}")
```

**Output:**
```
CPU Available: True
CUDA Available: False
```

> **Note:** CUDA akan `True` jika sistem memiliki GPU NVIDIA.

---

[← Back to Index](index.md) | [Next: Basics →](02_basics.md)
