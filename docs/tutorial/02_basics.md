# 2. Zenith Basics

## Import Modules

```python
# Core
import zenith
from zenith import backends

# Optimization
from zenith.optimization.qat import FakeQuantize, QATConfig

# Serving
from zenith.serving.triton_client import TritonClient, MockTritonClient
```

## Version Info

```python
import zenith
print(f"Version: {zenith.__version__}")
```

**Output:**
```
Version: 0.1.4
```

## Available Backends

```python
from zenith import backends

# Check availability
print(backends.is_cpu_available())   # True
print(backends.is_cuda_available())  # True/False

# Get all backends
print(backends.get_available_backends())
```

**Output:**
```
True
False
['cpu']
```

## NumPy Compatibility

Zenith bekerja dengan NumPy arrays:

```python
import numpy as np
from zenith.optimization.qat import FakeQuantize

# Create data
data = np.random.randn(100).astype(np.float32)

# Use with Zenith
fq = FakeQuantize(num_bits=8)
fq.observe(data)
```

---

[← Getting Started](01_getting_started.md) | [Next: Quantization →](03_quantization.md)
