# 2. Zenith Basics

## Import Modules

```python
# Core
import zenith
from zenith import backends
from zenith.core import GraphIR, DataType

# Optimization
from zenith.optimization import PassManager
from zenith.optimization.quantization import quantize

# Serving
from zenith.serving import TritonBackend, export_to_onnx
```

## Version Info

```python
import zenith
print(f"Version: {zenith.__version__}")
```

**Output:**
```
Version: 0.2.1
```

## Available Backends

```python
from zenith import backends

# Check availability
print(backends.is_cpu_available())   # True
print(backends.is_cuda_available())  # True/False (depends on GPU)

# Get all backends
print(backends.get_available_backends())  # ['cpu'] or ['cpu', 'cuda']
```

## Core Types

```python
from zenith.core import DataType, Shape

# Data types
DataType.Float32   # 32-bit floating point
DataType.Float16   # 16-bit floating point (Tensor Core)
DataType.Int8      # 8-bit integer (quantized)
DataType.Int32     # 32-bit integer

# Shapes
shape = Shape([1, 3, 224, 224])  # NCHW format
```

## GraphIR (Intermediate Representation)

```python
from zenith.core import GraphIR

# Create graph
graph = GraphIR(name="my_model")
print(f"Graph name: {graph.name}")
print(f"Num nodes: {graph.num_nodes}")
```

## NumPy Compatibility

Zenith bekerja dengan NumPy arrays:

```python
import numpy as np
from zenith.optimization.quantization import quantize

# Create data
data = np.random.randn(1, 768).astype(np.float32)

# Quantize to INT8
quantized, scale, zp = quantize(data, num_bits=8)

print(f"Original: {data.dtype}, {data.nbytes} bytes")
print(f"Quantized: {quantized.dtype}, {quantized.nbytes} bytes")
```

---

[← Getting Started](01_getting_started.md) | [Next: Quantization →](03_quantization.md)
