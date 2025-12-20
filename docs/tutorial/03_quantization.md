# 3. Quantization

Quantization mengurangi ukuran model dengan mengkonversi weights dari FP32 ke INT8.

## FakeQuantize

FakeQuantize mensimulasikan quantization selama training.

### Create FakeQuantize

```python
from zenith.optimization.qat import FakeQuantize

fq = FakeQuantize(num_bits=8, symmetric=True)
```

**Parameters:**
- `num_bits`: Bit-width (default: 8)
- `symmetric`: Use symmetric quantization (default: True)

### Observe Data

Sebelum quantize, observe data untuk kalibrasi:

```python
import numpy as np

data = np.random.randn(100).astype(np.float32)
fq.observe(data)
```

### Apply Quantization

```python
quantized = fq.forward(data)
```

### Complete Example

```python
from zenith.optimization.qat import FakeQuantize
import numpy as np

# Create FakeQuantize
fq = FakeQuantize(num_bits=8, symmetric=True)

# Generate data
data = np.random.randn(100).astype(np.float32)

# Calibrate
fq.observe(data)

# Quantize
quantized = fq.forward(data)

# Check error
error = np.mean(np.abs(data - quantized))
print(f"Mean Error: {error:.6f}")
```

**Output:**
```
Mean Error: 0.006094
```

### Get Quantization Parameters

```python
scale, zero_point = fq.get_quantization_params()
print(f"Scale: {scale}")
print(f"Zero Point: {zero_point}")
```

---

[← Basics](02_basics.md) | [Next: QAT Training →](04_qat.md)
