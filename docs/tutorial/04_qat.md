# 4. QAT Training

QAT (Quantization-Aware Training) melatih model dengan simulasi quantization.

## QAT Config

```python
from zenith.optimization.qat import QATConfig

config = QATConfig(
    weight_bits=8,
    activation_bits=8,
    symmetric_weights=True,
    per_channel_weights=True
)
```

**Parameters:**
- `weight_bits`: Bit-width untuk weights
- `activation_bits`: Bit-width untuk activations
- `symmetric_weights`: Symmetric quantization untuk weights
- `per_channel_weights`: Per-channel quantization

## Prepare Model for QAT

```python
from zenith.optimization.qat import prepare_model_for_qat

# Layer names
layer_names = ['fc1', 'fc2', 'fc3']

# Create trainer
trainer = prepare_model_for_qat(layer_names, config)
```

## Calibrate

```python
import numpy as np

# Your layer weights (from PyTorch/JAX)
weights = np.random.randn(256, 784).astype(np.float32)
activations = np.random.randn(100, 784).astype(np.float32)

trainer.calibrate('fc1', weights, activations)
```

## Convert to Quantized

```python
from zenith.optimization.qat import convert_qat_to_quantized

# Your weights dict
layer_weights = {
    'fc1': np.random.randn(256, 784).astype(np.float32),
    'fc2': np.random.randn(128, 256).astype(np.float32),
}

# Convert
quantized_weights = convert_qat_to_quantized(trainer, layer_weights)
```

## BatchNorm Folding

Fold BatchNorm into Conv for inference:

```python
from zenith.optimization.qat import fold_bn_into_conv
import numpy as np

# Conv weights
weight = np.random.randn(4, 3, 3, 3).astype(np.float32)
bias = np.random.randn(4).astype(np.float32)

# BN parameters
bn_mean = np.random.randn(4).astype(np.float32)
bn_var = np.abs(np.random.randn(4)) + 0.1
bn_gamma = np.random.randn(4).astype(np.float32)
bn_beta = np.random.randn(4).astype(np.float32)

# Fold
folded_weight, folded_bias = fold_bn_into_conv(
    weight, bias, bn_mean, bn_var, bn_gamma, bn_beta
)
print(f"Folded weight shape: {folded_weight.shape}")
```

**Output:**
```
Folded weight shape: (4, 3, 3, 3)
```

---

[← Quantization](03_quantization.md) | [Next: PyTorch →](05_pytorch.md)
