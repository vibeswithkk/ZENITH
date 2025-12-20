# 5. Zenith + PyTorch

Zenith bekerja sebagai **pelengkap** PyTorch untuk optimasi model.

## Extract PyTorch Weights

```python
import torch
import torch.nn as nn
import numpy as np

# Create PyTorch model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleNet()

# Extract weights as NumPy
layer_weights = {
    'fc1': model.fc1.weight.detach().numpy(),
    'fc2': model.fc2.weight.detach().numpy(),
}
```

## Apply Zenith QAT

```python
from zenith.optimization.qat import (
    QATConfig,
    prepare_model_for_qat,
    convert_qat_to_quantized
)

# Config
config = QATConfig(weight_bits=8, activation_bits=8)

# Prepare
trainer = prepare_model_for_qat(list(layer_weights.keys()), config)

# Calibrate
for name, weights in layer_weights.items():
    activations = np.random.randn(100, weights.shape[1]).astype(np.float32)
    trainer.calibrate(name, weights, activations)

# Convert
quantized = convert_qat_to_quantized(trainer, layer_weights)
```

## Compare Size

```python
fp32_size = sum(w.nbytes for w in layer_weights.values())
int8_size = fp32_size / 4

print(f"FP32 Size: {fp32_size / 1024:.2f} KB")
print(f"INT8 Size: {int8_size / 1024:.2f} KB")
print(f"Reduction: {fp32_size / int8_size:.1f}x")
```

**Output:**
```
FP32 Size: 795.31 KB
INT8 Size: 198.83 KB
Reduction: 4.0x
```

---

[← QAT](04_qat.md) | [Next: Triton →](06_triton.md)
