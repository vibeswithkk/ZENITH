# 8. PyTorch Training dengan Data Besar

Tutorial training PyTorch model dengan Zenith QAT menggunakan data dari Hugging Face (streaming).

## Install Dependencies

```bash
pip install datasets torch
```

## Load Data (Streaming)

Data di-stream dari Hugging Face, tidak tersimpan lokal:

```python
from datasets import load_dataset

# Streaming mode - tidak download ke lokal
dataset = load_dataset(
    "mnist",
    split="train",
    streaming=True  # Key: streaming mode!
)

# Ambil batch
batch = list(dataset.take(32))
print(f"Loaded {len(batch)} samples (streaming)")
```

## Create PyTorch Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MNISTNet()
```

## Apply Zenith QAT

```python
import numpy as np
from zenith.optimization.qat import (
    QATConfig,
    FakeQuantize,
    prepare_model_for_qat,
    convert_qat_to_quantized
)

# Config
config = QATConfig(
    weight_bits=8,
    activation_bits=8,
    symmetric_weights=True
)

# Extract layer weights
layer_weights = {
    'conv1': model.conv1.weight.detach().numpy(),
    'conv2': model.conv2.weight.detach().numpy(),
    'fc1': model.fc1.weight.detach().numpy(),
    'fc2': model.fc2.weight.detach().numpy(),
}

# Prepare QAT
trainer = prepare_model_for_qat(list(layer_weights.keys()), config)
```

## Training Loop dengan QAT

```python
import torch.optim as optim
from datasets import load_dataset

# Streaming dataset
dataset = load_dataset("mnist", split="train", streaming=True)

# Setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for i, batch in enumerate(dataset.take(100)):  # 100 samples
    # Preprocess
    image = torch.tensor(batch['image']).float().unsqueeze(0).unsqueeze(0)
    image = F.interpolate(image, size=(28, 28)) / 255.0
    label = torch.tensor([batch['label']])
    
    # Forward
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, label)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Calibrate QAT setiap 10 steps
    if i % 10 == 0:
        for name, param in model.named_parameters():
            if 'weight' in name:
                layer_name = name.replace('.weight', '')
                if layer_name in trainer.modules:
                    trainer.calibrate(
                        layer_name,
                        param.detach().numpy(),
                        np.random.randn(32, param.shape[1] if len(param.shape) > 1 else 1).astype(np.float32)
                    )
    
    if i % 20 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")

print("Training complete!")
```

## Convert ke INT8

```python
# Convert after training
quantized_weights = convert_qat_to_quantized(trainer, layer_weights)

# Size comparison
fp32_size = sum(w.nbytes for w in layer_weights.values())
int8_size = fp32_size / 4

print(f"FP32 Size: {fp32_size / 1024:.2f} KB")
print(f"INT8 Size: {int8_size / 1024:.2f} KB")
print(f"Reduction: 4.0x")
```

## Alternative Datasets (Hugging Face)

```python
# CIFAR-10
dataset = load_dataset("cifar10", split="train", streaming=True)

# ImageNet (subset)
dataset = load_dataset("imagenet-1k", split="train", streaming=True)

# Text datasets
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

# Custom dari URL
dataset = load_dataset("json", data_files="https://example.com/data.json", streaming=True)
```

## Keuntungan Streaming

| Aspect | Download | Streaming |
|--------|----------|-----------|
| Storage | GB-TB | 0 KB |
| Start time | Lama | Instant |
| Memory | High | Low |
| Shuffle | Full | Limited |

---

[← Autotuner](07_autotuner.md) | [Back to Index →](index.md)
