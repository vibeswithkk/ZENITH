# PyTorch Integration

Zenith integrates seamlessly with PyTorch models.

## Basic Integration

### Using zenith.compile()

```python
import zenith
import torch

# Any PyTorch model
model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
).cuda()

# Compile with Zenith
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp16",
)

# Use like normal PyTorch
x = torch.randn(32, 128, 512).cuda()
output = optimized(x)
```

### Using torch.compile Backend

```python
import torch

# Use Zenith as torch.compile backend
model = torch.nn.Linear(1024, 1024).cuda()
optimized = torch.compile(model, backend="zenith")

x = torch.randn(32, 1024).cuda()
output = optimized(x)
```

## Precision Modes

### FP16 (Recommended for inference)

```python
optimized = zenith.compile(model, precision="fp16")
```

### INT8 Quantization

```python
optimized = zenith.compile(model, precision="int8")
```

## Working with HuggingFace Models

```python
from transformers import BertModel
import zenith

# Load pretrained model
model = BertModel.from_pretrained("bert-base-uncased").cuda()
model.eval()

# Compile with Zenith
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp16",
)

# Run inference
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")

with torch.no_grad():
    output = optimized(**inputs)
```

## ONNX Export

```python
import zenith.torch as ztorch

# Export to ONNX
ztorch.to_onnx(
    model,
    sample_input=torch.randn(1, 768).cuda(),
    output_path="model.onnx",
)
```

## Best Practices

1. **Always use .eval()** for inference models
2. **Use torch.no_grad()** to disable gradient computation
3. **Start with FP16** for best performance/accuracy tradeoff
4. **Warm up** the model with a few forward passes before benchmarking

## Next Steps

- [Optimization Options](04_optimization_options.md)
- [Troubleshooting](05_troubleshooting.md)
