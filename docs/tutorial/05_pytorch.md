# 5. Zenith + PyTorch (Hybrid Mode)

Zenith bekerja sebagai **pelengkap** PyTorch untuk optimasi model.

## Quick Start

```python
import zenith
import torch

# Any PyTorch model
model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
).cuda().eval()

# Compile with Zenith
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp32"
)

# Use like normal PyTorch
x = torch.randn(32, 128, 512).cuda()
with torch.no_grad():
    output = optimized(x)
```

## zenith.compile() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | PyTorch, TensorFlow, or JAX model |
| `target` | str | "cpu" | "cpu" or "cuda" |
| `precision` | str | "fp32" | "fp32", "fp16", or "int8" |
| `opt_level` | int | 2 | 0=none, 1=basic, 2=full |

## FP16 Mode (Tensor Core)

Untuk performa maksimal pada GPU dengan Tensor Cores:

```python
# Convert model to FP16
model = model.half()

# Convert input to FP16
x = x.half()

# Run inference
with torch.no_grad():
    output = model(x)  # Uses Tensor Cores!
```

**Expected Speedup: 2-6x**

## Quantization (INT8)

```python
from zenith.optimization.quantization import quantize

# Quantize weights
for name, param in model.named_parameters():
    if 'weight' in name:
        quantized, scale, zp = quantize(
            param.detach().cpu().numpy()
        )
        print(f"{name}: {quantized.dtype}")
```

**Expected Size Reduction: 4x**

## Benchmark Example

```python
import time
import torch

def benchmark(model, x, warmup=10, runs=50):
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)

# Compare
t_fp32 = benchmark(model, x)
t_fp16 = benchmark(model.half(), x.half())
print(f"FP32: {t_fp32:.2f}ms, FP16: {t_fp16:.2f}ms")
print(f"Speedup: {t_fp32/t_fp16:.2f}x")
```

## HuggingFace Integration

```python
from transformers import BertModel, BertTokenizer
import zenith

# Load pretrained
model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Compile with Zenith
optimized = zenith.compile(model, target="cuda", precision="fp16")

# Inference
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
    sample_input=torch.randn(1, 128, 768).cuda(),
    output_path="model.onnx"
)
```

## Best Practices

1. **Always use `.eval()`** for inference models
2. **Use `torch.no_grad()`** to disable gradient computation
3. **Start with FP16** for best performance/accuracy tradeoff
4. **Warm up** the model with a few forward passes before benchmarking
5. **Use batch inference** when possible

---

[← QAT](04_qat.md) | [Next: Triton →](06_triton.md)
