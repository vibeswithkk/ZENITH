# Zenith Full Performance Mode

Tutorial untuk mencapai performa maksimal dengan Zenith.

## Overview

| Technique | Speedup | Memory Reduction | Best For |
|-----------|---------|------------------|----------|
| FP16 | 2-6x | 50% | GPU Inference |
| INT8 | 2-4x | 75% | Edge Deployment |
| Operator Fusion | 1.2-1.5x | - | All |
| Batch Processing | 2-10x | - | Throughput |

## 1. FP16 (Tensor Core)

Untuk GPU dengan Tensor Cores (NVIDIA V100, T4, A100, RTX 30xx+):

```python
import torch

# Convert model to FP16
model = model.half()

# Convert input to FP16
x = x.half()

# Run inference
with torch.no_grad():
    output = model(x)  # Uses Tensor Cores!
```

**Speedup: 2-6x pada Tensor Core GPUs**

## 2. INT8 Quantization

```python
from zenith.optimization.quantization import quantize, CalibrationMethod

# Quantize weights
weights = model.fc.weight.detach().cpu().numpy()
quantized, scale, zp = quantize(
    weights,
    num_bits=8,
    method=CalibrationMethod.MINMAX
)

print(f"Original: {weights.nbytes} bytes")
print(f"Quantized: {quantized.nbytes} bytes")
print(f"Reduction: {weights.nbytes/quantized.nbytes:.1f}x")
```

**Size Reduction: 4x**

## 3. Combine FP16 + Zenith Compile

```python
import zenith
import torch

# Full performance pipeline
model = MyModel().cuda().half().eval()

# Compile with Zenith
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp16",
    opt_level=2  # Maximum optimization
)

# Run inference
x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float16).cuda()
with torch.no_grad():
    output = optimized(x)
```

## 4. Complete Benchmark Example

```python
import torch
import torch.nn as nn
import time

# BERT-style block
class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x

# 4-layer model
class MiniTransformer(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Benchmark function
def benchmark(model, x, warmup=10, runs=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)

# Create models
model_fp32 = MiniTransformer().cuda().eval()
model_fp16 = MiniTransformer().cuda().half().eval()

# Input
x_fp32 = torch.randn(8, 128, 768).cuda()
x_fp16 = x_fp32.half()

# Benchmark
print("Full Performance Benchmark")
print("="*40)
t_fp32 = benchmark(model_fp32, x_fp32)
t_fp16 = benchmark(model_fp16, x_fp16)
print(f"FP32: {t_fp32:.2f} ms")
print(f"FP16: {t_fp16:.2f} ms")
print(f"Speedup: {t_fp32/t_fp16:.2f}x")
```

## 5. Memory Optimization

```python
# Enable memory efficient attention (PyTorch 2.0+)
import torch.nn.functional as F

# Use scaled_dot_product_attention
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=True
):
    output = F.scaled_dot_product_attention(q, k, v)
```

## Performance Checklist

- [ ] Model in `.eval()` mode
- [ ] Use `torch.no_grad()` context
- [ ] Convert to FP16 with `.half()`
- [ ] Use batch inference
- [ ] Warm up before benchmarking
- [ ] Synchronize GPU for accurate timing

## Expected Results (Tesla T4)

| Model | FP32 | FP16 | Speedup |
|-------|------|------|---------|
| BERT-Base (12 layers) | 10.60 ms | 4.80 ms | 2.2x |
| MiniTransformer (4 layers) | 3.50 ms | 0.85 ms | 4.1x |
| ResNet-50 | 8.20 ms | 2.10 ms | 3.9x |

---

[← JAX](10_jax.md) | [Back to Index →](index.md)
