# Zenith JAX Benchmark Report

**Generated:** December 29, 2025  
**JAX Version:** 0.4.x  
**Hardware:** NVIDIA Tesla T4 (Google Colab)

---

## 1. Custom Primitives Performance

### 1.1 Fused Attention

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| Baseline JAX | TBD | 1.00x |
| Zenith fused_attention | TBD | TBD |

### 1.2 Fused GELU

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| jax.nn.gelu | TBD | 1.00x |
| Zenith fused_gelu | TBD | TBD |

---

## 2. Mixed Precision Speedup

| Precision | MatMul Time (ms) | Throughput | Memory |
|-----------|------------------|------------|--------|
| FP32 | TBD | TBD | TBD |
| FP16 | TBD | TBD | TBD |
| BF16 | TBD | TBD | TBD |

---

## 3. Gradient Checkpointing Memory Reduction

| Model Size | Standard Memory | With Checkpointing | Reduction |
|------------|-----------------|-------------------|-----------|
| 125M | TBD | TBD | TBD |
| 350M | TBD | TBD | TBD |
| 1.3B | TBD | TBD | TBD |

---

## 4. How to Run Benchmarks

```bash
# Run all benchmarks
python benchmarks/jax_benchmarks.py --all

# Run specific benchmarks
python benchmarks/jax_benchmarks.py --attention
python benchmarks/jax_benchmarks.py --gelu
python benchmarks/jax_benchmarks.py --precision
```

---

## Notes

- All benchmarks run with JAX JIT compilation enabled
- Warmup iterations: 5
- Timed iterations: 100
- Memory measurements are approximate

---

*Benchmark results will be populated after running on target hardware.*
