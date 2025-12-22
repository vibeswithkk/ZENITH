# Optimization Options

Configure Zenith optimization for your specific needs.

## Precision Modes

| Mode | Description | Speedup | Accuracy |
|------|-------------|---------|----------|
| fp32 | Full precision | 1x | 100% |
| fp16 | Half precision | 2-3x | 99.9% |
| bf16 | Brain float | 2-3x | 99.9% |
| int8 | Quantized | 3-5x | 99% |

### Usage

```python
optimized = zenith.compile(model, precision="fp16")
```

## Compile Modes

### Default (Balanced)

```python
optimized = zenith.compile(model, mode="default")
```

### Reduce Overhead (Low Latency)

```python
optimized = zenith.compile(model, mode="reduce-overhead")
```

Uses CUDA Graphs to minimize CPU overhead.

### Max Autotune (Maximum Performance)

```python
optimized = zenith.compile(model, mode="max-autotune")
```

Longer compile time, best runtime performance.

## Backend Selection

```python
# CUDA GPU
optimized = zenith.compile(model, target="cuda")

# CPU
optimized = zenith.compile(model, target="cpu")

# Specific GPU
optimized = zenith.compile(model, target="cuda:1")
```

## Optimization Passes

Zenith applies these optimizations automatically:

1. **Operator Fusion**: Combine multiple ops into one kernel
2. **Dead Code Elimination**: Remove unused computations
3. **Layout Optimization**: NCHW to NHWC conversion
4. **Precision Conversion**: Cast to target precision
5. **Memory Planning**: Optimize memory allocation

## Configuration Examples

### Maximum Performance

```python
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp16",
    mode="max-autotune",
)
```

### Low Memory

```python
optimized = zenith.compile(
    model,
    target="cuda",
    precision="int8",
)
```

### Debug Mode

```python
zenith.set_verbosity(4)  # DEBUG
optimized = zenith.compile(model)
```

## Next Steps

- [Troubleshooting](05_troubleshooting.md)
