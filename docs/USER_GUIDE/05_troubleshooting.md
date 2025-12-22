# Troubleshooting

Common issues and solutions for Zenith.

## Installation Issues

### ImportError: No module named 'zenith'

```bash
pip install --upgrade pyzenith
```

### CUDA not available

1. Check CUDA installation: `nvcc --version`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Set CUDA path: `export CUDA_HOME=/usr/local/cuda`

## Compilation Errors

### CompilationError: GraphIR has no nodes

The model graph is empty. Ensure your model has layers:

```python
# Wrong
model = torch.nn.Sequential()

# Correct
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5)
)
```

### UnsupportedOperationError

Not all operations are supported. Solutions:

1. Check if the op is supported: `zenith.backends.list_supported_ops()`
2. Use a different precision
3. Decompose complex operations

## Runtime Errors

### KernelError: CUDA launch failed

1. Check input shapes match model expectations
2. Verify tensor device: `tensor.device`
3. Reduce batch size if out of memory

### ZenithMemoryError: Out of memory

1. Reduce batch size
2. Use FP16 precision
3. Clear unused tensors:

```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

## Performance Issues

### No speedup observed

1. Ensure model is on GPU: `model.cuda()`
2. Use FP16 precision
3. Warm up before benchmarking:

```python
# Warmup
for _ in range(5):
    _ = optimized(input)
```

### Compilation too slow

Use `mode="default"` instead of `mode="max-autotune"`:

```python
optimized = zenith.compile(model, mode="default")
```

## Debugging

### Enable verbose logging

```python
zenith.set_verbosity(4)  # DEBUG level
```

### Check error details

```python
try:
    optimized = zenith.compile(model)
except zenith.ZenithError as e:
    print(e)  # Includes suggestions
```

## Getting Help

- GitHub Issues: https://github.com/vibeswithkk/zenith/issues
- Documentation: https://zenith-ml.readthedocs.io
