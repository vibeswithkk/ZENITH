# Zenith API Reference

**Version:** 0.1.0  
**License:** Apache 2.0

---

## Overview

Zenith is a cross-platform ML optimization framework that provides unified interfaces for optimizing models from PyTorch, TensorFlow, and JAX.

---

## Installation

```bash
pip install zenith-ml

# With framework support
pip install zenith-ml[onnx,pytorch,tensorflow,jax]
```

---

## Quick Start

```python
import zenith
from zenith.core import GraphIR, DataType

# Create a computation graph
graph = GraphIR(name="my_model")

# Optimize model
optimized = zenith.optimize(graph, target="cuda")
```

---

## Core Module

### zenith.compile

```python
zenith.compile(
    model: Any,
    target: str = "cpu",
    precision: str = "fp32",
    opt_level: int = 2,
    tolerance: float = 1e-5
) -> CompiledModel
```

Compile a model for optimized execution.

**Parameters:**
- `model`: Model from PyTorch, TensorFlow, JAX, or ONNX
- `target`: Target device (`"cpu"`, `"cuda"`, `"rocm"`, `"tpu"`)
- `precision`: Precision level (`"fp32"`, `"fp16"`, `"bf16"`, `"int8"`)
- `opt_level`: Optimization aggressiveness (1-3)
- `tolerance`: Maximum numerical error tolerance

### zenith.optimize

```python
zenith.optimize(
    graph: GraphIR,
    passes: list[str] | None = None,
    target: str = "cpu"
) -> GraphIR
```

Apply optimization passes to a graph.

**Parameters:**
- `graph`: Input GraphIR
- `passes`: List of pass names to apply
- `target`: Target device for optimizations

---

## Core Types

### GraphIR

```python
from zenith.core import GraphIR

graph = GraphIR(name="model_name")
graph.add_input(tensor_descriptor)
graph.add_output(tensor_descriptor)
graph.add_node(op_type="Conv", name="conv1", inputs=[...], outputs=[...])
```

### DataType

```python
from zenith.core import DataType

DataType.Float32   # FP32
DataType.Float16   # FP16
DataType.BFloat16  # BF16
DataType.Int8      # INT8
DataType.Int32     # INT32
```

### Shape

```python
from zenith.core import Shape

shape = Shape([1, 3, 224, 224])  # NCHW format
len(shape)  # 4
shape[0]    # 1
```

### TensorDescriptor

```python
from zenith.core import TensorDescriptor, Shape, DataType

td = TensorDescriptor(
    name="input",
    shape=Shape([1, 3, 224, 224]),
    dtype=DataType.Float32
)
```

---

## Adapters

### ONNXAdapter

```python
from zenith.adapters import ONNXAdapter

adapter = ONNXAdapter()
graph = adapter.from_model("model.onnx")
graph = adapter.from_bytes(onnx_bytes)
```

### PyTorchAdapter

```python
from zenith.adapters import PyTorchAdapter

adapter = PyTorchAdapter()
graph = adapter.from_model(
    model,
    sample_input=torch.randn(1, 3, 224, 224)
)
```

### TensorFlowAdapter

```python
from zenith.adapters import TensorFlowAdapter

adapter = TensorFlowAdapter()
graph = adapter.from_saved_model("/path/to/saved_model")
graph = adapter.from_model(keras_model)
```

### JAXAdapter

```python
from zenith.adapters import JAXAdapter

adapter = JAXAdapter()
graph = adapter.from_model(
    jax_function,
    sample_input=jnp.ones((1, 10))
)
```

---

## Optimization

### PassManager

```python
from zenith.optimization import PassManager

pm = PassManager()
pm.add("constant_folding")
pm.add("dead_code_elimination")
pm.add("operator_fusion")
optimized = pm.run(graph)
```

### Available Passes

| Pass Name | Description |
|-----------|-------------|
| `constant_folding` | Fold constant expressions |
| `dead_code_elimination` | Remove unused operations |
| `operator_fusion` | Fuse Conv-BN-ReLU patterns |
| `layout_optimization` | Optimize memory layout |

### Quantization

```python
from zenith.optimization import Quantizer, QuantizationMode

quantizer = Quantizer(mode=QuantizationMode.STATIC)
quantized_graph = quantizer.quantize(graph, calibration_data)
```

### Mixed Precision

```python
from zenith.optimization import MixedPrecisionManager, PrecisionPolicy

policy = PrecisionPolicy.fp16_with_loss_scale()
mp = MixedPrecisionManager(policy)
fp16_graph = mp.convert(graph)
```

---

## Profiling

```python
from zenith.optimization import Profiler

profiler = Profiler()
with profiler.session("inference"):
    with profiler.measure("conv1", "Conv2D"):
        result = execute_conv()

profiler.export_json("profile.json")
```

---

## Error Handling

```python
from zenith.core import ZenithError, CompilationError

try:
    result = zenith.compile(model)
except CompilationError as e:
    print(f"Compilation failed: {e}")
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ZENITH_CACHE_DIR` | Cache directory | `~/.zenith/cache` |
| `ZENITH_LOG_LEVEL` | Logging level | `INFO` |
| `ZENITH_BACKEND` | Preferred backend | Auto-detect |
