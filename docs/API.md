# Zenith API Reference

**Version:** 0.1.4  
**Author:** Wahyu Ardiansyah  
**License:** Apache License 2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core API](#core-api)
   - [compile](#compile)
   - [optimize](#optimize)
   - [is_native](#is_native)
4. [Core Types](#core-types)
   - [DataType](#datatype)
   - [Layout](#layout)
   - [StatusCode](#statuscode)
   - [Status](#status)
   - [Shape](#shape)
   - [TensorDescriptor](#tensordescriptor)
   - [Node](#node)
   - [GraphIR](#graphir)
5. [Framework Adapters](#framework-adapters)
   - [PyTorchAdapter](#pytorchadapter)
   - [TensorFlowAdapter](#tensorflowadapter)
   - [JAXAdapter](#jaxadapter)
   - [ONNXAdapter](#onnxadapter)
6. [Optimization Passes](#optimization-passes)
   - [PassManager](#passmanager)
   - [ConstantFoldingPass](#constantfoldingpass)
   - [DeadCodeEliminationPass](#deadcodeeliminationpass)
   - [OperatorFusionPass](#operatorfusionpass)
7. [Execution Engine](#execution-engine)
   - [ONNXInterpreter](#onnxinterpreter)
   - [ExecutionContext](#executioncontext)
8. [Utility Functions](#utility-functions)
9. [Constants and Enumerations](#constants-and-enumerations)
10. [Error Handling](#error-handling)
11. [Examples](#examples)

---

## Overview

Zenith is a cross-platform Machine Learning optimization framework that provides a unified API for compiling and optimizing models from PyTorch, TensorFlow, JAX, and ONNX. The framework is designed to be model-agnostic and hardware-agnostic, enabling seamless deployment across CPU, CUDA, ROCm, and TPU backends.

### Core Principles

- **Model-Agnostic:** Works with models from any supported ML framework
- **Hardware-Agnostic:** Targets multiple backends through a unified API
- **Mathematical Guarantees:** Optimization preserves accuracy within configurable bounds
- **Non-Invasive:** Integrates without modifying existing model code

### Import Structure

```python
import zenith

# Core functions
zenith.compile(model, target="cuda", precision="fp16")
zenith.optimize(model)  # Alias for compile
zenith.is_native()      # Check native bindings availability

# Core types
from zenith import (
    DataType,
    Layout,
    Shape,
    Status,
    StatusCode,
    TensorDescriptor,
    Node,
    GraphIR,
)

# Adapters
from zenith import (
    PyTorchAdapter,
    TensorFlowAdapter,
    JAXAdapter,
    ONNXAdapter,
)
```

---

## Quick Start

### Basic Usage

```python
import zenith
import torch

# Create a PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 5)
)

# Compile with Zenith
sample_input = torch.randn(1, 10)
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp16",
    sample_input=sample_input
)

# Use the optimized model
output = optimized(sample_input)
```

### ONNX Model Optimization

```python
import zenith

# Load and optimize an ONNX model
optimized = zenith.compile(
    "model.onnx",
    target="cpu",
    opt_level=3
)
```

---

## Core API

### compile

```python
def compile(
    model: Any,
    target: str = "cpu",
    precision: str = "fp32",
    opt_level: int = 2,
    tolerance: float = 1e-6,
    sample_input: Any = None,
    **kwargs
) -> CompiledModel
```

Compile and optimize a model for the target platform.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | Required | Model from PyTorch, TensorFlow, JAX, or ONNX format |
| `target` | str | `"cpu"` | Target device specification |
| `precision` | str | `"fp32"` | Precision level for computations |
| `opt_level` | int | `2` | Optimization aggressiveness level (1-3) |
| `tolerance` | float | `1e-6` | Maximum relative error bound |
| `sample_input` | Any | `None` | Sample input for model tracing (required for PyTorch/JAX) |
| `**kwargs` | dict | - | Additional framework-specific options |

**Target Options:**

| Target | Description |
|--------|-------------|
| `"cpu"` | CPU with SIMD optimizations |
| `"cuda"` | Default CUDA device |
| `"cuda:0"` | Specific NVIDIA GPU (device 0) |
| `"cuda:1"` | NVIDIA GPU (device 1) |
| `"rocm:0"` | AMD GPU (planned) |
| `"tpu"` | Google TPU (planned) |

**Precision Options:**

| Precision | Description |
|-----------|-------------|
| `"fp32"` | Full precision (32-bit floating point) |
| `"fp16"` | Half precision (16-bit floating point) |
| `"bf16"` | Brain float 16 |
| `"int8"` | 8-bit integer quantization |

**Optimization Levels:**

| Level | Description |
|-------|-------------|
| `1` | Conservative - minimal transformations, fastest compilation |
| `2` | Standard - balanced optimization (default) |
| `3` | Aggressive - maximum optimization, longer compilation time |

**Returns:**

`CompiledModel` - An optimized model callable that can be used like the original model.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | Source framework is not installed |
| `ValueError` | Model format is not supported |
| `RuntimeError` | Optimization fails |

**Mathematical Guarantee:**

For tolerance parameter delta, the optimization guarantees:

```
|T(F)(x) - F(x)| / (|F(x)| + epsilon) <= delta
```

Where:
- `T(F)(x)` is the output of the optimized function
- `F(x)` is the output of the original function
- `epsilon` is a small constant to avoid division by zero

**Example:**

```python
import zenith
import torch

model = torch.nn.Linear(10, 5)
sample = torch.randn(1, 10)

optimized = zenith.compile(
    model,
    target="cuda:0",
    precision="fp16",
    opt_level=3,
    tolerance=1e-5,
    sample_input=sample
)

output = optimized(sample)
```

---

### optimize

```python
def optimize(model: Any, **kwargs) -> CompiledModel
```

Alias for `compile()`. Provided for convenience and API compatibility.

**Example:**

```python
optimized = zenith.optimize(model, target="cuda", precision="fp16")
```

---

### is_native

```python
def is_native() -> bool
```

Check if native C++ bindings are available.

When native bindings are available, Zenith uses high-performance C++ implementations for core operations. When not available, pure Python implementations are used as fallback.

**Returns:**

`bool` - True if native C++ bindings are loaded, False otherwise.

**Example:**

```python
if zenith.is_native():
    print("Using native C++ bindings")
else:
    print("Using pure Python implementations")
```

---

## Core Types

### DataType

```python
class DataType(Enum)
```

Enumeration of supported data types for tensors.

**Values:**

| Value | Description | Size (bytes) |
|-------|-------------|--------------|
| `DataType.Float32` | 32-bit floating point | 4 |
| `DataType.Float16` | 16-bit floating point | 2 |
| `DataType.BFloat16` | Brain float 16 | 2 |
| `DataType.Float64` | 64-bit floating point | 8 |
| `DataType.Int8` | 8-bit signed integer | 1 |
| `DataType.Int16` | 16-bit signed integer | 2 |
| `DataType.Int32` | 32-bit signed integer | 4 |
| `DataType.Int64` | 64-bit signed integer | 8 |
| `DataType.UInt8` | 8-bit unsigned integer | 1 |
| `DataType.Bool` | Boolean | 1 |

**Example:**

```python
from zenith import DataType

dtype = DataType.Float32
print(dtype.name)  # "Float32"
```

---

### Layout

```python
class Layout(Enum)
```

Memory layout for tensors.

**Values:**

| Value | Description | Typical Framework |
|-------|-------------|-------------------|
| `Layout.NCHW` | Batch, Channels, Height, Width | PyTorch default |
| `Layout.NHWC` | Batch, Height, Width, Channels | TensorFlow default |
| `Layout.NC` | Batch, Channels (1D) | - |

---

### StatusCode

```python
class StatusCode(Enum)
```

Result status codes for operations.

**Values:**

| Value | Description |
|-------|-------------|
| `StatusCode.Ok` | Operation completed successfully |
| `StatusCode.InvalidArgument` | Invalid argument provided |
| `StatusCode.NotFound` | Resource not found |
| `StatusCode.AlreadyExists` | Resource already exists |
| `StatusCode.OutOfMemory` | Memory allocation failed |
| `StatusCode.NotImplemented` | Feature not implemented |
| `StatusCode.InternalError` | Internal error occurred |
| `StatusCode.InvalidGraph` | Graph structure is invalid |
| `StatusCode.OptimizationFailed` | Optimization pass failed |

---

### Status

```python
@dataclass
class Status:
    code: StatusCode = StatusCode.Ok
    message: str = ""
```

Status class for operation results.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `ok()` | `bool` | Returns True if status code is Ok |
| `Ok()` | `Status` | Class method that creates an Ok status |
| `Error(code, message)` | `Status` | Class method that creates an error status |

**Example:**

```python
from zenith import Status, StatusCode

# Create success status
status = Status.Ok()
if status.ok():
    print("Success")

# Create error status
error = Status.Error(StatusCode.InvalidArgument, "Missing input")
print(error.message)  # "Missing input"
```

---

### Shape

```python
@dataclass
class Shape:
    dims: list[int] = field(default_factory=list)
```

Represents tensor dimensions.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `rank()` | `int` | Number of dimensions |
| `numel()` | `int` | Total number of elements |
| `is_dynamic()` | `bool` | True if shape has dynamic dimensions (negative values) |

**Example:**

```python
from zenith import Shape

shape = Shape([1, 3, 224, 224])  # Batch of 1, 3 channels, 224x224
print(shape.rank())   # 4
print(shape.numel())  # 150528
print(shape[0])       # 1
```

---

### TensorDescriptor

```python
@dataclass
class TensorDescriptor:
    name: str = ""
    shape: Shape = field(default_factory=Shape)
    dtype: DataType = DataType.Float32
    layout: Layout = Layout.NCHW
```

Describes a tensor's metadata without holding actual data.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `size_bytes()` | `int` | Calculate the size in bytes |
| `is_valid()` | `bool` | Check if tensor has a valid shape |

**Example:**

```python
from zenith import TensorDescriptor, Shape, DataType

tensor = TensorDescriptor(
    name="input",
    shape=Shape([1, 3, 224, 224]),
    dtype=DataType.Float32
)
print(tensor.size_bytes())  # 602112
```

---

### Node

```python
@dataclass
class Node:
    op_type: str = ""
    name: str = ""
    inputs: list[TensorDescriptor] = field(default_factory=list)
    outputs: list[TensorDescriptor] = field(default_factory=list)
    attrs: AttributeMap = field(default_factory=dict)
```

Represents a single operation (node) in the computation graph.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `num_inputs()` | `int` | Get number of inputs |
| `num_outputs()` | `int` | Get number of outputs |
| `is_op(op)` | `bool` | Check if this is a specific operation type |
| `add_input(tensor)` | `None` | Add an input tensor |
| `add_output(tensor)` | `None` | Add an output tensor |
| `set_attr(key, value)` | `None` | Set an attribute |
| `get_attr(key, default)` | `Any` | Get an attribute value |
| `has_attr(key)` | `bool` | Check if attribute exists |
| `clone()` | `Node` | Create a copy of this node |

---

### GraphIR

```python
@dataclass
class GraphIR:
    name: str = ""
    inputs: list[TensorDescriptor] = field(default_factory=list)
    outputs: list[TensorDescriptor] = field(default_factory=list)
    constants: dict[str, bytes] = field(default_factory=dict)
```

The unified intermediate representation for computation graphs.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_node(op_type, name, inputs, outputs, attrs)` | `Node` | Add a node to the graph |
| `get_node(name)` | `Optional[Node]` | Get node by name |
| `nodes` | `list[Node]` | Property to get all nodes |
| `num_nodes()` | `int` | Get number of nodes |
| `remove_node(name)` | `bool` | Remove a node by name |
| `set_inputs(inputs)` | `None` | Set graph input tensors |
| `set_outputs(outputs)` | `None` | Set graph output tensors |
| `add_input(tensor)` | `None` | Add a graph input |
| `add_output(tensor)` | `None` | Add a graph output |
| `add_constant(name, data)` | `None` | Add constant tensor data |
| `get_constant(name)` | `Optional[bytes]` | Get constant data |
| `find_nodes_by_op(op_type)` | `list[Node]` | Find all nodes of a specific operation type |
| `topological_order()` | `list[Node]` | Get nodes in topological order |
| `validate()` | `Status` | Validate the graph structure |
| `count_ops()` | `dict[str, int]` | Count nodes by operation type |
| `summary()` | `str` | Get a printable summary |
| `clone()` | `GraphIR` | Deep clone the graph |

**Example:**

```python
from zenith import GraphIR, TensorDescriptor, Shape, DataType

# Create a graph
graph = GraphIR(name="my_model")

# Add input
input_tensor = TensorDescriptor(
    name="x",
    shape=Shape([1, 10]),
    dtype=DataType.Float32
)
graph.add_input(input_tensor)

# Add output
output_tensor = TensorDescriptor(
    name="y",
    shape=Shape([1, 5]),
    dtype=DataType.Float32
)
graph.add_output(output_tensor)

# Validate
status = graph.validate()
if not status.ok():
    print(f"Validation failed: {status.message}")
```

---

## Framework Adapters

### PyTorchAdapter

```python
class PyTorchAdapter:
    is_available: bool
    config: ZenithPyTorchConfig
```

Enterprise-grade adapter for PyTorch models with full support for:
- PyTorch nn.Module and torch.jit models
- torch.compile backend integration (TorchDynamo)
- FX Graph capture and conversion (PyTorch 2.x)
- HuggingFace Transformers (PyTorch models)
- Training integration with Automatic Mixed Precision (AMP)

**Configuration:**

```python
@dataclass
class ZenithPyTorchConfig:
    target: str = "cuda"           # "cpu", "cuda", "cuda:0"
    precision: str = "fp32"        # "fp32", "fp16", "bf16", "int8"
    opt_level: int = 2             # 1-3
    opset_version: int = 17        # ONNX opset
    mode: str = "default"          # torch.compile mode
    fullgraph: bool = False        # Require full graph capture
    enable_amp: bool = False       # Automatic Mixed Precision
    gradient_checkpointing: bool = False
    tolerance: float = 1e-6
```

**Core Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(model, sample_input, **kwargs)` | `GraphIR` | Convert PyTorch model to GraphIR |
| `from_fx_graph(model, sample_input)` | `GraphIR` | Convert using FX Graph (PyTorch 2.x) |
| `from_transformers(model_name, task, **kwargs)` | `GraphIR` | Convert HuggingFace PyTorch model |
| `create_compile_backend(target, precision)` | `Callable` | Create torch.compile backend |
| `compile_function(func, target, precision)` | `Callable` | Compile function with optimizations |
| `wrap_training_step(fn, **kwargs)` | `Callable` | Wrap training step |
| `create_optimizer_wrapper(optimizer)` | `Wrapper` | Create optimized optimizer |
| `to_onnx(model, sample_input, **kwargs)` | `bytes` | Export to ONNX format |

**Basic Example:**

```python
from zenith import PyTorchAdapter
import torch

adapter = PyTorchAdapter()
model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10)
)
sample = torch.randn(1, 16)
graph_ir = adapter.from_model(model, sample)
```

**torch.compile Backend Integration:**

```python
import torch
import zenith.torch as ztorch

# Create Zenith backend for torch.compile
backend = ztorch.create_backend(target="cuda", precision="fp16")

# Use with torch.compile
model = torch.nn.Linear(10, 5)
compiled = torch.compile(model, backend=backend)

# Runs with Zenith optimizations
output = compiled(torch.randn(1, 10))
```

**HuggingFace Integration:**

```python
# Load HuggingFace PyTorch model directly
graph = adapter.from_transformers(
    "bert-base-uncased",
    task="text-classification",
    max_length=128
)
```

**Compilation Hook (like torch.compile):**

```python
import zenith.torch as ztorch

@ztorch.compile(target="cuda", precision="fp16")
def forward(x):
    return model(x)

# Compiled function runs with Zenith optimizations
output = forward(input_tensor)
```

**Training Integration:**

```python
import zenith.torch as ztorch

# Wrap training step with AMP
optimized_step = ztorch.wrap_training_step(
    train_step,
    enable_amp=True
)

# Create optimized optimizer wrapper
optimizer = torch.optim.Adam(model.parameters())
wrapped = ztorch.create_optimizer_wrapper(optimizer, enable_amp=True)
```

**Module-level API (zenith.torch):**

```python
import zenith.torch as ztorch

# Configuration
ztorch.configure(target="cuda", precision="fp16")

# Core functions
ztorch.compile(func)                 # Compile function
ztorch.create_backend()              # torch.compile backend
ztorch.from_model(model, sample)     # Convert model
ztorch.from_transformers(name)       # Load HuggingFace
ztorch.wrap_training_step(fn)        # Training integration
ztorch.to_onnx(model, sample)        # Export to ONNX
ztorch.is_available()                # Check PyTorch availability
ztorch.has_torch_compile()           # Check torch.compile support
```

---

### TensorFlowAdapter

```python
class TensorFlowAdapter:
    is_available: bool
    config: ZenithTFConfig
```

Enterprise-grade adapter for TensorFlow 2.x models with full support for:
- SavedModel and Keras models
- HuggingFace Transformers (TF models)
- `tf.function` compilation hook (like `torch.compile`)
- Inference and Training integration

**Configuration:**

```python
@dataclass
class ZenithTFConfig:
    target: str = "cuda"           # "cpu", "cuda", "cuda:0"
    precision: str = "fp32"        # "fp32", "fp16", "bf16", "int8"
    opt_level: int = 2             # 1-3
    opset_version: int = 17        # ONNX opset
    enable_gradient_optimization: bool = True
    enable_mixed_precision_training: bool = False
    gradient_checkpointing: bool = False
    tolerance: float = 1e-6
```

**Core Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(model, sample_input, **kwargs)` | `GraphIR` | Convert TensorFlow/Keras model to GraphIR |
| `from_saved_model(path, signature_key, **kwargs)` | `GraphIR` | Load and convert SavedModel |
| `from_transformers(model_name, task, **kwargs)` | `GraphIR` | Convert HuggingFace TF model |
| `compile_function(func, target, precision, **kwargs)` | `Callable` | Compile tf.function with optimizations |
| `create_training_callback(model, **kwargs)` | `Callback` | Create Keras training callback |
| `wrap_training_step(fn, model, optimizer, **kwargs)` | `Callable` | Wrap custom training step |
| `to_onnx(model, sample_input, **kwargs)` | `bytes` | Export to ONNX format |

**Basic Example:**

```python
from zenith import TensorFlowAdapter
import tensorflow as tf

adapter = TensorFlowAdapter()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
sample = tf.random.normal((1, 10))
graph_ir = adapter.from_model(model, sample_input=sample)
```

**HuggingFace Integration:**

```python
# Load HuggingFace TF model directly
graph = adapter.from_transformers(
    "bert-base-uncased",
    task="text-classification",
    max_length=128
)
```

**tf.function Compilation Hook (like torch.compile):**

```python
import zenith.tensorflow as ztf

@ztf.compile(target="cuda", precision="fp16")
@tf.function
def forward(x):
    return model(x)

# Compiled function runs with Zenith optimizations
output = forward(input_tensor)
```

**Training Integration:**

```python
# Create training callback for mixed precision
callback = adapter.create_training_callback(
    model,
    enable_mixed_precision=True
)
model.fit(X, y, callbacks=[callback.get_keras_callback()])

# Or wrap custom training loop
optimized_step = adapter.wrap_training_step(
    train_step,
    model,
    optimizer,
    enable_mixed_precision=True
)
```

**Module-level API (zenith.tensorflow):**

```python
import zenith.tensorflow as ztf

# Configuration
ztf.configure(target="cuda", precision="fp16")

# Core functions
ztf.compile(func)                  # Compile tf.function
ztf.compile_function(func)         # Same as compile
ztf.from_model(model)              # Convert model
ztf.from_transformers(name)        # Load HuggingFace model
ztf.create_training_callback()     # Training callback
ztf.wrap_training_step()           # Wrap training step
ztf.to_onnx(model)                 # Export to ONNX
ztf.is_available()                 # Check TF availability
```

---

### JAXAdapter

```python
class JAXAdapter:
    is_available: bool
    config: ZenithJAXConfig
```

Enterprise-grade adapter for JAX functions and Flax/Haiku models with full support for:
- Pure JAX functions with jax.jit
- Flax nn.Module models
- Haiku transformed functions
- HuggingFace Transformers (Flax models)
- Compilation hook (like torch.compile)
- Training state integration
- StableHLO native export

**Configuration:**

```python
@dataclass
class ZenithJAXConfig:
    target: str = "cuda"           # "cpu", "cuda", "tpu"
    precision: str = "fp32"        # "fp32", "fp16", "bf16", "int8"
    opt_level: int = 2             # 1-3
    opset_version: int = 17        # ONNX opset
    enable_xla: bool = True
    enable_donation: bool = False  # Buffer donation
    gradient_checkpointing: bool = False
    tolerance: float = 1e-6
```

**Core Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(fn, sample_input, params, **kwargs)` | `GraphIR` | Convert JAX function/Flax/Haiku to GraphIR |
| `from_flax_module(module, params, sample_input)` | `GraphIR` | Convert Flax nn.Module |
| `from_haiku(transformed_fn, params, sample_input)` | `GraphIR` | Convert Haiku function |
| `from_transformers(model_name, task, **kwargs)` | `GraphIR` | Convert HuggingFace Flax model |
| `from_stablehlo(model, sample_input, **kwargs)` | `GraphIR` | Export via StableHLO |
| `compile_function(func, target, precision)` | `Callable` | Compile with optimizations |
| `create_training_state(model, params, optimizer)` | `TrainState` | Create training state |
| `wrap_training_step(fn, **kwargs)` | `Callable` | Wrap training step |
| `to_onnx(model, sample_input, **kwargs)` | `bytes` | Export to ONNX format |

**Basic Example:**

```python
from zenith import JAXAdapter
import jax.numpy as jnp

adapter = JAXAdapter()

def mlp(x):
    w1 = jnp.ones((x.shape[-1], 32))
    return jnp.maximum(jnp.dot(x, w1), 0)

sample = jnp.ones((1, 16))
graph_ir = adapter.from_model(mlp, sample_input=sample)
```

**Flax nn.Module Integration:**

```python
from flax import linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

model = MLP()
params = model.init(jax.random.PRNGKey(0), sample)["params"]
graph = adapter.from_flax_module(model, params, sample)
```

**HuggingFace Flax Integration:**

```python
# Load HuggingFace Flax model directly
graph = adapter.from_transformers(
    "bert-base-uncased",
    task="text-classification",
    max_length=128
)
```

**Compilation Hook (like torch.compile):**

```python
import zenith.jax as zjax

@zjax.compile(target="cuda", precision="fp16")
@jax.jit
def forward(x):
    return model.apply(params, x)

# Compiled function runs with Zenith optimizations
output = forward(input_tensor)
```

**Training State Integration:**

```python
import optax

# Create Zenith training state
state = adapter.create_training_state(
    model,
    params,
    optax.adam(1e-4),
    enable_gradient_checkpointing=True
)

# Training loop
for batch in dataloader:
    grads = compute_gradients(state, batch)
    state = state.apply_gradients(grads)
```

**Module-level API (zenith.jax):**

```python
import zenith.jax as zjax

# Configuration
zjax.configure(target="cuda", precision="fp16")

# Core functions
zjax.compile(func)                  # Compile JAX function
zjax.from_model(fn, sample)         # Convert function
zjax.from_flax_module(m, p, s)      # Convert Flax module
zjax.from_haiku(fn, p, s)           # Convert Haiku
zjax.from_transformers(name)        # Load HuggingFace
zjax.create_training_state()        # Training state
zjax.to_onnx(fn, sample)            # Export to ONNX
zjax.is_available()                 # Check JAX availability
```


### ONNXAdapter

```python
class ONNXAdapter:
    is_available: bool
```

Adapter for loading ONNX models as GraphIR.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(model_or_path, **kwargs)` | `GraphIR` | Load ONNX model as GraphIR |

**Example:**

```python
from zenith import ONNXAdapter

adapter = ONNXAdapter()
graph_ir = adapter.from_model("model.onnx")
print(graph_ir.summary())
```

---

## Optimization Passes

### PassManager

```python
class PassManager:
    def __init__(self) -> None
```

Manages and orchestrates optimization passes.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_pass(opt_pass)` | `None` | Add an optimization pass |
| `run(graph, max_iterations)` | `tuple[GraphIR, dict]` | Run all registered passes |

**Example:**

```python
from zenith.optimization import (
    PassManager,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass
)

pm = PassManager()
pm.add_pass(ConstantFoldingPass())
pm.add_pass(DeadCodeEliminationPass())
pm.add_pass(OperatorFusionPass())

optimized_graph, stats = pm.run(graph_ir, max_iterations=10)

print(f"Optimization stats: {stats}")
```

---

### ConstantFoldingPass

```python
class ConstantFoldingPass(OptimizationPass)
```

Evaluates operations with all-constant inputs at compile time, replacing the operation node with a constant result.

**Foldable Operations:**

- Add, Sub, Mul, Div
- Neg
- Reshape, Transpose
- Cast, Identity

---

### DeadCodeEliminationPass

```python
class DeadCodeEliminationPass(OptimizationPass)
```

Removes nodes whose outputs are never used by any other node or by the graph outputs. Reduces memory usage and computation by eliminating unused work.

---

### OperatorFusionPass

```python
class OperatorFusionPass(OptimizationPass)
```

Combines sequences of operations into single fused operations to reduce memory bandwidth and kernel launch overhead.

**Supported Fusion Patterns:**

| Pattern | Sequence | Fused Operation |
|---------|----------|-----------------|
| conv_relu | Conv, Relu | ConvRelu |
| bn_relu | BatchNormalization, Relu | BnRelu |
| matmul_add | MatMul, Add | Gemm |
| conv_bn | Conv, BatchNormalization | ConvBn |

---

## Execution Engine

### ONNXInterpreter

```python
class ONNXInterpreter:
    def __init__(
        self,
        graph_ir: GraphIR,
        device: str = "cuda",
        strict: bool = False
    ) -> None
```

Executes ONNX graphs using Zenith CUDA operations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_ir` | `GraphIR` | Required | GraphIR representation of the model |
| `device` | `str` | `"cuda"` | Target device ("cuda" or "cpu") |
| `strict` | `bool` | `False` | If True, raise error for unsupported ops |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_fully_supported` | `bool` | Check if all operators are supported |
| `unsupported_operators` | `list[str]` | Get list of unsupported operator types |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__call__(**inputs)` | `dict[str, ndarray]` | Execute the graph with given inputs |
| `execute_with_timing(**inputs)` | `tuple[dict, dict]` | Execute with per-node timing |
| `summary()` | `str` | Get a summary of the interpreter state |

**Example:**

```python
from zenith.execution import ONNXInterpreter
import numpy as np

interpreter = ONNXInterpreter(graph_ir, device="cuda")

if interpreter.is_fully_supported:
    outputs = interpreter(input=np.random.randn(1, 3, 224, 224).astype(np.float32))
    print(outputs)
```

---

### ExecutionContext

```python
class ExecutionContext:
    def __init__(self, device: str = "cuda") -> None
```

Manages tensor storage during graph execution.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `set_tensor(name, value)` | `None` | Store a tensor value |
| `get_tensor(name)` | `ndarray` | Retrieve a tensor value |
| `has_tensor(name)` | `bool` | Check if tensor exists |
| `clear()` | `None` | Clear all stored tensors |

---

## Utility Functions

### dtype_size

```python
def dtype_size(dtype: DataType) -> int
```

Get the size in bytes for a data type.

**Example:**

```python
from zenith import DataType, dtype_size

size = dtype_size(DataType.Float32)  # Returns 4
```

---

### dtype_to_string

```python
def dtype_to_string(dtype: DataType) -> str
```

Get string representation of data type.

**Example:**

```python
from zenith import DataType, dtype_to_string

name = dtype_to_string(DataType.Float16)  # Returns "float16"
```

---

### get_version

```python
def get_version() -> str
```

Get the Zenith version string.

---

## Constants and Enumerations

### Operation Types (Ops)

Standard operation type names matching ONNX conventions.

```python
from zenith.core.node import Ops

# Activation operations
Ops.RELU          # "Relu"
Ops.GELU          # "Gelu"
Ops.SIGMOID       # "Sigmoid"
Ops.TANH          # "Tanh"
Ops.SOFTMAX       # "Softmax"

# Linear operations
Ops.MATMUL        # "MatMul"
Ops.GEMM          # "Gemm"
Ops.LINEAR        # "Linear"

# Convolution operations
Ops.CONV          # "Conv"
Ops.CONV_TRANSPOSE  # "ConvTranspose"

# Normalization operations
Ops.BATCH_NORM    # "BatchNormalization"
Ops.LAYER_NORM    # "LayerNormalization"
Ops.INSTANCE_NORM # "InstanceNormalization"

# Pooling operations
Ops.MAX_POOL      # "MaxPool"
Ops.AVG_POOL      # "AveragePool"
Ops.GLOBAL_AVG_POOL  # "GlobalAveragePool"

# Element-wise operations
Ops.ADD           # "Add"
Ops.SUB           # "Sub"
Ops.MUL           # "Mul"
Ops.DIV           # "Div"

# Shape operations
Ops.RESHAPE       # "Reshape"
Ops.TRANSPOSE     # "Transpose"
Ops.FLATTEN       # "Flatten"
Ops.CONCAT        # "Concat"

# Special
Ops.IDENTITY      # "Identity"
Ops.CONSTANT      # "Constant"
```

---

## Error Handling

Zenith uses a combination of exceptions and status codes for error handling.

### Exceptions

| Exception | Description |
|-----------|-------------|
| `ImportError` | Framework dependency not installed |
| `ValueError` | Invalid argument or unsupported format |
| `RuntimeError` | Execution or compilation failure |
| `NotImplementedError` | Feature not yet implemented |

### Status-Based Errors

For operations that return Status objects, check the status code:

```python
status = graph.validate()
if not status.ok():
    if status.code == StatusCode.InvalidGraph:
        # Handle invalid graph
        pass
    elif status.code == StatusCode.InvalidArgument:
        # Handle invalid argument
        pass
    print(f"Error: {status.message}")
```

---

## Examples

### Complete PyTorch Workflow

```python
import zenith
import torch
import torch.nn as nn

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create and compile
model = SimpleNet()
sample = torch.randn(1, 784)

optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp16",
    opt_level=3,
    sample_input=sample
)

# Inference
with torch.no_grad():
    output = optimized(sample.cuda())
    prediction = output.argmax(dim=1)
```

### ONNX Model Optimization Pipeline

```python
import zenith
from zenith import ONNXAdapter
from zenith.optimization import (
    PassManager,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass
)

# Load ONNX model
adapter = ONNXAdapter()
graph_ir = adapter.from_model("resnet50.onnx")

# Configure optimization passes
pm = PassManager()
pm.add_pass(ConstantFoldingPass())
pm.add_pass(DeadCodeEliminationPass())
pm.add_pass(OperatorFusionPass())

# Run optimization
optimized_graph, stats = pm.run(graph_ir, max_iterations=10)

# Print results
print(f"Original nodes: {graph_ir.num_nodes()}")
print(f"Optimized nodes: {optimized_graph.num_nodes()}")
print(f"Optimization passes applied: {stats}")
```

### Building a Custom Graph

```python
from zenith import GraphIR, TensorDescriptor, Shape, DataType

# Create empty graph
graph = GraphIR(name="custom_model")

# Define tensors
input_tensor = TensorDescriptor("x", Shape([1, 10]), DataType.Float32)
hidden_tensor = TensorDescriptor("h", Shape([1, 64]), DataType.Float32)
output_tensor = TensorDescriptor("y", Shape([1, 5]), DataType.Float32)

# Set graph inputs and outputs
graph.add_input(input_tensor)
graph.add_output(output_tensor)

# Add nodes
graph.add_node(
    op_type="MatMul",
    name="matmul_1",
    inputs=[input_tensor],
    outputs=[hidden_tensor],
    attrs={}
)

graph.add_node(
    op_type="Relu",
    name="relu_1",
    inputs=[hidden_tensor],
    outputs=[hidden_tensor],
    attrs={}
)

graph.add_node(
    op_type="MatMul",
    name="matmul_2",
    inputs=[hidden_tensor],
    outputs=[output_tensor],
    attrs={}
)

# Validate and summarize
status = graph.validate()
if status.ok():
    print(graph.summary())
else:
    print(f"Validation error: {status.message}")
```

---

## Hardware Backends

### Overview

Zenith provides a unified Hardware Abstraction Layer (HAL) supporting multiple hardware backends:

- **CPU**: Universal fallback, always available
- **CUDA**: NVIDIA GPU acceleration
- **ROCm**: AMD GPU acceleration (HIP runtime)
- **oneAPI**: Intel GPU/CPU acceleration (SYCL runtime)

### Backend Base Class

```python
class BaseBackend(ABC):
    name: str              # Backend identifier
    backend_type: BackendType  # Enum (CPU, CUDA, ROCM, ONEAPI)
    device_id: int         # Device index

    # Core methods
    def is_available() -> bool
    def initialize() -> bool
    def cleanup() -> None
    def get_device_properties() -> DeviceProperties

    # Memory management
    def allocate(size_bytes: int) -> int
    def deallocate(ptr: int) -> None
    def copy_to_device(dst, src, size_bytes) -> None
    def copy_to_host(dst, src, size_bytes) -> None

    # Synchronization
    def synchronize() -> None
```

### Availability Functions

```python
from zenith.backends import (
    is_cpu_available,    # Always True
    is_cuda_available,   # NVIDIA GPU
    is_rocm_available,   # AMD GPU
    is_oneapi_available, # Intel GPU/CPU
    get_available_backends,  # Returns list of names
)

# Example
print(get_available_backends())  # ['cpu', 'cuda']
```

### Device Management

```python
from zenith.backends import (
    get_device,
    set_device,
    list_devices,
    synchronize,
)

# Get specific device
cuda_device = get_device("cuda:0")
rocm_device = get_device("rocm:0")
cpu_device = get_device("cpu:0")

# List all available devices
devices = list_devices()  # ['cuda:0', 'cuda:1', 'cpu:0']

# Set default device
set_device("cuda:0")

# Synchronize device
synchronize("cuda:0")
```

### Backend Classes

#### CUDABackend

```python
from zenith.backends import CUDABackend

backend = CUDABackend(device_id=0)
if backend.is_available():
    backend.initialize()

    # Get device properties
    props = backend.get_device_properties()
    print(f"Device: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute: {props.compute_capability}")

    # Allocate memory
    ptr = backend.allocate(1024 * 1024)  # 1 MB

    # Copy data
    import numpy as np
    data = np.random.randn(1000).astype(np.float32)
    backend.copy_to_device(ptr, data, data.nbytes)

    backend.synchronize()
    backend.deallocate(ptr)
    backend.cleanup()
```

#### ROCmBackend

```python
from zenith.backends import ROCmBackend

backend = ROCmBackend(device_id=0)
if backend.is_available():
    backend.initialize()
    props = backend.get_device_properties()
    print(f"AMD GPU: {props.name}")
    print(f"Wavefront size: {props.warp_size}")  # 64 for AMD
```

#### OneAPIBackend

```python
from zenith.backends import OneAPIBackend

# GPU mode
gpu_backend = OneAPIBackend(device_id=0, device_type="gpu")

# CPU mode with SYCL
cpu_backend = OneAPIBackend(device_id=0, device_type="cpu")

if gpu_backend.is_available():
    gpu_backend.initialize()
    props = gpu_backend.get_device_properties()
    print(f"Intel device: {props.name}")
```

### Backend Registry

```python
from zenith.backends import BackendRegistry

registry = BackendRegistry()

# List registered backends
print(registry.list_backends())  # ['cpu', 'cuda', 'rocm', 'oneapi']

# Get backend by device string
backend = registry.get("cuda:0", auto_init=True)

# Get default (first available)
default = registry.get_default()

# Set fallback chain
registry.set_fallback_chain(["cuda", "rocm", "oneapi", "cpu"])
```

### DeviceProperties

```python
@dataclass
class DeviceProperties:
    name: str                    # Device name
    vendor: str                  # Vendor (NVIDIA, AMD, Intel)
    backend_type: BackendType    # Backend enum
    device_id: int               # Device index
    total_memory: int            # Total memory (bytes)
    free_memory: int             # Available memory (bytes)
    compute_capability: tuple    # (major, minor)
    max_threads_per_block: int
    warp_size: int               # 32 (NVIDIA), 64 (AMD)
    multiprocessor_count: int
    supports_fp16: bool
    supports_bf16: bool
    supports_fp64: bool
    is_available: bool
```

### Context Manager

```python
from zenith.backends import CUDABackend

with CUDABackend(device_id=0) as backend:
    ptr = backend.allocate(1024)
    # ... operations ...
    backend.deallocate(ptr)
# Backend automatically cleaned up
```

### Error Handling

```python
from zenith.backends import (
    BackendError,
    BackendNotAvailableError,
    BackendMemoryError,
    BackendExecutionError,
    create_backend,
)

try:
    backend = create_backend("cuda:0")
except BackendNotAvailableError:
    print("CUDA not available, falling back to CPU")
    backend = create_backend("cpu:0")

try:
    ptr = backend.allocate(1024 * 1024 * 1024 * 100)  # 100 GB
except BackendMemoryError as e:
    print(f"Allocation failed: {e}")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.4 | December 2024 | Current stable release |
| 0.1.3 | December 2024 | Added ONNX interpreter, FP16 support |
| 0.1.2 | December 2024 | Framework adapters, optimization passes |
| 0.1.1 | December 2024 | Core types and GraphIR implementation |
| 0.1.0 | December 2024 | Initial release |

---

## License

Apache License 2.0

Copyright 2025 Wahyu Ardiansyah

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
