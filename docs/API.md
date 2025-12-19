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
```

Adapter for converting PyTorch models to GraphIR.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(model, sample_input, **kwargs)` | `GraphIR` | Convert PyTorch model to GraphIR |

**Example:**

```python
from zenith import PyTorchAdapter
import torch

adapter = PyTorchAdapter()
if adapter.is_available:
    model = torch.nn.Linear(10, 5)
    sample = torch.randn(1, 10)
    graph_ir = adapter.from_model(model, sample)
```

---

### TensorFlowAdapter

```python
class TensorFlowAdapter:
    is_available: bool
```

Adapter for converting TensorFlow/Keras models to GraphIR.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(model, sample_input, **kwargs)` | `GraphIR` | Convert TensorFlow model to GraphIR |

---

### JAXAdapter

```python
class JAXAdapter:
    is_available: bool
```

Adapter for converting JAX functions to GraphIR.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(fn, sample_input, **kwargs)` | `GraphIR` | Convert JAX function to GraphIR |

---

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
