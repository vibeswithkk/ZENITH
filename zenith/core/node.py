# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Node (Pure Python Implementation)

Represents a single operation in the computation graph.
"""

from dataclasses import dataclass, field
from typing import Any
import itertools

from .types import AttributeMap
from .tensor import TensorDescriptor


# Node ID counter
_node_id_counter = itertools.count()


@dataclass
class Node:
    """
    Represents a single operation (node) in the computation graph.

    Based on the class diagram in section 4.4 of the blueprint.
    """

    op_type: str = ""
    name: str = ""
    inputs: list[TensorDescriptor] = field(default_factory=list)
    outputs: list[TensorDescriptor] = field(default_factory=list)
    attrs: AttributeMap = field(default_factory=dict)

    # Auto-generated ID
    id: int = field(default_factory=lambda: next(_node_id_counter), init=False)

    def num_inputs(self) -> int:
        """Get number of inputs."""
        return len(self.inputs)

    def num_outputs(self) -> int:
        """Get number of outputs."""
        return len(self.outputs)

    def is_op(self, op: str) -> bool:
        """Check if this is a specific operation type."""
        return self.op_type == op

    def add_input(self, tensor: TensorDescriptor) -> None:
        """Add an input tensor."""
        self.inputs.append(tensor)

    def add_output(self, tensor: TensorDescriptor) -> None:
        """Add an output tensor."""
        self.outputs.append(tensor)

    def set_attr(self, key: str, value: Any) -> None:
        """Set an attribute."""
        self.attrs[key] = value

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attrs.get(key, default)

    def has_attr(self, key: str) -> bool:
        """Check if attribute exists."""
        return key in self.attrs

    def clone(self) -> "Node":
        """Create a copy of this node."""
        return Node(
            op_type=self.op_type,
            name=self.name,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            attrs=dict(self.attrs),
        )

    def __repr__(self) -> str:
        return f"Node(op='{self.op_type}', name='{self.name}')"


# Common operation type constants
class Ops:
    """Standard operation type names (matching ONNX conventions)."""

    # Activation operations
    RELU = "Relu"
    GELU = "Gelu"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SOFTMAX = "Softmax"

    # Linear operations
    MATMUL = "MatMul"
    GEMM = "Gemm"
    LINEAR = "Linear"

    # Convolution operations
    CONV = "Conv"
    CONV_TRANSPOSE = "ConvTranspose"

    # Normalization operations
    BATCH_NORM = "BatchNormalization"
    LAYER_NORM = "LayerNormalization"
    INSTANCE_NORM = "InstanceNormalization"

    # Pooling operations
    MAX_POOL = "MaxPool"
    AVG_POOL = "AveragePool"
    GLOBAL_AVG_POOL = "GlobalAveragePool"

    # Element-wise operations
    ADD = "Add"
    SUB = "Sub"
    MUL = "Mul"
    DIV = "Div"

    # Shape operations
    RESHAPE = "Reshape"
    TRANSPOSE = "Transpose"
    FLATTEN = "Flatten"
    CONCAT = "Concat"

    # Special
    IDENTITY = "Identity"
    CONSTANT = "Constant"
