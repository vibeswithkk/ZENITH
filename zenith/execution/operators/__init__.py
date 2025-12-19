# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
ONNX Operator Implementations

This package contains implementations of ONNX operators
that execute using Zenith's CUDA operations.

Operators are organized by category:
- math_ops: MatMul, Gemm, Add, Sub, Mul, Div, Erf
- activation_ops: Relu, Sigmoid, Tanh, Softmax, Gelu
- conv_ops: Conv, BatchNormalization, MaxPool, AvgPool
- shape_ops: Reshape, Transpose, Flatten, Squeeze, Unsqueeze
- norm_ops: LayerNormalization, InstanceNormalization, GroupNormalization
- tensor_ops: Gather, Cast, Split, Concat, Slice, Pad, Where
"""

# Import all operator modules to register them
from . import math_ops
from . import activation_ops
from . import conv_ops
from . import shape_ops
from . import norm_ops
from . import tensor_ops

__all__ = [
    "math_ops",
    "activation_ops",
    "conv_ops",
    "shape_ops",
    "norm_ops",
    "tensor_ops",
]
