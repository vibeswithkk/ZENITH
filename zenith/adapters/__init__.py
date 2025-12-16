# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Framework Adapters Module

Provides adapters for converting models from various frameworks
(PyTorch, TensorFlow, JAX) to Zenith's unified GraphIR format.
"""

from .base import BaseAdapter
from .pytorch_adapter import PyTorchAdapter
from .tensorflow_adapter import TensorFlowAdapter
from .jax_adapter import JAXAdapter
from .onnx_adapter import ONNXAdapter

__all__ = [
    "BaseAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "JAXAdapter",
    "ONNXAdapter",
]
