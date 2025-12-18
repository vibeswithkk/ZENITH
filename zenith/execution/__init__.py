# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Execution Engine

This module provides the ONNX graph interpreter that executes
computation graphs using Zenith's CUDA operations.

Components:
- ExecutionContext: Manages tensor buffers during execution
- OperatorRegistry: Maps ONNX op_type to kernel functions
- ONNXInterpreter: Full graph execution engine
"""

from .context import ExecutionContext
from .registry import OperatorRegistry
from .interpreter import ONNXInterpreter

__all__ = [
    "ExecutionContext",
    "OperatorRegistry",
    "ONNXInterpreter",
]
