# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Runtime Module

The runtime is responsible for executing optimized GraphIR using
Zenith's high-performance CUDA kernels.

Architecture (inspired by TensorRT, ONNX Runtime, TVM):
- ZenithEngine: Compiles GraphIR into executable model
- GraphExecutor: Executes compiled model
- KernelDispatcher: Routes operations to appropriate kernels
- KernelRegistry: Registry of all available kernels
- ExecutionContext: Holds execution state
- MemoryManager: Manages GPU memory

Example:
    from zenith.runtime import ZenithEngine

    engine = ZenithEngine(backend="cuda")
    compiled = engine.compile(graph_ir, config)
    output = compiled(input_tensor)
"""

from .kernel_registry import KernelRegistry, KernelSpec
from .context import ExecutionContext
from .dispatcher import KernelDispatcher
from .memory_manager import MemoryManager
from .executor import GraphExecutor
from .engine import ZenithEngine, CompileConfig, CompiledModel

__all__ = [
    "ZenithEngine",
    "CompileConfig",
    "CompiledModel",
    "GraphExecutor",
    "KernelDispatcher",
    "KernelRegistry",
    "KernelSpec",
    "ExecutionContext",
    "MemoryManager",
]
