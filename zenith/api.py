# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Public API

Provides the main entry points for compiling and optimizing ML models.
"""

from typing import Any

from .core import GraphIR
from .adapters import (
    PyTorchAdapter,
    TensorFlowAdapter,
    JAXAdapter,
    ONNXAdapter,
)


def compile(
    model: Any,
    target: str = "cpu",
    precision: str = "fp32",
    opt_level: int = 2,
    tolerance: float = 1e-6,
    sample_input: Any = None,
    **kwargs,
) -> Any:
    """
    Compile and optimize a model for the target platform.

    This is the main entry point for Zenith. It automatically detects
    the source framework, converts the model to GraphIR, applies
    optimizations, and produces an optimized model for the target.

    Args:
        model: Model from PyTorch, TensorFlow, JAX, or ONNX format.
        target: Target device specification.
            - "cpu" - CPU with SIMD optimizations
            - "cuda:0" - NVIDIA GPU (device 0)
            - "cuda" - Default CUDA device
            - "rocm:0" - AMD GPU (not implemented yet)
            - "tpu" - Google TPU (not implemented yet)
        precision: Precision level for computations.
            - "fp32" - Full precision (default)
            - "fp16" - Half precision
            - "bf16" - Brain float 16
            - "int8" - 8-bit integer quantization
        opt_level: Optimization aggressiveness (1-3).
            - 1: Conservative, minimal transformations
            - 2: Standard optimizations (default)
            - 3: Aggressive, may increase compile time
        tolerance: Maximum relative error bound (delta from section 4.1).
            The optimization guarantees that output deviation is bounded:
            |T(F)(x) - F(x)| / (|F(x)| + epsilon) <= tolerance
        sample_input: Sample input for model tracing (required for PyTorch/JAX).
        **kwargs: Additional framework-specific options.

    Returns:
        Optimized model. The return type depends on the target:
        - For CPU/CUDA: Callable that can be used like the original model
        - For export: Optimized GraphIR or serialized format

    Raises:
        ImportError: If the source framework is not installed.
        ValueError: If the model format is not supported.
        RuntimeError: If optimization fails.

    Example:
        >>> import zenith
        >>> import torch
        >>>
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 64),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(64, 5)
        ... )
        >>> sample = torch.randn(1, 10)
        >>> optimized = zenith.compile(model, target="cuda", sample_input=sample)
    """
    # Step 1: Detect framework and convert to GraphIR
    graph_ir = _convert_to_graphir(model, sample_input, **kwargs)

    # Step 2: Apply optimizations based on opt_level
    optimized_ir = _optimize_graph(
        graph_ir,
        opt_level=opt_level,
        precision=precision,
        tolerance=tolerance,
    )

    # Step 3: Compile for target
    compiled = _compile_for_target(optimized_ir, target)

    return compiled


def optimize(model: Any, **kwargs) -> Any:
    """
    Alias for compile().

    Provided for convenience and API compatibility.
    """
    return compile(model, **kwargs)


def _detect_framework(model: Any) -> str:
    """Detect which ML framework the model comes from."""
    model_type = type(model).__module__

    if "torch" in model_type:
        return "pytorch"
    elif "tensorflow" in model_type or "keras" in model_type:
        return "tensorflow"
    elif "jax" in model_type or "flax" in model_type:
        return "jax"
    elif "onnx" in model_type:
        return "onnx"
    elif isinstance(model, (str, bytes)):
        # Could be file path or serialized ONNX
        return "onnx"
    else:
        raise ValueError(
            f"Unable to detect framework for model type: {type(model)}. "
            "Supported frameworks: PyTorch, TensorFlow, JAX, ONNX"
        )


def _convert_to_graphir(model: Any, sample_input: Any = None, **kwargs) -> GraphIR:
    """Convert model from any framework to GraphIR."""
    framework = _detect_framework(model)

    if framework == "pytorch":
        adapter = PyTorchAdapter()
        if not adapter.is_available:
            raise ImportError("PyTorch is not installed")
        return adapter.from_model(model, sample_input, **kwargs)

    elif framework == "tensorflow":
        adapter = TensorFlowAdapter()
        if not adapter.is_available:
            raise ImportError("TensorFlow is not installed")
        return adapter.from_model(model, sample_input, **kwargs)

    elif framework == "jax":
        adapter = JAXAdapter()
        if not adapter.is_available:
            raise ImportError("JAX is not installed")
        return adapter.from_model(model, sample_input, **kwargs)

    elif framework == "onnx":
        adapter = ONNXAdapter()
        return adapter.from_model(model, **kwargs)

    else:
        raise ValueError(f"Unsupported framework: {framework}")


def _optimize_graph(
    graph_ir: GraphIR,
    opt_level: int,
    precision: str,
    tolerance: float,
) -> GraphIR:
    """
    Apply optimization passes to the graph.

    This is a placeholder for the full optimization pipeline.
    Full implementation will include:
    - Graph simplification (dead code elimination, constant folding)
    - Operator fusion (Conv-BN-ReLU, MatMul-Add)
    - Precision conversion (FP32 -> FP16/INT8)
    - Layout optimization (NCHW <-> NHWC)
    """
    # For now, return the graph as-is
    # Full optimization will be implemented in Phase 2
    return graph_ir


def _compile_for_target(graph_ir: GraphIR, target: str) -> Any:
    """
    Compile GraphIR for the specified target.

    This is a placeholder for the full compilation pipeline.
    Full implementation will:
    - Select appropriate backend (CPU, CUDA, etc.)
    - Generate optimized kernels
    - Create executable or callable
    """
    # Parse target
    if target.startswith("cuda"):
        backend = "cuda"
        if ":" in target:
            # Parse device ID for future use
            _ = int(target.split(":")[1])
    elif target.startswith("rocm"):
        backend = "rocm"
        raise NotImplementedError("ROCm backend not implemented yet")
    elif target == "tpu":
        backend = "tpu"
        raise NotImplementedError("TPU backend not implemented yet")
    else:
        backend = "cpu"

    # For now, return the GraphIR with compilation metadata
    # Full compilation will be implemented in Phase 1
    return CompiledModel(graph_ir, backend, target)


class CompiledModel:
    """
    Represents a compiled and optimized model.

    This is a placeholder that will be replaced with actual
    compiled execution logic in Phase 1.
    """

    def __init__(self, graph_ir: GraphIR, backend: str, target: str):
        self.graph_ir = graph_ir
        self.backend = backend
        self.target = target

    def __call__(self, *args, **kwargs):
        """Execute the compiled model."""
        raise NotImplementedError(
            "Model execution not implemented yet. "
            "This will be available in Phase 1 with backend support."
        )

    def __repr__(self) -> str:
        return (
            f"CompiledModel("
            f"graph='{self.graph_ir.name}', "
            f"nodes={self.graph_ir.num_nodes()}, "
            f"backend='{self.backend}')"
        )

    def summary(self) -> str:
        """Get a summary of the compiled model."""
        return self.graph_ir.summary()
