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
    graph_ir, original_model = _convert_to_graphir(model, sample_input, **kwargs)

    # Step 2: Apply optimizations based on opt_level
    optimized_ir = _optimize_graph(
        graph_ir,
        opt_level=opt_level,
        precision=precision,
        tolerance=tolerance,
    )

    # Step 3: Compile for target (pass original model for execution)
    compiled = _compile_for_target(optimized_ir, target, original_model)

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


def _convert_to_graphir(
    model: Any, sample_input: Any = None, **kwargs
) -> tuple[GraphIR, Any]:
    """Convert model from any framework to GraphIR.

    Returns:
        Tuple of (GraphIR, original_model) - keeps original for execution
    """
    framework = _detect_framework(model)

    if framework == "pytorch":
        adapter = PyTorchAdapter()
        if not adapter.is_available:
            raise ImportError("PyTorch is not installed")
        graph_ir = adapter.from_model(model, sample_input, **kwargs)
        return graph_ir, model

    elif framework == "tensorflow":
        adapter = TensorFlowAdapter()
        if not adapter.is_available:
            raise ImportError("TensorFlow is not installed")
        graph_ir = adapter.from_model(model, sample_input, **kwargs)
        return graph_ir, model

    elif framework == "jax":
        adapter = JAXAdapter()
        if not adapter.is_available:
            raise ImportError("JAX is not installed")
        graph_ir = adapter.from_model(model, sample_input, **kwargs)
        return graph_ir, model

    elif framework == "onnx":
        adapter = ONNXAdapter()
        graph_ir = adapter.from_model(model, **kwargs)
        return graph_ir, model

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


def _compile_for_target(graph_ir: GraphIR, target: str, original_model: Any) -> Any:
    """
    Compile GraphIR for the specified target.

    Args:
        graph_ir: The optimized graph IR.
        target: Target device specification.
        original_model: Original model for execution fallback.

    Returns:
        CompiledModel that can be called like the original model.
    """
    # Parse target
    if target.startswith("cuda"):
        backend = "cuda"
        device_id = 0
        if ":" in target:
            device_id = int(target.split(":")[1])
    elif target.startswith("rocm"):
        backend = "rocm"
        raise NotImplementedError("ROCm backend not implemented yet")
    elif target == "tpu":
        backend = "tpu"
        raise NotImplementedError("TPU backend not implemented yet")
    else:
        backend = "cpu"
        device_id = 0

    return CompiledModel(graph_ir, backend, target, original_model)


class CompiledModel:
    """
    Represents a compiled and optimized model.

    Wraps the original model and provides Zenith optimization metadata.
    Currently delegates execution to the original model, with Zenith
    optimizations to be added incrementally.
    """

    def __init__(
        self,
        graph_ir: GraphIR,
        backend: str,
        target: str,
        original_model: Any = None,
    ):
        self.graph_ir = graph_ir
        self.backend = backend
        self.target = target
        self._original_model = original_model
        self._framework = None  # Lazy detection

    def _detect_framework(self) -> str:
        """Detect which ML framework the original model uses."""
        if self._original_model is None:
            return "unknown"
        module = type(self._original_model).__module__
        if "torch" in module:
            return "pytorch"
        elif "tensorflow" in module or "keras" in module:
            return "tensorflow"
        elif "jax" in module or "flax" in module or "haiku" in module:
            return "jax"
        else:
            return "unknown"

    def __call__(self, *args, **kwargs):
        """
        Execute the compiled model.

        Currently delegates to the original model. Future versions will
        use Zenith's optimized execution path.
        """
        if self._original_model is None:
            raise RuntimeError(
                "No original model available for execution. "
                "Provide sample_input during compilation."
            )

        framework = self._detect_framework()

        if framework == "pytorch":
            return self._execute_pytorch(*args, **kwargs)
        elif framework == "tensorflow":
            return self._execute_tensorflow(*args, **kwargs)
        elif framework == "jax":
            return self._execute_jax(*args, **kwargs)
        else:
            # Generic fallback
            return self._original_model(*args, **kwargs)

    def _execute_pytorch(self, *args, **kwargs):
        """Execute PyTorch model with device handling."""
        import torch

        # Move to target device if needed
        if self.backend == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            self._original_model.to(device)
            args = tuple(a.to(device) if hasattr(a, "to") else a for a in args)

        with torch.no_grad():
            return self._original_model(*args, **kwargs)

    def _execute_tensorflow(self, *args, **kwargs):
        """Execute TensorFlow/Keras model with device handling."""
        import tensorflow as tf

        # Handle device placement
        if self.backend == "cuda":
            device = "/GPU:0"
        else:
            device = "/CPU:0"

        with tf.device(device):
            return self._original_model(*args, **kwargs)

    def _execute_jax(self, *args, **kwargs):
        """Execute JAX function with device handling."""
        import jax

        # JAX automatically uses available accelerators
        # For explicit device control, we could use jax.devices()
        if self.backend == "cuda":
            # Ensure JAX uses GPU if available
            devices = jax.devices("gpu")
            if devices:
                with jax.default_device(devices[0]):
                    return self._original_model(*args, **kwargs)

        return self._original_model(*args, **kwargs)

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

    def to(self, device):
        """Move the model to a device (PyTorch compatibility)."""
        if self._detect_framework() == "pytorch" and self._original_model is not None:
            self._original_model.to(device)
        return self
