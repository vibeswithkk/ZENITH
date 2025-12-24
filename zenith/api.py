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

    # Step 3: Compile for target (pass original model and precision for execution)
    compiled = _compile_for_target(optimized_ir, target, original_model, precision)

    return compiled


def optimize(model: Any, **kwargs) -> Any:
    """
    Alias for compile().

    Provided for convenience and API compatibility.
    """
    return compile(model, **kwargs)


def _detect_framework(model: Any) -> str:
    """Detect which ML framework the model comes from."""
    # Try isinstance checks first (more reliable for user-defined classes)
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return "pytorch"
    except ImportError:
        pass

    try:
        import tensorflow as tf

        if isinstance(model, (tf.keras.Model, tf.Module)):
            return "tensorflow"
    except ImportError:
        pass

    try:
        import jax

        # JAX functions are typically callable
        if hasattr(model, "__call__") and "jax" in str(type(model)):
            return "jax"
    except ImportError:
        pass

    # Fallback to module name check
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

    adapter_classes = {
        "pytorch": PyTorchAdapter,
        "tensorflow": TensorFlowAdapter,
        "jax": JAXAdapter,
        "onnx": ONNXAdapter,
    }

    adapter_cls = adapter_classes.get(framework)
    if adapter_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    adapter = adapter_cls()

    if hasattr(adapter, "is_available") and not adapter.is_available:
        raise ImportError(f"{framework.capitalize()} is not installed")

    if framework == "onnx":
        graph_ir = adapter.from_model(model, **kwargs)
    else:
        graph_ir = adapter.from_model(model, sample_input, **kwargs)

    return graph_ir, model


def _optimize_graph(
    graph_ir: GraphIR,
    opt_level: int,
    precision: str,
    tolerance: float,
) -> GraphIR:
    """
    Apply optimization passes to the graph.

    Implements CetakBiru Bab 4.2 and 5.1 Fase 2:
    - Graph simplification (dead code elimination, constant folding)
    - Operator fusion (Conv-BN-ReLU, MatMul-Add)
    - Precision conversion (FP32 -> FP16/INT8)
    - Layout optimization (NCHW <-> NHWC)

    Args:
        graph_ir: Input graph IR to optimize
        opt_level: Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
        precision: Target precision (fp32, fp16, bf16, int8)
        tolerance: Maximum relative error tolerance

    Returns:
        Optimized GraphIR
    """
    if opt_level == 0:
        return graph_ir

    try:
        from .optimization import optimize_graph

        optimized_graph, stats = optimize_graph(
            graph_ir,
            opt_level=opt_level,
            max_iterations=10,
            precision=precision,
            tolerance=tolerance,
        )

        # Log optimization statistics
        total_passes = sum(stats.values())
        if total_passes > 0:
            import logging

            logger = logging.getLogger("zenith.optimization")
            logger.debug(f"Optimization stats: {stats}")

        return optimized_graph

    except ImportError:
        # Optimization module not available, return as-is
        return graph_ir
    except Exception as e:
        # Log error but don't fail - return original graph
        import logging

        logger = logging.getLogger("zenith.optimization")
        logger.warning(f"Optimization failed, using original graph: {e}")
        return graph_ir


def _compile_for_target(
    graph_ir: GraphIR, target: str, original_model: Any, precision: str = "fp32"
) -> Any:
    """
    Compile GraphIR for the specified target using Zenith Runtime.

    This function now uses ZenithEngine to create an executable model
    that actually uses Zenith's optimized CUDA kernels.

    Args:
        graph_ir: The optimized graph IR.
        target: Target device specification.
        original_model: Original model for fallback if needed.
        precision: Target precision (fp32, fp16, bf16, int8).

    Returns:
        CompiledModel that uses Zenith kernels for execution.
    """
    # Parse target
    if target.startswith("cuda"):
        backend = "cuda"
        device_id = 0
        if ":" in target:
            device_id = int(target.split(":")[1])
    elif target.startswith("rocm"):
        backend = "rocm"
        # ROCm support coming soon
    elif target == "tpu":
        backend = "tpu"
        raise NotImplementedError("TPU backend not implemented yet")
    else:
        backend = "cpu"
        device_id = 0

    # Try to use new ZenithEngine for compilation
    try:
        from .runtime import ZenithEngine, CompileConfig

        engine = ZenithEngine(backend=backend)
        config = CompileConfig(
            precision=precision,
            mode="default",
            verbose=2,
        )

        # Compile with ZenithEngine - this connects to actual kernels!
        compiled = engine.compile(
            graph_ir=graph_ir, config=config, original_model=original_model
        )

        return compiled

    except Exception as e:
        # Fallback to wrapper model if runtime fails
        import logging

        logger = logging.getLogger("zenith.compile")
        logger.warning(f"ZenithEngine compilation failed, using fallback: {e}")

        return CompiledModelLegacy(graph_ir, backend, target, original_model)


class CompiledModelLegacy:
    """
    Legacy compiled model wrapper (fallback).

    Wraps the original model and provides Zenith optimization metadata.
    This is used when ZenithEngine compilation fails.

    For full Zenith kernel execution, use ZenithEngine.compile().
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
        self._interpreter = None  # ONNX Interpreter for native execution
        self._use_interpreter = False

        # Try to create interpreter for native execution
        self._init_interpreter()

    def _init_interpreter(self) -> None:
        """Try to create an ONNX interpreter for native execution."""
        try:
            from .execution import ONNXInterpreter

            # Only use interpreter if:
            # 1. We have a valid graph
            # 2. Graph has nodes
            if self.graph_ir is not None and self.graph_ir.num_nodes() > 0:
                self._interpreter = ONNXInterpreter(
                    self.graph_ir,
                    device=self.backend,
                    strict=False,  # Allow fallback for unsupported ops
                )
                # Use interpreter only if all ops are supported
                self._use_interpreter = self._interpreter.is_fully_supported

        except Exception:
            # Interpreter not available, use wrapper approach
            self._interpreter = None
            self._use_interpreter = False

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

        Uses ONNX interpreter for native execution when all operators
        are supported, otherwise delegates to the original model.
        """
        # Try interpreter first (native Zenith execution)
        if self._use_interpreter and self._interpreter is not None:
            try:
                return self._execute_via_interpreter(*args, **kwargs)
            except Exception:
                # Fallback to wrapper on interpreter failure
                pass

        # Fallback: delegate to original model
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

    def _execute_via_interpreter(self, *args, **kwargs):
        """Execute model using native ONNX interpreter."""
        import numpy as np

        # Convert args to numpy
        inputs_dict = {}
        input_names = [inp.name for inp in self.graph_ir.inputs]

        for i, arg in enumerate(args):
            if i < len(input_names):
                name = input_names[i]
                if hasattr(arg, "numpy"):
                    # PyTorch tensor
                    inputs_dict[name] = arg.detach().cpu().numpy()
                elif hasattr(arg, "data"):
                    # TensorFlow tensor
                    inputs_dict[name] = np.asarray(arg)
                else:
                    inputs_dict[name] = np.asarray(arg)

        # Execute via interpreter
        outputs = self._interpreter(**inputs_dict)

        # Return first output (most common case)
        if len(outputs) == 1:
            output_val = list(outputs.values())[0]
            # Convert back to framework tensor if needed
            framework = self._detect_framework()
            if framework == "pytorch":
                import torch

                device = "cuda" if self.backend == "cuda" else "cpu"
                return torch.from_numpy(output_val).to(device)
            elif framework == "tensorflow":
                import tensorflow as tf

                return tf.constant(output_val)
            return output_val

        return outputs

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
