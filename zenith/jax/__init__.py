# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith JAX Integration Module

Provides convenient access to JAX-specific Zenith features.

Usage:
    import zenith.jax as zjax

    # Compile a jax.jit function (like torch.compile)
    @zjax.compile(target="cuda", precision="fp16")
    @jax.jit
    def forward(x):
        return model.apply(params, x)

    # Load HuggingFace Flax model
    graph = zjax.from_transformers("bert-base-uncased")

    # Create training state
    state = zjax.create_training_state(model, params, optimizer)
"""

from ..adapters.jax_adapter import (
    JAXAdapter,
    ZenithJAXConfig,
    ZenithCompiledJAXFunction,
    ZenithTrainState,
    OptimizationStats,
    ModelType,
    detect_model_type,
    compile,
)

# Import checkpointing module
from .checkpointing import (
    CheckpointPolicy,
    SelectionMethod,
    CheckpointConfig,
    CheckpointingStats,
    OptimalCheckpointSelector,
    ZenithCheckpointer,
    checkpoint,
    checkpoint_sequential,
    remat,
)

# Import memory management module
from .memory_manager import (
    EvictionPolicy,
    DeviceType,
    JAXMemoryConfig,
    ArrayMetadata,
    MemoryStats,
    JAXActivationStore,
    JAXMemoryManager,
    compute_array_size,
    get_device_memory_info,
)

# Import mixed precision module
from .mixed_precision import (
    PrecisionMode,
    MixedPrecisionPolicy,
    LossScalerConfig,
    LossScalerState,
    DynamicLossScaler,
    MixedPrecisionStats,
    ZenithMixedPrecision,
    create_policy,
    detect_best_precision,
)

# Import ONNX export module
from .onnx_export import (
    ONNXExportConfig,
    ONNXExportResult,
    JAXONNXExporter,
    export_to_onnx,
    validate_onnx_model,
    get_onnx_model_info,
)

# Import primitives module (Phase 3)
from .primitives import (
    PrimitiveConfig,
    ZenithPrimitiveRegistry,
    fused_attention,
    fused_layernorm,
    fused_gelu,
    fused_softmax,
    get_primitive_registry,
    list_primitives,
)

# Default adapter instance
_default_adapter: JAXAdapter = None


def _get_adapter() -> JAXAdapter:
    """Get or create the default adapter instance."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = JAXAdapter()
    return _default_adapter


def configure(**kwargs) -> ZenithJAXConfig:
    """
    Configure the default JAX adapter.

    Args:
        **kwargs: Configuration options (see ZenithJAXConfig).

    Returns:
        Updated configuration.

    Example:
        zenith.jax.configure(
            target="cuda",
            precision="fp16",
            enable_gradient_optimization=True,
        )
    """
    global _default_adapter
    config = ZenithJAXConfig(**kwargs)
    _default_adapter = JAXAdapter(config=config)
    return config


def from_model(model, sample_input, params=None, **kwargs):
    """
    Convert a JAX function or Flax/Haiku model to GraphIR.

    See JAXAdapter.from_model for full documentation.
    """
    return _get_adapter().from_model(model, sample_input, params=params, **kwargs)


def from_flax_module(module, params, sample_input, **kwargs):
    """
    Convert a Flax nn.Module to GraphIR.

    See JAXAdapter.from_flax_module for full documentation.
    """
    return _get_adapter().from_flax_module(module, params, sample_input, **kwargs)


def from_haiku(transformed_fn, params, sample_input, **kwargs):
    """
    Convert a Haiku transformed function to GraphIR.

    See JAXAdapter.from_haiku for full documentation.
    """
    return _get_adapter().from_haiku(transformed_fn, params, sample_input, **kwargs)


def from_transformers(
    model_name_or_path,
    task=None,
    sample_input=None,
    max_length=128,
    batch_size=1,
    **kwargs,
):
    """
    Convert a HuggingFace Transformers Flax model to GraphIR.

    Args:
        model_name_or_path: Model identifier (e.g., "bert-base-uncased").
        task: Task type for Auto classes (e.g., "text-classification").
        sample_input: Optional sample input.
        max_length: Maximum sequence length.
        batch_size: Batch size for sample.
        **kwargs: Additional options.

    Returns:
        GraphIR representation of the model.

    Example:
        graph = zenith.jax.from_transformers(
            "bert-base-uncased",
            task="text-classification"
        )
    """
    return _get_adapter().from_transformers(
        model_name_or_path,
        task=task,
        sample_input=sample_input,
        max_length=max_length,
        batch_size=batch_size,
        **kwargs,
    )


def from_stablehlo(model, sample_input, params=None, **kwargs):
    """
    Convert a JAX function using StableHLO export.

    See JAXAdapter.from_stablehlo for full documentation.
    """
    return _get_adapter().from_stablehlo(model, sample_input, params=params, **kwargs)


def compile_function(
    func=None, *, target="cuda", precision="fp32", opt_level=2, **kwargs
):
    """
    Compile a JAX function with Zenith optimizations.

    Works like torch.compile() - can be used as a decorator or function.

    Args:
        func: Function to compile (or None for decorator usage).
        target: Target device ("cpu", "cuda", "tpu").
        precision: Precision level ("fp32", "fp16", "bf16", "int8").
        opt_level: Optimization level (1-3).
        **kwargs: Additional options.

    Returns:
        Compiled function or decorator.

    Example:
        @zenith.jax.compile_function(target="cuda", precision="fp16")
        @jax.jit
        def forward(x):
            return model.apply(params, x)
    """
    return _get_adapter().compile_function(
        func,
        target=target,
        precision=precision,
        opt_level=opt_level,
        **kwargs,
    )


def create_training_state(
    model,
    params,
    optimizer,
    enable_gradient_checkpointing=False,
):
    """
    Create a Zenith-optimized training state for Flax models.

    Args:
        model: Flax module.
        params: Model parameters.
        optimizer: Optax optimizer.
        enable_gradient_checkpointing: Enable gradient checkpointing.

    Returns:
        ZenithTrainState for use in training loops.

    Example:
        state = zenith.jax.create_training_state(
            model,
            params,
            optax.adam(1e-4),
            enable_gradient_checkpointing=True
        )
    """
    return _get_adapter().create_training_state(
        model,
        params,
        optimizer,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )


def wrap_training_step(
    train_step_fn,
    enable_mixed_precision=False,
    gradient_accumulation_steps=1,
):
    """
    Wrap a custom training step with Zenith optimizations.

    Args:
        train_step_fn: Original training step function.
        enable_mixed_precision: Enable mixed precision.
        gradient_accumulation_steps: Gradient accumulation.

    Returns:
        Optimized training step function.

    Example:
        optimized_step = zenith.jax.wrap_training_step(
            train_step,
            enable_mixed_precision=True,
        )
    """
    return _get_adapter().wrap_training_step(
        train_step_fn,
        enable_mixed_precision=enable_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def to_onnx(model, sample_input, output_path=None, params=None, **kwargs):
    """
    Export JAX function to ONNX format.

    Args:
        model: JAX function or Flax model.
        sample_input: Sample input for tracing.
        output_path: Path to save ONNX file.
        params: Model parameters.
        **kwargs: Additional export options.

    Returns:
        ONNX model as bytes.
    """
    return _get_adapter().to_onnx(
        model, sample_input, output_path, params=params, **kwargs
    )


def is_available() -> bool:
    """Check if JAX is available."""
    return _get_adapter().is_available


# Expose key classes and functions
__all__ = [
    # Core adapter
    "JAXAdapter",
    "ZenithJAXConfig",
    "ZenithCompiledJAXFunction",
    "ZenithTrainState",
    "OptimizationStats",
    "ModelType",
    "detect_model_type",
    "configure",
    "compile",
    "compile_function",
    "from_model",
    "from_flax_module",
    "from_haiku",
    "from_transformers",
    "from_stablehlo",
    "create_training_state",
    "wrap_training_step",
    "to_onnx",
    "is_available",
    # Checkpointing
    "CheckpointPolicy",
    "SelectionMethod",
    "CheckpointConfig",
    "CheckpointingStats",
    "OptimalCheckpointSelector",
    "ZenithCheckpointer",
    "checkpoint",
    "checkpoint_sequential",
    "remat",
    # Memory management
    "EvictionPolicy",
    "DeviceType",
    "JAXMemoryConfig",
    "ArrayMetadata",
    "MemoryStats",
    "JAXActivationStore",
    "JAXMemoryManager",
    "compute_array_size",
    "get_device_memory_info",
    # Mixed precision
    "PrecisionMode",
    "MixedPrecisionPolicy",
    "LossScalerConfig",
    "LossScalerState",
    "DynamicLossScaler",
    "MixedPrecisionStats",
    "ZenithMixedPrecision",
    "create_policy",
    "detect_best_precision",
    # ONNX Export
    "ONNXExportConfig",
    "ONNXExportResult",
    "JAXONNXExporter",
    "export_to_onnx",
    "validate_onnx_model",
    "get_onnx_model_info",
    # Primitives (Phase 3)
    "PrimitiveConfig",
    "ZenithPrimitiveRegistry",
    "fused_attention",
    "fused_layernorm",
    "fused_gelu",
    "fused_softmax",
    "get_primitive_registry",
    "list_primitives",
]
