# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith TensorFlow Integration Module

Provides convenient access to TensorFlow-specific Zenith features.

Usage:
    import zenith.tensorflow as ztf

    # Compile a tf.function (like torch.compile)
    @ztf.compile(target="cuda", precision="fp16")
    @tf.function
    def forward(x):
        return model(x)

    # Load HuggingFace model
    graph = ztf.from_transformers("bert-base-uncased")

    # Create training callback
    callback = ztf.create_training_callback(model)
    model.fit(X, y, callbacks=[callback])
"""

from ..adapters.tensorflow_adapter import (
    TensorFlowAdapter,
    ZenithTFConfig,
    ZenithCompiledFunction,
    ZenithTrainingCallback,
    OptimizationStats,
    compile,
)

# Default adapter instance
_default_adapter: TensorFlowAdapter = None


def _get_adapter() -> TensorFlowAdapter:
    """Get or create the default adapter instance."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = TensorFlowAdapter()
    return _default_adapter


def configure(**kwargs) -> ZenithTFConfig:
    """
    Configure the default TensorFlow adapter.

    Args:
        **kwargs: Configuration options (see ZenithTFConfig).

    Returns:
        Updated configuration.

    Example:
        zenith.tensorflow.configure(
            target="cuda",
            precision="fp16",
            enable_mixed_precision_training=True,
        )
    """
    global _default_adapter
    config = ZenithTFConfig(**kwargs)
    _default_adapter = TensorFlowAdapter(config=config)
    return config


def from_model(model, sample_input=None, **kwargs):
    """
    Convert a TensorFlow/Keras model to GraphIR.

    See TensorFlowAdapter.from_model for full documentation.
    """
    return _get_adapter().from_model(model, sample_input, **kwargs)


def from_saved_model(saved_model_path, signature_key="serving_default", **kwargs):
    """
    Load and convert a TensorFlow SavedModel.

    See TensorFlowAdapter.from_saved_model for full documentation.
    """
    return _get_adapter().from_saved_model(saved_model_path, signature_key, **kwargs)


def from_transformers(
    model_name_or_path,
    task=None,
    sample_input=None,
    max_length=128,
    batch_size=1,
    **kwargs,
):
    """
    Convert a HuggingFace Transformers TF model to GraphIR.

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
        graph = zenith.tensorflow.from_transformers(
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


def compile_function(
    func=None, *, target="cuda", precision="fp32", opt_level=2, **kwargs
):
    """
    Compile a tf.function with Zenith optimizations.

    Works like torch.compile() - can be used as a decorator or function.

    Args:
        func: Function to compile (or None for decorator usage).
        target: Target device ("cpu", "cuda").
        precision: Precision level ("fp32", "fp16", "bf16", "int8").
        opt_level: Optimization level (1-3).
        **kwargs: Additional options.

    Returns:
        Compiled function or decorator.

    Example:
        @zenith.tensorflow.compile_function(target="cuda", precision="fp16")
        @tf.function
        def forward(x):
            return model(x)
    """
    return _get_adapter().compile_function(
        func,
        target=target,
        precision=precision,
        opt_level=opt_level,
        **kwargs,
    )


def create_training_callback(
    model,
    optimizer=None,
    enable_mixed_precision=False,
    gradient_accumulation_steps=1,
):
    """
    Create a Keras callback for Zenith-optimized training.

    Args:
        model: Keras model to optimize.
        optimizer: Optimizer (uses model's if None).
        enable_mixed_precision: Enable mixed precision training.
        gradient_accumulation_steps: Gradient accumulation.

    Returns:
        Keras callback for use with model.fit().

    Example:
        callback = zenith.tensorflow.create_training_callback(
            model,
            enable_mixed_precision=True,
        )
        model.fit(X, y, callbacks=[callback.get_keras_callback()])
    """
    return _get_adapter().create_training_callback(
        model,
        optimizer=optimizer,
        enable_mixed_precision=enable_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def wrap_training_step(
    train_step_fn,
    model,
    optimizer,
    enable_mixed_precision=False,
    gradient_checkpointing=False,
):
    """
    Wrap a custom training step with Zenith optimizations.

    Args:
        train_step_fn: Original training step function.
        model: Model being trained.
        optimizer: Optimizer to use.
        enable_mixed_precision: Enable mixed precision.
        gradient_checkpointing: Enable gradient checkpointing.

    Returns:
        Optimized training step function.

    Example:
        optimized_step = zenith.tensorflow.wrap_training_step(
            train_step,
            model,
            optimizer,
            enable_mixed_precision=True,
        )
    """
    return _get_adapter().wrap_training_step(
        train_step_fn,
        model,
        optimizer,
        enable_mixed_precision=enable_mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
    )


def to_onnx(model, sample_input=None, output_path=None, **kwargs):
    """
    Export TensorFlow model to ONNX format.

    Args:
        model: TensorFlow/Keras model.
        sample_input: Sample input for shape inference.
        output_path: Path to save ONNX file.
        **kwargs: Additional export options.

    Returns:
        ONNX model as bytes.
    """
    return _get_adapter().to_onnx(model, sample_input, output_path, **kwargs)


def is_available() -> bool:
    """Check if TensorFlow 2.x is available."""
    return _get_adapter().is_available


# Expose key classes
__all__ = [
    "TensorFlowAdapter",
    "ZenithTFConfig",
    "ZenithCompiledFunction",
    "ZenithTrainingCallback",
    "OptimizationStats",
    "configure",
    "compile",
    "compile_function",
    "from_model",
    "from_saved_model",
    "from_transformers",
    "create_training_callback",
    "wrap_training_step",
    "to_onnx",
    "is_available",
]
