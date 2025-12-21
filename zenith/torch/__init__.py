# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith PyTorch Integration Module

Provides convenient access to PyTorch-specific Zenith features including
the torch.compile backend integration as specified in CetakBiru.md.

Usage:
    import zenith.torch as ztorch

    # Create Zenith backend for torch.compile
    backend = ztorch.create_backend(target="cuda", precision="fp16")
    compiled = torch.compile(model, backend=backend)

    # Compile a function (decorator style)
    @ztorch.compile(target="cuda", precision="fp16")
    def forward(x):
        return model(x)

    # Load HuggingFace model
    graph = ztorch.from_transformers("bert-base-uncased")
"""

from ..adapters.pytorch_adapter import (
    PyTorchAdapter,
    ZenithPyTorchConfig,
    ZenithCompiledPyTorchFunction,
    ZenithOptimizerWrapper,
    OptimizationStats,
    compile,
    create_backend,
)

# Default adapter instance
_default_adapter: PyTorchAdapter = None


def _get_adapter() -> PyTorchAdapter:
    """Get or create the default adapter instance."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = PyTorchAdapter()
    return _default_adapter


def configure(**kwargs) -> ZenithPyTorchConfig:
    """
    Configure the default PyTorch adapter.

    Args:
        **kwargs: Configuration options (see ZenithPyTorchConfig).

    Returns:
        Updated configuration.

    Example:
        zenith.torch.configure(
            target="cuda",
            precision="fp16",
            enable_amp=True,
        )
    """
    global _default_adapter
    config = ZenithPyTorchConfig(**kwargs)
    _default_adapter = PyTorchAdapter(config=config)
    return config


def from_model(model, sample_input, **kwargs):
    """
    Convert a PyTorch nn.Module to GraphIR.

    See PyTorchAdapter.from_model for full documentation.
    """
    return _get_adapter().from_model(model, sample_input, **kwargs)


def from_fx_graph(model, sample_input, **kwargs):
    """
    Convert a PyTorch model using FX Graph (PyTorch 2.x).

    See PyTorchAdapter.from_fx_graph for full documentation.
    """
    return _get_adapter().from_fx_graph(model, sample_input, **kwargs)


def from_transformers(
    model_name_or_path,
    task=None,
    sample_input=None,
    max_length=128,
    batch_size=1,
    **kwargs,
):
    """
    Convert a HuggingFace Transformers PyTorch model to GraphIR.

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
        graph = zenith.torch.from_transformers(
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
    func=None,
    *,
    target="cuda",
    precision="fp32",
    opt_level=2,
    mode="default",
    fullgraph=False,
    **kwargs,
):
    """
    Compile a PyTorch function with Zenith optimizations.

    Works like torch.compile() - can be used as a decorator or function.

    Args:
        func: Function to compile (or None for decorator usage).
        target: Target device ("cpu", "cuda").
        precision: Precision level ("fp32", "fp16", "bf16", "int8").
        opt_level: Optimization level (1-3).
        mode: torch.compile mode.
        fullgraph: Require full graph capture.
        **kwargs: Additional options.

    Returns:
        Compiled function or decorator.

    Example:
        @zenith.torch.compile_function(target="cuda", precision="fp16")
        def forward(x):
            return model(x)
    """
    return _get_adapter().compile_function(
        func,
        target=target,
        precision=precision,
        opt_level=opt_level,
        mode=mode,
        fullgraph=fullgraph,
        **kwargs,
    )


def wrap_training_step(
    train_step_fn,
    enable_amp=False,
    gradient_accumulation_steps=1,
    grad_scaler=None,
):
    """
    Wrap a custom training step with Zenith optimizations.

    Args:
        train_step_fn: Original training step function.
        enable_amp: Enable Automatic Mixed Precision.
        gradient_accumulation_steps: Gradient accumulation.
        grad_scaler: Optional GradScaler.

    Returns:
        Optimized training step function.

    Example:
        optimized_step = zenith.torch.wrap_training_step(
            train_step,
            enable_amp=True,
        )
    """
    return _get_adapter().wrap_training_step(
        train_step_fn,
        enable_amp=enable_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        grad_scaler=grad_scaler,
    )


def create_optimizer_wrapper(optimizer, enable_amp=False):
    """
    Create a Zenith-optimized optimizer wrapper.

    Args:
        optimizer: PyTorch optimizer.
        enable_amp: Enable Automatic Mixed Precision.

    Returns:
        ZenithOptimizerWrapper for use in training.

    Example:
        wrapped = zenith.torch.create_optimizer_wrapper(
            optimizer,
            enable_amp=True
        )
    """
    return _get_adapter().create_optimizer_wrapper(optimizer, enable_amp)


def to_onnx(model, sample_input, output_path=None, **kwargs):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch nn.Module.
        sample_input: Sample input for tracing.
        output_path: Path to save ONNX file.
        **kwargs: Additional export options.

    Returns:
        ONNX model as bytes.
    """
    return _get_adapter().to_onnx(model, sample_input, output_path, **kwargs)


def is_available() -> bool:
    """Check if PyTorch is available."""
    return _get_adapter().is_available


def has_torch_compile() -> bool:
    """Check if torch.compile is available (PyTorch 2.0+)."""
    return _get_adapter()._has_torch_compile()


def has_torch_export() -> bool:
    """Check if torch.export is available (PyTorch 2.1+)."""
    return _get_adapter()._has_torch_export()


# Expose key classes and functions
__all__ = [
    # Classes
    "PyTorchAdapter",
    "ZenithPyTorchConfig",
    "ZenithCompiledPyTorchFunction",
    "ZenithOptimizerWrapper",
    "OptimizationStats",
    # Core functions
    "configure",
    "compile",
    "compile_function",
    "create_backend",
    "from_model",
    "from_fx_graph",
    "from_transformers",
    # Training
    "wrap_training_step",
    "create_optimizer_wrapper",
    # Export
    "to_onnx",
    # Utility
    "is_available",
    "has_torch_compile",
    "has_torch_export",
]
