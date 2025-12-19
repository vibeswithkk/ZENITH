# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Normalization Operators

Implements ONNX normalization operators:
- LayerNormalization: Layer normalization
- InstanceNormalization: Instance normalization
- GroupNormalization: Group normalization
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING
import numpy as np

from ..registry import OperatorRegistry

if TYPE_CHECKING:
    from ..context import ExecutionContext


def _get_cuda():
    """Get CUDA module, return None if not available."""
    try:
        from zenith._zenith_core import cuda

        return cuda
    except ImportError:
        return None


@OperatorRegistry.register("LayerNormalization")
def execute_layer_normalization(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Layer Normalization operator.

    ONNX Spec: Y = (X - Mean) / sqrt(Var + epsilon) * Scale + Bias

    Inputs:
        X: Input tensor
        Scale: Scale tensor (gamma)
        B: Bias tensor (beta) - optional

    Attributes:
        axis: First normalization dimension (default: -1)
        epsilon: Small value to avoid division by zero (default: 1e-5)
        stash_type: Precision for mean/variance (default: 1 = float)
    """
    # Get inputs
    x = ctx.get_tensor(inputs[0])
    scale = ctx.get_tensor(inputs[1]) if len(inputs) > 1 else None
    bias = ctx.get_tensor(inputs[2]) if len(inputs) > 2 else None

    # Get attributes
    axis = attrs.get("axis", -1)
    epsilon = attrs.get("epsilon", 1e-5)

    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    # Get normalization axes (from axis to end)
    norm_axes = tuple(range(axis, x.ndim))

    # Compute mean and variance
    mean = np.mean(x, axis=norm_axes, keepdims=True)
    variance = np.var(x, axis=norm_axes, keepdims=True)

    # Normalize
    x_normalized = (x - mean) / np.sqrt(variance + epsilon)

    # Apply scale and bias
    if scale is not None:
        # Reshape scale for broadcasting
        scale_shape = [1] * axis + list(scale.shape)
        scale = scale.reshape(scale_shape)
        x_normalized = x_normalized * scale

    if bias is not None:
        # Reshape bias for broadcasting
        bias_shape = [1] * axis + list(bias.shape)
        bias = bias.reshape(bias_shape)
        x_normalized = x_normalized + bias

    # Store output
    ctx.set_tensor(outputs[0], x_normalized.astype(x.dtype))

    # Optional outputs: mean and inv_std_dev
    if len(outputs) > 1:
        ctx.set_tensor(outputs[1], mean.astype(np.float32))
    if len(outputs) > 2:
        inv_std_dev = 1.0 / np.sqrt(variance + epsilon)
        ctx.set_tensor(outputs[2], inv_std_dev.astype(np.float32))


@OperatorRegistry.register("InstanceNormalization")
def execute_instance_normalization(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Instance Normalization operator.

    ONNX Spec: Y = (X - Mean) / sqrt(Var + epsilon) * Scale + Bias
    Normalization is done per instance (per batch, per channel).
    """
    x = ctx.get_tensor(inputs[0])
    scale = ctx.get_tensor(inputs[1]) if len(inputs) > 1 else None
    bias = ctx.get_tensor(inputs[2]) if len(inputs) > 2 else None

    epsilon = attrs.get("epsilon", 1e-5)

    # For 4D input (N, C, H, W), normalize over (H, W)
    if x.ndim == 4:
        norm_axes = (2, 3)
    elif x.ndim == 3:
        norm_axes = (2,)
    else:
        norm_axes = tuple(range(2, x.ndim))

    mean = np.mean(x, axis=norm_axes, keepdims=True)
    variance = np.var(x, axis=norm_axes, keepdims=True)

    x_normalized = (x - mean) / np.sqrt(variance + epsilon)

    if scale is not None:
        # Scale shape: (C,) -> reshape for broadcasting
        if x.ndim == 4:
            scale = scale.reshape(1, -1, 1, 1)
        elif x.ndim == 3:
            scale = scale.reshape(1, -1, 1)
        x_normalized = x_normalized * scale

    if bias is not None:
        if x.ndim == 4:
            bias = bias.reshape(1, -1, 1, 1)
        elif x.ndim == 3:
            bias = bias.reshape(1, -1, 1)
        x_normalized = x_normalized + bias

    ctx.set_tensor(outputs[0], x_normalized.astype(x.dtype))


@OperatorRegistry.register("GroupNormalization")
def execute_group_normalization(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Group Normalization operator.

    Divides channels into groups and normalizes within each group.
    """
    x = ctx.get_tensor(inputs[0])
    scale = ctx.get_tensor(inputs[1]) if len(inputs) > 1 else None
    bias = ctx.get_tensor(inputs[2]) if len(inputs) > 2 else None

    epsilon = attrs.get("epsilon", 1e-5)
    num_groups = attrs.get("num_groups", 32)

    # Input shape: (N, C, *)
    n, c = x.shape[:2]
    spatial_dims = x.shape[2:]

    # Reshape to (N, G, C//G, *)
    if c % num_groups != 0:
        raise ValueError(f"Channels {c} must be divisible by num_groups {num_groups}")

    channels_per_group = c // num_groups

    # Reshape for group normalization
    x_reshaped = x.reshape(n, num_groups, channels_per_group, *spatial_dims)

    # Normalize over (C//G, spatial_dims)
    norm_axes = tuple(range(2, x_reshaped.ndim))

    mean = np.mean(x_reshaped, axis=norm_axes, keepdims=True)
    variance = np.var(x_reshaped, axis=norm_axes, keepdims=True)

    x_normalized = (x_reshaped - mean) / np.sqrt(variance + epsilon)

    # Reshape back to (N, C, *)
    x_normalized = x_normalized.reshape(n, c, *spatial_dims)

    if scale is not None:
        scale = scale.reshape(1, -1, *([1] * len(spatial_dims)))
        x_normalized = x_normalized * scale

    if bias is not None:
        bias = bias.reshape(1, -1, *([1] * len(spatial_dims)))
        x_normalized = x_normalized + bias

    ctx.set_tensor(outputs[0], x_normalized.astype(x.dtype))
