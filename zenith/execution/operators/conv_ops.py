# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Convolution and Normalization Operators

Implements ONNX conv/norm operators:
- Conv: 2D Convolution
- BatchNormalization: Batch normalization
- MaxPool: Max pooling
- AveragePool: Average pooling
- GlobalAveragePool: Global average pooling
- LayerNormalization: Layer normalization
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
        from zenith import _zenith_core

        return _zenith_core.cuda
    except (ImportError, AttributeError):
        return None


def _ensure_nchw(X: np.ndarray) -> np.ndarray:
    """Ensure array is in NCHW format (4D)."""
    if X.ndim == 3:
        return X.reshape(1, *X.shape)
    elif X.ndim == 4:
        return X
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {X.ndim}D")


@OperatorRegistry.register("Conv", tier=1)
def execute_conv(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    2D Convolution operator.

    ONNX Spec: Y = Conv(X, W, B)
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    W = np.ascontiguousarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    B = None
    if len(inputs) > 2:
        B = np.ascontiguousarray(ctx.get_tensor(inputs[2]), dtype=np.float32)

    # Get attributes
    kernel_shape = attrs.get("kernel_shape", [W.shape[2], W.shape[3]])
    strides = attrs.get("strides", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]
    dilations = attrs.get("dilations", [1, 1])
    group = attrs.get("group", 1)

    # Ensure NCHW format
    X = _ensure_nchw(X)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            # Use CUDA conv2d
            stride = strides[0] if isinstance(strides, list) else strides
            padding = pads[0] if isinstance(pads, list) else pads

            result = cuda.conv2d(X, W, B, stride, padding)
        except Exception:
            # Fallback to basic numpy implementation
            result = _conv2d_numpy(X, W, B, strides, pads, dilations, group)
    else:
        result = _conv2d_numpy(X, W, B, strides, pads, dilations, group)

    ctx.set_tensor(outputs[0], result)


def _conv2d_numpy(
    X: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    strides: List[int],
    pads: List[int],
    dilations: List[int],
    group: int,
) -> np.ndarray:
    """Basic 2D convolution in pure numpy (slow but correct)."""
    N, C_in, H, W_in = X.shape
    C_out, C_in_group, Kh, Kw = W.shape

    stride_h, stride_w = strides[:2] if len(strides) >= 2 else (strides[0], strides[0])
    pad_top = pads[0] if len(pads) >= 1 else 0
    pad_left = pads[1] if len(pads) >= 2 else pad_top
    pad_bottom = pads[2] if len(pads) >= 3 else pad_top
    pad_right = pads[3] if len(pads) >= 4 else pad_left

    # Pad input
    X_padded = np.pad(
        X,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )

    # Output dimensions
    H_out = (H + pad_top + pad_bottom - Kh) // stride_h + 1
    W_out = (W_in + pad_left + pad_right - Kw) // stride_w + 1

    # Allocate output
    Y = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    # Convolution
    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride_h
                    w_start = w * stride_w
                    patch = X_padded[
                        n, :, h_start : h_start + Kh, w_start : w_start + Kw
                    ]
                    Y[n, c_out, h, w] = np.sum(patch * W[c_out])

    # Add bias
    if B is not None:
        Y += B.reshape(1, -1, 1, 1)

    return Y


@OperatorRegistry.register("BatchNormalization", tier=1)
def execute_batch_norm(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Batch Normalization operator.

    ONNX Spec: Y = BatchNormalization(X, scale, B, input_mean, input_var)
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    scale = np.ascontiguousarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    B = np.ascontiguousarray(ctx.get_tensor(inputs[2]), dtype=np.float32)
    input_mean = np.ascontiguousarray(ctx.get_tensor(inputs[3]), dtype=np.float32)
    input_var = np.ascontiguousarray(ctx.get_tensor(inputs[4]), dtype=np.float32)

    epsilon = attrs.get("epsilon", 1e-5)

    X = _ensure_nchw(X)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            result = cuda.batch_norm(X, scale, B, input_mean, input_var, epsilon)
        except Exception:
            result = _batch_norm_numpy(X, scale, B, input_mean, input_var, epsilon)
    else:
        result = _batch_norm_numpy(X, scale, B, input_mean, input_var, epsilon)

    ctx.set_tensor(outputs[0], result)


def _batch_norm_numpy(
    X: np.ndarray,
    scale: np.ndarray,
    B: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Batch normalization in pure numpy."""
    # Reshape parameters for broadcasting [1, C, 1, 1]
    scale = scale.reshape(1, -1, 1, 1)
    B = B.reshape(1, -1, 1, 1)
    mean = mean.reshape(1, -1, 1, 1)
    var = var.reshape(1, -1, 1, 1)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + epsilon)

    # Scale and shift
    return scale * X_norm + B


@OperatorRegistry.register("MaxPool", tier=1)
def execute_maxpool(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Max Pooling operator.

    ONNX Spec: Y = MaxPool(X, kernel_shape, strides, pads)
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    X = _ensure_nchw(X)

    kernel_shape = attrs.get("kernel_shape", [2, 2])
    strides = attrs.get("strides", kernel_shape)
    pads = attrs.get("pads", [0, 0, 0, 0])

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            kernel = kernel_shape[0] if isinstance(kernel_shape, list) else kernel_shape
            stride = strides[0] if isinstance(strides, list) else strides
            padding = pads[0] if isinstance(pads, list) else pads
            result = cuda.maxpool(X, kernel, stride, padding)
        except Exception:
            result = _maxpool_numpy(X, kernel_shape, strides, pads)
    else:
        result = _maxpool_numpy(X, kernel_shape, strides, pads)

    ctx.set_tensor(outputs[0], result)


def _maxpool_numpy(
    X: np.ndarray,
    kernel_shape: List[int],
    strides: List[int],
    pads: List[int],
) -> np.ndarray:
    """Max pooling in pure numpy."""
    N, C, H, W = X.shape
    Kh, Kw = kernel_shape[:2]
    stride_h = strides[0] if len(strides) >= 1 else Kh
    stride_w = strides[1] if len(strides) >= 2 else stride_h
    pad_top = pads[0] if len(pads) >= 1 else 0
    pad_left = pads[1] if len(pads) >= 2 else pad_top

    H_out = (H + 2 * pad_top - Kh) // stride_h + 1
    W_out = (W + 2 * pad_left - Kw) // stride_w + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride_h - pad_top
                    w_start = w * stride_w - pad_left
                    h_end = min(h_start + Kh, H)
                    w_end = min(w_start + Kw, W)
                    h_start = max(h_start, 0)
                    w_start = max(w_start, 0)
                    Y[n, c, h, w] = np.max(X[n, c, h_start:h_end, w_start:w_end])

    return Y


@OperatorRegistry.register("AveragePool", tier=2)
def execute_avgpool(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Average Pooling operator."""
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    X = _ensure_nchw(X)

    kernel_shape = attrs.get("kernel_shape", [2, 2])
    strides = attrs.get("strides", kernel_shape)
    pads = attrs.get("pads", [0, 0, 0, 0])

    N, C, H, W = X.shape
    Kh, Kw = kernel_shape[:2]
    stride_h = strides[0] if len(strides) >= 1 else Kh
    stride_w = strides[1] if len(strides) >= 2 else stride_h

    H_out = (H - Kh) // stride_h + 1
    W_out = (W - Kw) // stride_w + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride_h
                    w_start = w * stride_w
                    Y[n, c, h, w] = np.mean(
                        X[n, c, h_start : h_start + Kh, w_start : w_start + Kw]
                    )

    ctx.set_tensor(outputs[0], Y)


@OperatorRegistry.register("GlobalAveragePool", tier=2)
def execute_global_avgpool(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Global Average Pooling operator."""
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    X = _ensure_nchw(X)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            result = cuda.global_avgpool(X)
        except Exception:
            result = np.mean(X, axis=(2, 3), keepdims=True)
    else:
        result = np.mean(X, axis=(2, 3), keepdims=True)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("LayerNormalization", tier=3)
def execute_layer_norm(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Layer Normalization operator.

    ONNX Spec: Y = LayerNormalization(X, scale, B)
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    scale = np.ascontiguousarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    B = None
    if len(inputs) > 2:
        B = np.ascontiguousarray(ctx.get_tensor(inputs[2]), dtype=np.float32)

    axis = attrs.get("axis", -1)
    epsilon = attrs.get("epsilon", 1e-5)

    # Compute mean and variance along normalized axes
    mean = np.mean(X, axis=axis, keepdims=True)
    var = np.var(X, axis=axis, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + epsilon)

    # Scale and shift
    result = scale * X_norm
    if B is not None:
        result = result + B

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Dropout", tier=2)
def execute_dropout(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Dropout operator (inference mode - pass through).

    In inference mode, dropout is a no-op.
    """
    X = ctx.get_tensor(inputs[0])
    ctx.set_tensor(outputs[0], X)

    # Also set mask output if present
    if len(outputs) > 1:
        mask = np.ones_like(np.asarray(X), dtype=np.bool_)
        ctx.set_tensor(outputs[1], mask)
