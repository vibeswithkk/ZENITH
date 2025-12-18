# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Activation Operators

Implements ONNX activation operators:
- Relu: Rectified Linear Unit
- Sigmoid: Sigmoid activation
- Tanh: Hyperbolic tangent
- Softmax: Softmax normalization
- Gelu: Gaussian Error Linear Unit
- LeakyRelu: Leaky ReLU
- Clip: Clamp values to range
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


@OperatorRegistry.register("Relu", tier=1)
def execute_relu(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    ReLU activation operator.

    ONNX Spec: Y = Relu(X) = max(0, X)
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            # Upload to GPU
            X_gpu = ctx.upload_to_gpu("_relu_input", X, cache=False)
            result_gpu = cuda.relu_gpu(X_gpu)
            result = cuda.to_numpy(result_gpu)
        except Exception:
            result = np.maximum(0, X)
    else:
        result = np.maximum(0, X)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Sigmoid", tier=2)
def execute_sigmoid(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Sigmoid activation operator.

    ONNX Spec: Y = Sigmoid(X) = 1 / (1 + exp(-X))
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            X_gpu = ctx.upload_to_gpu("_sigmoid_input", X, cache=False)
            result_gpu = cuda.sigmoid_gpu(X_gpu)
            result = cuda.to_numpy(result_gpu)
        except Exception:
            result = 1.0 / (1.0 + np.exp(-X))
    else:
        result = 1.0 / (1.0 + np.exp(-X))

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Tanh", tier=2)
def execute_tanh(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Tanh activation operator.

    ONNX Spec: Y = Tanh(X) = (exp(2X) - 1) / (exp(2X) + 1)
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            X_gpu = ctx.upload_to_gpu("_tanh_input", X, cache=False)
            result_gpu = cuda.tanh_gpu(X_gpu)
            result = cuda.to_numpy(result_gpu)
        except Exception:
            result = np.tanh(X)
    else:
        result = np.tanh(X)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Softmax", tier=2)
def execute_softmax(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Softmax activation operator.

    ONNX Spec: Y = Softmax(X, axis)
    Computes softmax along specified axis.
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    axis = attrs.get("axis", -1)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            X_gpu = ctx.upload_to_gpu("_softmax_input", X, cache=False)
            result_gpu = cuda.softmax_gpu(X_gpu)
            result = cuda.to_numpy(result_gpu)
        except Exception:
            # NumPy softmax
            exp_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
            result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    else:
        exp_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
        result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Gelu", tier=3)
def execute_gelu(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    GELU activation operator.

    ONNX Spec: Y = Gelu(X)
    Uses the GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    X = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            X_gpu = ctx.upload_to_gpu("_gelu_input", X, cache=False)
            result_gpu = cuda.gelu_gpu(X_gpu)
            result = cuda.to_numpy(result_gpu)
        except Exception:
            # GELU approximation
            result = 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)))
    else:
        result = 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)))

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("LeakyRelu", tier=2)
def execute_leaky_relu(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Leaky ReLU activation operator.

    ONNX Spec: Y = LeakyRelu(X, alpha)
    Y = X if X >= 0, else alpha * X
    """
    X = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    alpha = attrs.get("alpha", 0.01)

    result = np.where(X >= 0, X, alpha * X)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Clip", tier=2)
def execute_clip(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Clip operator - clamps values to specified range.

    ONNX Spec: Y = Clip(X, min, max)
    """
    X = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    # Get min/max from inputs or attributes
    min_val = None
    max_val = None

    if len(inputs) > 1 and ctx.has_tensor(inputs[1]):
        min_val = float(np.asarray(ctx.get_tensor(inputs[1])))
    if len(inputs) > 2 and ctx.has_tensor(inputs[2]):
        max_val = float(np.asarray(ctx.get_tensor(inputs[2])))

    # Fallback to attributes (for older ONNX versions)
    if min_val is None:
        min_val = attrs.get("min", None)
    if max_val is None:
        max_val = attrs.get("max", None)

    result = np.clip(X, min_val, max_val)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Elu", tier=2)
def execute_elu(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    ELU activation operator.

    ONNX Spec: Y = Elu(X, alpha)
    Y = X if X >= 0, else alpha * (exp(X) - 1)
    """
    X = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    alpha = attrs.get("alpha", 1.0)

    result = np.where(X >= 0, X, alpha * (np.exp(X) - 1))
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Selu", tier=3)
def execute_selu(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    SELU activation operator.

    ONNX Spec: Y = Selu(X, alpha, gamma)
    Y = gamma * (X if X >= 0, else alpha * (exp(X) - 1))
    """
    X = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    alpha = attrs.get("alpha", 1.6732632423543772)
    gamma = attrs.get("gamma", 1.0507009873554805)

    result = gamma * np.where(X >= 0, X, alpha * (np.exp(X) - 1))
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("HardSigmoid", tier=3)
def execute_hard_sigmoid(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Hard sigmoid activation."""
    X = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    alpha = attrs.get("alpha", 0.2)
    beta = attrs.get("beta", 0.5)

    result = np.clip(alpha * X + beta, 0, 1)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("LogSoftmax", tier=2)
def execute_log_softmax(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Log softmax operator."""
    X = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    axis = attrs.get("axis", -1)

    # Log-sum-exp trick for numerical stability
    max_x = np.max(X, axis=axis, keepdims=True)
    exp_x = np.exp(X - max_x)
    result = X - max_x - np.log(np.sum(exp_x, axis=axis, keepdims=True))

    ctx.set_tensor(outputs[0], result)
