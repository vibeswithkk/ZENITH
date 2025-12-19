# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Mathematical Operators

Implements ONNX math operators:
- MatMul: Matrix multiplication
- Gemm: General matrix multiply (MatMul + bias)
- Add: Element-wise addition
- Sub: Element-wise subtraction
- Mul: Element-wise multiplication
- Div: Element-wise division
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


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D for matrix operations."""
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    elif arr.ndim > 2:
        # Flatten all but last dim for batched matmul
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _broadcast_shapes(shape_a: tuple, shape_b: tuple) -> tuple:
    """Compute broadcast shape for two arrays."""
    result = []
    for a, b in zip(reversed(shape_a), reversed(shape_b)):
        if a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        elif a == b:
            result.append(a)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} not broadcastable")
    return tuple(reversed(result))


@OperatorRegistry.register("MatMul", tier=1)
def execute_matmul(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Matrix multiplication operator.

    ONNX Spec: C = MatMul(A, B)
    Computes matrix product of A and B.
    """
    # Get inputs
    A = ctx.get_tensor(inputs[0])
    B = ctx.get_tensor(inputs[1])

    # Ensure numpy arrays
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        # Use CUDA matmul
        try:
            # Ensure 2D for cuBLAS
            A_2d = _ensure_2d(A)
            B_2d = _ensure_2d(B)
            result = cuda.matmul(A_2d, B_2d)
        except Exception:
            # Fallback to numpy
            result = np.matmul(A, B)
    else:
        # CPU fallback
        result = np.matmul(A, B)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Gemm", tier=1)
def execute_gemm(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    General Matrix Multiply operator.

    ONNX Spec: Y = alpha * A' * B' + beta * C
    Where A' = transpose(A) if transA, A otherwise
    And B' = transpose(B) if transB, B otherwise
    """
    # Get inputs
    A = np.ascontiguousarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    B = np.ascontiguousarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    C = None
    if len(inputs) > 2:
        C = np.ascontiguousarray(ctx.get_tensor(inputs[2]), dtype=np.float32)

    # Get attributes
    alpha = attrs.get("alpha", 1.0)
    beta = attrs.get("beta", 1.0)
    trans_a = attrs.get("transA", 0)
    trans_b = attrs.get("transB", 0)

    # Apply transposes
    if trans_a:
        A = A.T
    if trans_b:
        B = B.T

    cuda = _get_cuda()

    if cuda is not None and ctx.is_cuda_available:
        try:
            # Ensure 2D
            A_2d = _ensure_2d(A)
            B_2d = _ensure_2d(B)
            result = cuda.matmul(A_2d, B_2d)
            result = alpha * result
            if C is not None:
                result = result + beta * C
        except Exception:
            # Fallback
            result = alpha * np.matmul(A, B)
            if C is not None:
                result = result + beta * C
    else:
        result = alpha * np.matmul(A, B)
        if C is not None:
            result = result + beta * C

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Add", tier=1)
def execute_add(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Element-wise addition operator.

    ONNX Spec: C = Add(A, B)
    Supports broadcasting.
    """
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    B = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.float32)

    cuda = _get_cuda()

    # Check if shapes match for GPU add (requires same shape)
    if (
        cuda is not None
        and ctx.is_cuda_available
        and A.shape == B.shape
        and A.ndim == 4
    ):
        try:
            result = cuda.add(
                np.ascontiguousarray(A),
                np.ascontiguousarray(B),
            )
        except Exception:
            result = A + B
    else:
        # numpy handles broadcasting
        result = A + B

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Sub", tier=2)
def execute_sub(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise subtraction operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    B = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    result = A - B
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Mul", tier=2)
def execute_mul(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise multiplication operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    B = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    result = A * B
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Div", tier=2)
def execute_div(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise division operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    B = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    result = A / B
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Pow", tier=2)
def execute_pow(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise power operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    B = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.float32)
    result = np.power(A, B)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Sqrt", tier=2)
def execute_sqrt(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise square root operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    result = np.sqrt(A)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Neg", tier=2)
def execute_neg(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise negation operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    result = -A
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Abs", tier=2)
def execute_abs(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise absolute value operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    result = np.abs(A)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Exp", tier=2)
def execute_exp(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise exponential operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    result = np.exp(A)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Log", tier=2)
def execute_log(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise natural log operator."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    result = np.log(A)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("ReduceMean", tier=2)
def execute_reduce_mean(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Reduce mean along axes."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    axes = attrs.get("axes", None)
    keepdims = attrs.get("keepdims", 1)

    if axes is not None:
        result = np.mean(A, axis=tuple(axes), keepdims=bool(keepdims))
    else:
        result = np.mean(A, keepdims=bool(keepdims))

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("ReduceSum", tier=2)
def execute_reduce_sum(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Reduce sum along axes."""
    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    axes = attrs.get("axes", None)
    keepdims = attrs.get("keepdims", 1)

    if axes is not None:
        result = np.sum(A, axis=tuple(axes), keepdims=bool(keepdims))
    else:
        result = np.sum(A, keepdims=bool(keepdims))

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Erf", tier=2)
def execute_erf(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Error function operator.

    ONNX Spec: Y = erf(X)
    Used for exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    from scipy import special

    A = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    result = special.erf(A).astype(np.float32)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Equal", tier=2)
def execute_equal(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise equality comparison."""
    A = ctx.get_tensor(inputs[0])
    B = ctx.get_tensor(inputs[1])
    result = np.equal(A, B)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Less", tier=2)
def execute_less(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise less than comparison."""
    A = ctx.get_tensor(inputs[0])
    B = ctx.get_tensor(inputs[1])
    result = np.less(A, B)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Greater", tier=2)
def execute_greater(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Element-wise greater than comparison."""
    A = ctx.get_tensor(inputs[0])
    B = ctx.get_tensor(inputs[1])
    result = np.greater(A, B)
    ctx.set_tensor(outputs[0], result)
