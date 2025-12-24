# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
PyTorch GPU Kernels - GPU-accelerated operations using PyTorch.

This module provides GPU kernel implementations using PyTorch's optimized
CUDA operations. These serve as the primary execution path when native
CUDA bindings are not available (typical for PyPI installations).

Performance Characteristics:
- Uses PyTorch's cuBLAS/cuDNN backends
- Tensor Core utilization for FP16 operations
- Memory-efficient with minimal host-device transfers
"""

from typing import Any, Optional, Tuple
import numpy as np

# PyTorch import with graceful fallback
_HAS_TORCH = False
_HAS_CUDA = False
torch = None
F = None

try:
    import torch as _torch
    import torch.nn.functional as _F

    torch = _torch
    F = _F
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass


def is_available() -> bool:
    """Check if PyTorch GPU kernels are available."""
    return _HAS_TORCH and _HAS_CUDA


def _to_tensor(x: Any, device: str = "cuda") -> "torch.Tensor":
    """Convert input to PyTorch tensor on specified device."""
    if torch is None:
        raise RuntimeError("PyTorch not available")

    if isinstance(x, torch.Tensor):
        return x.to(device) if x.device.type != device else x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    return torch.tensor(x, device=device)


def _to_numpy(x: "torch.Tensor") -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    if x.device.type != "cpu":
        x = x.cpu()
    return x.detach().numpy()


# =============================================================================
# Linear / MatMul Operations
# =============================================================================


def matmul(a: Any, b: Any) -> np.ndarray:
    """
    Matrix multiplication: C = A @ B

    Uses cuBLAS for optimal performance on GPU.
    """
    a_t = _to_tensor(a)
    b_t = _to_tensor(b)

    with torch.no_grad():
        result = torch.matmul(a_t, b_t)

    return _to_numpy(result)


def linear(x: Any, weight: Any, bias: Optional[Any] = None) -> np.ndarray:
    """
    Linear transformation: y = x @ W^T + b

    Equivalent to torch.nn.functional.linear.
    """
    x_t = _to_tensor(x)
    w_t = _to_tensor(weight)
    b_t = _to_tensor(bias) if bias is not None else None

    with torch.no_grad():
        result = F.linear(x_t, w_t, b_t)

    return _to_numpy(result)


def linear_fp16(x: Any, weight: Any, bias: Optional[Any] = None) -> np.ndarray:
    """
    FP16 Linear transformation with Tensor Core acceleration.
    """
    x_t = _to_tensor(x).half()
    w_t = _to_tensor(weight).half()
    b_t = _to_tensor(bias).half() if bias is not None else None

    with torch.no_grad():
        result = F.linear(x_t, w_t, b_t)

    return _to_numpy(result.float())


# =============================================================================
# Convolution Operations
# =============================================================================


def conv2d(
    x: Any, weight: Any, bias: Optional[Any] = None, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """
    2D Convolution using cuDNN.
    """
    x_t = _to_tensor(x)
    w_t = _to_tensor(weight)
    b_t = _to_tensor(bias) if bias is not None else None

    with torch.no_grad():
        result = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding)

    return _to_numpy(result)


# =============================================================================
# Normalization Operations
# =============================================================================


def layer_norm(
    x: Any,
    normalized_shape: Tuple[int, ...],
    weight: Optional[Any] = None,
    bias: Optional[Any] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Layer Normalization.
    """
    x_t = _to_tensor(x)
    w_t = _to_tensor(weight) if weight is not None else None
    b_t = _to_tensor(bias) if bias is not None else None

    with torch.no_grad():
        result = F.layer_norm(x_t, normalized_shape, w_t, b_t, eps)

    return _to_numpy(result)


def batch_norm(
    x: Any,
    running_mean: Any,
    running_var: Any,
    weight: Optional[Any] = None,
    bias: Optional[Any] = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> np.ndarray:
    """
    Batch Normalization.
    """
    x_t = _to_tensor(x)
    rm_t = _to_tensor(running_mean)
    rv_t = _to_tensor(running_var)
    w_t = _to_tensor(weight) if weight is not None else None
    b_t = _to_tensor(bias) if bias is not None else None

    with torch.no_grad():
        result = F.batch_norm(x_t, rm_t, rv_t, w_t, b_t, False, momentum, eps)

    return _to_numpy(result)


# =============================================================================
# Activation Functions
# =============================================================================


def relu(x: Any) -> np.ndarray:
    """ReLU activation."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = F.relu(x_t)
    return _to_numpy(result)


def gelu(x: Any) -> np.ndarray:
    """GELU activation (used in Transformers)."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = F.gelu(x_t)
    return _to_numpy(result)


def sigmoid(x: Any) -> np.ndarray:
    """Sigmoid activation."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = torch.sigmoid(x_t)
    return _to_numpy(result)


def tanh(x: Any) -> np.ndarray:
    """Tanh activation."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = torch.tanh(x_t)
    return _to_numpy(result)


def softmax(x: Any, axis: int = -1) -> np.ndarray:
    """Softmax activation."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = F.softmax(x_t, dim=axis)
    return _to_numpy(result)


# =============================================================================
# Elementwise Operations
# =============================================================================


def add(a: Any, b: Any) -> np.ndarray:
    """Elementwise addition."""
    a_t = _to_tensor(a)
    b_t = _to_tensor(b)
    with torch.no_grad():
        result = a_t + b_t
    return _to_numpy(result)


def mul(a: Any, b: Any) -> np.ndarray:
    """Elementwise multiplication."""
    a_t = _to_tensor(a)
    b_t = _to_tensor(b)
    with torch.no_grad():
        result = a_t * b_t
    return _to_numpy(result)


def sub(a: Any, b: Any) -> np.ndarray:
    """Elementwise subtraction."""
    a_t = _to_tensor(a)
    b_t = _to_tensor(b)
    with torch.no_grad():
        result = a_t - b_t
    return _to_numpy(result)


# =============================================================================
# Attention Operations
# =============================================================================


def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> np.ndarray:
    """
    Scaled Dot-Product Attention.

    Uses PyTorch's optimized implementation (Flash Attention when available).
    """
    q_t = _to_tensor(query)
    k_t = _to_tensor(key)
    v_t = _to_tensor(value)
    mask_t = _to_tensor(attn_mask) if attn_mask is not None else None

    with torch.no_grad():
        result = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=mask_t, dropout_p=dropout_p, is_causal=is_causal
        )

    return _to_numpy(result)


def multihead_attention(
    query: Any, key: Any, value: Any, num_heads: int, embed_dim: int
) -> np.ndarray:
    """
    Multi-Head Attention.

    Reshapes Q, K, V for multi-head computation and applies attention.
    """
    q_t = _to_tensor(query)
    k_t = _to_tensor(key)
    v_t = _to_tensor(value)

    batch_size = q_t.shape[0]
    seq_len = q_t.shape[1]
    head_dim = embed_dim // num_heads

    with torch.no_grad():
        # Reshape to (batch, heads, seq, head_dim)
        q = q_t.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_t.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_t.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

    return _to_numpy(attn_output)


# =============================================================================
# Fused Operations (for performance)
# =============================================================================


def fused_add_relu(a: Any, b: Any) -> np.ndarray:
    """Fused add + ReLU."""
    a_t = _to_tensor(a)
    b_t = _to_tensor(b)
    with torch.no_grad():
        result = F.relu(a_t + b_t)
    return _to_numpy(result)


def fused_bias_relu(x: Any, bias: Any) -> np.ndarray:
    """Fused bias + ReLU."""
    x_t = _to_tensor(x)
    b_t = _to_tensor(bias)
    with torch.no_grad():
        result = F.relu(x_t + b_t)
    return _to_numpy(result)


def fused_bias_gelu(x: Any, bias: Any) -> np.ndarray:
    """Fused bias + GELU."""
    x_t = _to_tensor(x)
    b_t = _to_tensor(bias)
    with torch.no_grad():
        result = F.gelu(x_t + b_t)
    return _to_numpy(result)


def fused_add_layernorm(
    x: Any, residual: Any, weight: Any, bias: Any, eps: float = 1e-5
) -> np.ndarray:
    """Fused add + LayerNorm."""
    x_t = _to_tensor(x)
    r_t = _to_tensor(residual)
    w_t = _to_tensor(weight)
    b_t = _to_tensor(bias)

    with torch.no_grad():
        combined = x_t + r_t
        normalized_shape = (combined.shape[-1],)
        result = F.layer_norm(combined, normalized_shape, w_t, b_t, eps)

    return _to_numpy(result)


# =============================================================================
# Pooling Operations
# =============================================================================


def max_pool2d(x: Any, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    """2D Max Pooling."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = F.max_pool2d(x_t, kernel_size, stride)
    return _to_numpy(result)


def avg_pool2d(x: Any, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    """2D Average Pooling."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        result = F.avg_pool2d(x_t, kernel_size, stride)
    return _to_numpy(result)


# =============================================================================
# Reduction Operations
# =============================================================================


def reduce_sum(
    x: Any, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    """Sum reduction."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        if axis is None:
            result = x_t.sum()
        else:
            result = x_t.sum(dim=axis, keepdim=keepdims)
    return _to_numpy(result)


def reduce_mean(
    x: Any, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    """Mean reduction."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        if axis is None:
            result = x_t.mean()
        else:
            result = x_t.mean(dim=axis, keepdim=keepdims)
    return _to_numpy(result)


def reduce_max(
    x: Any, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    """Max reduction."""
    x_t = _to_tensor(x)
    with torch.no_grad():
        if axis is None:
            result = x_t.max()
        else:
            result = x_t.max(dim=axis, keepdim=keepdims).values
    return _to_numpy(result)


# =============================================================================
# Kernel Registry Integration
# =============================================================================


def get_kernel_map() -> dict:
    """
    Get mapping of operation types to PyTorch kernel functions.

    Returns dictionary suitable for KernelRegistry registration.
    """
    if not is_available():
        return {}

    return {
        # Linear
        "MatMul": matmul,
        "Gemm": linear,
        "Linear": linear,
        "LinearFP16": linear_fp16,
        # Convolution
        "Conv": conv2d,
        "Conv2d": conv2d,
        # Normalization
        "LayerNorm": layer_norm,
        "LayerNormalization": layer_norm,
        "BatchNorm": batch_norm,
        "BatchNormalization": batch_norm,
        # Activation
        "Relu": relu,
        "ReLU": relu,
        "Gelu": gelu,
        "GELU": gelu,
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "Softmax": softmax,
        # Elementwise
        "Add": add,
        "Mul": mul,
        "Sub": sub,
        # Attention
        "Attention": scaled_dot_product_attention,
        "MultiHeadAttention": multihead_attention,
        # Fused
        "FusedAddReLU": fused_add_relu,
        "FusedBiasReLU": fused_bias_relu,
        "FusedBiasGeLU": fused_bias_gelu,
        "FusedAddLayerNorm": fused_add_layernorm,
        # Pooling
        "MaxPool": max_pool2d,
        "MaxPool2d": max_pool2d,
        "AvgPool2d": avg_pool2d,
        # Reduction
        "Sum": reduce_sum,
        "ReduceSum": reduce_sum,
        "Mean": reduce_mean,
        "ReduceMean": reduce_mean,
        "Max": reduce_max,
        "ReduceMax": reduce_max,
    }
