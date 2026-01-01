# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Triton Kernels - High-Performance Fused GPU Operations.

This module provides Triton-based GPU kernels that achieve real speedup
through kernel fusion and optimized memory access patterns.

Key optimizations:
1. Kernel Fusion - Multiple ops in single kernel (no intermediate memory)
2. Auto-tuning - Hardware-optimal block sizes
3. Memory Coalescing - Efficient GPU memory access

Performance targets:
- Fused Linear+GELU: 1.5-2x vs separate ops
- Fused Linear+ReLU: 1.3-1.5x vs separate ops
- Fused LayerNorm: 1.2-1.4x vs PyTorch
"""

import logging
from typing import Optional, Callable
from functools import lru_cache

logger = logging.getLogger("zenith.runtime.triton_kernels")

# Check Triton availability
_HAS_TRITON = False
_TRITON_VERSION = None
triton = None
tl = None

try:
    import triton as _triton
    import triton.language as _tl

    triton = _triton
    tl = _tl
    _HAS_TRITON = True
    _TRITON_VERSION = getattr(_triton, "__version__", "unknown")
    logger.info(f"Triton {_TRITON_VERSION} available for kernel generation")
except ImportError:
    logger.debug("Triton not available, using fallback kernels")

# Check PyTorch availability
_HAS_TORCH = False
torch = None

try:
    import torch as _torch

    torch = _torch
    _HAS_TORCH = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if Triton kernels are available."""
    return _HAS_TRITON and _HAS_TORCH


def get_version() -> Optional[str]:
    """Get Triton version if available."""
    return _TRITON_VERSION


# =============================================================================
# FUSED LINEAR + GELU KERNEL
# =============================================================================

if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fused_linear_gelu_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_outm,
        stride_outn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused Linear + GELU kernel.

        Computes: GELU(X @ W.T + bias)

        Instead of:
        1. tmp = X @ W.T (memory write)
        2. tmp = tmp + bias (memory read + write)
        3. out = gelu(tmp) (memory read + write)

        We do everything in one pass with data in registers.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_offs = k + offs_k

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn

            mask_m = offs_m[:, None] < M
            mask_k = k_offs[None, :] < K
            mask_n = offs_n[None, :] < N

            x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)
            w = tl.load(w_ptrs, mask=(k_offs[:, None] < K) & mask_n, other=0.0)

            acc += tl.dot(x, w)

        if HAS_BIAS:
            b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc = acc + b[None, :]

        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        SQRT_2_OVER_PI = 0.7978845608028654
        x = acc
        x3 = x * x * x
        inner = SQRT_2_OVER_PI * (x + 0.044715 * x3)
        out = 0.5 * x * (1.0 + tl.libdevice.tanh(inner))

        out_ptrs = (
            out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        )
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out, mask=mask_out)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fused_linear_relu_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_outm,
        stride_outn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused Linear + ReLU kernel."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_offs = k + offs_k

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn

            x = tl.load(
                x_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0
            )
            w = tl.load(
                w_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0
            )

            acc += tl.dot(x, w)

        if HAS_BIAS:
            b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc = acc + b[None, :]

        # ReLU
        out = tl.maximum(acc, 0.0)

        out_ptrs = (
            out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        )
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out, mask=mask_out)


# =============================================================================
# PYTHON WRAPPER FUNCTIONS
# =============================================================================


def fused_linear_gelu(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    bias: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """
    Fused Linear + GELU operation.

    Computes: GELU(x @ weight.T + bias)

    Args:
        x: Input tensor of shape (M, K)
        weight: Weight tensor of shape (N, K)
        bias: Optional bias tensor of shape (N,)

    Returns:
        Output tensor of shape (M, N)
    """
    if not is_available():
        import torch.nn.functional as F

        out = torch.nn.functional.linear(x, weight, bias)
        return F.gelu(out)

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: x has {K} cols, weight has {K_w} rows"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_linear_gelu_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
    )

    return out


def fused_linear_relu(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    bias: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """
    Fused Linear + ReLU operation.

    Computes: ReLU(x @ weight.T + bias)
    """
    if not is_available():
        import torch.nn.functional as F

        out = torch.nn.functional.linear(x, weight, bias)
        return F.relu(out)

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_linear_relu_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
    )

    return out


# =============================================================================
# KERNEL REGISTRY INTEGRATION
# =============================================================================


def get_triton_kernel_map() -> dict:
    """
    Get map of operation types to Triton kernel functions.

    Returns:
        Dict mapping op names to kernel functions.
    """
    if not is_available():
        return {}

    return {
        "FusedLinearGELU": fused_linear_gelu,
        "FusedLinearReLU": fused_linear_relu,
        "Linear+GELU": fused_linear_gelu,
        "Linear+ReLU": fused_linear_relu,
    }


def register_triton_kernels(registry) -> int:
    """
    Register Triton kernels with the kernel registry.

    Args:
        registry: KernelRegistry instance.

    Returns:
        Number of kernels registered.
    """
    if not is_available():
        logger.debug("Triton not available, skipping kernel registration")
        return 0

    from .kernel_registry import KernelSpec, Precision

    count = 0

    registry.register(
        KernelSpec(
            name="triton_fused_linear_gelu",
            op_types=["FusedLinearGELU", "Linear+GELU", "Gemm+GELU"],
            precision=Precision.FP32,
            kernel_fn=fused_linear_gelu,
            priority=30,
            requires_gpu=True,
        )
    )
    count += 1

    registry.register(
        KernelSpec(
            name="triton_fused_linear_relu",
            op_types=["FusedLinearReLU", "Linear+ReLU", "Gemm+ReLU"],
            precision=Precision.FP32,
            kernel_fn=fused_linear_relu,
            priority=30,
            requires_gpu=True,
        )
    )
    count += 1

    logger.info(f"Registered {count} Triton kernels with priority 30")
    return count


# =============================================================================
# BENCHMARKING
# =============================================================================


def benchmark_fused_linear_gelu(
    M: int = 1024, N: int = 4096, K: int = 1024, runs: int = 100
):
    """
    Benchmark fused vs separate Linear+GELU.

    Returns dict with timing comparison.
    """
    if not is_available():
        return {"error": "Triton not available"}

    import time

    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, K, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = fused_linear_gelu(x, weight, bias)
        _ = torch.nn.functional.gelu(torch.nn.functional.linear(x, weight, bias))
    torch.cuda.synchronize()

    # Benchmark fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = fused_linear_gelu(x, weight, bias)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - t0) / runs * 1000

    # Benchmark separate
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        tmp = torch.nn.functional.linear(x, weight, bias)
        _ = torch.nn.functional.gelu(tmp)
    torch.cuda.synchronize()
    separate_time = (time.perf_counter() - t0) / runs * 1000

    speedup = separate_time / fused_time if fused_time > 0 else 0

    return {
        "fused_ms": fused_time,
        "separate_ms": separate_time,
        "speedup": speedup,
        "shape": f"({M}, {K}) x ({N}, {K})",
    }
