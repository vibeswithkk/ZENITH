# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
FX Graph Pattern Replacement for torch.compile Backend

This module provides pattern matching and replacement functionality
for PyTorch FX graphs, enabling:
- Attention pattern detection and replacement with Zenith kernels
- Gemm/Linear fusion
- Custom operator dispatch

Uses PyTorch's torch.fx.subgraph_rewriter for pattern matching.
"""

import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger("zenith.optimization.fx_patterns")

# Check for torch availability
_HAS_TORCH = False
torch = None
F = None

try:
    import torch as _torch
    import torch.nn.functional as _F

    torch = _torch
    F = _F
    _HAS_TORCH = True
except ImportError:
    pass


@dataclass
class FXPattern:
    """Defines an FX graph pattern for replacement."""

    name: str
    pattern_fn: Callable
    replacement_fn: Callable
    priority: int = 10
    enabled: bool = True


def _check_torch_available():
    """Raise error if torch not available."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for FX pattern optimization")


# =============================================================================
# Attention Patterns
# =============================================================================


def sdpa_pattern(q, k, v):
    """
    Standard Scaled Dot-Product Attention pattern.

    Matches: softmax(Q @ K^T / sqrt(d)) @ V
    """
    _check_torch_available()
    # Standard attention computation
    scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def zenith_sdpa_replacement(q, k, v):
    """
    Replacement using PyTorch's optimized SDPA (which uses FlashAttention).

    This is a validated replacement that:
    1. Uses PyTorch's F.scaled_dot_product_attention
    2. Automatically enables FlashAttention when available
    3. Falls back gracefully on unsupported hardware
    """
    _check_torch_available()
    # PyTorch 2.0+ SDPA with automatic FlashAttention
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def causal_sdpa_pattern(q, k, v, mask):
    """Causal attention pattern with mask."""
    _check_torch_available()
    scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = attn_weights + mask  # Causal mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def zenith_causal_sdpa_replacement(q, k, v, mask):
    """Causal replacement using SDPA."""
    _check_torch_available()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)


# =============================================================================
# GELU Patterns
# =============================================================================


def gelu_tanh_pattern(x):
    """
    GELU approximation using tanh.

    Matches: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    _check_torch_available()
    return (
        0.5
        * x
        * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3))))
    )


def zenith_gelu_replacement(x):
    """Replacement using PyTorch's optimized GELU."""
    _check_torch_available()
    return F.gelu(x, approximate="tanh")


# =============================================================================
# LayerNorm Patterns
# =============================================================================


def layernorm_pattern(x, weight, bias, eps=1e-5):
    """
    Manual LayerNorm computation pattern.

    Matches: (x - mean) / sqrt(var + eps) * weight + bias
    """
    _check_torch_available()
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * weight + bias


def zenith_layernorm_replacement(x, weight, bias, eps=1e-5):
    """Replacement using PyTorch's fused LayerNorm."""
    _check_torch_available()
    return F.layer_norm(x, weight.shape, weight, bias, eps)


# =============================================================================
# Pattern Registry
# =============================================================================


def get_attention_patterns() -> list[FXPattern]:
    """Get all attention-related patterns."""
    return [
        FXPattern(
            name="sdpa_to_flash",
            pattern_fn=sdpa_pattern,
            replacement_fn=zenith_sdpa_replacement,
            priority=20,
        ),
    ]


def get_activation_patterns() -> list[FXPattern]:
    """Get activation function patterns."""
    return [
        FXPattern(
            name="gelu_tanh_to_native",
            pattern_fn=gelu_tanh_pattern,
            replacement_fn=zenith_gelu_replacement,
            priority=10,
        ),
    ]


def get_normalization_patterns() -> list[FXPattern]:
    """Get normalization patterns."""
    return [
        FXPattern(
            name="manual_layernorm_to_native",
            pattern_fn=layernorm_pattern,
            replacement_fn=zenith_layernorm_replacement,
            priority=10,
        ),
    ]


def get_all_patterns() -> list[FXPattern]:
    """Get all available FX patterns."""
    patterns = []
    patterns.extend(get_attention_patterns())
    patterns.extend(get_activation_patterns())
    patterns.extend(get_normalization_patterns())
    return patterns
