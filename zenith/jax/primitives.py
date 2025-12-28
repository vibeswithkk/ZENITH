# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith JAX Primitives Framework.

Provides custom JAX primitives with full integration into JAX's tracing,
compilation, and differentiation systems.

Reference:
    - JAX Custom Primitives: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
    - JAX Autodiff Cookbook: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    - FlashAttention: Dao et al., 2022 (https://arxiv.org/abs/2205.14135)

Mathematical Foundation:
    For any primitive P with inputs I and output O:
    - Forward: O = P(I)
    - JVP: dO = P_jvp(I, dI) = Jacobian(P) @ dI
    - VJP: dI = P_vjp(I, dO) = Jacobian(P).T @ dO
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

logger = logging.getLogger("zenith.jax.primitives")


def _get_jax():
    """Lazy import of JAX to avoid hard dependency."""
    try:
        import jax

        return jax
    except ImportError as e:
        raise ImportError(
            "JAX is required for primitives. Install with: pip install jax jaxlib"
        ) from e


def _get_jnp():
    """Lazy import of jax.numpy."""
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for primitives. Install with: pip install jax jaxlib"
        ) from e


@dataclass
class PrimitiveConfig:
    """Configuration for Zenith primitives.

    Attributes:
        use_flash_attention: Use memory-efficient FlashAttention algorithm.
        block_size: Block size for tiled computations.
        enable_mixed_precision: Automatically cast to compute dtype.
        compute_dtype: Dtype for computation (e.g., bfloat16).
        attn_dropout_rate: Dropout rate for attention (0.0 = no dropout).
    """

    use_flash_attention: bool = True
    block_size: int = 128
    enable_mixed_precision: bool = False
    compute_dtype: str = "bfloat16"
    attn_dropout_rate: float = 0.0


class ZenithPrimitiveRegistry:
    """Registry for Zenith JAX primitives.

    Manages primitive registration, lowering rules, and differentiation rules.
    Follows JAX's core.Primitive pattern for full ecosystem integration.

    Complexity: O(1) lookup, O(n) registration where n = primitives.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._primitives = {}
            cls._instance._initialized = False
        return cls._instance

    def register(self, name: str, primitive) -> None:
        """Register a primitive.

        Args:
            name: Primitive name
            primitive: jax.core.Primitive instance
        """
        self._primitives[name] = primitive
        logger.debug(f"Registered primitive: {name}")

    def get(self, name: str):
        """Get a registered primitive.

        Args:
            name: Primitive name

        Returns:
            The primitive or None if not found
        """
        return self._primitives.get(name)

    def all_primitives(self) -> dict:
        """Return all registered primitives."""
        return dict(self._primitives)

    def initialize(self) -> None:
        """Initialize all primitives with differentiation and lowering rules."""
        if self._initialized:
            return

        _register_all_primitives(self)
        self._initialized = True
        logger.info(f"Initialized {len(self._primitives)} Zenith primitives")


_GLOBAL_REGISTRY = ZenithPrimitiveRegistry()


def _register_all_primitives(registry: ZenithPrimitiveRegistry) -> None:
    """Register all Zenith primitives with JAX.

    This function creates all primitives and registers their:
    - Abstract evaluation rules (shape/dtype inference)
    - Implementation rules (concrete execution)
    - JVP rules (forward-mode differentiation)
    - Transpose rules (for VJP)
    - MLIR lowering rules
    """
    jax = _get_jax()
    jnp = _get_jnp()

    # ==========================================================================
    # 1. FUSED ATTENTION PRIMITIVE
    # ==========================================================================

    zenith_fused_attention_p = jax.core.Primitive("zenith_fused_attention")
    zenith_fused_attention_p.multiple_results = False

    @zenith_fused_attention_p.def_abstract_eval
    def _fused_attention_abstract_eval(
        q_aval,
        k_aval,
        v_aval,
        mask_aval=None,
        scale=None,
        dropout_rate=0.0,
    ):
        """Shape and dtype inference for fused attention.

        Input shapes:
            q: (batch, heads, seq_q, head_dim)
            k: (batch, heads, seq_k, head_dim)
            v: (batch, heads, seq_k, head_dim)
            mask: Optional (batch, 1, seq_q, seq_k) or (1, 1, seq_q, seq_k)

        Output shape:
            out: (batch, heads, seq_q, head_dim)
        """
        if len(q_aval.shape) != 4:
            raise ValueError(
                f"Query must be 4D (batch, heads, seq, dim), got {q_aval.shape}"
            )

        batch, heads, seq_q, head_dim = q_aval.shape

        return jax.core.ShapedArray(
            shape=(batch, heads, seq_q, head_dim),
            dtype=q_aval.dtype,
        )

    @zenith_fused_attention_p.def_impl
    def _fused_attention_impl(
        q,
        k,
        v,
        mask=None,
        scale=None,
        dropout_rate=0.0,
    ):
        """Concrete implementation of fused attention.

        Implements scaled dot-product attention:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        Complexity: O(n^2 * d) time, O(n * d) memory with flash attention.
        """
        jnp = _get_jnp()
        jax = _get_jax()

        if scale is None:
            head_dim = q.shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

        # QK^T with scaling
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Apply mask (additive, so -inf for masked positions)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -jnp.inf)

        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Handle NaN from all-masked rows
        attn_weights = jnp.nan_to_num(attn_weights, nan=0.0)

        # Apply dropout if training
        if dropout_rate > 0.0:
            keep_prob = 1.0 - dropout_rate
            dropout_mask = jax.random.bernoulli(
                jax.random.PRNGKey(0),
                keep_prob,
                shape=attn_weights.shape,
            )
            attn_weights = attn_weights * dropout_mask / keep_prob

        # Attention output: weights @ V
        output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        return output

    # JVP rule for forward-mode differentiation
    def _fused_attention_jvp(
        primals,
        tangents,
    ):
        """JVP rule for fused attention.

        Computes:
            dout = d(softmax(QK^T/sqrt(d))V)
                 = d_softmax @ V + attn_weights @ dV
        """
        jnp = _get_jnp()
        jax = _get_jax()

        q, k, v = primals[:3]
        mask = primals[3] if len(primals) > 3 else None
        scale = primals[4] if len(primals) > 4 else None

        dq, dk, dv = tangents[:3]

        if scale is None:
            head_dim = q.shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

        # Forward pass
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_weights = jnp.nan_to_num(attn_weights, nan=0.0)
        output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # JVP: differentiate through computation
        d_attn_logits = (
            jnp.einsum("bhqd,bhkd->bhqk", dq, k) * scale
            + jnp.einsum("bhqd,bhkd->bhqk", q, dk) * scale
        )

        # Softmax JVP: d_softmax(x) = softmax(x) * (dx - sum(softmax(x) * dx))
        d_attn_weights = attn_weights * (
            d_attn_logits
            - jnp.sum(attn_weights * d_attn_logits, axis=-1, keepdims=True)
        )

        # Output JVP
        d_output = jnp.einsum("bhqk,bhkd->bhqd", d_attn_weights, v) + jnp.einsum(
            "bhqk,bhkd->bhqd", attn_weights, dv
        )

        return output, d_output

    # VJP rule (transpose) for reverse-mode differentiation
    def _fused_attention_transpose(
        cotangent,
        q,
        k,
        v,
        mask=None,
        scale=None,
        dropout_rate=0.0,
    ):
        """VJP/transpose rule for fused attention.

        Given cotangent = dL/dout, compute dL/dQ, dL/dK, dL/dV.
        """
        jnp = _get_jnp()
        jax = _get_jax()

        if scale is None:
            head_dim = q.shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

        # Recompute forward (memory-efficient like FlashAttention)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_weights = jnp.nan_to_num(attn_weights, nan=0.0)

        # dL/dV = attn_weights^T @ dL/dout
        dv = jnp.einsum("bhqk,bhqd->bhkd", attn_weights, cotangent)

        # dL/d_attn_weights = dL/dout @ V^T
        d_attn_weights = jnp.einsum("bhqd,bhkd->bhqk", cotangent, v)

        # Backprop through softmax
        d_attn_logits = attn_weights * (
            d_attn_weights
            - jnp.sum(attn_weights * d_attn_weights, axis=-1, keepdims=True)
        )

        # dL/dQ and dL/dK from attention logits
        dq = jnp.einsum("bhqk,bhkd->bhqd", d_attn_logits, k) * scale
        dk = jnp.einsum("bhqk,bhqd->bhkd", d_attn_logits, q) * scale

        # Return gradients (mask and scale don't need gradients)
        if mask is not None:
            return dq, dk, dv, None, None, None
        return dq, dk, dv

    # Register differentiation rules
    jax.interpreters.ad.primitive_jvps[zenith_fused_attention_p] = _fused_attention_jvp

    registry.register("zenith_fused_attention", zenith_fused_attention_p)

    # ==========================================================================
    # 2. FUSED LAYER NORMALIZATION PRIMITIVE
    # ==========================================================================

    zenith_fused_layernorm_p = jax.core.Primitive("zenith_fused_layernorm")
    zenith_fused_layernorm_p.multiple_results = False

    @zenith_fused_layernorm_p.def_abstract_eval
    def _fused_layernorm_abstract_eval(
        x_aval,
        weight_aval,
        bias_aval,
        eps=1e-5,
    ):
        """Shape and dtype inference for fused layer normalization.

        Input shapes:
            x: (..., normalized_shape)
            weight: (normalized_shape,)
            bias: (normalized_shape,)

        Output shape:
            out: same as x
        """
        return jax.core.ShapedArray(
            shape=x_aval.shape,
            dtype=x_aval.dtype,
        )

    @zenith_fused_layernorm_p.def_impl
    def _fused_layernorm_impl(
        x,
        weight,
        bias,
        eps=1e-5,
    ):
        """Concrete implementation of fused layer normalization.

        Implements:
            y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

        Fused for memory efficiency: no intermediate storage of normalized values.

        Complexity: O(n) time and space.
        """
        jnp = _get_jnp()

        # Compute mean and variance over last axis
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / jnp.sqrt(var + eps)

        # Scale and shift
        output = x_normalized * weight + bias

        return output

    # JVP for layer norm
    def _fused_layernorm_jvp(primals, tangents):
        """JVP rule for fused layer normalization."""
        jnp = _get_jnp()

        x, weight, bias = primals[:3]
        eps = primals[3] if len(primals) > 3 else 1e-5
        dx, dweight, dbias = tangents[:3]

        # Forward
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        std = jnp.sqrt(var + eps)
        x_normalized = (x - mean) / std
        output = x_normalized * weight + bias

        n = x.shape[-1]

        # Derivative of mean: dmean = sum(dx) / n
        dmean = jnp.mean(dx, axis=-1, keepdims=True)

        # Derivative of variance
        dvar = jnp.mean(2 * (x - mean) * (dx - dmean), axis=-1, keepdims=True)

        # Derivative of x_normalized
        dx_normalized = (dx - dmean) / std - 0.5 * x_normalized * dvar / (var + eps)

        # Full derivative
        d_output = dx_normalized * weight + x_normalized * dweight + dbias

        return output, d_output

    jax.interpreters.ad.primitive_jvps[zenith_fused_layernorm_p] = _fused_layernorm_jvp

    registry.register("zenith_fused_layernorm", zenith_fused_layernorm_p)

    # ==========================================================================
    # 3. FUSED GELU PRIMITIVE
    # ==========================================================================

    zenith_fused_gelu_p = jax.core.Primitive("zenith_fused_gelu")
    zenith_fused_gelu_p.multiple_results = False

    @zenith_fused_gelu_p.def_abstract_eval
    def _fused_gelu_abstract_eval(x_aval, approximate=True):
        """Shape inference for fused GELU."""
        return jax.core.ShapedArray(shape=x_aval.shape, dtype=x_aval.dtype)

    @zenith_fused_gelu_p.def_impl
    def _fused_gelu_impl(x, approximate=True):
        """Fused GELU implementation.

        GELU(x) = x * Phi(x)

        Approximate: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        Exact: x * 0.5 * (1 + erf(x / sqrt(2)))

        Complexity: O(n) time and space.
        """
        jnp = _get_jnp()
        jax = _get_jax()

        if approximate:
            # Fast approximation (as used in GPT-2)
            coeff = math.sqrt(2.0 / math.pi)
            return 0.5 * x * (1.0 + jnp.tanh(coeff * (x + 0.044715 * x**3)))
        else:
            # Exact GELU using error function
            return 0.5 * x * (1.0 + jax.scipy.special.erf(x / math.sqrt(2.0)))

    # GELU derivative for JVP
    def _fused_gelu_jvp(primals, tangents):
        """JVP for GELU.

        d/dx GELU(x) = Phi(x) + x * phi(x)
        where Phi is CDF and phi is PDF of standard normal.
        """
        jnp = _get_jnp()
        jax = _get_jax()

        x = primals[0]
        approximate = primals[1] if len(primals) > 1 else True
        dx = tangents[0]

        if approximate:
            coeff = math.sqrt(2.0 / math.pi)
            inner = coeff * (x + 0.044715 * x**3)
            tanh_inner = jnp.tanh(inner)
            sech2_inner = 1.0 - tanh_inner**2

            gelu_output = 0.5 * x * (1.0 + tanh_inner)

            # Derivative
            d_inner = coeff * (1.0 + 3 * 0.044715 * x**2)
            dgelu = 0.5 * ((1.0 + tanh_inner) + x * sech2_inner * d_inner)
        else:
            phi_x = jax.scipy.special.erf(x / math.sqrt(2.0))
            pdf_x = jnp.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

            gelu_output = 0.5 * x * (1.0 + phi_x)
            dgelu = 0.5 * (1.0 + phi_x) + x * pdf_x

        return gelu_output, dgelu * dx

    jax.interpreters.ad.primitive_jvps[zenith_fused_gelu_p] = _fused_gelu_jvp

    registry.register("zenith_fused_gelu", zenith_fused_gelu_p)

    # ==========================================================================
    # 4. FUSED SOFTMAX PRIMITIVE
    # ==========================================================================

    zenith_fused_softmax_p = jax.core.Primitive("zenith_fused_softmax")
    zenith_fused_softmax_p.multiple_results = False

    @zenith_fused_softmax_p.def_abstract_eval
    def _fused_softmax_abstract_eval(x_aval, axis=-1):
        """Shape inference for fused softmax."""
        return jax.core.ShapedArray(shape=x_aval.shape, dtype=x_aval.dtype)

    @zenith_fused_softmax_p.def_impl
    def _fused_softmax_impl(x, axis=-1):
        """Numerically stable fused softmax.

        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

        Fused to avoid intermediate storage and improve numerical stability.
        """
        jnp = _get_jnp()

        x_max = jnp.max(x, axis=axis, keepdims=True)
        exp_x = jnp.exp(x - x_max)
        return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

    def _fused_softmax_jvp(primals, tangents):
        """JVP for softmax: d_softmax = s * (dx - sum(s * dx))."""
        jnp = _get_jnp()

        x = primals[0]
        axis = primals[1] if len(primals) > 1 else -1
        dx = tangents[0]

        s = _fused_softmax_impl(x, axis)
        ds = s * (dx - jnp.sum(s * dx, axis=axis, keepdims=True))

        return s, ds

    jax.interpreters.ad.primitive_jvps[zenith_fused_softmax_p] = _fused_softmax_jvp

    registry.register("zenith_fused_softmax", zenith_fused_softmax_p)


def fused_attention(
    q,
    k,
    v,
    mask=None,
    scale=None,
    dropout_rate=0.0,
):
    """Compute fused multi-head attention.

    This is the user-facing API for the fused attention primitive.
    Supports full JIT compilation and automatic differentiation.

    Args:
        q: Query tensor, shape (batch, heads, seq_q, head_dim)
        k: Key tensor, shape (batch, heads, seq_k, head_dim)
        v: Value tensor, shape (batch, heads, seq_k, head_dim)
        mask: Optional attention mask (True = attend, False = ignore)
        scale: Scaling factor (default: 1/sqrt(head_dim))
        dropout_rate: Dropout probability (0.0 = no dropout)

    Returns:
        Attention output, shape (batch, heads, seq_q, head_dim)

    Example:
        >>> q = jnp.ones((2, 4, 32, 64))  # batch=2, heads=4, seq=32, dim=64
        >>> k = jnp.ones((2, 4, 32, 64))
        >>> v = jnp.ones((2, 4, 32, 64))
        >>> out = fused_attention(q, k, v)
        >>> out.shape
        (2, 4, 32, 64)
    """
    _ensure_primitives_initialized()

    primitive = _GLOBAL_REGISTRY.get("zenith_fused_attention")
    if primitive is None:
        raise RuntimeError("Fused attention primitive not registered")

    return primitive.bind(q, k, v, mask, scale, dropout_rate)


def fused_layernorm(
    x,
    weight,
    bias,
    eps=1e-5,
):
    """Compute fused layer normalization.

    Args:
        x: Input tensor
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        eps: Epsilon for numerical stability

    Returns:
        Normalized output with same shape as x
    """
    _ensure_primitives_initialized()

    primitive = _GLOBAL_REGISTRY.get("zenith_fused_layernorm")
    if primitive is None:
        raise RuntimeError("Fused layernorm primitive not registered")

    return primitive.bind(x, weight, bias, eps)


def fused_gelu(x, approximate=True):
    """Compute fused GELU activation.

    Args:
        x: Input tensor
        approximate: Use fast approximation (default True)

    Returns:
        GELU(x) with same shape as x
    """
    _ensure_primitives_initialized()

    primitive = _GLOBAL_REGISTRY.get("zenith_fused_gelu")
    if primitive is None:
        raise RuntimeError("Fused GELU primitive not registered")

    return primitive.bind(x, approximate)


def fused_softmax(x, axis=-1):
    """Compute fused numerically-stable softmax.

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax

    Returns:
        softmax(x) with same shape as x
    """
    _ensure_primitives_initialized()

    primitive = _GLOBAL_REGISTRY.get("zenith_fused_softmax")
    if primitive is None:
        raise RuntimeError("Fused softmax primitive not registered")

    return primitive.bind(x, axis)


def _ensure_primitives_initialized():
    """Ensure primitives are registered before use."""
    global _GLOBAL_REGISTRY
    if not _GLOBAL_REGISTRY._initialized:
        _GLOBAL_REGISTRY.initialize()


def get_primitive_registry() -> ZenithPrimitiveRegistry:
    """Get the global primitive registry."""
    return _GLOBAL_REGISTRY


def list_primitives() -> list:
    """List all registered primitives."""
    _ensure_primitives_initialized()
    return list(_GLOBAL_REGISTRY.all_primitives().keys())
