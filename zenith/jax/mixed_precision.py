# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith JAX Mixed Precision Training Module.

Provides automatic mixed precision (AMP) support for JAX with
BF16 and FP16 precision policies.

Technical Foundation:
--------------------
Based on:
- Micikevicius et al., 2018: "Mixed Precision Training" (ICLR)
- Google DeepMind jmp library patterns
- NVIDIA Apex AMP concepts

Precision Types:
---------------
- BF16 (bfloat16): Same dynamic range as FP32, reduced precision
    Range: [-3.39e38, 3.39e38]
    Recommended for TPU and Ampere+ GPUs
    Does NOT require loss scaling

- FP16 (float16): IEEE half precision
    Range: [-65504, 65504]
    Requires loss scaling to prevent underflow/overflow
    Better hardware support on older GPUs

Mathematical Foundation:
-----------------------
For gradient g in FP16 training with loss scale S:
    g_scaled = g * S
    g_unscaled = g_scaled / S

Precision preserved when:
    |g| * S <= 65504      (FP16 max)
    |g| * S >= 2^-24      (FP16 min positive)

Dynamic loss scaling adjusts S based on gradient health:
    If gradients contain inf/nan: S = S * backoff_factor
    Else after growth_interval good steps: S = S * growth_factor

References:
----------
1. Micikevicius et al., 2018: "Mixed Precision Training"
2. DeepMind jmp: https://github.com/deepmind/jmp
3. JAX dtype promotion rules
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

logger = logging.getLogger("zenith.jax.mixed_precision")

F = TypeVar("F", bound=Callable[..., Any])
PyTree = Any


def _get_jax():
    """Lazy import of JAX with error handling."""
    try:
        import jax

        return jax
    except ImportError as e:
        raise ImportError(
            "JAX is required for mixed precision training. "
            "Install with: pip install jax jaxlib"
        ) from e


def _get_jnp():
    """Lazy import of jax.numpy."""
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for mixed precision training. "
            "Install with: pip install jax jaxlib"
        ) from e


class PrecisionMode(Enum):
    """
    Supported precision modes for training.

    FP32: Standard full precision (no mixed precision)
    FP16: Half precision with loss scaling
    BF16: Brain floating point (recommended for TPU/Ampere+)
    FP8: 8-bit float (experimental, requires H100+)
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


@dataclass
class MixedPrecisionPolicy:
    """
    Mixed precision policy specification.

    Defines dtype for different stages of computation:
    - param_dtype: Storage dtype for model parameters
    - compute_dtype: Dtype for forward/backward computation
    - output_dtype: Dtype for outputs and activations

    Based on Google DeepMind's jmp library pattern.

    Attributes:
        param_dtype: Parameter storage dtype (typically float32)
        compute_dtype: Computation dtype (bfloat16 or float16)
        output_dtype: Output dtype (typically same as compute_dtype)
        mode: Precision mode enum
    """

    param_dtype: str = "float32"
    compute_dtype: str = "float32"
    output_dtype: str = "float32"
    mode: PrecisionMode = PrecisionMode.FP32

    @classmethod
    def fp32(cls) -> "MixedPrecisionPolicy":
        """
        Full precision policy (no mixed precision).

        Use when:
        - Maximum precision is required
        - Debugging numerical issues
        - Model is sensitive to precision
        """
        return cls(
            param_dtype="float32",
            compute_dtype="float32",
            output_dtype="float32",
            mode=PrecisionMode.FP32,
        )

    @classmethod
    def bf16(cls) -> "MixedPrecisionPolicy":
        """
        BF16 mixed precision policy.

        Recommended for:
        - Google TPU
        - NVIDIA Ampere+ GPUs (A100, H100, RTX 30xx+)
        - AMD MI200+

        Advantages:
        - Same dynamic range as FP32, no loss scaling needed
        - 2x memory reduction
        - Up to 2x speedup on supported hardware
        """
        return cls(
            param_dtype="float32",
            compute_dtype="bfloat16",
            output_dtype="bfloat16",
            mode=PrecisionMode.BF16,
        )

    @classmethod
    def fp16(cls) -> "MixedPrecisionPolicy":
        """
        FP16 mixed precision policy.

        Recommended for:
        - NVIDIA Volta/Turing GPUs (V100, T4, RTX 20xx)
        - Older GPU architectures

        Advantages:
        - 2x memory reduction
        - Up to 2x speedup on Tensor Cores

        Note: Requires loss scaling for stability
        """
        return cls(
            param_dtype="float32",
            compute_dtype="float16",
            output_dtype="float16",
            mode=PrecisionMode.FP16,
        )

    def get_jax_dtype(self, dtype_name: str):
        """Convert dtype name to JAX dtype."""
        jnp = _get_jnp()

        dtype_map = {
            "float32": jnp.float32,
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
            "float64": jnp.float64,
        }

        return dtype_map.get(dtype_name, jnp.float32)

    @property
    def compute_jax_dtype(self):
        """Get compute dtype as JAX dtype."""
        return self.get_jax_dtype(self.compute_dtype)

    @property
    def param_jax_dtype(self):
        """Get param dtype as JAX dtype."""
        return self.get_jax_dtype(self.param_dtype)

    @property
    def output_jax_dtype(self):
        """Get output dtype as JAX dtype."""
        return self.get_jax_dtype(self.output_dtype)

    @property
    def requires_loss_scaling(self) -> bool:
        """Check if this policy requires loss scaling."""
        return self.mode == PrecisionMode.FP16


@dataclass
class LossScalerConfig:
    """
    Configuration for dynamic loss scaling.

    Attributes:
        initial_scale: Initial loss scale value (power of 2)
        growth_factor: Factor to increase scale on good steps
        backoff_factor: Factor to decrease scale on inf/nan
        growth_interval: Steps between scale increases
        min_scale: Minimum allowed scale value
        max_scale: Maximum allowed scale value
    """

    initial_scale: float = 2.0**15  # 32768
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    min_scale: float = 1.0
    max_scale: float = 2.0**24  # 16777216


@dataclass
class LossScalerState:
    """
    State for dynamic loss scaler.

    Maintained across training steps for scale adjustment.
    """

    scale: float
    good_steps: int = 0
    total_steps: int = 0
    overflow_count: int = 0
    underflow_count: int = 0


class DynamicLossScaler:
    """
    Dynamic loss scaling for FP16 training stability.

    Automatically adjusts the loss scale based on gradient health:
    - Increases scale when gradients are healthy for growth_interval steps
    - Decreases scale when inf/nan gradients are detected

    This prevents gradient underflow (gradients too small for FP16)
    and overflow (gradients too large for FP16).

    Mathematical basis:
    For gradient g and scale S:
        scaled_loss = loss * S
        scaled_grads = grads * S
        unscaled_grads = scaled_grads / S

    The scale S should satisfy:
        2^-24 <= |g| * S <= 65504

    Example:
        scaler = DynamicLossScaler()

        for batch in dataloader:
            # Scale loss before backward
            scaled_loss = scaler.scale_loss(loss)
            grads = jax.grad(lambda p: scaled_loss)(params)

            # Check for inf/nan and unscale
            grads, is_finite = scaler.unscale_grads(grads)

            if is_finite:
                params = optimizer.update(params, grads)

            # Update scale
            scaler.update(is_finite)
    """

    def __init__(self, config: Optional[LossScalerConfig] = None):
        """
        Initialize dynamic loss scaler.

        Args:
            config: Loss scaler configuration (uses defaults if None)
        """
        self._config = config if config else LossScalerConfig()
        self._state = LossScalerState(scale=self._config.initial_scale)

    @property
    def scale(self) -> float:
        """Get current loss scale."""
        return self._state.scale

    @property
    def state(self) -> LossScalerState:
        """Get current scaler state."""
        return self._state

    def scale_loss(self, loss: Any) -> Any:
        """
        Scale loss before backward pass.

        Args:
            loss: The loss value to scale

        Returns:
            Scaled loss
        """
        return loss * self._state.scale

    def unscale_grads(self, grads: PyTree) -> Tuple[PyTree, bool]:
        """
        Unscale gradients and check for inf/nan.

        Args:
            grads: Gradient pytree from backward pass

        Returns:
            Tuple of (unscaled_grads, is_finite)
        """
        jax = _get_jax()
        jnp = _get_jnp()

        inv_scale = 1.0 / self._state.scale
        unscaled = jax.tree_util.tree_map(lambda g: g * inv_scale, grads)

        def check_finite(g):
            return jnp.all(jnp.isfinite(g))

        finite_checks = jax.tree_util.tree_map(check_finite, unscaled)
        is_finite = jax.tree_util.tree_reduce(
            lambda a, b: jnp.logical_and(a, b),
            finite_checks,
            initializer=jnp.array(True),
        )

        return unscaled, bool(is_finite)

    def update(self, grads_finite: bool) -> None:
        """
        Update loss scale based on gradient health.

        Args:
            grads_finite: Whether gradients were finite (no inf/nan)
        """
        self._state.total_steps += 1

        if grads_finite:
            self._state.good_steps += 1

            if self._state.good_steps >= self._config.growth_interval:
                new_scale = self._state.scale * self._config.growth_factor
                self._state.scale = min(new_scale, self._config.max_scale)
                self._state.good_steps = 0

                logger.debug(f"Loss scale increased to {self._state.scale}")
        else:
            self._state.overflow_count += 1
            new_scale = self._state.scale * self._config.backoff_factor
            self._state.scale = max(new_scale, self._config.min_scale)
            self._state.good_steps = 0

            logger.warning(
                f"Gradient overflow detected. Loss scale reduced to {self._state.scale}"
            )

    def reset(self) -> None:
        """Reset scaler to initial state."""
        self._state = LossScalerState(scale=self._config.initial_scale)


@dataclass
class MixedPrecisionStats:
    """Statistics from mixed precision training."""

    total_steps: int = 0
    scale_updates: int = 0
    overflow_count: int = 0
    current_scale: float = 1.0
    dtype_cast_count: int = 0


class ZenithMixedPrecision:
    """
    High-level mixed precision API for JAX training.

    Provides:
    - Automatic dtype casting for forward/backward passes
    - Dynamic loss scaling for FP16 stability
    - Parameter management (master weights)
    - Gradient handling

    Example:
        # Create mixed precision handler
        mp = ZenithMixedPrecision(policy="bf16")

        # Cast params to compute dtype for forward pass
        compute_params = mp.cast_to_compute(params)

        # Run forward pass
        loss, grads = jax.value_and_grad(loss_fn)(compute_params, x)

        # Handle gradients (with scaling for FP16)
        grads = mp.handle_grads(grads)

        # Cast gradients back to param dtype for optimizer
        grads = mp.cast_to_param(grads)

        # Update parameters
        params = optimizer.update(params, grads)
    """

    def __init__(
        self,
        policy: Union[str, MixedPrecisionPolicy] = "bf16",
        loss_scaler_config: Optional[LossScalerConfig] = None,
    ):
        """
        Initialize mixed precision handler.

        Args:
            policy: Precision policy ("fp32", "fp16", "bf16") or MixedPrecisionPolicy
            loss_scaler_config: Configuration for loss scaler (FP16 only)
        """
        if isinstance(policy, str):
            if policy == "fp32":
                self._policy = MixedPrecisionPolicy.fp32()
            elif policy == "fp16":
                self._policy = MixedPrecisionPolicy.fp16()
            elif policy == "bf16":
                self._policy = MixedPrecisionPolicy.bf16()
            else:
                raise ValueError(f"Unknown precision policy: {policy}")
        else:
            self._policy = policy

        self._scaler: Optional[DynamicLossScaler] = None
        if self._policy.requires_loss_scaling:
            self._scaler = DynamicLossScaler(loss_scaler_config)

        self._stats = MixedPrecisionStats()

    @property
    def policy(self) -> MixedPrecisionPolicy:
        """Get current precision policy."""
        return self._policy

    @property
    def scaler(self) -> Optional[DynamicLossScaler]:
        """Get loss scaler if using FP16."""
        return self._scaler

    @property
    def stats(self) -> MixedPrecisionStats:
        """Get runtime statistics."""
        stats = self._stats
        if self._scaler:
            stats.current_scale = self._scaler.scale
            stats.overflow_count = self._scaler.state.overflow_count
        return stats

    def cast_to_compute(self, pytree: PyTree) -> PyTree:
        """
        Cast pytree to compute dtype.

        Used before forward pass to convert params to compute precision.

        Args:
            pytree: Pytree of arrays to cast

        Returns:
            Pytree with arrays cast to compute dtype
        """
        jax = _get_jax()
        jnp = _get_jnp()

        compute_dtype = self._policy.compute_jax_dtype

        def cast_array(x):
            if hasattr(x, "dtype") and x.dtype in (jnp.float32, jnp.float64):
                self._stats.dtype_cast_count += 1
                return x.astype(compute_dtype)
            return x

        return jax.tree_util.tree_map(cast_array, pytree)

    def cast_to_param(self, pytree: PyTree) -> PyTree:
        """
        Cast pytree to param dtype.

        Used after gradient computation to convert back to full precision.

        Args:
            pytree: Pytree of arrays to cast

        Returns:
            Pytree with arrays cast to param dtype
        """
        jax = _get_jax()
        jnp = _get_jnp()

        param_dtype = self._policy.param_jax_dtype

        def cast_array(x):
            if hasattr(x, "dtype") and x.dtype in (jnp.float16, jnp.bfloat16):
                self._stats.dtype_cast_count += 1
                return x.astype(param_dtype)
            return x

        return jax.tree_util.tree_map(cast_array, pytree)

    def scale_loss(self, loss: Any) -> Any:
        """
        Scale loss before backward pass (FP16 only).

        Args:
            loss: Loss value to scale

        Returns:
            Scaled loss (or original if not using FP16)
        """
        if self._scaler is not None:
            return self._scaler.scale_loss(loss)
        return loss

    def unscale_grads(self, grads: PyTree) -> Tuple[PyTree, bool]:
        """
        Unscale gradients after backward pass (FP16 only).

        Args:
            grads: Gradient pytree

        Returns:
            Tuple of (unscaled_grads, is_finite)
        """
        if self._scaler is not None:
            return self._scaler.unscale_grads(grads)
        return grads, True

    def update_scale(self, grads_finite: bool) -> None:
        """
        Update loss scale after gradient step (FP16 only).

        Args:
            grads_finite: Whether gradients were finite
        """
        if self._scaler is not None:
            self._scaler.update(grads_finite)
            self._stats.scale_updates += 1

        self._stats.total_steps += 1

    def handle_grads(self, grads: PyTree) -> Tuple[PyTree, bool]:
        """
        Handle gradients with scaling and unscaling.

        Convenience method that combines unscale_grads and dtype casting.

        Args:
            grads: Gradient pytree from backward pass

        Returns:
            Tuple of (processed_grads, is_finite)
        """
        grads, is_finite = self.unscale_grads(grads)

        grads = self.cast_to_param(grads)

        return grads, is_finite

    def wrap_forward(
        self,
        fn: F,
        *,
        cast_inputs: bool = True,
        cast_outputs: bool = False,
    ) -> F:
        """
        Wrap a forward function with automatic dtype casting.

        Args:
            fn: Forward function to wrap
            cast_inputs: Whether to cast inputs to compute dtype
            cast_outputs: Whether to cast outputs to output dtype

        Returns:
            Wrapped function with automatic casting

        Example:
            @mp.wrap_forward
            def forward(params, x):
                return model.apply(params, x)
        """

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if cast_inputs:
                args = tuple(self.cast_to_compute(a) for a in args)
                kwargs = {k: self.cast_to_compute(v) for k, v in kwargs.items()}

            result = fn(*args, **kwargs)

            if cast_outputs:
                result_dtype = self._policy.output_jax_dtype
                jax = _get_jax()
                result = jax.tree_util.tree_map(
                    lambda x: x.astype(result_dtype) if hasattr(x, "astype") else x,
                    result,
                )

            return result

        return wrapped

    def create_train_state(
        self,
        params: PyTree,
        optimizer,
        apply_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Create a training state with mixed precision support.

        Args:
            params: Model parameters
            optimizer: Optax optimizer
            apply_fn: Optional apply function for the model

        Returns:
            Training state dict with params, opt_state, and metadata
        """
        opt_state = optimizer.init(params)

        return {
            "params": params,
            "opt_state": opt_state,
            "step": 0,
            "apply_fn": apply_fn,
            "policy": self._policy,
            "scaler_state": self._scaler.state if self._scaler else None,
        }


def create_policy(mode: str) -> MixedPrecisionPolicy:
    """
    Create a mixed precision policy by name.

    Args:
        mode: Precision mode ("fp32", "fp16", "bf16")

    Returns:
        MixedPrecisionPolicy instance
    """
    if mode == "fp32":
        return MixedPrecisionPolicy.fp32()
    elif mode == "fp16":
        return MixedPrecisionPolicy.fp16()
    elif mode == "bf16":
        return MixedPrecisionPolicy.bf16()
    else:
        raise ValueError(f"Unknown precision mode: {mode}")


def detect_best_precision() -> str:
    """
    Detect the best precision mode for the current hardware.

    Returns:
        Recommended precision mode string
    """
    jax = _get_jax()

    try:
        devices = jax.devices()
        if not devices:
            return "fp32"

        device = devices[0]
        platform = device.platform.lower()

        if platform == "tpu":
            return "bf16"

        if platform == "gpu":
            return "bf16"

        return "fp32"

    except Exception:
        return "fp32"


__all__ = [
    "PrecisionMode",
    "MixedPrecisionPolicy",
    "LossScalerConfig",
    "LossScalerState",
    "DynamicLossScaler",
    "MixedPrecisionStats",
    "ZenithMixedPrecision",
    "create_policy",
    "detect_best_precision",
]
