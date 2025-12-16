"""
Mixed Precision Management System

Implements automatic mixed precision with:
- FP16/BF16 precision policies
- Loss scaling for FP16
- Operation-level precision assignment
- Automatic precision fallback

Based on CetakBiru Section 5.1 Fase 3 requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable

from ..core import GraphIR, Node, DataType


class Precision(Enum):
    """Supported precision levels."""

    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"


@dataclass
class PrecisionPolicy:
    """Policy for automatic precision assignment."""

    name: str
    compute_dtype: Precision
    output_dtype: Precision
    loss_scale: float = 1.0
    use_dynamic_loss_scaling: bool = False

    @classmethod
    def fp16_with_loss_scale(cls, initial_scale: float = 2**16) -> "PrecisionPolicy":
        """FP16 policy with loss scaling for gradient stability."""
        return cls(
            name="fp16_loss_scale",
            compute_dtype=Precision.FP16,
            output_dtype=Precision.FP32,
            loss_scale=initial_scale,
            use_dynamic_loss_scaling=True,
        )

    @classmethod
    def bf16(cls) -> "PrecisionPolicy":
        """BF16 policy (no loss scaling needed)."""
        return cls(
            name="bf16",
            compute_dtype=Precision.BF16,
            output_dtype=Precision.FP32,
            loss_scale=1.0,
            use_dynamic_loss_scaling=False,
        )

    @classmethod
    def fp32(cls) -> "PrecisionPolicy":
        """Full FP32 precision (no mixed precision)."""
        return cls(
            name="fp32",
            compute_dtype=Precision.FP32,
            output_dtype=Precision.FP32,
            loss_scale=1.0,
            use_dynamic_loss_scaling=False,
        )


# Operations that should stay in FP32 for numerical stability
FP32_REQUIRED_OPS = {
    "Softmax",
    "LayerNormalization",
    "BatchNormalization",
    "Loss",
    "CrossEntropyLoss",
    "Exp",
    "Log",
    "Pow",
    "Sqrt",
    "ReduceSum",
    "ReduceMean",
}

# Operations safe for lower precision
LOW_PRECISION_SAFE_OPS = {
    "Conv",
    "Conv2D",
    "MatMul",
    "Gemm",
    "Relu",
    "Add",
    "Mul",
    "MaxPool",
    "AvgPool",
    "Concat",
    "Reshape",
    "Transpose",
}


class DynamicLossScaler:
    """
    Dynamic loss scaling for FP16 training stability.

    Automatically adjusts the loss scale based on gradient overflow:
    - Increase scale when no overflow for consecutive steps
    - Decrease scale when overflow detected
    """

    def __init__(
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.current_scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0

    def scale_loss(self, loss: np.ndarray) -> np.ndarray:
        """Scale loss value for backward pass."""
        return loss * self.current_scale

    def unscale_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Unscale gradients after backward pass."""
        return gradients / self.current_scale

    def update(self, overflow_detected: bool) -> None:
        """Update scale based on overflow detection."""
        if overflow_detected:
            self.current_scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.current_scale *= self.growth_factor
                self._growth_tracker = 0

    @staticmethod
    def check_overflow(tensor: np.ndarray) -> bool:
        """Check if tensor contains inf or nan values."""
        return bool(np.any(~np.isfinite(tensor)))


class MixedPrecisionManager:
    """
    Manages mixed precision for graph execution.

    Usage:
        mp_manager = MixedPrecisionManager(policy=PrecisionPolicy.bf16())

        # Assign precision to graph operations
        precision_map = mp_manager.assign_precision(graph)

        # Cast tensor to operation's precision
        x_casted = mp_manager.cast(x, "MatMul")
    """

    def __init__(self, policy: PrecisionPolicy | None = None):
        self.policy = policy or PrecisionPolicy.bf16()
        self.loss_scaler: DynamicLossScaler | None = None
        self.precision_map: dict[str, Precision] = {}

        if self.policy.use_dynamic_loss_scaling:
            self.loss_scaler = DynamicLossScaler(initial_scale=self.policy.loss_scale)

    def assign_precision(self, graph: GraphIR) -> dict[str, Precision]:
        """
        Assign precision to each operation in the graph.

        Args:
            graph: Graph to assign precision to

        Returns:
            Dictionary mapping operation names to precision
        """
        self.precision_map = {}

        for node in graph.nodes:
            precision = self._get_op_precision(node.op_type)
            self.precision_map[node.name] = precision

        return self.precision_map

    def _get_op_precision(self, op_type: str) -> Precision:
        """Determine precision for an operation type."""
        if op_type in FP32_REQUIRED_OPS:
            return Precision.FP32
        elif op_type in LOW_PRECISION_SAFE_OPS:
            return self.policy.compute_dtype
        else:
            # Unknown op: use compute dtype but be cautious
            return self.policy.compute_dtype

    def cast(
        self,
        tensor: np.ndarray,
        target_op: str | None = None,
        target_precision: Precision | None = None,
    ) -> np.ndarray:
        """
        Cast tensor to target precision.

        Args:
            tensor: Input tensor
            target_op: Target operation name (uses precision map)
            target_precision: Explicit target precision

        Returns:
            Casted tensor
        """
        if target_precision is None:
            if target_op and target_op in self.precision_map:
                target_precision = self.precision_map[target_op]
            else:
                target_precision = self.policy.compute_dtype

        return self._cast_to_precision(tensor, target_precision)

    def _cast_to_precision(
        self, tensor: np.ndarray, precision: Precision
    ) -> np.ndarray:
        """Cast tensor to specified precision."""
        dtype_map = {
            Precision.FP32: np.float32,
            Precision.FP16: np.float16,
            Precision.BF16: np.float32,  # NumPy doesn't support bfloat16
            Precision.INT8: np.int8,
        }

        target_dtype = dtype_map.get(precision, np.float32)

        # Handle bfloat16 simulation
        if precision == Precision.BF16:
            # Simulate BF16 by truncating mantissa
            return self._simulate_bf16(tensor)

        return tensor.astype(target_dtype)

    def _simulate_bf16(self, tensor: np.ndarray) -> np.ndarray:
        """Simulate bfloat16 by truncating mantissa."""
        # BF16 has 7 mantissa bits vs FP32's 23
        # Truncate lower 16 bits of the mantissa
        fp32 = tensor.astype(np.float32)
        int_view = fp32.view(np.uint32)
        # Clear lower 16 bits (preserving sign, exponent, and top 7 mantissa)
        truncated = int_view & 0xFFFF0000
        return truncated.view(np.float32)

    def scale_loss(self, loss: np.ndarray) -> np.ndarray:
        """Scale loss for FP16 training."""
        if self.loss_scaler:
            return self.loss_scaler.scale_loss(loss)
        return loss

    def unscale_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Unscale gradients after backward pass."""
        if self.loss_scaler:
            return self.loss_scaler.unscale_gradients(gradients)
        return gradients

    def step(self, gradients: np.ndarray) -> bool:
        """
        Update loss scaler based on gradient values.

        Args:
            gradients: Gradient tensor to check

        Returns:
            True if step is valid (no overflow)
        """
        if self.loss_scaler:
            overflow = DynamicLossScaler.check_overflow(gradients)
            self.loss_scaler.update(overflow)
            return not overflow
        return True

    def get_stats(self) -> dict:
        """Get mixed precision statistics."""
        stats = {
            "policy": self.policy.name,
            "compute_dtype": self.policy.compute_dtype.value,
            "output_dtype": self.policy.output_dtype.value,
        }
        if self.loss_scaler:
            stats["current_loss_scale"] = self.loss_scaler.current_scale
        return stats


def convert_to_fp16(tensor: np.ndarray) -> np.ndarray:
    """Convert tensor to FP16."""
    return tensor.astype(np.float16)


def convert_to_bf16(tensor: np.ndarray) -> np.ndarray:
    """Simulate BF16 conversion (using FP32 with truncated mantissa)."""
    fp32 = tensor.astype(np.float32)
    int_view = fp32.view(np.uint32)
    truncated = int_view & 0xFFFF0000
    return truncated.view(np.float32)


def check_precision_safety(
    fp32_result: np.ndarray,
    reduced_result: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> tuple[bool, float]:
    """
    Check if reduced precision result is within tolerance.

    Args:
        fp32_result: Reference FP32 result
        reduced_result: Reduced precision result
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (is_safe, max_error)
    """
    # Convert to FP32 for comparison
    reduced_fp32 = reduced_result.astype(np.float32)
    max_error = float(np.max(np.abs(fp32_result - reduced_fp32)))
    is_close = np.allclose(fp32_result, reduced_fp32, rtol=rtol, atol=atol)
    return is_close, max_error
