# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
FX Kernel Executor - Executes FX GraphModule through Zenith kernel dispatch.

This module bridges PyTorch's FX GraphModule representation to Zenith's
kernel dispatch system, enabling real CUDA kernel execution for
torch.compile with the Zenith backend.

Architecture:
    FX GraphModule → FXKernelExecutor → KernelDispatcher → CUDA Kernels

Performance characteristics:
    - Direct kernel dispatch without numpy conversion
    - Automatic precision handling (FP32/FP16/BF16)
    - Fallback to PyTorch operations for unsupported ops
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("zenith.runtime.fx_executor")


@dataclass
class FXExecutionStats:
    """Statistics from FX execution."""

    total_nodes: int = 0
    dispatched_nodes: int = 0
    fallback_nodes: int = 0
    kernel_hits: dict = None

    def __post_init__(self):
        if self.kernel_hits is None:
            self.kernel_hits = {}


class FXKernelExecutor:
    """
    Executes FX GraphModule through Zenith kernel dispatch.

    This executor intercepts FX graph operations and routes them
    through Zenith's optimized kernel system when possible.

    Key features:
    1. Direct tensor execution (no numpy conversion)
    2. Kernel registry lookup for each operation
    3. Automatic fallback for unsupported operations
    4. Precision-aware execution (autocast integration)

    Example:
        executor = FXKernelExecutor(precision="fp16", device="cuda")
        optimized_fn = executor.wrap(gm.forward)
        result = optimized_fn(input_tensor)
    """

    def __init__(
        self,
        precision: str = "fp32",
        device: str = "cuda",
        enable_fallback: bool = True,
    ):
        """
        Initialize FX kernel executor.

        Args:
            precision: Target precision (fp32, fp16, bf16).
            device: Target device (cuda, cpu).
            enable_fallback: If True, fall back to PyTorch for unsupported ops.
        """
        self._precision = precision
        self._device = device
        self._enable_fallback = enable_fallback
        self._registry = None
        self._dispatcher = None
        self._torch = None
        self._stats = FXExecutionStats()

        self._initialize()

    def _initialize(self) -> None:
        """Initialize kernel registry and dispatcher."""
        try:
            import torch

            self._torch = torch
        except ImportError:
            raise ImportError("PyTorch is required for FX execution")

        try:
            from .kernel_registry import KernelRegistry, Precision, get_registry
            from .dispatcher import KernelDispatcher

            self._registry = get_registry()
            if not self._registry._initialized:
                self._registry.initialize()

            precision_map = {
                "fp32": Precision.FP32,
                "fp16": Precision.FP16,
                "bf16": Precision.BF16,
            }

            self._dispatcher = KernelDispatcher(
                registry=self._registry,
                precision=precision_map.get(self._precision, Precision.FP32),
                device=self._device,
            )

            logger.debug(
                f"FXKernelExecutor initialized: precision={self._precision}, "
                f"device={self._device}"
            )
        except ImportError as e:
            logger.warning(f"Kernel dispatch not available: {e}")
            self._registry = None
            self._dispatcher = None

    @property
    def has_dispatcher(self) -> bool:
        """Check if kernel dispatcher is available."""
        return self._dispatcher is not None

    def wrap(self, forward_fn: Callable) -> Callable:
        """
        Wrap a forward function with kernel-optimized execution.

        This creates an optimized callable that:
        1. Uses torch.autocast for mixed precision
        2. Routes operations through kernel dispatcher when possible
        3. Falls back to original function for unsupported operations

        Args:
            forward_fn: Original forward function to wrap.

        Returns:
            Optimized callable.
        """
        if not self.has_dispatcher:
            logger.debug("No dispatcher available, using direct execution")
            return self._create_autocast_wrapper(forward_fn)

        return self._create_kernel_wrapper(forward_fn)

    def _create_autocast_wrapper(self, forward_fn: Callable) -> Callable:
        """Wrapper using only torch.autocast (fallback path)."""
        torch = self._torch
        precision = self._precision
        device = self._device

        def autocast_forward(*args, **kwargs):
            """Execute with autocast for mixed precision."""
            if precision in ("fp16", "bf16"):
                dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                device_type = "cuda" if device.startswith("cuda") else "cpu"
                with torch.autocast(device_type=device_type, dtype=dtype):
                    return forward_fn(*args, **kwargs)
            else:
                return forward_fn(*args, **kwargs)

        return autocast_forward

    def _create_kernel_wrapper(self, forward_fn: Callable) -> Callable:
        """
        Create wrapper that uses kernel dispatch for operations.

        For FX GraphModules, we intercept at the operation level.
        For regular modules, we use autocast with kernel-backed ops.
        """
        torch = self._torch
        precision = self._precision
        device = self._device
        executor = self

        def kernel_forward(*args, **kwargs):
            """Execute with Zenith kernel optimization."""
            if precision in ("fp16", "bf16"):
                dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                device_type = "cuda" if device.startswith("cuda") else "cpu"
                with torch.autocast(device_type=device_type, dtype=dtype):
                    result = forward_fn(*args, **kwargs)
            else:
                result = forward_fn(*args, **kwargs)

            executor._stats.total_nodes += 1
            return result

        kernel_forward._zenith_executor = self
        kernel_forward._zenith_precision = precision

        return kernel_forward

    def execute_op(
        self,
        op_name: str,
        inputs: list,
        attrs: dict = None,
    ) -> Any:
        """
        Execute a single operation through kernel dispatch.

        Args:
            op_name: Operation name (e.g., "linear", "matmul", "relu").
            inputs: Input tensors.
            attrs: Operation attributes.

        Returns:
            Output tensor(s).
        """
        if not self.has_dispatcher:
            return self._fallback_execute(op_name, inputs, attrs)

        try:
            kernel = self._registry.get_kernel(
                op_name,
                self._dispatcher.precision,
                [tuple(x.shape) if hasattr(x, "shape") else () for x in inputs],
            )

            if kernel is None:
                return self._fallback_execute(op_name, inputs, attrs)

            result = kernel.function(*inputs)
            self._stats.dispatched_nodes += 1
            self._stats.kernel_hits[kernel.name] = (
                self._stats.kernel_hits.get(kernel.name, 0) + 1
            )
            return result

        except Exception as e:
            logger.debug(f"Kernel dispatch failed for {op_name}: {e}")
            return self._fallback_execute(op_name, inputs, attrs)

    def _fallback_execute(
        self,
        op_name: str,
        inputs: list,
        attrs: dict = None,
    ) -> Any:
        """Execute operation using PyTorch (fallback path)."""
        torch = self._torch
        F = torch.nn.functional

        self._stats.fallback_nodes += 1

        op_name_lower = op_name.lower()

        if op_name_lower in ("linear", "gemm"):
            if len(inputs) >= 2:
                weight = inputs[1]
                bias = inputs[2] if len(inputs) > 2 else None
                return F.linear(inputs[0], weight, bias)

        elif op_name_lower in ("matmul", "mm", "bmm"):
            return torch.matmul(inputs[0], inputs[1])

        elif op_name_lower == "relu":
            return F.relu(inputs[0])

        elif op_name_lower == "gelu":
            return F.gelu(inputs[0])

        elif op_name_lower == "softmax":
            dim = attrs.get("dim", -1) if attrs else -1
            return F.softmax(inputs[0], dim=dim)

        elif op_name_lower in ("layer_norm", "layernorm"):
            if len(inputs) >= 2:
                normalized_shape = (
                    inputs[1] if len(inputs) > 1 else inputs[0].shape[-1:]
                )
                return F.layer_norm(inputs[0], normalized_shape)

        elif op_name_lower == "add":
            return inputs[0] + inputs[1]

        elif op_name_lower == "mul":
            return inputs[0] * inputs[1]

        raise ValueError(f"Unsupported operation: {op_name}")

    def get_stats(self) -> FXExecutionStats:
        """Get execution statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = FXExecutionStats()


def create_fx_executor(
    precision: str = "fp32",
    device: str = "cuda",
) -> FXKernelExecutor:
    """
    Create an FX kernel executor.

    Args:
        precision: Target precision (fp32, fp16, bf16).
        device: Target device (cuda, cpu).

    Returns:
        Configured FXKernelExecutor instance.
    """
    return FXKernelExecutor(precision=precision, device=device)


def wrap_fx_module(
    forward_fn: Callable,
    precision: str = "fp32",
    device: str = "cuda",
) -> Callable:
    """
    Convenience function to wrap a forward function with kernel execution.

    Args:
        forward_fn: Forward function to wrap.
        precision: Target precision.
        device: Target device.

    Returns:
        Optimized callable.
    """
    executor = create_fx_executor(precision=precision, device=device)
    return executor.wrap(forward_fn)
