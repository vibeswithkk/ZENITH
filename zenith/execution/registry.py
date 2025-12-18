# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Operator Registry

Maps ONNX operator types to their kernel implementations.
Uses a decorator-based registration pattern for extensibility.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ExecutionContext


# Type alias for operator function signature
# Signature: (context: ExecutionContext, inputs: List[str],
#             outputs: List[str], attrs: Dict) -> None
OperatorFunc = Callable[["ExecutionContext", List[str], List[str], Dict], None]


class OperatorRegistry:
    """
    Registry for ONNX operator implementations.

    Maps ONNX op_type strings to their kernel execution functions.
    Uses a decorator pattern for easy operator registration.

    Example:
        @OperatorRegistry.register("MatMul")
        def execute_matmul(ctx, inputs, outputs, attrs):
            A = ctx.get_gpu_tensor(inputs[0])
            B = ctx.get_gpu_tensor(inputs[1])
            result = cuda.matmul_gpu(A, B)
            ctx.set_tensor(outputs[0], result)

        # Later, execute the operator
        kernel = OperatorRegistry.get_kernel("MatMul")
        kernel(ctx, ["A", "B"], ["C"], {})
    """

    _registry: Dict[str, OperatorFunc] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        op_type: str,
        aliases: Optional[List[str]] = None,
        tier: int = 1,
    ) -> Callable[[OperatorFunc], OperatorFunc]:
        """
        Decorator to register an operator implementation.

        Args:
            op_type: ONNX operator type (e.g., "MatMul", "Conv").
            aliases: Alternative names for the operator.
            tier: Implementation priority tier (1=critical, 2=common, 3=advanced).

        Returns:
            Decorator function.

        Example:
            @OperatorRegistry.register("MatMul", tier=1)
            def execute_matmul(ctx, inputs, outputs, attrs):
                ...
        """

        def decorator(func: OperatorFunc) -> OperatorFunc:
            cls._registry[op_type] = func
            cls._metadata[op_type] = {
                "tier": tier,
                "func_name": func.__name__,
            }

            # Register aliases
            if aliases:
                for alias in aliases:
                    cls._registry[alias] = func
                    cls._metadata[alias] = cls._metadata[op_type]

            return func

        return decorator

    @classmethod
    def get_kernel(cls, op_type: str) -> OperatorFunc:
        """
        Get the kernel function for an operator.

        Args:
            op_type: ONNX operator type.

        Returns:
            Operator execution function.

        Raises:
            KeyError: If operator not registered.
        """
        if op_type not in cls._registry:
            raise KeyError(
                f"Operator '{op_type}' not registered. "
                f"Supported operators: {cls.list_operators()}"
            )
        return cls._registry[op_type]

    @classmethod
    def is_supported(cls, op_type: str) -> bool:
        """Check if an operator is supported."""
        return op_type in cls._registry

    @classmethod
    def list_operators(cls) -> List[str]:
        """List all registered operators."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_tier(cls, op_type: str) -> int:
        """Get the tier of an operator."""
        if op_type in cls._metadata:
            return cls._metadata[op_type].get("tier", 0)
        return 0

    @classmethod
    def list_by_tier(cls, tier: int) -> List[str]:
        """List operators in a specific tier."""
        return [op for op, meta in cls._metadata.items() if meta.get("tier") == tier]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operators (for testing)."""
        cls._registry.clear()
        cls._metadata.clear()

    @classmethod
    def count(cls) -> int:
        """Get number of registered operators."""
        return len(cls._registry)

    @classmethod
    def get_unsupported_ops(cls, op_types: List[str]) -> List[str]:
        """
        Get list of operators that are not supported.

        Args:
            op_types: List of ONNX operator types to check.

        Returns:
            List of unsupported operator types.
        """
        return [op for op in op_types if not cls.is_supported(op)]


# Register a fallback/placeholder operator for debugging
@OperatorRegistry.register("Identity", tier=1)
def execute_identity(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Identity operator - passes input through unchanged."""
    input_value = ctx.get_tensor(inputs[0])
    ctx.set_tensor(outputs[0], input_value)


@OperatorRegistry.register("Constant", tier=1)
def execute_constant(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Constant operator - output is stored in attrs."""
    # ONNX Constant stores value in attributes
    if "value" in attrs:
        ctx.set_tensor(outputs[0], attrs["value"])
    else:
        # Value might already be in context (loaded from initializer)
        pass
