# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Error Hierarchy

Provides comprehensive error types for the Zenith framework with:
- Clear error categorization
- Helpful error messages with suggestions
- Context information for debugging

Error Categories:
- ZenithError: Base class for all Zenith errors
- CompilationError: Errors during model compilation
- UnsupportedOperationError: Operation not supported by backend
- PrecisionError: Numerical precision violations
- KernelError: Errors during kernel execution
- ZenithMemoryError: GPU/CPU memory allocation errors
- ValidationError: Input validation errors
- ConfigurationError: Configuration/setup errors
"""

from typing import Optional


class ZenithError(Exception):
    """
    Base class for all Zenith errors.

    Provides consistent error formatting and context tracking.

    Attributes:
        message: Human-readable error message
        suggestions: List of suggestions to fix the error
        context: Optional context dictionary for debugging
    """

    def __init__(
        self,
        message: str,
        suggestions: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ):
        self.message = message
        self.suggestions = suggestions or []
        self.context = context or {}

        full_message = self._format_message()
        super().__init__(full_message)

    def _format_message(self) -> str:
        """Format the error message with suggestions."""
        lines = [self.message]

        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        if self.context:
            lines.append("")
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class CompilationError(ZenithError):
    """
    Error during model compilation.

    Raised when:
    - Graph validation fails
    - Unsupported patterns detected
    - Optimization passes fail
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        node_name: Optional[str] = None,
        suggestions: Optional[list[str]] = None,
    ):
        context = {}
        if model_name:
            context["model_name"] = model_name
        if node_name:
            context["node_name"] = node_name

        default_suggestions = [
            "Check that all operations are supported",
            "Verify input shapes are valid",
            "Try using a different precision mode",
        ]

        super().__init__(
            message=f"Compilation failed: {message}",
            suggestions=suggestions or default_suggestions,
            context=context,
        )


class UnsupportedOperationError(ZenithError):
    """
    Operation not supported by current backend.

    Raised when an operation is not implemented for the target hardware.
    """

    def __init__(
        self,
        op_type: str,
        backend: Optional[str] = None,
        supported_ops: Optional[list[str]] = None,
    ):
        self.op_type = op_type
        self.backend = backend
        self.supported_ops = supported_ops or []

        context = {"operation": op_type}
        if backend:
            context["backend"] = backend

        suggestions = [
            f"Use a different operation instead of '{op_type}'",
            "Check if the operation is available in a different precision",
            "Consider decomposing the operation into supported primitives",
        ]

        if supported_ops:
            similar = self._find_similar_ops(op_type, supported_ops)
            if similar:
                suggestions.insert(0, f"Try using: {', '.join(similar)}")

        super().__init__(
            message=f"Operation '{op_type}' is not supported",
            suggestions=suggestions,
            context=context,
        )

    @staticmethod
    def _find_similar_ops(op_type: str, supported_ops: list[str]) -> list[str]:
        """Find similar supported operations."""
        op_lower = op_type.lower()
        similar = []
        for op in supported_ops:
            if op_lower in op.lower() or op.lower() in op_lower:
                similar.append(op)
        return similar[:3]


class PrecisionError(ZenithError):
    """
    Error related to numerical precision.

    Raised when:
    - Output differs too much from reference
    - Precision requirements cannot be met
    - Numerical instability detected
    """

    def __init__(
        self,
        expected_tolerance: float,
        actual_error: float,
        operation: Optional[str] = None,
    ):
        self.expected_tolerance = expected_tolerance
        self.actual_error = actual_error

        context = {
            "expected_tolerance": f"{expected_tolerance:.2e}",
            "actual_error": f"{actual_error:.2e}",
        }
        if operation:
            context["operation"] = operation

        suggestions = [
            "Use higher precision (fp32 instead of fp16)",
            "Increase numerical tolerance if acceptable",
            "Check for numerical instability in the model",
            "Consider using mixed precision with critical ops in fp32",
        ]

        super().__init__(
            message=(
                f"Numerical precision violated: "
                f"expected error < {expected_tolerance:.2e}, "
                f"got {actual_error:.2e}"
            ),
            suggestions=suggestions,
            context=context,
        )


class KernelError(ZenithError):
    """
    Error during kernel execution.

    Raised when:
    - Kernel launch fails
    - Invalid kernel parameters
    - Runtime execution errors
    """

    def __init__(
        self,
        message: str,
        kernel_name: Optional[str] = None,
        op_type: Optional[str] = None,
        input_shapes: Optional[list] = None,
    ):
        context = {}
        if kernel_name:
            context["kernel"] = kernel_name
        if op_type:
            context["operation"] = op_type
        if input_shapes:
            context["input_shapes"] = str(input_shapes)

        suggestions = [
            "Check that input shapes are valid for this kernel",
            "Verify memory alignment requirements",
            "Try a different precision mode",
        ]

        super().__init__(
            message=f"Kernel execution failed: {message}",
            suggestions=suggestions,
            context=context,
        )


class ZenithMemoryError(ZenithError):
    """
    GPU/CPU memory allocation error.

    Raised when:
    - Out of memory during allocation
    - Memory fragmentation issues
    - Invalid memory access
    """

    def __init__(
        self,
        message: str,
        requested_bytes: Optional[int] = None,
        available_bytes: Optional[int] = None,
        device: Optional[str] = None,
    ):
        context = {}
        if requested_bytes is not None:
            context["requested_mb"] = f"{requested_bytes / (1024 * 1024):.2f}"
        if available_bytes is not None:
            context["available_mb"] = f"{available_bytes / (1024 * 1024):.2f}"
        if device:
            context["device"] = device

        suggestions = [
            "Reduce batch size",
            "Use gradient checkpointing for large models",
            "Enable memory-efficient attention",
            "Try using fp16 to reduce memory usage",
            "Clear unused tensors with gc.collect()",
        ]

        super().__init__(
            message=f"Memory error: {message}",
            suggestions=suggestions,
            context=context,
        )


class ValidationError(ZenithError):
    """
    Input validation error.

    Raised when:
    - Invalid input shapes
    - Invalid data types
    - Configuration parameter out of range
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        expected: Optional[str] = None,
        received: Optional[str] = None,
    ):
        context = {}
        if parameter:
            context["parameter"] = parameter
        if expected:
            context["expected"] = expected
        if received:
            context["received"] = received

        suggestions = [
            "Check the parameter value and type",
            "Review the API documentation",
        ]

        super().__init__(
            message=f"Validation failed: {message}",
            suggestions=suggestions,
            context=context,
        )


class ConfigurationError(ZenithError):
    """
    Configuration or setup error.

    Raised when:
    - Invalid configuration parameters
    - Missing required dependencies
    - Backend initialization failure
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
    ):
        context = {}
        if config_key:
            context["config_key"] = config_key
        if config_value:
            context["config_value"] = str(config_value)

        suggestions = [
            "Check configuration parameters",
            "Verify all dependencies are installed",
            "Review the setup documentation",
        ]

        super().__init__(
            message=f"Configuration error: {message}",
            suggestions=suggestions,
            context=context,
        )


def format_shape_mismatch(
    expected_shape: tuple,
    actual_shape: tuple,
    tensor_name: Optional[str] = None,
) -> ValidationError:
    """Create a ValidationError for shape mismatch."""
    msg = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
    return ValidationError(
        message=msg,
        parameter=tensor_name or "tensor",
        expected=str(expected_shape),
        received=str(actual_shape),
    )


def format_dtype_mismatch(
    expected_dtype: str,
    actual_dtype: str,
    tensor_name: Optional[str] = None,
) -> ValidationError:
    """Create a ValidationError for dtype mismatch."""
    msg = f"Dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
    return ValidationError(
        message=msg,
        parameter=tensor_name or "tensor",
        expected=expected_dtype,
        received=actual_dtype,
    )
