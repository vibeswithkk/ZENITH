# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for Zenith Error Handling

Validates:
- Error hierarchy
- Error message formatting
- Suggestions in error messages
- Context information
"""

import pytest

from zenith.errors import (
    ZenithError,
    CompilationError,
    UnsupportedOperationError,
    PrecisionError,
    KernelError,
    ZenithMemoryError,
    ValidationError,
    ConfigurationError,
    format_shape_mismatch,
    format_dtype_mismatch,
)


class TestZenithError:
    """Tests for ZenithError base class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ZenithError("Test error")
        assert "Test error" in str(error)

    def test_error_with_suggestions(self):
        """Test error with suggestions."""
        error = ZenithError(
            "Test error",
            suggestions=["Fix A", "Fix B"],
        )
        msg = str(error)
        assert "Test error" in msg
        assert "Suggestions:" in msg
        assert "Fix A" in msg
        assert "Fix B" in msg

    def test_error_with_context(self):
        """Test error with context."""
        error = ZenithError(
            "Test error",
            context={"key1": "value1", "key2": "value2"},
        )
        msg = str(error)
        assert "Context:" in msg
        assert "key1: value1" in msg
        assert "key2: value2" in msg

    def test_error_attributes(self):
        """Test error attributes."""
        error = ZenithError(
            "Test error",
            suggestions=["Fix A"],
            context={"key": "value"},
        )
        assert error.message == "Test error"
        assert error.suggestions == ["Fix A"]
        assert error.context == {"key": "value"}


class TestCompilationError:
    """Tests for CompilationError."""

    def test_basic_compilation_error(self):
        """Test basic compilation error."""
        error = CompilationError("Invalid graph")
        assert "Compilation failed:" in str(error)
        assert "Invalid graph" in str(error)

    def test_compilation_error_with_model_name(self):
        """Test compilation error with model name."""
        error = CompilationError(
            "Invalid graph",
            model_name="bert-base",
            node_name="layer_0/attention",
        )
        msg = str(error)
        assert "model_name: bert-base" in msg
        assert "node_name: layer_0/attention" in msg

    def test_compilation_error_has_suggestions(self):
        """Test compilation error has default suggestions."""
        error = CompilationError("Invalid graph")
        msg = str(error)
        assert "Suggestions:" in msg


class TestUnsupportedOperationError:
    """Tests for UnsupportedOperationError."""

    def test_basic_unsupported_op(self):
        """Test basic unsupported operation error."""
        error = UnsupportedOperationError("CustomOp")
        assert "CustomOp" in str(error)
        assert "not supported" in str(error)

    def test_unsupported_op_with_backend(self):
        """Test unsupported operation with backend."""
        error = UnsupportedOperationError(
            "CustomOp",
            backend="cuda",
        )
        msg = str(error)
        assert "backend: cuda" in msg

    def test_similar_ops_suggestion(self):
        """Test similar ops in suggestions."""
        error = UnsupportedOperationError(
            "MatMulTranspose",
            supported_ops=["MatMul", "Transpose", "Conv2D"],
        )
        msg = str(error)
        assert "MatMul" in msg or "Transpose" in msg

    def test_op_type_attribute(self):
        """Test op_type attribute."""
        error = UnsupportedOperationError("CustomOp")
        assert error.op_type == "CustomOp"


class TestPrecisionError:
    """Tests for PrecisionError."""

    def test_basic_precision_error(self):
        """Test basic precision error."""
        error = PrecisionError(
            expected_tolerance=1e-5,
            actual_error=1e-3,
        )
        msg = str(error)
        assert "precision violated" in msg.lower()
        assert "1.00e-05" in msg or "1e-05" in msg
        assert "1.00e-03" in msg or "1e-03" in msg

    def test_precision_error_with_operation(self):
        """Test precision error with operation."""
        error = PrecisionError(
            expected_tolerance=1e-5,
            actual_error=1e-3,
            operation="LayerNorm",
        )
        msg = str(error)
        assert "operation: LayerNorm" in msg

    def test_precision_error_attributes(self):
        """Test precision error attributes."""
        error = PrecisionError(
            expected_tolerance=1e-5,
            actual_error=1e-3,
        )
        assert error.expected_tolerance == 1e-5
        assert error.actual_error == 1e-3


class TestKernelError:
    """Tests for KernelError."""

    def test_basic_kernel_error(self):
        """Test basic kernel error."""
        error = KernelError("CUDA launch failed")
        assert "Kernel execution failed" in str(error)
        assert "CUDA launch failed" in str(error)

    def test_kernel_error_with_context(self):
        """Test kernel error with context."""
        error = KernelError(
            "Invalid parameters",
            kernel_name="flash_attention_v2",
            op_type="Attention",
            input_shapes=[(1, 128, 768), (1, 128, 768)],
        )
        msg = str(error)
        assert "kernel: flash_attention_v2" in msg
        assert "operation: Attention" in msg


class TestZenithMemoryError:
    """Tests for ZenithMemoryError."""

    def test_basic_memory_error(self):
        """Test basic memory error."""
        error = ZenithMemoryError("Out of memory")
        assert "Memory error" in str(error)
        assert "Out of memory" in str(error)

    def test_memory_error_with_sizes(self):
        """Test memory error with sizes."""
        error = ZenithMemoryError(
            "Allocation failed",
            requested_bytes=1024 * 1024 * 100,
            available_bytes=1024 * 1024 * 50,
            device="cuda:0",
        )
        msg = str(error)
        assert "requested_mb:" in msg
        assert "available_mb:" in msg
        assert "device: cuda:0" in msg


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Invalid input")
        assert "Validation failed" in str(error)

    def test_validation_error_with_details(self):
        """Test validation error with details."""
        error = ValidationError(
            "Invalid shape",
            parameter="input_tensor",
            expected="(batch, seq, hidden)",
            received="(batch, seq)",
        )
        msg = str(error)
        assert "parameter: input_tensor" in msg
        assert "expected:" in msg
        assert "received:" in msg


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_config_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid precision")
        assert "Configuration error" in str(error)

    def test_config_error_with_details(self):
        """Test configuration error with details."""
        error = ConfigurationError(
            "Invalid precision",
            config_key="precision",
            config_value="fp64",
        )
        msg = str(error)
        assert "config_key: precision" in msg
        assert "config_value: fp64" in msg


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_shape_mismatch(self):
        """Test shape mismatch helper."""
        error = format_shape_mismatch(
            expected_shape=(1, 128, 768),
            actual_shape=(1, 64, 768),
            tensor_name="input",
        )
        assert isinstance(error, ValidationError)
        assert "(1, 128, 768)" in str(error)
        assert "(1, 64, 768)" in str(error)

    def test_format_dtype_mismatch(self):
        """Test dtype mismatch helper."""
        error = format_dtype_mismatch(
            expected_dtype="float32",
            actual_dtype="float16",
            tensor_name="weights",
        )
        assert isinstance(error, ValidationError)
        assert "float32" in str(error)
        assert "float16" in str(error)


class TestErrorInheritance:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_zenith_error(self):
        """Test all error types inherit from ZenithError."""
        errors = [
            CompilationError("test"),
            UnsupportedOperationError("test"),
            PrecisionError(1e-5, 1e-3),
            KernelError("test"),
            ZenithMemoryError("test"),
            ValidationError("test"),
            ConfigurationError("test"),
        ]
        for error in errors:
            assert isinstance(error, ZenithError)
            assert isinstance(error, Exception)

    def test_errors_can_be_caught_as_zenith_error(self):
        """Test all errors can be caught as ZenithError."""
        try:
            raise CompilationError("test")
        except ZenithError as e:
            assert "test" in str(e)

        try:
            raise UnsupportedOperationError("CustomOp")
        except ZenithError as e:
            assert "CustomOp" in str(e)


class TestNoSilentFailures:
    """Tests to ensure no silent failures."""

    def test_empty_suggestions_ok(self):
        """Test empty suggestions list works."""
        error = ZenithError("Test", suggestions=[])
        assert "Suggestions:" not in str(error)

    def test_empty_context_ok(self):
        """Test empty context works."""
        error = ZenithError("Test", context={})
        assert "Context:" not in str(error)

    def test_none_values_handled(self):
        """Test None values are handled."""
        error = CompilationError(
            "Test",
            model_name=None,
            node_name=None,
        )
        assert "Compilation failed:" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
