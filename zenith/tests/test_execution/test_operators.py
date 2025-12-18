# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit tests for ONNX operator implementations.

Tests each operator against numpy reference implementations
to ensure numerical correctness.
"""

import pytest
import numpy as np

from zenith.execution.context import ExecutionContext
from zenith.execution.registry import OperatorRegistry

# Import operators to register them
from zenith.execution.operators import (  # noqa: F401
    math_ops,
    activation_ops,
    conv_ops,
    shape_ops,
)


class TestMathOperators:
    """Test mathematical operators."""

    def test_matmul_2d(self):
        """Test 2D matrix multiplication."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(8, 6).astype(np.float32)

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)

        kernel = OperatorRegistry.get_kernel("MatMul")
        kernel(ctx, ["A", "B"], ["C"], {})

        result = ctx.get_tensor("C")
        expected = np.matmul(A, B)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_gemm_with_bias(self):
        """Test GEMM (General Matrix Multiply with bias)."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(8, 6).astype(np.float32)
        C = np.random.randn(6).astype(np.float32)

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)
        ctx.set_tensor("C", C)

        kernel = OperatorRegistry.get_kernel("Gemm")
        kernel(ctx, ["A", "B", "C"], ["Y"], {"alpha": 1.0, "beta": 1.0})

        result = ctx.get_tensor("Y")
        expected = np.matmul(A, B) + C

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_add(self):
        """Test element-wise addition."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)

        kernel = OperatorRegistry.get_kernel("Add")
        kernel(ctx, ["A", "B"], ["C"], {})

        result = ctx.get_tensor("C")
        expected = A + B

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sub(self):
        """Test element-wise subtraction."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)

        kernel = OperatorRegistry.get_kernel("Sub")
        kernel(ctx, ["A", "B"], ["C"], {})

        result = ctx.get_tensor("C")
        expected = A - B

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mul(self):
        """Test element-wise multiplication."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)

        kernel = OperatorRegistry.get_kernel("Mul")
        kernel(ctx, ["A", "B"], ["C"], {})

        result = ctx.get_tensor("C")
        expected = A * B

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_div(self):
        """Test element-wise division."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(4, 8).astype(np.float32) + 0.1  # Avoid div by zero

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)

        kernel = OperatorRegistry.get_kernel("Div")
        kernel(ctx, ["A", "B"], ["C"], {})

        result = ctx.get_tensor("C")
        expected = A / B

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sqrt(self):
        """Test element-wise square root."""
        ctx = ExecutionContext(device="cpu")
        A = np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1

        ctx.set_tensor("A", A)

        kernel = OperatorRegistry.get_kernel("Sqrt")
        kernel(ctx, ["A"], ["B"], {})

        result = ctx.get_tensor("B")
        expected = np.sqrt(A)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_exp(self):
        """Test element-wise exponential."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("A", A)

        kernel = OperatorRegistry.get_kernel("Exp")
        kernel(ctx, ["A"], ["B"], {})

        result = ctx.get_tensor("B")
        expected = np.exp(A)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_reduce_mean(self):
        """Test reduce mean along axes."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8, 6).astype(np.float32)

        ctx.set_tensor("A", A)

        kernel = OperatorRegistry.get_kernel("ReduceMean")
        kernel(ctx, ["A"], ["B"], {"axes": [1], "keepdims": 1})

        result = ctx.get_tensor("B")
        expected = np.mean(A, axis=(1,), keepdims=True)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestActivationOperators:
    """Test activation operators."""

    def test_relu(self):
        """Test ReLU activation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Relu")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")
        expected = np.maximum(0, X)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sigmoid(self):
        """Test sigmoid activation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Sigmoid")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")
        expected = 1.0 / (1.0 + np.exp(-X))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_tanh(self):
        """Test tanh activation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Tanh")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")
        expected = np.tanh(X)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_softmax(self):
        """Test softmax activation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Softmax")
        kernel(ctx, ["X"], ["Y"], {"axis": -1})

        result = ctx.get_tensor("Y")

        # Check softmax properties
        assert np.allclose(np.sum(result, axis=-1), 1.0, rtol=1e-5)
        assert np.all(result >= 0)

    def test_gelu(self):
        """Test GELU activation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Gelu")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")
        # GELU approximation
        expected = 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_leaky_relu(self):
        """Test Leaky ReLU activation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("LeakyRelu")
        kernel(ctx, ["X"], ["Y"], {"alpha": 0.1})

        result = ctx.get_tensor("Y")
        expected = np.where(X >= 0, X, 0.1 * X)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_clip(self):
        """Test clip operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32) * 10

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Clip")
        kernel(ctx, ["X"], ["Y"], {"min": -1.0, "max": 1.0})

        result = ctx.get_tensor("Y")
        expected = np.clip(X, -1.0, 1.0)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestConvOperators:
    """Test convolution and normalization operators."""

    def test_batch_norm(self):
        """Test batch normalization."""
        ctx = ExecutionContext(device="cpu")
        N, C, H, W = 2, 4, 8, 8
        X = np.random.randn(N, C, H, W).astype(np.float32)
        scale = np.random.randn(C).astype(np.float32)
        B = np.random.randn(C).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        var = np.abs(np.random.randn(C).astype(np.float32)) + 0.1
        eps = 1e-5

        ctx.set_tensor("X", X)
        ctx.set_tensor("scale", scale)
        ctx.set_tensor("B", B)
        ctx.set_tensor("mean", mean)
        ctx.set_tensor("var", var)

        kernel = OperatorRegistry.get_kernel("BatchNormalization")
        kernel(ctx, ["X", "scale", "B", "mean", "var"], ["Y"], {"epsilon": eps})

        result = ctx.get_tensor("Y")

        # Compute expected
        scale_r = scale.reshape(1, -1, 1, 1)
        B_r = B.reshape(1, -1, 1, 1)
        mean_r = mean.reshape(1, -1, 1, 1)
        var_r = var.reshape(1, -1, 1, 1)
        expected = scale_r * (X - mean_r) / np.sqrt(var_r + eps) + B_r

        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_global_avgpool(self):
        """Test global average pooling."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(2, 4, 8, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("GlobalAveragePool")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")
        expected = np.mean(X, axis=(2, 3), keepdims=True)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_dropout_inference(self):
        """Test dropout in inference mode (pass-through)."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Dropout")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")

        np.testing.assert_array_equal(result, X)


class TestShapeOperators:
    """Test shape manipulation operators."""

    def test_reshape(self):
        """Test reshape operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)
        shape = np.array([2, 16], dtype=np.int64)

        ctx.set_tensor("X", X)
        ctx.set_tensor("shape", shape)

        kernel = OperatorRegistry.get_kernel("Reshape")
        kernel(ctx, ["X", "shape"], ["Y"], {})

        result = ctx.get_tensor("Y")
        expected = X.reshape(2, 16)

        np.testing.assert_array_equal(result, expected)

    def test_transpose(self):
        """Test transpose operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8, 6).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Transpose")
        kernel(ctx, ["X"], ["Y"], {"perm": [2, 0, 1]})

        result = ctx.get_tensor("Y")
        expected = np.transpose(X, (2, 0, 1))

        np.testing.assert_array_equal(result, expected)

    def test_flatten(self):
        """Test flatten operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Flatten")
        kernel(ctx, ["X"], ["Y"], {"axis": 1})

        result = ctx.get_tensor("Y")
        expected = X.reshape(2, -1)

        np.testing.assert_array_equal(result, expected)

    def test_squeeze(self):
        """Test squeeze operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 1, 8, 1).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Squeeze")
        kernel(ctx, ["X"], ["Y"], {"axes": [1, 3]})

        result = ctx.get_tensor("Y")
        expected = np.squeeze(X, axis=(1, 3))

        np.testing.assert_array_equal(result, expected)

    def test_unsqueeze(self):
        """Test unsqueeze operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Unsqueeze")
        kernel(ctx, ["X"], ["Y"], {"axes": [0, 3]})

        result = ctx.get_tensor("Y")

        assert result.shape == (1, 4, 8, 1)

    def test_concat(self):
        """Test concatenation."""
        ctx = ExecutionContext(device="cpu")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(4, 8).astype(np.float32)

        ctx.set_tensor("A", A)
        ctx.set_tensor("B", B)

        kernel = OperatorRegistry.get_kernel("Concat")
        kernel(ctx, ["A", "B"], ["Y"], {"axis": 0})

        result = ctx.get_tensor("Y")
        expected = np.concatenate([A, B], axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_gather(self):
        """Test gather operation."""
        ctx = ExecutionContext(device="cpu")
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([0, 1, 0], dtype=np.int64)

        ctx.set_tensor("X", X)
        ctx.set_tensor("indices", indices)

        kernel = OperatorRegistry.get_kernel("Gather")
        kernel(ctx, ["X", "indices"], ["Y"], {"axis": 0})

        result = ctx.get_tensor("Y")
        expected = np.take(X, indices, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_shape(self):
        """Test shape operator."""
        ctx = ExecutionContext(device="cpu")
        X = np.random.randn(4, 8, 6).astype(np.float32)

        ctx.set_tensor("X", X)

        kernel = OperatorRegistry.get_kernel("Shape")
        kernel(ctx, ["X"], ["Y"], {})

        result = ctx.get_tensor("Y")
        expected = np.array([4, 8, 6], dtype=np.int64)

        np.testing.assert_array_equal(result, expected)


class TestOperatorRegistry:
    """Test the OperatorRegistry class."""

    def test_is_supported(self):
        """Test operator support check."""
        assert OperatorRegistry.is_supported("MatMul")
        assert OperatorRegistry.is_supported("Relu")
        assert not OperatorRegistry.is_supported("NonExistentOp")

    def test_list_operators(self):
        """Test listing operators."""
        ops = OperatorRegistry.list_operators()
        assert "MatMul" in ops
        assert "Relu" in ops
        assert "Add" in ops

    def test_get_unsupported_ops(self):
        """Test finding unsupported operators."""
        ops = ["MatMul", "Relu", "FakeOp1", "FakeOp2"]
        unsupported = OperatorRegistry.get_unsupported_ops(ops)

        assert "FakeOp1" in unsupported
        assert "FakeOp2" in unsupported
        assert "MatMul" not in unsupported


class TestExecutionContext:
    """Test the ExecutionContext class."""

    def test_set_get_tensor(self):
        """Test setting and getting tensors."""
        ctx = ExecutionContext(device="cpu")
        data = np.array([1, 2, 3], dtype=np.float32)

        ctx.set_tensor("test", data)
        result = ctx.get_tensor("test")

        np.testing.assert_array_equal(result, data)

    def test_has_tensor(self):
        """Test tensor existence check."""
        ctx = ExecutionContext(device="cpu")
        ctx.set_tensor("exists", np.array([1]))

        assert ctx.has_tensor("exists")
        assert not ctx.has_tensor("not_exists")

    def test_constants(self):
        """Test constant handling."""
        ctx = ExecutionContext(device="cpu")
        weight = np.random.randn(4, 8).astype(np.float32)

        ctx.set_constant("weight", weight)

        assert ctx.has_tensor("weight")
        result = ctx.get_tensor("weight")
        np.testing.assert_array_equal(result, weight)

    def test_clear(self):
        """Test clearing tensors."""
        ctx = ExecutionContext(device="cpu")
        ctx.set_tensor("test", np.array([1]))
        ctx.set_constant("const", np.array([2]))

        ctx.clear()

        assert not ctx.has_tensor("test")
        assert ctx.has_tensor("const")  # Constants should persist

    def test_clear_all(self):
        """Test clearing all including constants."""
        ctx = ExecutionContext(device="cpu")
        ctx.set_tensor("test", np.array([1]))
        ctx.set_constant("const", np.array([2]))

        ctx.clear_all()

        assert not ctx.has_tensor("test")
        assert not ctx.has_tensor("const")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
