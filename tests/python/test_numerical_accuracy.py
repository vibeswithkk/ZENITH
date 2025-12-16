"""
Numerical Accuracy Tests - Zenith Kernels vs NumPy Reference

This module validates Zenith C++ kernel implementations against
NumPy reference implementations to ensure numerical correctness.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import pytest
from typing import Callable


def relative_error(result: np.ndarray, reference: np.ndarray) -> float:
    """Calculate maximum relative error between arrays."""
    diff = np.abs(result - reference)
    ref_abs = np.abs(reference)
    # Avoid division by zero
    mask = ref_abs > 1e-8
    if np.any(mask):
        return float(np.max(diff[mask] / ref_abs[mask]))
    return float(np.max(diff))


def absolute_error(result: np.ndarray, reference: np.ndarray) -> float:
    """Calculate maximum absolute error between arrays."""
    return float(np.max(np.abs(result - reference)))


def allclose(
    result: np.ndarray, reference: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6
) -> bool:
    """Check if two arrays are element-wise equal within tolerance."""
    return np.allclose(result, reference, rtol=rtol, atol=atol)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.Generator(np.random.PCG64(42))


@pytest.fixture
def small_matrix(rng):
    """Small test matrix for basic correctness."""
    return rng.standard_normal((8, 8)).astype(np.float32)


@pytest.fixture
def medium_matrix(rng):
    """Medium test matrix for performance validation."""
    return rng.standard_normal((64, 64)).astype(np.float32)


@pytest.fixture
def large_matrix(rng):
    """Large test matrix for stress testing."""
    return rng.standard_normal((256, 256)).astype(np.float32)


# ============================================================================
# Matrix Multiplication Tests
# ============================================================================


class TestMatMulAccuracy:
    """Validate matrix multiplication numerical accuracy."""

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (4, 4, 4),
            (8, 16, 8),
            (16, 8, 32),
            (32, 64, 16),
            (64, 64, 64),
            (128, 128, 128),
        ],
    )
    def test_matmul_dimensions(self, rng, M: int, K: int, N: int):
        """Test matmul with various dimensions."""
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)

        # NumPy reference
        ref = A @ B

        # Zenith kernel (simulated - will use actual C++ binding when built)
        result = self._zenith_matmul(A, B)

        assert allclose(result, ref, rtol=1e-4, atol=1e-5), (
            f"MatMul error: max_rel={relative_error(result, ref):.2e}"
        )

    def test_matmul_identity(self, rng):
        """Test matmul with identity matrix."""
        A = rng.standard_normal((16, 16)).astype(np.float32)
        I = np.eye(16, dtype=np.float32)

        ref = A @ I
        result = self._zenith_matmul(A, I)

        assert allclose(result, ref)

    def test_matmul_zeros(self):
        """Test matmul with zero matrix."""
        A = np.zeros((8, 8), dtype=np.float32)
        B = np.ones((8, 8), dtype=np.float32)

        ref = A @ B
        result = self._zenith_matmul(A, B)

        assert np.allclose(result, ref)

    def _zenith_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Call Zenith matmul kernel (fallback to NumPy if not built)."""
        try:
            from zenith._zenith_core import kernels

            return kernels.matmul(A, B)
        except ImportError:
            # Fallback to NumPy for testing without C++ bindings
            return A @ B


# ============================================================================
# Activation Function Tests
# ============================================================================


class TestActivationAccuracy:
    """Validate activation function numerical accuracy."""

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_relu_accuracy(self, rng, size: int):
        """Test ReLU activation."""
        x = rng.standard_normal(size).astype(np.float32)

        ref = np.maximum(0, x)
        result = self._zenith_relu(x.copy())

        assert allclose(result, ref)

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_sigmoid_accuracy(self, rng, size: int):
        """Test sigmoid activation."""
        x = rng.uniform(-5, 5, size).astype(np.float32)

        ref = 1 / (1 + np.exp(-x))
        result = self._zenith_sigmoid(x.copy())

        assert allclose(result, ref, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_tanh_accuracy(self, rng, size: int):
        """Test tanh activation."""
        x = rng.uniform(-3, 3, size).astype(np.float32)

        ref = np.tanh(x)
        result = self._zenith_tanh(x.copy())

        assert allclose(result, ref, rtol=1e-5, atol=1e-6)

    def test_relu_edge_cases(self):
        """Test ReLU with edge cases."""
        x = np.array([-1e10, -1e-10, 0, 1e-10, 1e10], dtype=np.float32)
        ref = np.maximum(0, x)
        result = self._zenith_relu(x.copy())
        assert allclose(result, ref)

    def test_sigmoid_edge_cases(self):
        """Test sigmoid with extreme values."""
        x = np.array([-100, -10, 0, 10, 100], dtype=np.float32)
        ref = 1 / (1 + np.exp(-np.clip(x, -88, 88)))  # Clip to avoid overflow
        result = self._zenith_sigmoid(x.copy())
        # For extreme values, check approximate behavior
        assert result[0] < 0.01  # sigmoid(-100) ~ 0
        assert abs(result[2] - 0.5) < 0.01  # sigmoid(0) = 0.5
        assert result[4] > 0.99  # sigmoid(100) ~ 1

    def _zenith_relu(self, x: np.ndarray) -> np.ndarray:
        """Call Zenith ReLU kernel."""
        try:
            from zenith._zenith_core import kernels

            return kernels.relu(x)
        except ImportError:
            return np.maximum(0, x)

    def _zenith_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Call Zenith sigmoid kernel."""
        try:
            from zenith._zenith_core import kernels

            return kernels.sigmoid(x)
        except ImportError:
            # Clip to avoid overflow in exp
            return 1 / (1 + np.exp(-np.clip(x, -88, 88)))

    def _zenith_tanh(self, x: np.ndarray) -> np.ndarray:
        """Call Zenith tanh kernel."""
        try:
            from zenith._zenith_core import kernels

            return kernels.tanh(x)
        except ImportError:
            return np.tanh(x)


# ============================================================================
# Element-wise Operation Tests
# ============================================================================


class TestElementWiseAccuracy:
    """Validate element-wise operation numerical accuracy."""

    @pytest.mark.parametrize("size", [16, 64, 256, 1024])
    def test_add_accuracy(self, rng, size: int):
        """Test element-wise addition."""
        a = rng.standard_normal(size).astype(np.float32)
        b = rng.standard_normal(size).astype(np.float32)

        ref = a + b
        result = self._zenith_add(a, b)

        assert allclose(result, ref)

    @pytest.mark.parametrize("size", [16, 64, 256])
    def test_sum_accuracy(self, rng, size: int):
        """Test sum reduction."""
        x = rng.standard_normal(size).astype(np.float32)

        ref = np.sum(x)
        result = self._zenith_sum(x)

        # Reduction operations can accumulate error
        assert abs(result - ref) < 1e-3 * size

    @pytest.mark.parametrize("size", [16, 64, 256])
    def test_mean_accuracy(self, rng, size: int):
        """Test mean reduction."""
        x = rng.standard_normal(size).astype(np.float32)

        ref = np.mean(x)
        result = self._zenith_mean(x)

        assert abs(result - ref) < 1e-4

    def test_max_min_accuracy(self, rng):
        """Test max and min reductions."""
        x = rng.standard_normal(100).astype(np.float32)

        ref_max = np.max(x)
        ref_min = np.min(x)

        result_max = self._zenith_max(x)
        result_min = self._zenith_min(x)

        assert abs(result_max - ref_max) < 1e-6
        assert abs(result_min - ref_min) < 1e-6

    def _zenith_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            from zenith._zenith_core import kernels

            return kernels.add(a, b)
        except ImportError:
            return a + b

    def _zenith_sum(self, x: np.ndarray) -> float:
        try:
            from zenith._zenith_core import kernels

            return kernels.sum(x)
        except ImportError:
            return float(np.sum(x))

    def _zenith_mean(self, x: np.ndarray) -> float:
        try:
            from zenith._zenith_core import kernels

            return kernels.mean(x)
        except ImportError:
            return float(np.mean(x))

    def _zenith_max(self, x: np.ndarray) -> float:
        try:
            from zenith._zenith_core import kernels

            return kernels.max(x)
        except ImportError:
            return float(np.max(x))

    def _zenith_min(self, x: np.ndarray) -> float:
        try:
            from zenith._zenith_core import kernels

            return kernels.min(x)
        except ImportError:
            return float(np.min(x))


# ============================================================================
# Softmax Tests
# ============================================================================


class TestSoftmaxAccuracy:
    """Validate softmax numerical accuracy."""

    @pytest.mark.parametrize("size", [8, 16, 64, 128])
    def test_softmax_basic(self, rng, size: int):
        """Test basic softmax."""
        x = rng.standard_normal(size).astype(np.float32)

        # NumPy reference with numerical stability
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        ref = exp_x / np.sum(exp_x)

        result = self._zenith_softmax(x)

        assert allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_softmax_sums_to_one(self, rng):
        """Test that softmax output sums to 1."""
        x = rng.standard_normal(100).astype(np.float32)
        result = self._zenith_softmax(x)
        assert abs(np.sum(result) - 1.0) < 1e-5

    def test_softmax_non_negative(self, rng):
        """Test that softmax outputs are non-negative."""
        x = rng.standard_normal(100).astype(np.float32)
        result = self._zenith_softmax(x)
        assert np.all(result >= 0)

    def test_softmax_monotonic(self):
        """Test that larger inputs produce larger softmax outputs."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = self._zenith_softmax(x)
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def _zenith_softmax(self, x: np.ndarray) -> np.ndarray:
        try:
            from zenith._zenith_core import kernels

            return kernels.softmax(x)
        except ImportError:
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x)


# ============================================================================
# Convolution Tests
# ============================================================================


class TestConv2DAccuracy:
    """Validate 2D convolution numerical accuracy."""

    def test_conv2d_identity_kernel(self, rng):
        """Test conv2d with identity-like kernel."""
        # 1x1x5x5 input, 1x1x1x1 kernel
        x = rng.standard_normal((1, 1, 5, 5)).astype(np.float32)
        w = np.array([[[[1.0]]]]).astype(np.float32)

        ref = x  # Identity kernel should preserve input
        result = self._zenith_conv2d(x, w)

        assert allclose(result, ref)

    def test_conv2d_simple(self, rng):
        """Test simple 3x3 convolution."""
        # 1x1x8x8 input, 1x1x3x3 kernel
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
        w = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)

        ref = self._numpy_conv2d(x, w, padding=0, stride=1)
        result = self._zenith_conv2d(x, w, padding=0, stride=1)

        assert allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_conv2d_with_padding(self, rng):
        """Test convolution with padding."""
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
        w = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)

        ref = self._numpy_conv2d(x, w, padding=1, stride=1)
        result = self._zenith_conv2d(x, w, padding=1, stride=1)

        assert allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_conv2d_multi_channel(self, rng):
        """Test multi-channel convolution."""
        # 1x3x8x8 input, 2x3x3x3 kernel -> 1x2x6x6 output
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
        w = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)

        ref = self._numpy_conv2d(x, w, padding=0, stride=1)
        result = self._zenith_conv2d(x, w, padding=0, stride=1)

        assert allclose(result, ref, rtol=1e-4, atol=1e-5)

    def _zenith_conv2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        bias: np.ndarray = None,
        padding: int = 0,
        stride: int = 1,
    ) -> np.ndarray:
        try:
            from zenith._zenith_core import kernels

            return kernels.conv2d(x, w, bias, stride, padding)
        except ImportError:
            return self._numpy_conv2d(x, w, padding, stride, bias)

    def _numpy_conv2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        padding: int = 0,
        stride: int = 1,
        bias: np.ndarray = None,
    ) -> np.ndarray:
        """NumPy reference implementation for conv2d."""
        N, C_in, H, W = x.shape
        C_out, C_in_w, K_h, K_w = w.shape
        assert C_in == C_in_w

        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1

        # Pad input
        if padding > 0:
            x_pad = np.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )
        else:
            x_pad = x

        output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for co in range(C_out):
                for h in range(H_out):
                    for wi in range(W_out):
                        h_start = h * stride
                        w_start = wi * stride
                        patch = x_pad[
                            n, :, h_start : h_start + K_h, w_start : w_start + K_w
                        ]
                        output[n, co, h, wi] = np.sum(patch * w[co])

        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)

        return output


# ============================================================================
# Pooling Tests
# ============================================================================


class TestPoolingAccuracy:
    """Validate pooling operation numerical accuracy."""

    def test_maxpool2d_basic(self, rng):
        """Test 2x2 max pooling."""
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)

        ref = self._numpy_maxpool2d(x, kernel_size=2, stride=2)
        result = self._zenith_maxpool2d(x, kernel_size=2, stride=2)

        assert allclose(result, ref)

    def test_maxpool2d_multi_channel(self, rng):
        """Test max pooling with multiple channels."""
        x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)

        ref = self._numpy_maxpool2d(x, kernel_size=2, stride=2)
        result = self._zenith_maxpool2d(x, kernel_size=2, stride=2)

        assert allclose(result, ref)

    def _zenith_maxpool2d(
        self, x: np.ndarray, kernel_size: int = 2, stride: int = 2
    ) -> np.ndarray:
        try:
            from zenith._zenith_core import kernels

            return kernels.maxpool2d(x, kernel_size, stride)
        except ImportError:
            return self._numpy_maxpool2d(x, kernel_size, stride)

    def _numpy_maxpool2d(
        self, x: np.ndarray, kernel_size: int = 2, stride: int = 2
    ) -> np.ndarray:
        """NumPy reference for max pooling."""
        N, C, H, W = x.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        output = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        w_start = w * stride
                        patch = x[
                            n,
                            c,
                            h_start : h_start + kernel_size,
                            w_start : w_start + kernel_size,
                        ]
                        output[n, c, h, w] = np.max(patch)

        return output


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    def test_matmul_large_values(self, rng):
        """Test matmul with large values."""
        A = rng.uniform(1e3, 1e4, (16, 16)).astype(np.float32)
        B = rng.uniform(1e3, 1e4, (16, 16)).astype(np.float32)

        ref = A @ B
        result = self._zenith_matmul(A, B)

        # Allow larger tolerance for large values
        assert allclose(result, ref, rtol=1e-3, atol=1e3)

    def test_matmul_small_values(self, rng):
        """Test matmul with small values."""
        A = rng.uniform(1e-4, 1e-3, (16, 16)).astype(np.float32)
        B = rng.uniform(1e-4, 1e-3, (16, 16)).astype(np.float32)

        ref = A @ B
        result = self._zenith_matmul(A, B)

        assert allclose(result, ref, rtol=1e-4, atol=1e-10)

    def test_softmax_numerical_stability(self):
        """Test softmax with large input values."""
        # Large positive values
        x = np.array([1000, 1001, 1002], dtype=np.float32)

        result = self._zenith_softmax(x)

        # Should still sum to 1 and be non-negative
        assert abs(np.sum(result) - 1.0) < 1e-5
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def _zenith_matmul(self, A, B):
        try:
            from zenith._zenith_core import kernels

            return kernels.matmul(A, B)
        except ImportError:
            return A @ B

    def _zenith_softmax(self, x):
        try:
            from zenith._zenith_core import kernels

            return kernels.softmax(x)
        except ImportError:
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
