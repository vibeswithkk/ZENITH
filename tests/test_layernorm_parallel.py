# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for Parallel LayerNorm Kernel Optimization.

Tests numerical correctness of the optimized parallel LayerNorm kernel
against a reference NumPy implementation.

Test Strategy:
1. Basic Correctness: Compare against numpy reference
2. Hidden Sizes: Test various hidden dimensions
3. Batch Sizes: Test various batch dimensions
4. Edge Cases: Minimum dimensions
5. Numerical Stability: Large values, small epsilon
"""

import numpy as np
import pytest


def numpy_layernorm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Reference LayerNorm implementation in NumPy."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * x_normalized + beta


class TestLayerNormCorrectness:
    """Test numerical correctness of parallel LayerNorm."""

    @pytest.fixture
    def cuda_layernorm(self):
        """Get CUDA LayerNorm function if available."""
        try:
            from zenith._zenith_core import cuda

            return cuda.layernorm_f32
        except ImportError:
            pytest.skip("CUDA module not available")

    @pytest.mark.parametrize(
        "batch,hidden",
        [
            (1, 64),
            (1, 128),
            (1, 256),
            (1, 768),
            (8, 768),
            (32, 768),
            (128, 768),
            (1, 1024),
            (16, 1024),
        ],
    )
    def test_layernorm_correctness(self, cuda_layernorm, batch, hidden):
        """Test LayerNorm output matches reference implementation."""
        np.random.seed(42)

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        expected = numpy_layernorm(x, gamma, beta, eps)

        output = np.zeros_like(x)
        cuda_layernorm(x, output, gamma, beta, batch, hidden, eps)

        np.testing.assert_allclose(
            output,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"LayerNorm mismatch for batch={batch}, hidden={hidden}",
        )

    @pytest.mark.parametrize("hidden", [32, 64, 128, 256, 512, 768, 1024])
    def test_hidden_dimensions(self, cuda_layernorm, hidden):
        """Test various hidden dimensions."""
        batch = 16
        np.random.seed(42)

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.random.randn(hidden).astype(np.float32)
        beta = np.random.randn(hidden).astype(np.float32)
        eps = 1e-5

        expected = numpy_layernorm(x, gamma, beta, eps)

        output = np.zeros_like(x)
        cuda_layernorm(x, output, gamma, beta, batch, hidden, eps)

        max_diff = np.abs(output - expected).max()
        mean_diff = np.abs(output - expected).mean()

        assert max_diff < 1e-4, f"Max diff {max_diff} for hidden={hidden}"
        assert mean_diff < 1e-5, f"Mean diff {mean_diff} for hidden={hidden}"

    def test_numerical_stability_large_values(self, cuda_layernorm):
        """Test numerical stability with large input values."""
        batch, hidden = 8, 768
        np.random.seed(42)

        x = np.random.randn(batch, hidden).astype(np.float32) * 100.0
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        expected = numpy_layernorm(x, gamma, beta, eps)

        output = np.zeros_like(x)
        cuda_layernorm(x, output, gamma, beta, batch, hidden, eps)

        np.testing.assert_allclose(
            output,
            expected,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Numerical stability issue with large values",
        )

    def test_edge_case_minimum_dimensions(self, cuda_layernorm):
        """Test edge case with minimum dimensions."""
        batch, hidden = 1, 32
        np.random.seed(42)

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        expected = numpy_layernorm(x, gamma, beta, eps)

        output = np.zeros_like(x)
        cuda_layernorm(x, output, gamma, beta, batch, hidden, eps)

        np.testing.assert_allclose(
            output,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Edge case failure for minimum dimensions",
        )


class TestLayerNormBenchmark:
    """Benchmark tests to verify performance improvement."""

    @pytest.fixture
    def cuda_layernorm(self):
        """Get CUDA LayerNorm function if available."""
        try:
            from zenith._zenith_core import cuda

            return cuda.layernorm_f32
        except ImportError:
            pytest.skip("CUDA module not available")

    def test_bert_base_config(self, cuda_layernorm):
        """Test BERT-base configuration (batch=32, hidden=768)."""
        batch, hidden = 32, 768
        np.random.seed(42)

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-12

        expected = numpy_layernorm(x, gamma, beta, eps)

        output = np.zeros_like(x)
        cuda_layernorm(x, output, gamma, beta, batch, hidden, eps)

        max_diff = np.abs(output - expected).max()
        assert max_diff < 1e-4, f"BERT config failed with max_diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
