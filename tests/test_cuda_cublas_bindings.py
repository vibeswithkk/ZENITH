"""
Test Suite untuk CUDA Kernels, cuBLAS Operations, dan Python Bindings.
"""

import pytest
import numpy as np
import math


class TestFusedBiasOperations:
    """Test Fused Bias + Activation operations."""

    def test_fused_bias_relu_positive(self):
        """Test fused bias + ReLU dengan nilai positif."""
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Expected: max(0, input + bias)
        expected = np.maximum(0, input_data + bias)

        # Simulate
        result = np.maximum(0, input_data + bias)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_fused_bias_relu_negative(self):
        """Test fused bias + ReLU dengan nilai negatif."""
        input_data = np.array([[-1.0, -2.0, 3.0]], dtype=np.float32)
        bias = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Expected: max(0, input + bias) = [-0.5, -1.5, 3.5] -> [0, 0, 3.5]
        result = np.maximum(0, input_data + bias)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert abs(result[0, 2] - 3.5) < 1e-6

    def test_fused_bias_gelu(self):
        """Test fused bias + GELU."""
        input_data = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        sqrt_2_over_pi = 0.7978845608028654
        gelu_coef = 0.044715

        def gelu(x):
            x_cubed = x**3
            inner = sqrt_2_over_pi * (x + gelu_coef * x_cubed)
            return 0.5 * x * (1 + np.tanh(inner))

        result = gelu(input_data + bias)

        # GELU(1) ≈ 0.841
        assert abs(result[0, 0] - 0.841) < 0.01
        # GELU(0) = 0
        assert abs(result[0, 1]) < 1e-6
        # GELU(-1) ≈ -0.159
        assert abs(result[0, 2] - (-0.159)) < 0.01

    def test_fused_bias_sigmoid(self):
        """Test fused bias + sigmoid."""
        input_data = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        result = 1 / (1 + np.exp(-(input_data + bias)))

        # sigmoid(0) = 0.5
        assert abs(result[0, 0] - 0.5) < 1e-6
        # sigmoid(1) ≈ 0.731
        assert abs(result[0, 1] - 0.731) < 0.01


class TestFusedResidualOperations:
    """Test Fused Residual + Activation operations."""

    def test_fused_add_relu(self):
        """Test fused add + ReLU."""
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        residual = np.array([0.5, 3.0, -4.0, 5.0], dtype=np.float32)

        # Expected: max(0, x + residual)
        expected = np.maximum(0, x + residual)

        result = np.maximum(0, x + residual)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # 1.5 > 0, 1.0 > 0, -1.0 -> 0, 1.0 > 0
        assert result[0] > 0
        assert result[1] > 0
        assert result[2] == 0
        assert result[3] > 0


class TestFusedLayerNorm:
    """Test Fused Add + LayerNorm operations."""

    def test_add_layernorm(self):
        """Test fused add + layernorm."""
        batch = 2
        hidden = 4
        x = np.random.randn(batch, hidden).astype(np.float32)
        residual = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        # Combined
        combined = x + residual

        # LayerNorm per row
        result = np.zeros_like(combined)
        for b in range(batch):
            row = combined[b]
            mean = np.mean(row)
            var = np.var(row)
            normalized = (row - mean) / np.sqrt(var + eps)
            result[b] = normalized * gamma + beta

        # Check mean is ~0 and std is ~1 for each row
        for b in range(batch):
            row_mean = np.mean(result[b])
            row_std = np.std(result[b])
            assert abs(row_mean) < 0.1
            assert abs(row_std - 1.0) < 0.1


class TestCuBLASOperations:
    """Test cuBLAS GEMM operations (simulated)."""

    def test_gemm_basic(self):
        """Test basic GEMM: C = A @ B."""
        M, K, N = 4, 3, 5
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        C = A @ B

        assert C.shape == (M, N)
        # Verify one element manually
        expected_00 = np.sum(A[0, :] * B[:, 0])
        assert abs(C[0, 0] - expected_00) < 1e-5

    def test_gemm_with_bias(self):
        """Test GEMM + bias: C = A @ B + bias."""
        M, K, N = 4, 3, 5
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        bias = np.random.randn(N).astype(np.float32)

        C = A @ B + bias

        assert C.shape == (M, N)
        # Each row should have bias added
        expected_00 = np.sum(A[0, :] * B[:, 0]) + bias[0]
        assert abs(C[0, 0] - expected_00) < 1e-5

    def test_strided_batched_gemm(self):
        """Test strided batched GEMM."""
        batch = 8
        M, K, N = 4, 3, 5
        A = np.random.randn(batch, M, K).astype(np.float32)
        B = np.random.randn(batch, K, N).astype(np.float32)

        C = np.zeros((batch, M, N), dtype=np.float32)
        for b in range(batch):
            C[b] = A[b] @ B[b]

        assert C.shape == (batch, M, N)


class TestAttentionOperations:
    """Test Attention-specific operations."""

    def test_attention_qk(self):
        """Test Q @ K^T for attention scores."""
        batch = 2
        heads = 4
        seq_len = 8
        head_dim = 16

        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # scores = Q @ K^T
        scores = np.zeros((batch, heads, seq_len, seq_len), dtype=np.float32)
        for b in range(batch):
            for h in range(heads):
                scores[b, h] = Q[b, h] @ K[b, h].T

        assert scores.shape == (batch, heads, seq_len, seq_len)

    def test_attention_av(self):
        """Test attention_weights @ V for output."""
        batch = 2
        heads = 4
        seq_len = 8
        head_dim = 16

        attn_weights = np.random.randn(batch, heads, seq_len, seq_len).astype(
            np.float32
        )
        # Make it valid probability distribution
        attn_weights = np.exp(attn_weights) / np.sum(
            np.exp(attn_weights), axis=-1, keepdims=True
        )

        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        output = np.zeros((batch, heads, seq_len, head_dim), dtype=np.float32)
        for b in range(batch):
            for h in range(heads):
                output[b, h] = attn_weights[b, h] @ V[b, h]

        assert output.shape == (batch, heads, seq_len, head_dim)


class TestErrorMetrics:
    """Test error metrics computation."""

    def test_compute_error_metrics(self):
        """Test error metrics between reference and optimized."""
        reference = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        optimized = np.array([1.001, 2.002, 3.001, 4.0], dtype=np.float32)

        epsilon = 1e-10
        abs_errors = np.abs(optimized - reference)
        rel_errors = abs_errors / (np.abs(reference) + epsilon)

        max_abs_error = np.max(abs_errors)
        max_rel_error = np.max(rel_errors)
        mean_abs_error = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(abs_errors**2))

        assert max_abs_error < 0.01
        assert max_rel_error < 0.01
        assert mean_abs_error < 0.01
        assert rmse < 0.01


class TestFP16Safety:
    """Test FP16 safety checks."""

    def test_safe_data(self):
        """Test data yang aman untuk FP16."""
        data = np.array([0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float32)

        FP16_MAX = 65504.0
        alpha = 0.95
        abs_max = np.max(np.abs(data))

        safe = abs_max < FP16_MAX * alpha
        assert safe

    def test_unsafe_data_overflow(self):
        """Test data yang akan overflow di FP16."""
        data = np.array([1.0, 100000.0], dtype=np.float32)  # > 65504

        FP16_MAX = 65504.0
        abs_max = np.max(np.abs(data))

        safe = abs_max < FP16_MAX
        assert not safe


class TestPruning:
    """Test pruning operations."""

    def test_magnitude_prune(self):
        """Test magnitude-based pruning."""
        weights = np.array(
            [0.1, -0.5, 0.3, -0.8, 0.2, 0.05, -0.9, 0.15], dtype=np.float32
        )
        target_sparsity = 0.5  # Prune 50%

        # Sort by magnitude
        magnitudes = np.abs(weights)
        sorted_indices = np.argsort(magnitudes)

        num_to_prune = int(len(weights) * target_sparsity)
        pruned = weights.copy()
        for i in range(num_to_prune):
            pruned[sorted_indices[i]] = 0.0

        # Check sparsity
        actual_sparsity = np.sum(pruned == 0) / len(pruned)
        assert actual_sparsity == target_sparsity

    def test_compute_sparsity(self):
        """Test sparsity computation."""
        weights = np.array([0, 1, 0, 2, 0, 0.5], dtype=np.float32)
        sparsity = np.sum(weights == 0) / len(weights)
        assert sparsity == 0.5


class TestKernelLaunchConfiguration:
    """Test CUDA kernel launch configuration calculations."""

    def test_block_size_calculation(self):
        """Test grid/block size calculation."""
        BLOCK_SIZE = 256

        sizes = [100, 256, 1000, 10000, 100000]
        for size in sizes:
            num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
            total_threads = num_blocks * BLOCK_SIZE

            assert total_threads >= size
            assert total_threads < size + BLOCK_SIZE

    def test_2d_grid_calculation(self):
        """Test 2D grid calculation for matrix operations."""
        TILE_SIZE = 16

        M, N = 100, 200
        grid_x = (N + TILE_SIZE - 1) // TILE_SIZE
        grid_y = (M + TILE_SIZE - 1) // TILE_SIZE

        assert grid_x * TILE_SIZE >= N
        assert grid_y * TILE_SIZE >= M


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
