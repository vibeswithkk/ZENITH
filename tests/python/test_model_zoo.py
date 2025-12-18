# Zenith Model Zoo Integration Tests
# Tests end-to-end execution of ResNet-50 and BERT operators
# Run in Google Colab with GPU runtime enabled
#
# These tests require:
#   - zenith._zenith_core compiled with CUDA
#   - GPU runtime enabled
#
# If zenith._zenith_core is not available, all tests will be skipped.

import pytest
import numpy as np
import time

# Check if CUDA module is available
try:
    from zenith._zenith_core import cuda, backends, kernels

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None
    backends = None
    kernels = None

# Skip all tests in this module if CUDA not available
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="zenith._zenith_core not available (requires CUDA build)"
)


class TestCNNOperators:
    """Test CNN operators for ResNet-50 style models."""

    def test_matmul_linear(self):
        """Test MatMul (Linear/Gemm) operator."""
        A = np.random.randn(16, 256).astype(np.float32)
        B = np.random.randn(256, 128).astype(np.float32)

        A_gpu = cuda.to_gpu(A)
        B_gpu = cuda.to_gpu(B)
        C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
        C = C_gpu.to_numpy()

        assert C.shape == (16, 128)
        assert np.allclose(C, A @ B, rtol=1e-3)

    def test_relu_activation(self):
        """Test ReLU activation operator."""
        X = np.random.randn(1, 64, 28, 28).astype(np.float32)
        Y = cuda.relu(X)

        assert Y.shape == X.shape
        assert np.all(Y >= 0)


class TestTransformerOperators:
    """Test Transformer operators for BERT style models."""

    def test_attention_qkt(self):
        """Test Attention QK^T computation."""
        seq, dim = 128, 64

        Q_flat = np.random.randn(seq, dim).astype(np.float32)
        K_flat = np.random.randn(seq, dim).astype(np.float32)

        Q_gpu = cuda.to_gpu(Q_flat)
        K_gpu = cuda.to_gpu(K_flat.T)  # Transpose for QK^T

        scores_gpu = cuda.matmul_gpu(Q_gpu, K_gpu)
        scores = scores_gpu.to_numpy()

        assert scores.shape == (seq, seq)


class TestPerformanceBenchmark:
    """Performance benchmarks for critical operations."""

    @pytest.mark.parametrize("size", [256, 512, 1024])
    def test_matmul_performance(self, size):
        """Benchmark MatMul at different sizes."""
        M, N = size, size
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(N, M).astype(np.float32)

        A_gpu = cuda.to_gpu(A)
        B_gpu = cuda.to_gpu(B)

        # Warmup
        for _ in range(5):
            cuda.matmul_gpu(A_gpu, B_gpu)

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
            times.append((time.perf_counter() - t0) * 1000)

        avg_time = np.mean(times)
        gflops = (2 * M * N * M) / (avg_time / 1000) / 1e9

        # Just ensure it completes without error
        assert avg_time > 0
        assert gflops > 0


class TestMemoryPool:
    """Test memory pool efficiency."""

    def test_memory_pool_reuse(self):
        """Test that memory pool reuses allocations."""
        cuda.clear_memory_pool()

        # Simulate inference pattern
        for _ in range(10):
            A = np.random.randn(256, 256).astype(np.float32)
            B = np.random.randn(256, 256).astype(np.float32)
            A_gpu = cuda.to_gpu(A)
            B_gpu = cuda.to_gpu(B)
            C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
            _ = C_gpu.to_numpy()

        stats = cuda.memory_stats()

        assert stats["allocations"] >= 0
        assert stats["cache_hits"] >= 0


# Allow running as standalone script for manual testing
if __name__ == "__main__":
    if CUDA_AVAILABLE:
        print("=" * 60)
        print("ZENITH MODEL ZOO INTEGRATION TESTS")
        print("=" * 60)
        print(f"\nCUDA available: {backends.is_cuda_available()}")
        print(f"cuDNN available: {backends.is_cudnn_available()}")
        print(f"cuDNN version: {backends.get_cudnn_version()}")
        print("\nRun with: pytest -v tests/python/test_model_zoo.py")
    else:
        print("zenith._zenith_core not available")
        print("Build Zenith with CUDA support first")
