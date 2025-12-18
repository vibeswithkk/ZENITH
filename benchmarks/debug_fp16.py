# Debug FP16 Accuracy Layer by Layer

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("FP16 ACCURACY DEBUG")
print("=" * 70)

from zenith._zenith_core import cuda

# First, test individual FP16 operations against FP32


def test_linear():
    """Test linear_fp16_gpu vs linear_gpu"""
    print("\n[1] Testing linear_fp16...")
    M, K, N = 32, 768, 768

    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(N, K).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    X_gpu = cuda.to_gpu(np.ascontiguousarray(X))
    W_gpu = cuda.to_gpu(np.ascontiguousarray(W))
    bias_gpu = cuda.to_gpu(np.ascontiguousarray(bias))

    # FP32 reference
    out_fp32 = cuda.linear_gpu(X_gpu, W_gpu, bias_gpu)
    out_fp32_np = out_fp32.to_numpy()

    # FP16
    out_fp16 = cuda.linear_fp16_gpu(X_gpu, W_gpu, bias_gpu)
    out_fp16_np = out_fp16.to_numpy()

    max_diff = np.max(np.abs(out_fp32_np - out_fp16_np))
    print(f"  FP32 vs FP16 linear: max_diff = {max_diff:.2e}")
    print(f"  Status: {'PASS' if max_diff < 0.1 else 'FAIL'}")
    return max_diff < 0.1


def test_gelu():
    """Test gelu_fp16_gpu vs gelu_gpu"""
    print("\n[2] Testing gelu_fp16...")
    size = 32 * 768

    X = np.random.randn(size).astype(np.float32)
    X_gpu = cuda.to_gpu(np.ascontiguousarray(X.reshape(32, 768)))

    out_fp32 = cuda.gelu_gpu(X_gpu)
    out_fp32_np = out_fp32.to_numpy()

    out_fp16 = cuda.gelu_fp16_gpu(X_gpu)
    out_fp16_np = out_fp16.to_numpy()

    max_diff = np.max(np.abs(out_fp32_np - out_fp16_np))
    print(f"  FP32 vs FP16 GELU: max_diff = {max_diff:.2e}")
    print(f"  Status: {'PASS' if max_diff < 0.01 else 'FAIL'}")
    return max_diff < 0.01


def test_layernorm():
    """Test layernorm_fp16_gpu vs layernorm_gpu"""
    print("\n[3] Testing layernorm_fp16...")
    batch, hidden = 32, 768

    X = np.random.randn(batch, hidden).astype(np.float32)
    gamma = np.ones(hidden).astype(np.float32)
    beta = np.zeros(hidden).astype(np.float32)

    X_gpu = cuda.to_gpu(np.ascontiguousarray(X))
    gamma_gpu = cuda.to_gpu(np.ascontiguousarray(gamma))
    beta_gpu = cuda.to_gpu(np.ascontiguousarray(beta))

    out_fp32 = cuda.layernorm_gpu(X_gpu, gamma_gpu, beta_gpu)
    out_fp32_np = out_fp32.to_numpy()

    out_fp16 = cuda.layernorm_fp16_gpu(X_gpu, gamma_gpu, beta_gpu, 1e-5)
    out_fp16_np = out_fp16.to_numpy()

    max_diff = np.max(np.abs(out_fp32_np - out_fp16_np))
    print(f"  FP32 vs FP16 LayerNorm: max_diff = {max_diff:.2e}")
    print(f"  Status: {'PASS' if max_diff < 0.01 else 'FAIL'}")
    return max_diff < 0.01


def test_attention():
    """Test attention_full_fp16_gpu vs cublas_attention_gpu"""
    print("\n[4] Testing attention_fp16...")
    batch, heads, seq, dim = 1, 12, 32, 64

    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 0.1
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 0.1
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 0.1

    Q_gpu = cuda.to_gpu(np.ascontiguousarray(Q))
    K_gpu = cuda.to_gpu(np.ascontiguousarray(K))
    V_gpu = cuda.to_gpu(np.ascontiguousarray(V))

    out_fp32 = cuda.cublas_attention_gpu(Q_gpu, K_gpu, V_gpu)
    out_fp32_np = out_fp32.to_numpy()

    out_fp16 = cuda.attention_full_fp16_gpu(Q_gpu, K_gpu, V_gpu)
    out_fp16_np = out_fp16.to_numpy()

    max_diff = np.max(np.abs(out_fp32_np - out_fp16_np))
    print(f"  FP32 vs FP16 Attention: max_diff = {max_diff:.2e}")
    print(f"  Status: {'PASS' if max_diff < 0.1 else 'FAIL'}")
    return max_diff < 0.1


def test_add():
    """Test add_fp16_gpu vs add_2d_gpu"""
    print("\n[5] Testing add_fp16...")
    size = 32 * 768

    A = np.random.randn(32, 768).astype(np.float32)
    B = np.random.randn(32, 768).astype(np.float32)

    A_gpu = cuda.to_gpu(np.ascontiguousarray(A))
    B_gpu = cuda.to_gpu(np.ascontiguousarray(B))

    out_fp32 = cuda.add_2d_gpu(A_gpu, B_gpu)
    out_fp32_np = out_fp32.to_numpy()

    out_fp16 = cuda.add_fp16_gpu(A_gpu, B_gpu)
    out_fp16_np = out_fp16.to_numpy()

    max_diff = np.max(np.abs(out_fp32_np - out_fp16_np))
    print(f"  FP32 vs FP16 Add: max_diff = {max_diff:.2e}")
    print(f"  Status: {'PASS' if max_diff < 0.001 else 'FAIL'}")
    return max_diff < 0.001


if __name__ == "__main__":
    results = []

    results.append(("Linear", test_linear()))
    results.append(("GELU", test_gelu()))
    results.append(("LayerNorm", test_layernorm()))
    results.append(("Attention", test_attention()))
    results.append(("Add", test_add()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:15}: {status}")

    all_pass = all(r[1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)
