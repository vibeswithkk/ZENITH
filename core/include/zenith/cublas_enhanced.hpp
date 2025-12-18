// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// cuBLAS Enhanced Operations - High Performance Linear Algebra
// Berdasarkan CetakBiru.md: Zenith-MKL dengan cuDNN/cuBLAS integration
// Referensi: NVIDIA cuBLAS Developer Guide, cuBLASLt, CUTLASS

#ifndef ZENITH_CUBLAS_ENHANCED_HPP
#define ZENITH_CUBLAS_ENHANCED_HPP

#include "cublas_ops.hpp"
#include "types.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zenith {
namespace cublas {

// ============================================================================
// Enhanced GEMM with Epilogue Fusion Support
// Menggunakan cuBLASLt untuk Tensor Core acceleration dan epilogue fusion
// ============================================================================

/// Compute Type for mixed precision
enum class ComputeType {
  FP32,      // Standard FP32
  TF32,      // Tensor Float 32 (Ampere+)
  FP16,      // Half precision with FP32 accumulation
  FP16_FAST, // Half precision with FP16 accumulation
  BF16,      // BFloat16 (Ampere+)
};

/// GEMM Configuration
struct GemmConfig {
  ComputeType compute_type = ComputeType::TF32;
  bool use_tensor_cores = true;
  bool allow_mixed_precision = true;
};

/// Result containing timing and status
struct GemmResult {
  bool success = false;
  float elapsed_ms = 0.0f;
  std::string error_message;
};

// ============================================================================
// GEMM Operations
// ============================================================================

/// Standard GEMM: C = alpha * A * B + beta * C
/// A: [M, K], B: [K, N], C: [M, N]
inline GemmResult gemm_f32(cublasHandle_t handle, const float *A,
                           const float *B, float *C, int M, int N, int K,
                           float alpha = 1.0f, float beta = 0.0f,
                           bool transA = false, bool transB = false) {
  GemmResult result;

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Note: cuBLAS uses column-major, so we compute B^T * A^T = (A * B)^T
  // For row-major: swap A and B, swap M and N
  int lda = transA ? M : K;
  int ldb = transB ? K : N;
  int ldc = N;

  cublasStatus_t status = cublasSgemm(handle, opB, opA, // Swapped for row-major
                                      N, M, K,          // Swapped M and N
                                      &alpha, B, ldb, A, lda, &beta, C, ldc);

  result.success = (status == CUBLAS_STATUS_SUCCESS);
  if (!result.success) {
    result.error_message =
        "cublasSgemm failed with status " + std::to_string(status);
  }

  return result;
}

/// Batch GEMM: C[i] = alpha * A[i] * B[i] + beta * C[i]
inline GemmResult batched_gemm_f32(cublasHandle_t handle, const float *const *A,
                                   const float *const *B, float *const *C,
                                   int M, int N, int K, int batch_count,
                                   float alpha = 1.0f, float beta = 0.0f) {
  GemmResult result;

  int lda = K;
  int ldb = N;
  int ldc = N;

  cublasStatus_t status =
      cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                         ldb, A, lda, &beta, C, ldc, batch_count);

  result.success = (status == CUBLAS_STATUS_SUCCESS);
  if (!result.success) {
    result.error_message = "cublasSgemmBatched failed";
  }

  return result;
}

/// Strided Batch GEMM (lebih efisien untuk memory-contiguous batches)
inline GemmResult strided_batched_gemm_f32(
    cublasHandle_t handle, const float *A, const float *B, float *C, int M,
    int N, int K, int batch_count, long long strideA, long long strideB,
    long long strideC, float alpha = 1.0f, float beta = 0.0f) {
  GemmResult result;

  cublasStatus_t status = cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, strideB, A, K,
      strideA, &beta, C, N, strideC, batch_count);

  result.success = (status == CUBLAS_STATUS_SUCCESS);
  if (!result.success) {
    result.error_message = "cublasSgemmStridedBatched failed";
  }

  return result;
}

// ============================================================================
// FP16 GEMM with Tensor Cores
// ============================================================================

/// FP16 GEMM dengan FP32 accumulation untuk accuracy
/// Menggunakan Tensor Cores pada hardware yang mendukung
inline GemmResult gemm_fp16(cublasHandle_t handle, const __half *A,
                            const __half *B, __half *C, int M, int N, int K,
                            float alpha = 1.0f, float beta = 0.0f) {
  GemmResult result;

  cublasStatus_t status =
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                   CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N,
                   CUBLAS_COMPUTE_32F, // FP32 accumulation for accuracy
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP // Allow Tensor Cores
      );

  result.success = (status == CUBLAS_STATUS_SUCCESS);
  if (!result.success) {
    result.error_message =
        "cublasGemmEx (FP16) failed with status " + std::to_string(status);
  }

  return result;
}

// ============================================================================
// Vector Operations (untuk bias addition, etc.)
// ============================================================================

/// Vector addition: y = alpha * x + y (SAXPY)
inline void vector_add_f32(cublasHandle_t handle, int n, float alpha,
                           const float *x, float *y) {
  cublasSaxpy(handle, n, &alpha, x, 1, y, 1);
}

/// Scale vector: x = alpha * x (SSCAL)
inline void vector_scale_f32(cublasHandle_t handle, int n, float alpha,
                             float *x) {
  cublasSscal(handle, n, &alpha, x, 1);
}

/// Dot product: result = x . y
inline float dot_product_f32(cublasHandle_t handle, int n, const float *x,
                             const float *y) {
  float result;
  cublasSdot(handle, n, x, 1, y, 1, &result);
  return result;
}

/// L2 norm: ||x||_2
inline float norm_f32(cublasHandle_t handle, int n, const float *x) {
  float result;
  cublasSnrm2(handle, n, x, 1, &result);
  return result;
}

// ============================================================================
// GEMM + Fused Epilogue (using external fused kernels)
// ============================================================================

// Forward declarations untuk fused kernels (implemented in fused_kernels.cu)
extern "C" {
void fused_gemm_epilogue_relu(float *C, const float *bias, int M, int N);
void fused_gemm_epilogue_gelu(float *C, const float *bias, int M, int N);
}

/// Linear + ReLU: Y = ReLU(X @ W + bias)
inline GemmResult linear_relu(cublasHandle_t handle, const float *X,
                              const float *W, const float *bias, float *Y,
                              int batch, int in_features, int out_features) {
  // Step 1: GEMM
  GemmResult result =
      gemm_f32(handle, X, W, Y, batch, out_features, in_features);
  if (!result.success)
    return result;

  // Step 2: Fused bias + ReLU
  fused_gemm_epilogue_relu(Y, bias, batch, out_features);

  return result;
}

/// Linear + GELU: Y = GELU(X @ W + bias)
inline GemmResult linear_gelu(cublasHandle_t handle, const float *X,
                              const float *W, const float *bias, float *Y,
                              int batch, int in_features, int out_features) {
  // Step 1: GEMM
  GemmResult result =
      gemm_f32(handle, X, W, Y, batch, out_features, in_features);
  if (!result.success)
    return result;

  // Step 2: Fused bias + GELU
  fused_gemm_epilogue_gelu(Y, bias, batch, out_features);

  return result;
}

// ============================================================================
// Attention-Specific Operations
// ============================================================================

/// Batched matrix multiply for attention: scores = Q @ K^T
inline GemmResult attention_qk(cublasHandle_t handle, const float *Q,
                               const float *K, float *scores, int batch,
                               int heads, int seq_len, int head_dim) {
  // Q: [batch, heads, seq_len, head_dim]
  // K: [batch, heads, seq_len, head_dim]
  // scores: [batch, heads, seq_len, seq_len]

  int total_batches = batch * heads;
  long long strideQ = static_cast<long long>(seq_len) * head_dim;
  long long strideK = static_cast<long long>(seq_len) * head_dim;
  long long strideS = static_cast<long long>(seq_len) * seq_len;

  // scores = Q @ K^T -> need transB
  return strided_batched_gemm_f32(handle, Q, K, scores, seq_len, seq_len,
                                  head_dim, total_batches, strideQ, strideK,
                                  strideS, 1.0f, 0.0f);
}

/// Batched matrix multiply for attention output: output = attn_weights @ V
inline GemmResult attention_av(cublasHandle_t handle, const float *attn_weights,
                               const float *V, float *output, int batch,
                               int heads, int seq_len, int head_dim) {
  int total_batches = batch * heads;
  long long strideA = static_cast<long long>(seq_len) * seq_len;
  long long strideV = static_cast<long long>(seq_len) * head_dim;
  long long strideO = static_cast<long long>(seq_len) * head_dim;

  return strided_batched_gemm_f32(handle, attn_weights, V, output, seq_len,
                                  head_dim, seq_len, total_batches, strideA,
                                  strideV, strideO, 1.0f, 0.0f);
}

} // namespace cublas
} // namespace zenith

#endif // ZENITH_CUBLAS_ENHANCED_HPP
