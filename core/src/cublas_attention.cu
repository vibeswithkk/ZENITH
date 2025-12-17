// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// cuBLAS-Based Attention for Zenith Framework
// Uses cuBLAS GEMM for high-performance matrix multiplication

#ifndef ZENITH_CUBLAS_ATTENTION_CU
#define ZENITH_CUBLAS_ATTENTION_CU

#ifdef ZENITH_HAS_CUDA

#include <cfloat>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace zenith {
namespace cublas_attention {

// ============================================================================
// cuBLAS Handle (singleton)
// ============================================================================

static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  static bool initialized = false;
  if (!initialized) {
    cublasCreate(&handle);
    // Enable Tensor Cores for FP32 (TF32 mode)
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    initialized = true;
  }
  return handle;
}

// ============================================================================
// Softmax Kernel (row-wise)
// ============================================================================

__global__ void softmax_kernel(float *data, const int rows, const int cols,
                               const float scale) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows)
    return;

  float *row_data = data + row * cols;

  // Scale and find max
  float max_val = -FLT_MAX;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_data[j] *= scale;
    max_val = fmaxf(max_val, row_data[j]);
  }

  // Warp reduce for max
  for (int offset = 16; offset > 0; offset /= 2) {
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
  }
  max_val = __shfl_sync(0xffffffff, max_val, 0);

  // Compute exp and sum
  float sum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_data[j] = expf(row_data[j] - max_val);
    sum += row_data[j];
  }

  // Warp reduce for sum
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  sum = __shfl_sync(0xffffffff, sum, 0);

  // Normalize
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_data[j] /= sum;
  }
}

// ============================================================================
// cuBLAS-Based Multi-Head Attention
// ============================================================================

// Q, K, V: [batch * num_heads, seq_len, head_dim]
// Output: [batch * num_heads, seq_len, head_dim]
void cublas_attention_forward(
    const float *Q, const float *K, const float *V, float *O,
    float *workspace, // Temporary storage for scores [batch*heads, seq, seq]
    int batch_heads,  // batch_size * num_heads
    int seq_len, int head_dim) {
  cublasHandle_t handle = get_cublas_handle();

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float alpha = 1.0f;
  float beta = 0.0f;

  // ========================================
  // Step 1: Scores = Q @ K^T   [B*H, S, S]
  // ========================================
  // Each "batch" is one attention head
  // Q: [B*H, S, D], K: [B*H, S, D] -> Scores: [B*H, S, S]

  long long stride_q = seq_len * head_dim;
  long long stride_k = seq_len * head_dim;
  long long stride_scores = seq_len * seq_len;

  // cuBLAS is column-major, for row-major we compute: K @ Q^T = (Q @ K^T)^T
  // But we want row-major output, so we compute: Q @ K^T
  // For row-major storage: cublasSgemmStridedBatched with trans_b
  cublasSgemmStridedBatched(handle,
                            CUBLAS_OP_T, // K transposed
                            CUBLAS_OP_N, // Q not transposed
                            seq_len,  // m (rows of K^T = cols of K = seq_len)
                            seq_len,  // n (rows of Q = seq_len)
                            head_dim, // k (common dimension)
                            &alpha, K, head_dim, stride_k, // K: [S, D] each
                            Q, head_dim, stride_q,         // Q: [S, D] each
                            &beta, workspace, seq_len,
                            stride_scores, // Scores: [S, S] each
                            batch_heads);

  // ========================================
  // Step 2: Softmax(Scores / sqrt(d))
  // ========================================
  dim3 softmax_block(32, 8);
  dim3 softmax_grid((batch_heads * seq_len + 7) / 8);
  softmax_kernel<<<softmax_grid, softmax_block>>>(
      workspace, batch_heads * seq_len, seq_len, scale);

  // ========================================
  // Step 3: Output = Attn @ V  [B*H, S, D]
  // ========================================
  long long stride_v = seq_len * head_dim;
  long long stride_o = seq_len * head_dim;

  cublasSgemmStridedBatched(handle,
                            CUBLAS_OP_N, // V not transposed
                            CUBLAS_OP_N, // Attn not transposed
                            head_dim,    // m (cols of V = head_dim)
                            seq_len,     // n (rows of Attn = seq_len)
                            seq_len,     // k (common dimension = seq_len)
                            &alpha, V, head_dim, stride_v, // V: [S, D] each
                            workspace, seq_len,
                            stride_scores,                // Attn: [S, S] each
                            &beta, O, head_dim, stride_o, // O: [S, D] each
                            batch_heads);

  cudaDeviceSynchronize();
}

// Wrapper with standard interface (allocates workspace)
void cublas_attention_forward_alloc(const float *Q, // [batch, heads, seq, dim]
                                    const float *K, const float *V, float *O,
                                    int batch_size, int num_heads, int seq_len,
                                    int head_dim) {
  int batch_heads = batch_size * num_heads;
  size_t workspace_size = batch_heads * seq_len * seq_len * sizeof(float);

  float *workspace;
  cudaMalloc(&workspace, workspace_size);

  cublas_attention_forward(Q, K, V, O, workspace, batch_heads, seq_len,
                           head_dim);

  cudaFree(workspace);
}

} // namespace cublas_attention
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_CUBLAS_ATTENTION_CU
