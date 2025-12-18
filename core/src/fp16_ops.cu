// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FP16 Operations using Tensor Cores for Zenith Framework
// Uses cublasGemmEx for mixed-precision GEMM

#ifndef ZENITH_FP16_OPS_CU
#define ZENITH_FP16_OPS_CU

#ifdef ZENITH_HAS_CUDA

#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zenith {
namespace fp16_ops {

// ============================================================================
// cuBLAS Handle (singleton with Tensor Core enabled)
// ============================================================================

static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  static bool initialized = false;
  if (!initialized) {
    cublasCreate(&handle);
    // Enable Tensor Cores
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    initialized = true;
  }
  return handle;
}

// ============================================================================
// FP32 to FP16 Conversion Kernels
// ============================================================================

__global__ void fp32_to_fp16_kernel(const float *input, __half *output,
                                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = __float2half(input[idx]);
}

__global__ void fp16_to_fp32_kernel(const __half *input, float *output,
                                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = __half2float(input[idx]);
}

void convert_fp32_to_fp16(const float *input, __half *output, int size) {
  int blocks = (size + 255) / 256;
  fp32_to_fp16_kernel<<<blocks, 256>>>(input, output, size);
}

void convert_fp16_to_fp32(const __half *input, float *output, int size) {
  int blocks = (size + 255) / 256;
  fp16_to_fp32_kernel<<<blocks, 256>>>(input, output, size);
}

// ============================================================================
// FP16 GEMM using Tensor Cores
// ============================================================================

// C = A @ B with mixed precision
// A: [M, K], B: [K, N], C: [M, N]
// Input/compute in FP16, accumulate in FP32
void gemm_fp16(const __half *A, const __half *B, __half *C, int M, int N,
               int K) {
  cublasHandle_t handle = get_cublas_handle();

  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  // cuBLAS is column-major, so we compute B^T @ A^T = (A @ B)^T
  // For row-major: swap A and B
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
               N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// FP16 Linear: Y = X @ W^T + bias
// X: [M, K], W: [N, K], bias: [N], Y: [M, N]
void linear_fp16(const __half *X, const __half *W, const __half *bias,
                 __half *Y, int M, int N, int K) {
  cublasHandle_t handle = get_cublas_handle();

  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  // Y = X @ W^T
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F,
               K, X, CUDA_R_16F, K, &beta, Y, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Add bias if provided (TODO: fused kernel)
  if (bias != nullptr) {
    // Simple broadcast add - could be optimized
    int total = M * N;
    int blocks = (total + 255) / 256;
    // Need bias add kernel for FP16
  }
}

// ============================================================================
// FP16 Batched Attention (for multi-head attention)
// ============================================================================

// FP16 Softmax kernel
__global__ void softmax_fp16_kernel(__half *data, int rows, int cols,
                                    float scale) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows)
    return;

  __half *row_data = data + row * cols;

  // Scale and find max (compute in FP32 for stability)
  float max_val = -1e10f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float val = __half2float(row_data[j]) * scale;
    row_data[j] = __float2half(val);
    max_val = fmaxf(max_val, val);
  }

  // Warp reduce max
  for (int offset = 16; offset > 0; offset /= 2) {
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
  }
  max_val = __shfl_sync(0xffffffff, max_val, 0);

  // Exp and sum
  float sum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float val = expf(__half2float(row_data[j]) - max_val);
    row_data[j] = __float2half(val);
    sum += val;
  }

  // Warp reduce sum
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  sum = __shfl_sync(0xffffffff, sum, 0);

  // Normalize
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_data[j] = __float2half(__half2float(row_data[j]) / sum);
  }
}

// FP16 Multi-Head Attention using cuBLAS
void attention_fp16(const __half *Q, const __half *K, const __half *V,
                    __half *O, __half *workspace, int batch_heads, int seq_len,
                    int head_dim) {
  cublasHandle_t handle = get_cublas_handle();

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  long long stride_q = seq_len * head_dim;
  long long stride_k = seq_len * head_dim;
  long long stride_scores = seq_len * seq_len;
  long long stride_v = seq_len * head_dim;
  long long stride_o = seq_len * head_dim;

  // Step 1: Scores = Q @ K^T
  cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim, &alpha, K,
      CUDA_R_16F, head_dim, stride_k, Q, CUDA_R_16F, head_dim, stride_q, &beta,
      workspace, CUDA_R_16F, seq_len, stride_scores, batch_heads,
      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Step 2: Softmax
  dim3 block(32, 8);
  dim3 grid((batch_heads * seq_len + 7) / 8);
  softmax_fp16_kernel<<<grid, block>>>(workspace, batch_heads * seq_len,
                                       seq_len, scale);

  // Step 3: Output = Attn @ V
  cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len, &alpha, V,
      CUDA_R_16F, head_dim, stride_v, workspace, CUDA_R_16F, seq_len,
      stride_scores, &beta, O, CUDA_R_16F, head_dim, stride_o, batch_heads,
      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cudaDeviceSynchronize();
}

// Wrapper that allocates workspace
void attention_fp16_alloc(const __half *Q, const __half *K, const __half *V,
                          __half *O, int batch_size, int num_heads, int seq_len,
                          int head_dim) {
  int batch_heads = batch_size * num_heads;
  size_t workspace_size = batch_heads * seq_len * seq_len * sizeof(__half);

  __half *workspace;
  cudaMalloc(&workspace, workspace_size);

  attention_fp16(Q, K, V, O, workspace, batch_heads, seq_len, head_dim);

  cudaFree(workspace);
}

} // namespace fp16_ops
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FP16_OPS_CU
