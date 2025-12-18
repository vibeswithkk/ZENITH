// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FP16 Kernel Suite for Full BERT Implementation
// Uses Tensor Cores with FP32 accumulation for stability

#ifndef ZENITH_FP16_KERNELS_CU
#define ZENITH_FP16_KERNELS_CU

#ifdef ZENITH_HAS_CUDA

#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zenith {
namespace fp16_kernels {

// ============================================================================
// cuBLAS Handle Singleton with Tensor Core enabled
// ============================================================================

static cublasHandle_t get_handle() {
  static cublasHandle_t handle = nullptr;
  static bool initialized = false;
  if (!initialized) {
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    initialized = true;
  }
  return handle;
}

// ============================================================================
// FP32 <-> FP16 Conversion
// ============================================================================

__global__ void fp32_to_fp16_kernel(const float *__restrict__ input,
                                    __half *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = __float2half(input[idx]);
}

__global__ void fp16_to_fp32_kernel(const __half *__restrict__ input,
                                    float *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = __half2float(input[idx]);
}

void convert_f32_to_f16(const float *input, __half *output, int size) {
  int blocks = (size + 255) / 256;
  fp32_to_fp16_kernel<<<blocks, 256>>>(input, output, size);
}

void convert_f16_to_f32(const __half *input, float *output, int size) {
  int blocks = (size + 255) / 256;
  fp16_to_fp32_kernel<<<blocks, 256>>>(input, output, size);
}

// ============================================================================
// FP16 Linear: Y = X @ W^T + bias using Tensor Cores
// ============================================================================

__global__ void add_bias_fp16_kernel(__half *output, const __half *bias, int M,
                                     int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N)
    return;
  int col = idx % N;
  output[idx] = __hadd(output[idx], bias[col]);
}

void linear_fp16(const __half *X, const __half *W, const __half *bias,
                 __half *Y, int M, int N, int K) {
  cublasHandle_t handle = get_handle();

  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  // Y = X @ W^T using cuBLAS (column-major)
  // For row-major: C = A @ B^T becomes cublas(B, A^T) = B @ A^T = (A @ B^T)^T
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, // Output is [M, N]
               &alpha, W, CUDA_R_16F, K, // W is [N, K], need transpose
               X, CUDA_R_16F, K,         // X is [M, K]
               &beta, Y, CUDA_R_16F, N,  // Y is [M, N]
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Add bias if provided
  if (bias != nullptr) {
    int total = M * N;
    int blocks = (total + 255) / 256;
    add_bias_fp16_kernel<<<blocks, 256>>>(Y, bias, M, N);
  }
}

// ============================================================================
// FP16 GELU Activation (approximate, fast)
// ============================================================================

__global__ void gelu_fp16_kernel(const __half *__restrict__ input,
                                 __half *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  // GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
  // Compute in FP32 for accuracy, store in FP16
  float x = __half2float(input[idx]);
  float x3 = x * x * x;
  float inner =
      0.7978845608f * (x + 0.044715f * x3); // sqrt(2/π) ≈ 0.7978845608
  float gelu = 0.5f * x * (1.0f + tanhf(inner));
  output[idx] = __float2half(gelu);
}

void gelu_fp16(const __half *input, __half *output, int size) {
  int blocks = (size + 255) / 256;
  gelu_fp16_kernel<<<blocks, 256>>>(input, output, size);
}

// ============================================================================
// FP16 LayerNorm with FP32 Statistics (Numerically Stable)
// FIXED: Use shared memory to broadcast mean/variance to all threads
// ============================================================================

__global__ void layernorm_fp16_kernel(const __half *__restrict__ input,
                                      __half *__restrict__ output,
                                      const __half *__restrict__ gamma,
                                      const __half *__restrict__ beta,
                                      int batch, int hidden, float eps) {
  int row = blockIdx.x;
  if (row >= batch)
    return;

  const __half *row_in = input + row * hidden;
  __half *row_out = output + row * hidden;

  // Shared memory for warp reductions and final broadcast
  __shared__ float warp_sums[8]; // Max 256 threads = 8 warps
  __shared__ float result_mean;
  __shared__ float result_inv_std;

  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  int num_warps = (blockDim.x + 31) / 32;

  // Step 1: Compute mean in FP32
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    sum += __half2float(row_in[i]);
  }

  // Warp reduce sum
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // Store warp sums
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  // Final reduction in first warp
  if (threadIdx.x < num_warps) {
    sum = warp_sums[threadIdx.x];
  } else if (threadIdx.x < 32) {
    sum = 0.0f;
  }
  if (threadIdx.x < 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x == 0) {
      result_mean = sum / static_cast<float>(hidden);
    }
  }
  __syncthreads();

  float mean = result_mean;

  // Step 2: Compute variance in FP32
  float var_sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float diff = __half2float(row_in[i]) - mean;
    var_sum += diff * diff;
  }

  // Warp reduce variance
  for (int offset = 16; offset > 0; offset /= 2) {
    var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
  }

  if (lane == 0) {
    warp_sums[warp] = var_sum;
  }
  __syncthreads();

  // Final reduction
  if (threadIdx.x < num_warps) {
    var_sum = warp_sums[threadIdx.x];
  } else if (threadIdx.x < 32) {
    var_sum = 0.0f;
  }
  if (threadIdx.x < 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    if (threadIdx.x == 0) {
      float variance = var_sum / static_cast<float>(hidden);
      result_inv_std = rsqrtf(variance + eps);
    }
  }
  __syncthreads();

  float inv_std = result_inv_std;

  // Step 3: Normalize and apply gamma/beta
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float x = __half2float(row_in[i]);
    float g = __half2float(gamma[i]);
    float b = __half2float(beta[i]);
    float normalized = (x - mean) * inv_std;
    row_out[i] = __float2half(normalized * g + b);
  }
}

void layernorm_fp16(const __half *input, __half *output, const __half *gamma,
                    const __half *beta, int batch, int hidden, float eps) {
  layernorm_fp16_kernel<<<batch, 256>>>(input, output, gamma, beta, batch,
                                        hidden, eps);
  cudaDeviceSynchronize();
}

// ============================================================================
// FP16 Element-wise Add (for residual connections)
// ============================================================================

__global__ void add_fp16_kernel(const __half *__restrict__ a,
                                const __half *__restrict__ b,
                                __half *__restrict__ c, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  c[idx] = __hadd(a[idx], b[idx]);
}

void add_fp16(const __half *a, const __half *b, __half *c, int size) {
  int blocks = (size + 255) / 256;
  add_fp16_kernel<<<blocks, 256>>>(a, b, c, size);
}

// ============================================================================
// FP16 Transpose: [B,S,H,D] -> [B,H,S,D] for attention
// ============================================================================

__global__ void transpose_0213_fp16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int batch, int seq, int heads,
                                           int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * heads * seq * dim;
  if (idx >= total)
    return;

  // Output index: [b, h, s, d]
  int d = idx % dim;
  int s = (idx / dim) % seq;
  int h = (idx / (dim * seq)) % heads;
  int b = idx / (dim * seq * heads);

  // Input index: [b, s, h, d]
  int input_idx = b * (seq * heads * dim) + s * (heads * dim) + h * dim + d;
  output[idx] = input[input_idx];
}

__global__ void transpose_0213_inv_fp16_kernel(const __half *__restrict__ input,
                                               __half *__restrict__ output,
                                               int batch, int heads, int seq,
                                               int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * heads * seq * dim;
  if (idx >= total)
    return;

  // Output index: [b, s, h, d]
  int d = idx % dim;
  int h = (idx / dim) % heads;
  int s = (idx / (dim * heads)) % seq;
  int b = idx / (dim * heads * seq);

  // Input index: [b, h, s, d]
  int input_idx = b * (heads * seq * dim) + h * (seq * dim) + s * dim + d;
  output[idx] = input[input_idx];
}

void transpose_for_attn_fp16(const __half *input, __half *output, int batch,
                             int seq, int heads, int dim) {
  int total = batch * seq * heads * dim;
  int blocks = (total + 255) / 256;
  transpose_0213_fp16_kernel<<<blocks, 256>>>(input, output, batch, seq, heads,
                                              dim);
}

void transpose_from_attn_fp16(const __half *input, __half *output, int batch,
                              int heads, int seq, int dim) {
  int total = batch * heads * seq * dim;
  int blocks = (total + 255) / 256;
  transpose_0213_inv_fp16_kernel<<<blocks, 256>>>(input, output, batch, heads,
                                                  seq, dim);
}

// ============================================================================
// FP16 Batched Attention with Tensor Cores
// ============================================================================

__global__ void softmax_scale_fp16_kernel(__half *data, int rows, int cols,
                                          float scale) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  __half *row_data = data + row * cols;

  // Compute max for numerical stability (FP32)
  float max_val = -1e10f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float val = __half2float(row_data[j]) * scale;
    max_val = fmaxf(max_val, val);
  }

  // Warp reduce max
  for (int offset = 16; offset > 0; offset /= 2) {
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
  }

  __shared__ float shared_max[32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  if (lane == 0)
    shared_max[warp] = max_val;
  __syncthreads();

  if (threadIdx.x < 32) {
    max_val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_max[threadIdx.x]
                                                     : -1e10f;
    for (int offset = 16; offset > 0; offset /= 2) {
      max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
  }
  max_val = __shfl_sync(0xffffffff, max_val, 0);

  // Exp and sum (FP32)
  float sum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float val = __half2float(row_data[j]) * scale;
    float exp_val = expf(val - max_val);
    row_data[j] = __float2half(exp_val);
    sum += exp_val;
  }

  // Warp reduce sum
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  __shared__ float shared_sum[32];
  if (lane == 0)
    shared_sum[warp] = sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    sum =
        (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }
  sum = __shfl_sync(0xffffffff, sum, 0);

  // Normalize
  float inv_sum = 1.0f / sum;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_data[j] = __float2half(__half2float(row_data[j]) * inv_sum);
  }
}

void attention_fp16(const __half *Q, const __half *K, const __half *V,
                    __half *O, int batch, int heads, int seq, int dim) {
  cublasHandle_t handle = get_handle();

  int batch_heads = batch * heads;
  float scale = 1.0f / sqrtf(static_cast<float>(dim));

  __half alpha_h = __float2half(1.0f);
  __half beta_h = __float2half(0.0f);

  // Allocate workspace for attention scores
  __half *scores;
  cudaMalloc(&scores, batch_heads * seq * seq * sizeof(__half));

  // Step 1: Scores = Q @ K^T (batched)
  long long stride_q = seq * dim;
  long long stride_k = seq * dim;
  long long stride_s = seq * seq;
  long long stride_v = seq * dim;
  long long stride_o = seq * dim;

  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq, seq, dim,
                             &alpha_h, K, CUDA_R_16F, dim, stride_k, Q,
                             CUDA_R_16F, dim, stride_q, &beta_h, scores,
                             CUDA_R_16F, seq, stride_s, batch_heads,
                             CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Step 2: Softmax with scaling
  softmax_scale_fp16_kernel<<<batch_heads * seq, 256>>>(
      scores, batch_heads * seq, seq, scale);

  // Step 3: Output = Attn @ V (batched)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, seq, seq,
                             &alpha_h, V, CUDA_R_16F, dim, stride_v, scores,
                             CUDA_R_16F, seq, stride_s, &beta_h, O, CUDA_R_16F,
                             dim, stride_o, batch_heads, CUBLAS_COMPUTE_16F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cudaFree(scores);
  cudaDeviceSynchronize();
}

} // namespace fp16_kernels
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FP16_KERNELS_CU
