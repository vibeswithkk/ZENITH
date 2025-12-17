// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Transformer CUDA Kernels for Zenith Framework
// Provides optimized CUDA kernels for Transformer operations not available
// in cuDNN: GELU, LayerNorm, etc.
//
// Reference: FlashAttention paper (Dao et al., 2022)
// Reference: NVIDIA FasterTransformer implementation

#ifndef ZENITH_TRANSFORMER_KERNELS_CU
#define ZENITH_TRANSFORMER_KERNELS_CU

#ifdef ZENITH_HAS_CUDA

#include <cmath>
#include <cuda_runtime.h>

namespace zenith {
namespace cuda_kernels {

// ============================================================================
// GELU Activation Kernel
// ============================================================================

// GELU approximation using tanh:
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// Constants: sqrt(2/π) ≈ 0.7978845608028654

__global__ void gelu_kernel(const float *__restrict__ input,
                            float *__restrict__ output, size_t size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const float x = input[idx];
    const float cdf =
        0.5f *
        (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
  }
}

/// Launch GELU kernel
/// @param input Input tensor, device pointer
/// @param output Output tensor, device pointer
/// @param size Number of elements
/// @param stream CUDA stream (0 for default)
inline void gelu_f32(const float *input, float *output, size_t size,
                     cudaStream_t stream = 0) {
  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  gelu_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, size);
}

// ============================================================================
// Layer Normalization Kernel
// ============================================================================

// Two-pass LayerNorm:
// 1. Compute mean and variance per row
// 2. Normalize: y = (x - mean) / sqrt(var + eps) * gamma + beta

// Warp reduce for sum
__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  }
  return val;
}

// Block reduce for sum (uses shared memory)
__inline__ __device__ float block_reduce_sum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (wid == 0) {
    val = warp_reduce_sum(val);
  }

  return val;
}

// LayerNorm kernel - each block processes one row
__global__ void layernorm_kernel(const float *__restrict__ input,
                                 float *__restrict__ output,
                                 const float *__restrict__ gamma,
                                 const float *__restrict__ beta, int batch_size,
                                 int hidden_size, float epsilon) {
  const int row = blockIdx.x;
  if (row >= batch_size)
    return;

  const float *row_input = input + row * hidden_size;
  float *row_output = output + row * hidden_size;

  // Compute mean
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    local_sum += row_input[i];
  }
  float mean = block_reduce_sum(local_sum) / hidden_size;

  // Compute variance
  float local_var_sum = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float diff = row_input[i] - mean;
    local_var_sum += diff * diff;
  }
  float var = block_reduce_sum(local_var_sum) / hidden_size;

  // Normalize
  float inv_std = rsqrtf(var + epsilon);
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float normalized = (row_input[i] - mean) * inv_std;
    row_output[i] = normalized * gamma[i] + beta[i];
  }
}

/// Launch LayerNorm kernel
/// @param input Input tensor [batch_size, hidden_size], device pointer
/// @param output Output tensor [batch_size, hidden_size], device pointer
/// @param gamma Scale parameter [hidden_size], device pointer
/// @param beta Bias parameter [hidden_size], device pointer
/// @param batch_size Number of rows
/// @param hidden_size Hidden dimension (columns per row)
/// @param epsilon Small value for numerical stability
/// @param stream CUDA stream
inline void layernorm_f32(const float *input, float *output, const float *gamma,
                          const float *beta, int batch_size, int hidden_size,
                          float epsilon, cudaStream_t stream = 0) {
  // One block per row, up to 1024 threads
  const int block_size = min(1024, hidden_size);
  layernorm_kernel<<<batch_size, block_size, 0, stream>>>(
      input, output, gamma, beta, batch_size, hidden_size, epsilon);
}

// ============================================================================
// Softmax Kernel (for attention)
// ============================================================================

// Softmax over the last dimension
__global__ void softmax_kernel(const float *__restrict__ input,
                               float *__restrict__ output, int batch_size,
                               int seq_len) {
  const int row = blockIdx.x;
  if (row >= batch_size)
    return;

  const float *row_input = input + row * seq_len;
  float *row_output = output + row * seq_len;

  // Find max for numerical stability
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
    local_max = fmaxf(local_max, row_input[i]);
  }

  // Reduce max across block
  __shared__ float shared_max[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local_max =
        fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask, 32));
  }

  if (lane == 0)
    shared_max[wid] = local_max;
  __syncthreads();

  float max_val = shared_max[0];
  for (int i = 1; i < blockDim.x / 32; ++i) {
    max_val = fmaxf(max_val, shared_max[i]);
  }

  // Compute exp and sum
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
    float exp_val = expf(row_input[i] - max_val);
    row_output[i] = exp_val;
    local_sum += exp_val;
  }

  float sum = block_reduce_sum(local_sum);

  // Normalize
  for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
    row_output[i] /= sum;
  }
}

/// Launch Softmax kernel
inline void softmax_f32(const float *input, float *output, int batch_size,
                        int seq_len, cudaStream_t stream = 0) {
  const int block_size = min(1024, seq_len);
  softmax_kernel<<<batch_size, block_size, 0, stream>>>(input, output,
                                                        batch_size, seq_len);
}

// ============================================================================
// Element-wise Add Kernel (for residual connections)
// ============================================================================

__global__ void add_kernel(const float *__restrict__ a,
                           const float *__restrict__ b,
                           float *__restrict__ output, size_t size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = a[idx] + b[idx];
  }
}

/// Launch element-wise add kernel
inline void add_f32(const float *a, const float *b, float *output, size_t size,
                    cudaStream_t stream = 0) {
  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  add_kernel<<<num_blocks, block_size, 0, stream>>>(a, b, output, size);
}

// ============================================================================
// Embedding Lookup Kernel
// ============================================================================

__global__ void embedding_kernel(const int *__restrict__ indices,
                                 const float *__restrict__ embedding_table,
                                 float *__restrict__ output, int batch_size,
                                 int seq_len, int hidden_size) {
  const int batch = blockIdx.x;
  const int pos = blockIdx.y;
  if (batch >= batch_size || pos >= seq_len)
    return;

  const int token_id = indices[batch * seq_len + pos];
  const float *src = embedding_table + token_id * hidden_size;
  float *dst = output + (batch * seq_len + pos) * hidden_size;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

/// Launch embedding lookup kernel
inline void embedding_f32(const int *indices, const float *embedding_table,
                          float *output, int batch_size, int seq_len,
                          int hidden_size, cudaStream_t stream = 0) {
  dim3 grid(batch_size, seq_len);
  const int block_size = min(256, hidden_size);
  embedding_kernel<<<grid, block_size, 0, stream>>>(
      indices, embedding_table, output, batch_size, seq_len, hidden_size);
}

} // namespace cuda_kernels
} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_TRANSFORMER_KERNELS_CU
