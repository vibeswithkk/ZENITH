// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FlashAttention CUDA Kernel - FP16 Optimized Version
// Uses half precision (FP16) for 2x memory bandwidth and Tensor Core potential

#ifndef ZENITH_FLASH_ATTENTION_CU
#define ZENITH_FLASH_ATTENTION_CU

#ifdef ZENITH_HAS_CUDA

#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zenith {
namespace flash_attention {

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int HEAD_DIM_MAX = 128;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;

// ============================================================================
// Warp-Level Reduction (FP32 for accuracy)
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
}

// ============================================================================
// FP16 Optimized Attention Kernel
// Uses __half for storage, FP32 for accumulation
// ============================================================================

__global__ void fp16_tiled_attention_kernel(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, half *__restrict__ O, const int batch_size,
    const int num_heads, const int seq_len, const int head_dim,
    const float scale) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads)
    return;

  const int tid = threadIdx.x;
  const int head_offset =
      (batch_idx * num_heads + head_idx) * seq_len * head_dim;

  const half *Q_head = Q + head_offset;
  const half *K_head = K + head_offset;
  const half *V_head = V + head_offset;
  half *O_head = O + head_offset;

  const int q_start = q_tile_idx * BLOCK_M;

  // Shared memory for K and V tiles (using half for 2x capacity)
  extern __shared__ char smem_raw[];
  half *K_tile = reinterpret_cast<half *>(smem_raw);
  half *V_tile = K_tile + BLOCK_N * head_dim;

  // Each thread handles one query position
  const int q_local = tid % BLOCK_M;
  const int global_q = q_start + q_local;
  bool valid_q = (global_q < seq_len) && (q_local < BLOCK_M);

  // Load query into registers (FP32 for computation, scaled)
  float q_vec[HEAD_DIM_MAX];
  if (valid_q) {
    for (int d = 0; d < head_dim; d++) {
      q_vec[d] = __half2float(Q_head[global_q * head_dim + d]) * scale;
    }
  }

  // Accumulators (FP32 for precision)
  float output[HEAD_DIM_MAX];
  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  for (int d = 0; d < head_dim; d++) {
    output[d] = 0.0f;
  }

  const int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;

    // Cooperative loading of K and V tiles (half precision)
    for (int i = tid; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
      const int row = i / head_dim;
      const int col = i % head_dim;
      const int global_kv = kv_start + row;

      if (global_kv < seq_len) {
        K_tile[row * head_dim + col] = K_head[global_kv * head_dim + col];
        V_tile[row * head_dim + col] = V_head[global_kv * head_dim + col];
      } else {
        K_tile[row * head_dim + col] = __float2half(0.0f);
        V_tile[row * head_dim + col] = __float2half(0.0f);
      }
    }
    __syncthreads();

    if (!valid_q) {
      __syncthreads();
      continue;
    }

    // Compute attention scores
    float tile_scores[BLOCK_N];
    float tile_max = -FLT_MAX;

#pragma unroll 4
    for (int k = 0; k < BLOCK_N; k++) {
      const int global_k = kv_start + k;

      if (global_k < seq_len) {
        float score = 0.0f;
#pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
          score += q_vec[d] * __half2float(K_tile[k * head_dim + d]);
        }
        tile_scores[k] = score;
        tile_max = fmaxf(tile_max, score);
      } else {
        tile_scores[k] = -FLT_MAX;
      }
    }

    // Online softmax update
    float m_new = fmaxf(m_i, tile_max);
    float correction = expf(m_i - m_new);

    for (int d = 0; d < head_dim; d++) {
      output[d] *= correction;
    }
    l_i *= correction;

    float tile_sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < BLOCK_N; k++) {
      const int global_k = kv_start + k;
      if (global_k < seq_len) {
        float p = expf(tile_scores[k] - m_new);
        tile_sum += p;

#pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
          output[d] += p * __half2float(V_tile[k * head_dim + d]);
        }
      }
    }

    l_i += tile_sum;
    m_i = m_new;

    __syncthreads();
  }

  // Write output (convert back to half)
  if (valid_q && l_i > 0.0f) {
    for (int d = 0; d < head_dim; d++) {
      O_head[global_q * head_dim + d] = __float2half(output[d] / l_i);
    }
  }
}

// ============================================================================
// FP32 Version (for compatibility)
// ============================================================================

__global__ void tiled_attention_kernel(const float *__restrict__ Q,
                                       const float *__restrict__ K,
                                       const float *__restrict__ V,
                                       float *__restrict__ O,
                                       const int batch_size,
                                       const int num_heads, const int seq_len,
                                       const int head_dim, const float scale) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads)
    return;

  const int tid = threadIdx.x;
  const int head_offset =
      (batch_idx * num_heads + head_idx) * seq_len * head_dim;

  const float *Q_head = Q + head_offset;
  const float *K_head = K + head_offset;
  const float *V_head = V + head_offset;
  float *O_head = O + head_offset;

  const int q_start = q_tile_idx * BLOCK_M;

  extern __shared__ float smem[];
  float *K_tile = smem;
  float *V_tile = smem + BLOCK_N * head_dim;

  const int q_local = tid % BLOCK_M;
  const int global_q = q_start + q_local;
  bool valid_q = (global_q < seq_len) && (q_local < BLOCK_M);

  float q_vec[HEAD_DIM_MAX];
  if (valid_q) {
    for (int d = 0; d < head_dim; d++) {
      q_vec[d] = Q_head[global_q * head_dim + d] * scale;
    }
  }

  float output[HEAD_DIM_MAX];
  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  for (int d = 0; d < head_dim; d++) {
    output[d] = 0.0f;
  }

  const int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;

    for (int i = tid; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
      const int row = i / head_dim;
      const int col = i % head_dim;
      const int global_kv = kv_start + row;

      if (global_kv < seq_len) {
        K_tile[row * head_dim + col] = K_head[global_kv * head_dim + col];
        V_tile[row * head_dim + col] = V_head[global_kv * head_dim + col];
      } else {
        K_tile[row * head_dim + col] = 0.0f;
        V_tile[row * head_dim + col] = 0.0f;
      }
    }
    __syncthreads();

    if (!valid_q) {
      __syncthreads();
      continue;
    }

    float tile_scores[BLOCK_N];
    float tile_max = -FLT_MAX;

#pragma unroll 4
    for (int k = 0; k < BLOCK_N; k++) {
      const int global_k = kv_start + k;

      if (global_k < seq_len) {
        float score = 0.0f;
#pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
          score += q_vec[d] * K_tile[k * head_dim + d];
        }
        tile_scores[k] = score;
        tile_max = fmaxf(tile_max, score);
      } else {
        tile_scores[k] = -FLT_MAX;
      }
    }

    float m_new = fmaxf(m_i, tile_max);
    float correction = expf(m_i - m_new);

    for (int d = 0; d < head_dim; d++) {
      output[d] *= correction;
    }
    l_i *= correction;

    float tile_sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < BLOCK_N; k++) {
      const int global_k = kv_start + k;
      if (global_k < seq_len) {
        float p = expf(tile_scores[k] - m_new);
        tile_sum += p;

#pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
          output[d] += p * V_tile[k * head_dim + d];
        }
      }
    }

    l_i += tile_sum;
    m_i = m_new;

    __syncthreads();
  }

  if (valid_q && l_i > 0.0f) {
    for (int d = 0; d < head_dim; d++) {
      O_head[global_q * head_dim + d] = output[d] / l_i;
    }
  }
}

// ============================================================================
// Wrapper Functions
// ============================================================================

void flash_attention_forward(const float *Q, const float *K, const float *V,
                             float *O, int batch_size, int num_heads,
                             int seq_len, int head_dim) {
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  int num_q_tiles = (seq_len + BLOCK_M - 1) / BLOCK_M;
  dim3 grid(num_q_tiles, num_heads, batch_size);
  dim3 block(THREADS_PER_BLOCK);

  size_t smem_size = 2 * BLOCK_N * head_dim * sizeof(float);

  tiled_attention_kernel<<<grid, block, smem_size>>>(
      Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale);
}

void flash_attention_forward_fp16(const half *Q, const half *K, const half *V,
                                  half *O, int batch_size, int num_heads,
                                  int seq_len, int head_dim) {
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  int num_q_tiles = (seq_len + BLOCK_M - 1) / BLOCK_M;
  dim3 grid(num_q_tiles, num_heads, batch_size);
  dim3 block(THREADS_PER_BLOCK);

  size_t smem_size = 2 * BLOCK_N * head_dim * sizeof(half);

  fp16_tiled_attention_kernel<<<grid, block, smem_size>>>(
      Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale);
}

} // namespace flash_attention
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FLASH_ATTENTION_CU
