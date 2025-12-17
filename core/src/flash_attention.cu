// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FlashAttention CUDA Kernel for Zenith Framework
// Memory-efficient attention using tiling and online softmax
// Based on FlashAttention paper: https://arxiv.org/abs/2205.14135

#ifndef ZENITH_FLASH_ATTENTION_CU
#define ZENITH_FLASH_ATTENTION_CU

#ifdef ZENITH_HAS_CUDA

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zenith {
namespace flash_attention {

// ============================================================================
// Configuration Constants
// ============================================================================

// Block sizes for tiling - optimized for T4/A100 SRAM
constexpr int BLOCK_M = 64; // Query block size
constexpr int BLOCK_N = 64; // Key/Value block size
constexpr int BLOCK_D = 64; // Head dimension block

// Thread configuration
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_WARP = 32;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * THREADS_PER_WARP;

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
}

// ============================================================================
// FlashAttention Forward Kernel
// ============================================================================

// Simplified FlashAttention for single head
// Q, K, V: [seq_len, head_dim]
// Output: [seq_len, head_dim]
__global__ void flash_attention_forward_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O,
    float *__restrict__ L, // logsumexp for backward
    float *__restrict__ M, // max for backward
    const int seq_len, const int head_dim, const float scale) {
  // Shared memory for Q, K, V tiles
  extern __shared__ float smem[];

  float *Q_tile = smem;                        // [BLOCK_M, head_dim]
  float *K_tile = smem + BLOCK_M * head_dim;   // [BLOCK_N, head_dim]
  float *V_tile = K_tile + BLOCK_N * head_dim; // [BLOCK_N, head_dim]
  float *S_tile = V_tile + BLOCK_N * head_dim; // [BLOCK_M, BLOCK_N]

  const int tid = threadIdx.x;
  const int q_block_idx = blockIdx.x;
  const int q_start = q_block_idx * BLOCK_M;

  // Initialize output accumulator and statistics per query
  float O_acc[BLOCK_M] = {0.0f};
  float m_i = -INFINITY; // Max score seen so far
  float l_i = 0.0f;      // Sum of exp(scores - max) seen so far

  // Load Q tile to shared memory
  for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
    int row = i / head_dim;
    int col = i % head_dim;
    int global_row = q_start + row;
    if (global_row < seq_len) {
      Q_tile[i] = Q[global_row * head_dim + col] * scale;
    } else {
      Q_tile[i] = 0.0f;
    }
  }
  __syncthreads();

  // Iterate over K, V blocks
  for (int kv_block_idx = 0; kv_block_idx < (seq_len + BLOCK_N - 1) / BLOCK_N;
       kv_block_idx++) {
    int kv_start = kv_block_idx * BLOCK_N;

    // Load K, V tiles to shared memory
    for (int i = tid; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
      int row = i / head_dim;
      int col = i % head_dim;
      int global_row = kv_start + row;
      if (global_row < seq_len) {
        K_tile[i] = K[global_row * head_dim + col];
        V_tile[i] = V[global_row * head_dim + col];
      } else {
        K_tile[i] = 0.0f;
        V_tile[i] = 0.0f;
      }
    }
    __syncthreads();

    // Compute S = Q @ K^T for this block
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS_PER_BLOCK) {
      int q_idx = i / BLOCK_N;
      int k_idx = i % BLOCK_N;

      float sum = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        sum += Q_tile[q_idx * head_dim + d] * K_tile[k_idx * head_dim + d];
      }
      S_tile[i] = sum;
    }
    __syncthreads();

    // Online softmax update
    // For each query position, update m_i, l_i, and O_acc
    for (int q_idx = tid; q_idx < BLOCK_M; q_idx += THREADS_PER_BLOCK) {
      int global_q = q_start + q_idx;
      if (global_q >= seq_len)
        continue;

      // Find max in this block
      float m_ij = -INFINITY;
      for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
        int global_k = kv_start + k_idx;
        if (global_k < seq_len) {
          m_ij = fmaxf(m_ij, S_tile[q_idx * BLOCK_N + k_idx]);
        }
      }

      // Update running max
      float m_new = fmaxf(m_i, m_ij);

      // Compute P = exp(S - m_new) and sum
      float l_ij = 0.0f;
      for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
        int global_k = kv_start + k_idx;
        if (global_k < seq_len) {
          float p = expf(S_tile[q_idx * BLOCK_N + k_idx] - m_new);
          l_ij += p;

          // Accumulate O = P @ V
          for (int d = 0; d < head_dim; d++) {
            O_acc[q_idx * head_dim + d] += p * V_tile[k_idx * head_dim + d];
          }
        }
      }

      // Rescale previous O_acc
      float alpha = expf(m_i - m_new);
      for (int d = 0; d < head_dim; d++) {
        O_acc[q_idx * head_dim + d] *= alpha;
      }

      // Update statistics
      l_i = alpha * l_i + l_ij;
      m_i = m_new;
    }
    __syncthreads();
  }

  // Normalize output and write to global memory
  for (int q_idx = tid; q_idx < BLOCK_M; q_idx += THREADS_PER_BLOCK) {
    int global_q = q_start + q_idx;
    if (global_q < seq_len) {
      for (int d = 0; d < head_dim; d++) {
        O[global_q * head_dim + d] = O_acc[q_idx * head_dim + d] / l_i;
      }
      // Store statistics for backward pass
      if (L != nullptr)
        L[global_q] = logf(l_i) + m_i;
      if (M != nullptr)
        M[global_q] = m_i;
    }
  }
}

// ============================================================================
// Multi-Head FlashAttention Kernel
// ============================================================================

// Q, K, V: [batch, num_heads, seq_len, head_dim]
// Output: [batch, num_heads, seq_len, head_dim]
__global__ void flash_attention_multihead_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O, const int batch_size,
    const int num_heads, const int seq_len, const int head_dim,
    const float scale) {
  // Block handles one (batch, head, query_block) combination
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_block_idx = blockIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads)
    return;

  // Offset to this head's data
  const int head_offset =
      (batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const float *Q_head = Q + head_offset;
  const float *K_head = K + head_offset;
  const float *V_head = V + head_offset;
  float *O_head = O + head_offset;

  // Shared memory
  extern __shared__ float smem[];
  float *Q_tile = smem;
  float *K_tile = smem + BLOCK_M * head_dim;
  float *V_tile = K_tile + BLOCK_N * head_dim;

  const int tid = threadIdx.x;
  const int q_start = q_block_idx * BLOCK_M;

  // Per-thread accumulators
  float O_acc[8] = {0.0f}; // Assuming head_dim <= 8 for simplicity
  float m_i = -INFINITY;
  float l_i = 0.0f;

  // Load Q tile
  for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
    int row = i / head_dim;
    int col = i % head_dim;
    int global_row = q_start + row;
    if (global_row < seq_len) {
      Q_tile[i] = Q_head[global_row * head_dim + col] * scale;
    } else {
      Q_tile[i] = 0.0f;
    }
  }
  __syncthreads();

  // Iterate over K, V blocks
  for (int kv_block = 0; kv_block < (seq_len + BLOCK_N - 1) / BLOCK_N;
       kv_block++) {
    int kv_start = kv_block * BLOCK_N;

    // Load K, V
    for (int i = tid; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
      int row = i / head_dim;
      int col = i % head_dim;
      int global_row = kv_start + row;
      if (global_row < seq_len) {
        K_tile[i] = K_head[global_row * head_dim + col];
        V_tile[i] = V_head[global_row * head_dim + col];
      } else {
        K_tile[i] = 0.0f;
        V_tile[i] = 0.0f;
      }
    }
    __syncthreads();

    // Compute attention for assigned query positions
    int q_per_thread = (BLOCK_M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int qt = 0; qt < q_per_thread; qt++) {
      int q_local = tid * q_per_thread + qt;
      if (q_local >= BLOCK_M)
        continue;

      int global_q = q_start + q_local;
      if (global_q >= seq_len)
        continue;

      // Compute scores for this query
      float scores[BLOCK_N];
      float max_score = -INFINITY;

      for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
        int global_k = kv_start + k_idx;
        if (global_k >= seq_len) {
          scores[k_idx] = -INFINITY;
          continue;
        }

        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          sum += Q_tile[q_local * head_dim + d] * K_tile[k_idx * head_dim + d];
        }
        scores[k_idx] = sum;
        max_score = fmaxf(max_score, sum);
      }

      // Online softmax
      float m_new = fmaxf(m_i, max_score);
      float l_new = 0.0f;

      // Rescale previous accumulator
      float alpha = expf(m_i - m_new);
      for (int d = 0; d < head_dim && d < 8; d++) {
        O_acc[d] *= alpha;
      }
      l_i *= alpha;

      // Add new contributions
      for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
        int global_k = kv_start + k_idx;
        if (global_k >= seq_len)
          continue;

        float p = expf(scores[k_idx] - m_new);
        l_new += p;

        for (int d = 0; d < head_dim && d < 8; d++) {
          O_acc[d] += p * V_tile[k_idx * head_dim + d];
        }
      }

      l_i += l_new;
      m_i = m_new;
    }
    __syncthreads();
  }

  // Write output
  int q_per_thread = (BLOCK_M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  for (int qt = 0; qt < q_per_thread; qt++) {
    int q_local = tid * q_per_thread + qt;
    if (q_local >= BLOCK_M)
      continue;

    int global_q = q_start + q_local;
    if (global_q >= seq_len)
      continue;

    for (int d = 0; d < head_dim && d < 8; d++) {
      O_head[global_q * head_dim + d] = O_acc[d] / l_i;
    }
  }
}

// ============================================================================
// Wrapper Function
// ============================================================================

void flash_attention_forward(const float *Q, const float *K, const float *V,
                             float *O, int batch_size, int num_heads,
                             int seq_len, int head_dim) {
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // Grid: (num_query_blocks, num_heads, batch_size)
  int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
  dim3 grid(num_q_blocks, num_heads, batch_size);
  dim3 block(THREADS_PER_BLOCK);

  // Shared memory size
  size_t smem_size = (BLOCK_M + 2 * BLOCK_N) * head_dim * sizeof(float);

  flash_attention_multihead_kernel<<<grid, block, smem_size>>>(
      Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale);
}

} // namespace flash_attention
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FLASH_ATTENTION_CU
