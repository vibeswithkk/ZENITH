// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FlashAttention CUDA Kernel for Zenith Framework
// Simplified version for correctness first, then optimize

#ifndef ZENITH_FLASH_ATTENTION_CU
#define ZENITH_FLASH_ATTENTION_CU

#ifdef ZENITH_HAS_CUDA

#include <cmath>
#include <cuda_runtime.h>

namespace zenith {
namespace flash_attention {

// ============================================================================
// Simple Multi-Head Attention Kernel (Correct First, Then Optimize)
// ============================================================================

// Each thread handles one query position
// Q, K, V: [batch, num_heads, seq_len, head_dim]
__global__ void simple_attention_kernel(const float *__restrict__ Q,
                                        const float *__restrict__ K,
                                        const float *__restrict__ V,
                                        float *__restrict__ O,
                                        const int batch_size,
                                        const int num_heads, const int seq_len,
                                        const int head_dim, const float scale) {
  // Thread handles one (batch, head, query) position
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_queries = batch_size * num_heads * seq_len;

  if (idx >= total_queries)
    return;

  // Decode position
  const int q_pos = idx % seq_len;
  const int temp = idx / seq_len;
  const int head_idx = temp % num_heads;
  const int batch_idx = temp / num_heads;

  // Pointers to this head's data
  const int head_offset =
      (batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const float *Q_head = Q + head_offset;
  const float *K_head = K + head_offset;
  const float *V_head = V + head_offset;
  float *O_head = O + head_offset;

  // Get query vector
  const float *q_vec = Q_head + q_pos * head_dim;

  // Step 1: Compute attention scores and find max
  float max_score = -INFINITY;
  for (int k_pos = 0; k_pos < seq_len; k_pos++) {
    const float *k_vec = K_head + k_pos * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += q_vec[d] * k_vec[d];
    }
    score *= scale;
    max_score = fmaxf(max_score, score);
  }

  // Step 2: Compute softmax and weighted sum
  float sum_exp = 0.0f;

  // Temporary storage for output (use registers)
  float output[128]; // Max head_dim = 128
  for (int d = 0; d < head_dim; d++) {
    output[d] = 0.0f;
  }

  for (int k_pos = 0; k_pos < seq_len; k_pos++) {
    const float *k_vec = K_head + k_pos * head_dim;
    const float *v_vec = V_head + k_pos * head_dim;

    // Recompute score
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += q_vec[d] * k_vec[d];
    }
    score *= scale;

    // Softmax
    float p = expf(score - max_score);
    sum_exp += p;

    // Accumulate weighted values
    for (int d = 0; d < head_dim; d++) {
      output[d] += p * v_vec[d];
    }
  }

  // Normalize and write output
  float *o_vec = O_head + q_pos * head_dim;
  for (int d = 0; d < head_dim; d++) {
    o_vec[d] = output[d] / sum_exp;
  }
}

// ============================================================================
// Wrapper Function
// ============================================================================

void flash_attention_forward(const float *Q, const float *K, const float *V,
                             float *O, int batch_size, int num_heads,
                             int seq_len, int head_dim) {
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // One thread per query position
  int total_queries = batch_size * num_heads * seq_len;
  int threads_per_block = 256;
  int num_blocks = (total_queries + threads_per_block - 1) / threads_per_block;

  simple_attention_kernel<<<num_blocks, threads_per_block>>>(
      Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale);
}

} // namespace flash_attention
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FLASH_ATTENTION_CU
