// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Flash Attention V2 Header
// Memory-efficient attention with O(N) memory instead of O(N^2)

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zenith {
namespace flash_attention {

/// Flash Attention V2 Configuration
struct FlashAttentionConfig {
  int batch_size;
  int num_heads;
  int seq_len;
  int head_dim;
  float scale;        // 1/sqrt(head_dim), computed automatically if 0
  bool causal;        // Use causal masking for autoregressive models
  float dropout_prob; // Dropout probability (0 = no dropout)

  FlashAttentionConfig()
      : batch_size(1), num_heads(12), seq_len(512), head_dim(64), scale(0.0f),
        causal(false), dropout_prob(0.0f) {}
};

/// Flash Attention V2 forward pass (FP32)
///
/// Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
/// Uses tiled computation to reduce memory from O(N^2) to O(N)
///
/// @param Q Query tensor [batch, num_heads, seq_len, head_dim]
/// @param K Key tensor [batch, num_heads, seq_len, head_dim]
/// @param V Value tensor [batch, num_heads, seq_len, head_dim]
/// @param O Output tensor [batch, num_heads, seq_len, head_dim]
/// @param config Attention configuration
void flash_attention_v2_forward(const float *Q, const float *K, const float *V,
                                float *O, const FlashAttentionConfig &config);

/// Flash Attention V2 forward pass (FP16)
/// Uses Tensor Cores for optimal performance on Ampere/Hopper GPUs
void flash_attention_v2_forward_fp16(const __half *Q, const __half *K,
                                     const __half *V, __half *O,
                                     const FlashAttentionConfig &config);

/// Flash Attention V2 backward pass (for training)
/// Computes gradients dQ, dK, dV from dO
void flash_attention_v2_backward(const float *Q, const float *K, const float *V,
                                 const float *O, const float *dO, float *dQ,
                                 float *dK, float *dV,
                                 const FlashAttentionConfig &config);

/// C-API wrappers for Python bindings
extern "C" {

void flash_attention_v2_forward_c(const float *Q, const float *K,
                                  const float *V, float *O, int batch_size,
                                  int num_heads, int seq_len, int head_dim,
                                  bool causal);

void flash_attention_v2_forward_fp16_c(const __half *Q, const __half *K,
                                       const __half *V, __half *O,
                                       int batch_size, int num_heads,
                                       int seq_len, int head_dim, bool causal);

} // extern "C"

// Block size constants (must match the .cu file)
constexpr int FLASH_BLOCK_M = 64; // Q block size
constexpr int FLASH_BLOCK_N = 64; // KV block size
constexpr int FLASH_BLOCK_D = 64; // Max head dimension

// Check if head_dim is supported
inline bool is_head_dim_supported(int head_dim) {
  return head_dim <= FLASH_BLOCK_D;
}

/// Estimate memory usage for Flash Attention V2
/// Returns bytes needed for workspace (0 for most cases)
inline size_t estimate_workspace_bytes(const FlashAttentionConfig &config) {
  // Flash Attention V2 doesn't need external workspace
  // All computation happens in shared memory
  return 0;
}

/// Estimate FLOPs for attention computation
inline double estimate_flops(const FlashAttentionConfig &config) {
  double seq = config.seq_len;
  double dim = config.head_dim;
  double bh = config.batch_size * config.num_heads;

  // Q @ K.T: 2 * seq * seq * dim FLOPs per head
  // Softmax: ~5 * seq * seq FLOPs per head
  // P @ V: 2 * seq * seq * dim FLOPs per head
  return bh * (4.0 * seq * seq * dim + 5.0 * seq * seq);
}

} // namespace flash_attention
} // namespace zenith
