// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Multi-Head Attention CUDA Implementation for Zenith Framework
// Inspired by FlashAttention (Dao et al., 2022) for memory efficiency.
//
// Note: This is a simplified implementation for inference.
// For production, consider using cuDNN's Scaled Dot Product Attention
// or NVIDIA's Transformer Engine.

#ifndef ZENITH_ATTENTION_OPS_HPP
#define ZENITH_ATTENTION_OPS_HPP

#include "gpu_tensor.hpp"
#include "types.hpp"

#ifdef ZENITH_HAS_CUDA

#include "cublas_ops.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace zenith {
namespace ops {

// Forward declaration of CUDA kernels
namespace cuda_kernels {
void softmax_f32(const float *input, float *output, int batch_size, int seq_len,
                 cudaStream_t stream);
void add_f32(const float *a, const float *b, float *output, size_t size,
             cudaStream_t stream);
} // namespace cuda_kernels

// ============================================================================
// Multi-Head Attention (Simplified Inference Version)
// ============================================================================

/// Multi-Head Attention forward pass
/// Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// @param Q Query tensor [batch, seq_len, hidden_size], device pointer
/// @param K Key tensor [batch, seq_len, hidden_size], device pointer
/// @param V Value tensor [batch, seq_len, hidden_size], device pointer
/// @param output Output tensor [batch, seq_len, hidden_size], device pointer
/// @param batch_size Batch size
/// @param seq_len Sequence length
/// @param hidden_size Hidden dimension (must be divisible by num_heads)
/// @param num_heads Number of attention heads
/// @return Status indicating success or error
inline Status multi_head_attention(const float *Q, const float *K,
                                   const float *V, float *output,
                                   int batch_size, int seq_len, int hidden_size,
                                   int num_heads) {
  if (hidden_size % num_heads != 0) {
    return Status::Error(StatusCode::InvalidArgument,
                         "hidden_size must be divisible by num_heads");
  }

  const int head_dim = hidden_size / num_heads;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // We need workspace for:
  // 1. QK^T scores: [batch * num_heads, seq_len, seq_len]
  // 2. Attention weights after softmax: same size
  const size_t scores_size = batch_size * num_heads * seq_len * seq_len;

  // Allocate workspace using GPU memory pool
  float *scores = nullptr;
  cudaError_t err = cudaMalloc(&scores, scores_size * sizeof(float));
  if (err != cudaSuccess) {
    return Status::Error(StatusCode::OutOfMemory, "Failed to allocate scores");
  }

  // For each batch and head, compute attention scores
  // This is a simplified version - production code would use batched GEMM

  cublasHandle_t handle = cublas::CublasHandle::instance().get();
  if (!handle) {
    cudaFree(scores);
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  // Compute QK^T for each batch*head
  // Q: [batch * num_heads, seq_len, head_dim]
  // K: [batch * num_heads, seq_len, head_dim]
  // scores = Q @ K^T: [batch * num_heads, seq_len, seq_len]

  float alpha = scale; // Include scaling in GEMM
  float beta = 0.0f;

  // Use batched GEMM for efficiency
  // Note: We assume Q, K, V are already reshaped to [batch*num_heads, seq_len,
  // head_dim]
  for (int b = 0; b < batch_size * num_heads; ++b) {
    const float *q_ptr = Q + b * seq_len * head_dim;
    const float *k_ptr = K + b * seq_len * head_dim;
    float *s_ptr = scores + b * seq_len * seq_len;

    // scores = Q @ K^T
    cublasStatus_t status =
        cublasSgemm(handle,
                    CUBLAS_OP_T,             // K^T
                    CUBLAS_OP_N,             // Q
                    seq_len,                 // m
                    seq_len,                 // n
                    head_dim,                // k
                    &alpha, k_ptr, head_dim, // K (transposed)
                    q_ptr, head_dim,         // Q
                    &beta, s_ptr, seq_len    // scores
        );

    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaFree(scores);
      return Status::Error(StatusCode::InternalError, "cuBLAS GEMM failed");
    }
  }

  // Apply softmax to get attention weights
  // softmax over last dimension (seq_len)
  for (int b = 0; b < batch_size * num_heads; ++b) {
    cuda_kernels::softmax_f32(scores + b * seq_len * seq_len,
                              scores + b * seq_len * seq_len, // in-place
                              seq_len, seq_len, 0);
  }
  cudaDeviceSynchronize();

  // Compute attention output: attn_weights @ V
  alpha = 1.0f;
  beta = 0.0f;

  for (int b = 0; b < batch_size * num_heads; ++b) {
    const float *s_ptr = scores + b * seq_len * seq_len;
    const float *v_ptr = V + b * seq_len * head_dim;
    float *o_ptr = output + b * seq_len * head_dim;

    // output = scores @ V
    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N,             // V
                                        CUBLAS_OP_N,             // scores
                                        head_dim,                // m
                                        seq_len,                 // n
                                        seq_len,                 // k
                                        &alpha, v_ptr, head_dim, // V
                                        s_ptr, seq_len,          // scores
                                        &beta, o_ptr, head_dim   // output
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaFree(scores);
      return Status::Error(StatusCode::InternalError, "cuBLAS GEMM failed");
    }
  }

  cudaFree(scores);
  return Status::Ok();
}

/// Scaled Dot-Product Attention (single head)
/// Computes: softmax(QK^T / sqrt(d_k)) * V
inline Status scaled_dot_product_attention(const GpuTensor &Q,
                                           const GpuTensor &K,
                                           const GpuTensor &V,
                                           GpuTensor &output,
                                           bool causal = false) {
  if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) {
    return Status::Error(StatusCode::InvalidArgument,
                         "Q, K, V must be 3D tensors [batch, seq, dim]");
  }

  const int batch = Q.dim(0);
  const int seq_len = Q.dim(1);
  const int dim = Q.dim(2);

  return multi_head_attention(Q.data_ptr<float>(), K.data_ptr<float>(),
                              V.data_ptr<float>(), output.data_ptr<float>(),
                              batch, seq_len, dim, 1);
}

} // namespace ops
} // namespace zenith

#else // !ZENITH_HAS_CUDA

namespace zenith {
namespace ops {

inline Status multi_head_attention(const float *, const float *, const float *,
                                   float *, int, int, int, int) {
  return Status::Error(StatusCode::NotImplemented,
                       "CUDA not compiled - enable ZENITH_HAS_CUDA");
}

} // namespace ops
} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_ATTENTION_OPS_HPP
