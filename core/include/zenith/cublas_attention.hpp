// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// cuBLAS Attention Header

#ifndef ZENITH_CUBLAS_ATTENTION_HPP
#define ZENITH_CUBLAS_ATTENTION_HPP

#ifdef ZENITH_HAS_CUDA

namespace zenith {
namespace cublas_attention {

/// cuBLAS-based multi-head attention (high performance)
/// Uses cuBLAS GEMM + Tensor Cores for Q@K^T and attn@V
/// Q, K, V: [batch, num_heads, seq_len, head_dim]
/// O: output [batch, num_heads, seq_len, head_dim]
void cublas_attention_forward_alloc(const float *Q, const float *K,
                                    const float *V, float *O, int batch_size,
                                    int num_heads, int seq_len, int head_dim);

} // namespace cublas_attention
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_CUBLAS_ATTENTION_HPP
