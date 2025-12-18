// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FP16 Kernel Suite Header

#ifndef ZENITH_FP16_KERNELS_HPP
#define ZENITH_FP16_KERNELS_HPP

#ifdef ZENITH_HAS_CUDA
#include <cuda_fp16.h>

namespace zenith {
namespace fp16_kernels {

// ============================================================================
// Conversion Functions
// ============================================================================

/// Convert FP32 array to FP16
void convert_f32_to_f16(const float *input, __half *output, int size);

/// Convert FP16 array to FP32
void convert_f16_to_f32(const __half *input, float *output, int size);

// ============================================================================
// FP16 Neural Network Operations
// ============================================================================

/// FP16 Linear: Y = X @ W^T + bias using Tensor Cores
/// X: [M, K], W: [N, K], bias: [N], Y: [M, N]
void linear_fp16(const __half *X, const __half *W, const __half *bias,
                 __half *Y, int M, int N, int K);

/// FP16 GELU activation (approximate)
void gelu_fp16(const __half *input, __half *output, int size);

/// FP16 LayerNorm with FP32 statistics for numerical stability
void layernorm_fp16(const __half *input, __half *output, const __half *gamma,
                    const __half *beta, int batch, int hidden,
                    float eps = 1e-5f);

/// FP16 Element-wise add: C = A + B
void add_fp16(const __half *a, const __half *b, __half *c, int size);

// ============================================================================
// FP16 Attention Operations
// ============================================================================

/// Transpose [B, S, H, D] -> [B, H, S, D] for attention
void transpose_for_attn_fp16(const __half *input, __half *output, int batch,
                             int seq, int heads, int dim);

/// Transpose [B, H, S, D] -> [B, S, H, D] from attention
void transpose_from_attn_fp16(const __half *input, __half *output, int batch,
                              int heads, int seq, int dim);

/// FP16 Multi-Head Attention with Tensor Cores
void attention_fp16(const __half *Q, const __half *K, const __half *V,
                    __half *O, int batch, int heads, int seq, int dim);

} // namespace fp16_kernels
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FP16_KERNELS_HPP
