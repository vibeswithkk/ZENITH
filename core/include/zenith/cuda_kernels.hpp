// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// CUDA Kernels Header - Forward declarations for bindings
// This header declares the kernel wrapper functions without CUDA syntax

#ifndef ZENITH_CUDA_KERNELS_HPP
#define ZENITH_CUDA_KERNELS_HPP

#ifdef ZENITH_HAS_CUDA

#include <cstddef>

namespace zenith {
namespace cuda_kernels {

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation (in-place)
void relu_f32(float *data, size_t size);

/// Sigmoid activation (in-place)
void sigmoid_f32(float *data, size_t size);

/// Tanh activation (in-place)
void tanh_f32(float *data, size_t size);

/// GELU activation
void gelu_f32(const float *input, float *output, size_t size);

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise addition: C = A + B
void add_f32(const float *A, const float *B, float *C, size_t size);

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
void matmul_f32(const float *A, const float *B, float *C, int M, int N, int K);

// ============================================================================
// Convolution Operations
// ============================================================================

/// 2D Convolution
void conv2d_f32(const float *input, const float *weight, const float *bias,
                float *output, int N, int C_in, int H, int W, int C_out,
                int K_h, int K_w, int stride_h, int stride_w, int pad_h,
                int pad_w);

// ============================================================================
// Pooling Operations
// ============================================================================

/// Max Pooling 2D
void maxpool2d_f32(const float *input, float *output, int N, int C, int H,
                   int W, int K_h, int K_w, int stride_h, int stride_w);

// ============================================================================
// Transformer/BERT Operations
// ============================================================================

/// Layer Normalization
void layernorm_f32(const float *input, float *output, const float *gamma,
                   const float *beta, int batch, int hidden, float eps);

/// Softmax (2D)
void softmax_2d_f32(const float *input, float *output, int batch, int len);

/// Add bias: output[i,j] += bias[j] for each row i
void add_bias_f32(float *output, const float *bias, int M, int N);

/// Element-wise add 2D: C = A + B
void add_2d_f32(const float *A, const float *B, float *C, int M, int N);

/// Transpose [batch, seq, heads, dim] -> [batch, heads, seq, dim]
void transpose_0213_f32(const float *input, float *output, int batch, int seq,
                        int heads, int dim);

/// Inverse transpose [batch, heads, seq, dim] -> [batch, seq, heads, dim]
void transpose_0213_inv_f32(const float *input, float *output, int batch,
                            int heads, int seq, int dim);

} // namespace cuda_kernels
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_CUDA_KERNELS_HPP
