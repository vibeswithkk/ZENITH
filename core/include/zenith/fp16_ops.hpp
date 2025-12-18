// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FP16 Operations Header

#ifndef ZENITH_FP16_OPS_HPP
#define ZENITH_FP16_OPS_HPP

#ifdef ZENITH_HAS_CUDA
#include <cuda_fp16.h>

namespace zenith {
namespace fp16_ops {

/// Convert FP32 array to FP16
void convert_fp32_to_fp16(const float *input, __half *output, int size);

/// Convert FP16 array to FP32
void convert_fp16_to_fp32(const __half *input, float *output, int size);

/// FP16 GEMM: C = A @ B using Tensor Cores
void gemm_fp16(const __half *A, const __half *B, __half *C, int M, int N,
               int K);

/// FP16 Linear: Y = X @ W^T + bias
void linear_fp16(const __half *X, const __half *W, const __half *bias,
                 __half *Y, int M, int N, int K);

/// FP16 Attention with Tensor Cores
void attention_fp16_alloc(const __half *Q, const __half *K, const __half *V,
                          __half *O, int batch_size, int num_heads, int seq_len,
                          int head_dim);

} // namespace fp16_ops
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FP16_OPS_HPP
