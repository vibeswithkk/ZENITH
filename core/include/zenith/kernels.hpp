// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_KERNELS_HPP
#define ZENITH_KERNELS_HPP

#include "types.hpp"
#include <cstddef>
#include <cstdint>

namespace zenith {
namespace kernels {

// ============================================================================
// SIMD Detection and Configuration
// ============================================================================

/// CPU feature detection flags
struct CpuFeatures {
  bool has_sse = false;
  bool has_sse2 = false;
  bool has_sse3 = false;
  bool has_ssse3 = false;
  bool has_sse4_1 = false;
  bool has_sse4_2 = false;
  bool has_avx = false;
  bool has_avx2 = false;
  bool has_avx512f = false;
  bool has_fma = false;
  bool has_neon = false; // ARM
};

/// Detect CPU features at runtime
CpuFeatures detect_cpu_features();

/// Get static instance of detected features
const CpuFeatures &get_cpu_features();

// ============================================================================
// Memory Utilities
// ============================================================================

/// Allocate aligned memory (32-byte for AVX2, 64-byte for AVX512)
void *aligned_alloc(size_t size, size_t alignment = 32);

/// Free aligned memory
void aligned_free(void *ptr);

/// Check if pointer is aligned to given boundary
inline bool is_aligned(const void *ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: C = A @ B
/// @param A Input matrix A [M x K], row-major
/// @param B Input matrix B [K x N], row-major
/// @param C Output matrix C [M x N], row-major
/// @param M Number of rows in A
/// @param N Number of columns in B
/// @param K Number of columns in A / rows in B
void matmul_f32(const float *A, const float *B, float *C, int M, int N, int K);

/// AVX2 optimized matrix multiplication
void matmul_f32_avx2(const float *A, const float *B, float *C, int M, int N,
                     int K);

/// Naive reference implementation (for testing)
void matmul_f32_naive(const float *A, const float *B, float *C, int M, int N,
                      int K);

// ============================================================================
// Convolution Operations
// ============================================================================

/// 2D Convolution with NCHW layout
/// @param input Input tensor [N, C_in, H, W]
/// @param weight Filter weights [C_out, C_in, K_h, K_w]
/// @param bias Bias vector [C_out] (can be nullptr)
/// @param output Output tensor [N, C_out, H_out, W_out]
/// @param N Batch size
/// @param C_in Input channels
/// @param H Input height
/// @param W Input width
/// @param C_out Output channels
/// @param K_h Kernel height
/// @param K_w Kernel width
/// @param stride_h Vertical stride
/// @param stride_w Horizontal stride
/// @param pad_h Vertical padding
/// @param pad_w Horizontal padding
void conv2d_f32(const float *input, const float *weight, const float *bias,
                float *output, int N, int C_in, int H, int W, int C_out,
                int K_h, int K_w, int stride_h, int stride_w, int pad_h,
                int pad_w);

/// AVX2 optimized convolution
void conv2d_f32_avx2(const float *input, const float *weight, const float *bias,
                     float *output, int N, int C_in, int H, int W, int C_out,
                     int K_h, int K_w, int stride_h, int stride_w, int pad_h,
                     int pad_w);

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation: y = max(0, x)
void relu_f32(float *data, size_t size);

/// AVX2 optimized ReLU
void relu_f32_avx2(float *data, size_t size);

/// Sigmoid activation: y = 1 / (1 + exp(-x))
void sigmoid_f32(float *data, size_t size);

/// Tanh activation: y = tanh(x)
void tanh_f32(float *data, size_t size);

/// Leaky ReLU: y = x if x > 0 else alpha * x
void leaky_relu_f32(float *data, size_t size, float alpha = 0.01f);

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise addition: C = A + B
void add_f32(const float *A, const float *B, float *C, size_t size);

/// AVX2 optimized addition
void add_f32_avx2(const float *A, const float *B, float *C, size_t size);

/// Element-wise subtraction: C = A - B
void sub_f32(const float *A, const float *B, float *C, size_t size);

/// Element-wise multiplication: C = A * B
void mul_f32(const float *A, const float *B, float *C, size_t size);

/// Element-wise division: C = A / B
void div_f32(const float *A, const float *B, float *C, size_t size);

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum reduction
float sum_f32(const float *data, size_t size);

/// Mean reduction
float mean_f32(const float *data, size_t size);

/// Max reduction
float max_f32(const float *data, size_t size);

/// Min reduction
float min_f32(const float *data, size_t size);

// ============================================================================
// Softmax
// ============================================================================

/// Softmax along last axis
/// @param input Input tensor
/// @param output Output tensor (can be same as input for in-place)
/// @param size Total size
/// @param axis_size Size of the softmax axis
void softmax_f32(const float *input, float *output, size_t size,
                 size_t axis_size);

// ============================================================================
// Pooling Operations
// ============================================================================

/// Max pooling 2D with NCHW layout
void maxpool2d_f32(const float *input, float *output, int N, int C, int H,
                   int W, int pool_h, int pool_w, int stride_h, int stride_w);

/// Average pooling 2D with NCHW layout
void avgpool2d_f32(const float *input, float *output, int N, int C, int H,
                   int W, int pool_h, int pool_w, int stride_h, int stride_w);

} // namespace kernels
} // namespace zenith

#endif // ZENITH_KERNELS_HPP
