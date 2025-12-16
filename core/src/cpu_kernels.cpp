// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include "zenith/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

// SIMD headers - conditionally included based on architecture
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#define ZENITH_X86_SIMD 1
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define ZENITH_ARM_NEON 1
#endif

namespace zenith {
namespace kernels {

// ============================================================================
// CPU Feature Detection
// ============================================================================

#ifdef ZENITH_X86_SIMD
#ifdef _MSC_VER
#include <intrin.h>
static void cpuid(int info[4], int func_id) { __cpuid(info, func_id); }
#else
#include <cpuid.h>
static void cpuid(int info[4], int func_id) {
  __cpuid(func_id, info[0], info[1], info[2], info[3]);
}
#endif
#endif

CpuFeatures detect_cpu_features() {
  CpuFeatures features;

#ifdef ZENITH_X86_SIMD
  int info[4];

  // Get basic CPU info
  cpuid(info, 0);
  int max_func = info[0];

  if (max_func >= 1) {
    cpuid(info, 1);
    features.has_sse = (info[3] & (1 << 25)) != 0;
    features.has_sse2 = (info[3] & (1 << 26)) != 0;
    features.has_sse3 = (info[2] & (1 << 0)) != 0;
    features.has_ssse3 = (info[2] & (1 << 9)) != 0;
    features.has_sse4_1 = (info[2] & (1 << 19)) != 0;
    features.has_sse4_2 = (info[2] & (1 << 20)) != 0;
    features.has_avx = (info[2] & (1 << 28)) != 0;
    features.has_fma = (info[2] & (1 << 12)) != 0;
  }

  if (max_func >= 7) {
    cpuid(info, 7);
    features.has_avx2 = (info[1] & (1 << 5)) != 0;
    features.has_avx512f = (info[1] & (1 << 16)) != 0;
  }
#endif

#ifdef ZENITH_ARM_NEON
  features.has_neon = true;
#endif

  return features;
}

const CpuFeatures &get_cpu_features() {
  static CpuFeatures features = detect_cpu_features();
  return features;
}

// ============================================================================
// Memory Utilities
// ============================================================================

void *aligned_alloc(size_t size, size_t alignment) {
#ifdef _MSC_VER
  return _aligned_malloc(size, alignment);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void aligned_free(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

// ============================================================================
// Naive Reference Implementations
// ============================================================================

void matmul_f32_naive(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  // Initialize output to zero
  std::memset(C, 0, M * N * sizeof(float));

  // Classic triple-loop matrix multiplication
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a_ik = A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }
}

// ============================================================================
// AVX2 Optimized Matrix Multiplication
// ============================================================================

#ifdef ZENITH_X86_SIMD
void matmul_f32_avx2(const float *A, const float *B, float *C, int M, int N,
                     int K) {
  // Initialize output to zero
  std::memset(C, 0, M * N * sizeof(float));

  // Process 8 columns at a time with AVX2
  const int N8 = (N / 8) * 8;

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      // Broadcast A[i,k] to all 8 lanes
      __m256 a_ik = _mm256_set1_ps(A[i * K + k]);

      // Process 8 columns at a time
      int j = 0;
      for (; j < N8; j += 8) {
        // Load 8 elements from B
        __m256 b_kj = _mm256_loadu_ps(&B[k * N + j]);

        // Load current C values
        __m256 c_ij = _mm256_loadu_ps(&C[i * N + j]);

        // FMA: C += A * B
        c_ij = _mm256_fmadd_ps(a_ik, b_kj, c_ij);

        // Store result
        _mm256_storeu_ps(&C[i * N + j], c_ij);
      }

      // Handle remaining columns
      float a_val = A[i * K + k];
      for (; j < N; ++j) {
        C[i * N + j] += a_val * B[k * N + j];
      }
    }
  }
}
#else
void matmul_f32_avx2(const float *A, const float *B, float *C, int M, int N,
                     int K) {
  // Fallback to naive implementation
  matmul_f32_naive(A, B, C, M, N, K);
}
#endif

void matmul_f32(const float *A, const float *B, float *C, int M, int N, int K) {
  // Use AVX2 if available, otherwise fall back to naive
  if (get_cpu_features().has_avx2 && get_cpu_features().has_fma) {
    matmul_f32_avx2(A, B, C, M, N, K);
  } else {
    matmul_f32_naive(A, B, C, M, N, K);
  }
}

// ============================================================================
// ReLU Activation
// ============================================================================

void relu_f32_naive(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = std::max(0.0f, data[i]);
  }
}

#ifdef ZENITH_X86_SIMD
void relu_f32_avx2(float *data, size_t size) {
  __m256 zero = _mm256_setzero_ps();

  size_t i = 0;
  const size_t size8 = (size / 8) * 8;

  // Process 8 elements at a time
  for (; i < size8; i += 8) {
    __m256 x = _mm256_loadu_ps(&data[i]);
    x = _mm256_max_ps(zero, x);
    _mm256_storeu_ps(&data[i], x);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    data[i] = std::max(0.0f, data[i]);
  }
}
#else
void relu_f32_avx2(float *data, size_t size) { relu_f32_naive(data, size); }
#endif

void relu_f32(float *data, size_t size) {
  if (get_cpu_features().has_avx2) {
    relu_f32_avx2(data, size);
  } else {
    relu_f32_naive(data, size);
  }
}

// ============================================================================
// Sigmoid Activation
// ============================================================================

void sigmoid_f32(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = 1.0f / (1.0f + std::exp(-data[i]));
  }
}

// ============================================================================
// Tanh Activation
// ============================================================================

void tanh_f32(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = std::tanh(data[i]);
  }
}

// ============================================================================
// Leaky ReLU
// ============================================================================

void leaky_relu_f32(float *data, size_t size, float alpha) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = data[i] > 0.0f ? data[i] : alpha * data[i];
  }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

void add_f32_naive(const float *A, const float *B, float *C, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    C[i] = A[i] + B[i];
  }
}

#ifdef ZENITH_X86_SIMD
void add_f32_avx2(const float *A, const float *B, float *C, size_t size) {
  size_t i = 0;
  const size_t size8 = (size / 8) * 8;

  for (; i < size8; i += 8) {
    __m256 a = _mm256_loadu_ps(&A[i]);
    __m256 b = _mm256_loadu_ps(&B[i]);
    __m256 c = _mm256_add_ps(a, b);
    _mm256_storeu_ps(&C[i], c);
  }

  for (; i < size; ++i) {
    C[i] = A[i] + B[i];
  }
}
#else
void add_f32_avx2(const float *A, const float *B, float *C, size_t size) {
  add_f32_naive(A, B, C, size);
}
#endif

void add_f32(const float *A, const float *B, float *C, size_t size) {
  if (get_cpu_features().has_avx2) {
    add_f32_avx2(A, B, C, size);
  } else {
    add_f32_naive(A, B, C, size);
  }
}

void sub_f32(const float *A, const float *B, float *C, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    C[i] = A[i] - B[i];
  }
}

void mul_f32(const float *A, const float *B, float *C, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    C[i] = A[i] * B[i];
  }
}

void div_f32(const float *A, const float *B, float *C, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    C[i] = A[i] / B[i];
  }
}

// ============================================================================
// Reduction Operations
// ============================================================================

float sum_f32(const float *data, size_t size) {
  float result = 0.0f;

#ifdef ZENITH_X86_SIMD
  if (get_cpu_features().has_avx2 && size >= 8) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    const size_t size8 = (size / 8) * 8;

    for (; i < size8; i += 8) {
      __m256 x = _mm256_loadu_ps(&data[i]);
      sum = _mm256_add_ps(sum, x);
    }

    // Horizontal sum
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 sum4 = _mm_add_ps(low, high);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    result = _mm_cvtss_f32(sum4);

    for (; i < size; ++i) {
      result += data[i];
    }
  } else
#endif
  {
    for (size_t i = 0; i < size; ++i) {
      result += data[i];
    }
  }

  return result;
}

float mean_f32(const float *data, size_t size) {
  return size > 0 ? sum_f32(data, size) / static_cast<float>(size) : 0.0f;
}

float max_f32(const float *data, size_t size) {
  if (size == 0)
    return 0.0f;

  float result = data[0];
  for (size_t i = 1; i < size; ++i) {
    result = std::max(result, data[i]);
  }
  return result;
}

float min_f32(const float *data, size_t size) {
  if (size == 0)
    return 0.0f;

  float result = data[0];
  for (size_t i = 1; i < size; ++i) {
    result = std::min(result, data[i]);
  }
  return result;
}

// ============================================================================
// Softmax
// ============================================================================

void softmax_f32(const float *input, float *output, size_t size,
                 size_t axis_size) {
  size_t outer = size / axis_size;

  for (size_t o = 0; o < outer; ++o) {
    const float *in_ptr = input + o * axis_size;
    float *out_ptr = output + o * axis_size;

    // Find max for numerical stability
    float max_val = in_ptr[0];
    for (size_t i = 1; i < axis_size; ++i) {
      max_val = std::max(max_val, in_ptr[i]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < axis_size; ++i) {
      out_ptr[i] = std::exp(in_ptr[i] - max_val);
      sum += out_ptr[i];
    }

    // Normalize
    for (size_t i = 0; i < axis_size; ++i) {
      out_ptr[i] /= sum;
    }
  }
}

// ============================================================================
// Convolution 2D
// ============================================================================

void conv2d_f32(const float *input, const float *weight, const float *bias,
                float *output, int N, int C_in, int H, int W, int C_out,
                int K_h, int K_w, int stride_h, int stride_w, int pad_h,
                int pad_w) {
  // Calculate output dimensions
  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  // For each batch
  for (int n = 0; n < N; ++n) {
    // For each output channel
    for (int c_out = 0; c_out < C_out; ++c_out) {
      // For each output spatial location
      for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
          float sum = bias ? bias[c_out] : 0.0f;

          // Convolve
          for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K_h; ++kh) {
              for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                  int in_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                  int w_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                  sum += input[in_idx] * weight[w_idx];
                }
              }
            }
          }

          int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
          output[out_idx] = sum;
        }
      }
    }
  }
}

void conv2d_f32_avx2(const float *input, const float *weight, const float *bias,
                     float *output, int N, int C_in, int H, int W, int C_out,
                     int K_h, int K_w, int stride_h, int stride_w, int pad_h,
                     int pad_w) {
  // For simplicity, use the naive implementation
  // Full AVX2 optimization would use im2col + GEMM approach
  conv2d_f32(input, weight, bias, output, N, C_in, H, W, C_out, K_h, K_w,
             stride_h, stride_w, pad_h, pad_w);
}

// ============================================================================
// Pooling Operations
// ============================================================================

void maxpool2d_f32(const float *input, float *output, int N, int C, int H,
                   int W, int pool_h, int pool_w, int stride_h, int stride_w) {
  int H_out = (H - pool_h) / stride_h + 1;
  int W_out = (W - pool_w) / stride_w + 1;

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
          float max_val = -std::numeric_limits<float>::infinity();

          for (int ph = 0; ph < pool_h; ++ph) {
            for (int pw = 0; pw < pool_w; ++pw) {
              int h_in = h_out * stride_h + ph;
              int w_in = w_out * stride_w + pw;
              int idx = ((n * C + c) * H + h_in) * W + w_in;
              max_val = std::max(max_val, input[idx]);
            }
          }

          int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
          output[out_idx] = max_val;
        }
      }
    }
  }
}

void avgpool2d_f32(const float *input, float *output, int N, int C, int H,
                   int W, int pool_h, int pool_w, int stride_h, int stride_w) {
  int H_out = (H - pool_h) / stride_h + 1;
  int W_out = (W - pool_w) / stride_w + 1;
  float pool_size = static_cast<float>(pool_h * pool_w);

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
          float sum = 0.0f;

          for (int ph = 0; ph < pool_h; ++ph) {
            for (int pw = 0; pw < pool_w; ++pw) {
              int h_in = h_out * stride_h + ph;
              int w_in = w_out * stride_w + pw;
              int idx = ((n * C + c) * H + h_in) * W + w_in;
              sum += input[idx];
            }
          }

          int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
          output[out_idx] = sum / pool_size;
        }
      }
    }
  }
}

} // namespace kernels
} // namespace zenith
