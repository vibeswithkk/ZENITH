/**
 * @file rocblas_ops.hpp
 * @brief rocBLAS Operations for AMD GPUs
 *
 * Provides a CUDA-compatible interface to rocBLAS operations.
 * When ZENITH_HAS_ROCBLAS is not defined, functions return error status.
 *
 * Build with: -DZENITH_HAS_ROCM=1 -DZENITH_HAS_ROCBLAS=1
 * Link with: -lrocblas
 *
 * Copyright 2025 Wahyu Ardiansyah
 * Licensed under the Apache License, Version 2.0
 */

#ifndef ZENITH_ROCBLAS_OPS_HPP
#define ZENITH_ROCBLAS_OPS_HPP

#include "types.hpp"
#include <string>

#ifdef ZENITH_HAS_ROCBLAS
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace zenith {
namespace rocblas_ops {

/**
 * @brief Check if rocBLAS is available.
 */
inline bool is_rocblas_available() {
#ifdef ZENITH_HAS_ROCBLAS
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  return (err == hipSuccess && device_count > 0);
#else
  return false;
#endif
}

/**
 * @brief Get rocBLAS version.
 */
inline int get_rocblas_version() {
#ifdef ZENITH_HAS_ROCBLAS
  size_t size = 0;
  rocblas_get_version_string_size(&size);
  return static_cast<int>(size > 0 ? 1 : 0);
#else
  return 0;
#endif
}

#ifdef ZENITH_HAS_ROCBLAS

/**
 * @brief RAII wrapper for rocBLAS handle.
 */
class RocBlasHandle {
public:
  RocBlasHandle() : handle_(nullptr), valid_(false) {
    rocblas_status status = rocblas_create_handle(&handle_);
    valid_ = (status == rocblas_status_success);
  }

  ~RocBlasHandle() {
    if (valid_ && handle_) {
      rocblas_destroy_handle(handle_);
    }
  }

  // Non-copyable
  RocBlasHandle(const RocBlasHandle &) = delete;
  RocBlasHandle &operator=(const RocBlasHandle &) = delete;

  // Movable
  RocBlasHandle(RocBlasHandle &&other) noexcept
      : handle_(other.handle_), valid_(other.valid_) {
    other.handle_ = nullptr;
    other.valid_ = false;
  }

  RocBlasHandle &operator=(RocBlasHandle &&other) noexcept {
    if (this != &other) {
      if (valid_ && handle_) {
        rocblas_destroy_handle(handle_);
      }
      handle_ = other.handle_;
      valid_ = other.valid_;
      other.handle_ = nullptr;
      other.valid_ = false;
    }
    return *this;
  }

  rocblas_handle get() const { return handle_; }
  bool is_valid() const { return valid_; }

private:
  rocblas_handle handle_;
  bool valid_;
};

/**
 * @brief Get or create the global rocBLAS handle.
 */
inline RocBlasHandle &get_global_handle() {
  static RocBlasHandle handle;
  return handle;
}

/**
 * @brief GEMM operation: C = alpha * A * B + beta * C
 *
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows of A and C
 * @param N Number of columns of B and C
 * @param K Number of columns of A and rows of B
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @return Status indicating success or failure
 */
inline Status gemm_f32(const float *A, const float *B, float *C, int M, int N,
                       int K, float alpha = 1.0f, float beta = 0.0f) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "rocBLAS handle not initialized");
  }

  // rocBLAS uses column-major by default, we use row-major
  // C = A * B in row-major = C^T = B^T * A^T in column-major
  // So we swap A and B and transpose the result
  rocblas_status status = rocblas_sgemm(
      handle.get(), rocblas_operation_none, rocblas_operation_none,
      N,            // Number of rows of B^T = columns of B
      M,            // Number of columns of A^T = rows of A
      K,            // Shared dimension
      &alpha, B, N, // B^T in column-major
      A, K,         // A^T in column-major
      &beta, C, N   // C^T in column-major
  );

  if (status != rocblas_status_success) {
    return Status(StatusCode::InternalError, "rocblas_sgemm failed");
  }

  return Status::ok();
}

/**
 * @brief Batched GEMM for attention QK^T computation.
 */
inline Status batched_gemm_f32(const float *A, const float *B, float *C,
                               int batch_size, int M, int N, int K,
                               int64_t stride_A, int64_t stride_B,
                               int64_t stride_C, float alpha = 1.0f,
                               float beta = 0.0f) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "rocBLAS handle not initialized");
  }

  rocblas_status status = rocblas_sgemm_strided_batched(
      handle.get(), rocblas_operation_none, rocblas_operation_none, N, M, K,
      &alpha, B, N, stride_B, A, K, stride_A, &beta, C, N, stride_C,
      batch_size);

  if (status != rocblas_status_success) {
    return Status(StatusCode::InternalError,
                  "rocblas_sgemm_strided_batched failed");
  }

  return Status::ok();
}

/**
 * @brief GEMM with FP16 input, FP32 accumulate (for Tensor Cores).
 */
inline Status gemm_f16_f32(const void *A, const void *B, float *C, int M, int N,
                           int K, float alpha = 1.0f, float beta = 0.0f) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "rocBLAS handle not initialized");
  }

  // Use rocblas_gemm_ex for mixed precision
  rocblas_status status = rocblas_gemm_ex(
      handle.get(), rocblas_operation_none, rocblas_operation_none, N, M, K,
      &alpha, B, rocblas_datatype_f16_r, N, A, rocblas_datatype_f16_r, K, &beta,
      C, rocblas_datatype_f32_r, N, C, rocblas_datatype_f32_r, N,
      rocblas_datatype_f32_r, // Compute type
      rocblas_gemm_algo_standard, 0, 0);

  if (status != rocblas_status_success) {
    return Status(StatusCode::InternalError, "rocblas_gemm_ex failed");
  }

  return Status::ok();
}

#else // No rocBLAS

// Stub implementations
inline Status gemm_f32(const float *, const float *, float *, int, int, int,
                       float = 1.0f, float = 0.0f) {
  return Status(StatusCode::NotImplemented, "rocBLAS not available");
}

inline Status batched_gemm_f32(const float *, const float *, float *, int, int,
                               int, int, int64_t, int64_t, int64_t,
                               float = 1.0f, float = 0.0f) {
  return Status(StatusCode::NotImplemented, "rocBLAS not available");
}

inline Status gemm_f16_f32(const void *, const void *, float *, int, int, int,
                           float = 1.0f, float = 0.0f) {
  return Status(StatusCode::NotImplemented, "rocBLAS not available");
}

#endif // ZENITH_HAS_ROCBLAS

} // namespace rocblas_ops
} // namespace zenith

#endif // ZENITH_ROCBLAS_OPS_HPP
