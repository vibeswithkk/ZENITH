// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// cuBLAS Operations Wrapper for Zenith Framework
// Provides optimized BLAS operations using NVIDIA cuBLAS library.
// Per CetakBiru Section 4.3: Use optimized primitives instead of custom
// kernels.

#ifndef ZENITH_CUBLAS_OPS_HPP
#define ZENITH_CUBLAS_OPS_HPP

#include "types.hpp"

#ifdef ZENITH_HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>

namespace zenith {
namespace cublas {

// ============================================================================
// cuBLAS Handle Manager (Thread-Safe Singleton)
// ============================================================================

class CublasHandle {
public:
  static CublasHandle &instance() {
    static CublasHandle handle;
    return handle;
  }

  cublasHandle_t get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      cublasStatus_t status = cublasCreate(&handle_);
      if (status != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
      }
      // Enable TF32 Tensor Core math for ~3x FP32 speedup on Ampere+
      // This uses Tensor Cores with TF32 precision (19-bit mantissa)
      cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH);
      initialized_ = true;
    }
    return handle_;
  }

  bool is_available() const { return initialized_ && handle_ != nullptr; }

  ~CublasHandle() {
    if (initialized_ && handle_) {
      cublasDestroy(handle_);
    }
  }

  // Non-copyable
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;

private:
  CublasHandle() : handle_(nullptr), initialized_(false) {}

  cublasHandle_t handle_;
  bool initialized_;
  mutable std::mutex mutex_;
};

// ============================================================================
// Error Handling
// ============================================================================

inline const char *cublas_get_error_string(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  default:
    return "Unknown cuBLAS error";
  }
}

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      return Status::Error(StatusCode::InternalError,                          \
                           std::string("cuBLAS error: ") +                     \
                               cublas_get_error_string(status));               \
    }                                                                          \
  } while (0)

// ============================================================================
// GEMM Operations
// ============================================================================

/// Single-precision General Matrix Multiply: C = alpha * A * B + beta * C
/// @param A Input matrix A [M x K], device pointer
/// @param B Input matrix B [K x N], device pointer
/// @param C Output matrix C [M x N], device pointer (also input when beta != 0)
/// @param M Number of rows in A and C
/// @param N Number of columns in B and C
/// @param K Number of columns in A / rows in B
/// @param alpha Scalar multiplier for A*B
/// @param beta Scalar multiplier for C
/// @param trans_a Whether to transpose A (false = N, true = T)
/// @param trans_b Whether to transpose B (false = N, true = T)
inline Status gemm_f32(const float *A, const float *B, float *C, int M, int N,
                       int K, float alpha = 1.0f, float beta = 0.0f,
                       bool trans_a = false, bool trans_b = false) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  // cuBLAS uses column-major order, so we compute B^T * A^T = (A*B)^T
  // For row-major: cublasSgemm(handle, opB, opA, N, M, K, ...)
  cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Leading dimensions for row-major storage
  int lda = trans_a ? M : K;
  int ldb = trans_b ? K : N;
  int ldc = N;

  // For row-major matrices, swap A and B
  CUBLAS_CHECK(cublasSgemm(handle, op_b, op_a, N, M, K, &alpha, B, ldb, A, lda,
                           &beta, C, ldc));

  return Status::Ok();
}

/// Double-precision General Matrix Multiply
inline Status gemm_f64(const double *A, const double *B, double *C, int M,
                       int N, int K, double alpha = 1.0, double beta = 0.0,
                       bool trans_a = false, bool trans_b = false) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = trans_a ? M : K;
  int ldb = trans_b ? K : N;
  int ldc = N;

  CUBLAS_CHECK(cublasDgemm(handle, op_b, op_a, N, M, K, &alpha, B, ldb, A, lda,
                           &beta, C, ldc));

  return Status::Ok();
}

// ============================================================================
// Vector Operations
// ============================================================================

/// Vector addition: y = alpha * x + y (SAXPY)
inline Status axpy_f32(int n, float alpha, const float *x, float *y) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, x, 1, y, 1));
  return Status::Ok();
}

/// Vector scaling: x = alpha * x (SSCAL)
inline Status scal_f32(int n, float alpha, float *x) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  CUBLAS_CHECK(cublasSscal(handle, n, &alpha, x, 1));
  return Status::Ok();
}

/// Dot product: result = x . y
inline Status dot_f32(int n, const float *x, const float *y, float *result) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  CUBLAS_CHECK(cublasSdot(handle, n, x, 1, y, 1, result));
  return Status::Ok();
}

/// Vector 2-norm: result = ||x||_2
inline Status nrm2_f32(int n, const float *x, float *result) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  CUBLAS_CHECK(cublasSnrm2(handle, n, x, 1, result));
  return Status::Ok();
}

// ============================================================================
// Matrix-Vector Operations
// ============================================================================

/// Matrix-Vector multiply: y = alpha * A * x + beta * y
inline Status gemv_f32(const float *A, const float *x, float *y, int M, int N,
                       float alpha = 1.0f, float beta = 0.0f,
                       bool trans = false) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  cublasOperation_t op = trans ? CUBLAS_OP_T : CUBLAS_OP_N;

  // For row-major, we need to swap M and N and use transpose
  CUBLAS_CHECK(cublasSgemv(handle, trans ? CUBLAS_OP_N : CUBLAS_OP_T, N, M,
                           &alpha, A, N, x, 1, &beta, y, 1));

  return Status::Ok();
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batched GEMM for multiple matrix multiplications
inline Status gemm_batched_f32(const float **A, const float **B, float **C,
                               int M, int N, int K, float alpha, float beta,
                               int batch_count) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, B, N, A, K, &beta, C, N,
                                  batch_count));

  return Status::Ok();
}

/// Strided Batched GEMM (more efficient for contiguous batches)
inline Status gemm_strided_batched_f32(const float *A, const float *B, float *C,
                                       int M, int N, int K, float alpha,
                                       float beta, int batch_count) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  long long int stride_a = static_cast<long long int>(M) * K;
  long long int stride_b = static_cast<long long int>(K) * N;
  long long int stride_c = static_cast<long long int>(M) * N;

  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, stride_b, A, K,
      stride_a, &beta, C, N, stride_c, batch_count));

  return Status::Ok();
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if cuBLAS is available
inline bool is_cublas_available() {
  return CublasHandle::instance().get() != nullptr;
}

/// Set cuBLAS math mode (for Tensor Core usage on Volta+)
inline Status set_math_mode(cublasMath_t mode) {
  cublasHandle_t handle = CublasHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuBLAS not initialized");
  }

  CUBLAS_CHECK(cublasSetMathMode(handle, mode));
  return Status::Ok();
}

/// Enable Tensor Core operations (FP32 with TF32)
inline Status enable_tensor_cores() {
  return set_math_mode(CUBLAS_TF32_TENSOR_OP_MATH);
}

/// Use default math mode (pure FP32)
inline Status disable_tensor_cores() {
  return set_math_mode(CUBLAS_DEFAULT_MATH);
}

} // namespace cublas
} // namespace zenith

#else // !ZENITH_HAS_CUDA

namespace zenith {
namespace cublas {

inline bool is_cublas_available() { return false; }

inline Status gemm_f32(const float *, const float *, float *, int, int, int,
                       float = 1.0f, float = 0.0f, bool = false, bool = false) {
  return Status::Error(StatusCode::NotImplemented,
                       "cuBLAS not compiled - CUDA support disabled");
}

} // namespace cublas
} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_CUBLAS_OPS_HPP
