/**
 * @file miopen_ops.hpp
 * @brief MIOpen Operations for AMD GPUs
 *
 * Provides a cuDNN-compatible interface to MIOpen operations.
 * Supports convolution, pooling, batch normalization, and activations.
 *
 * Build with: -DZENITH_HAS_ROCM=1 -DZENITH_HAS_MIOPEN=1
 * Link with: -lMIOpen
 *
 * Copyright 2025 Wahyu Ardiansyah
 * Licensed under the Apache License, Version 2.0
 */

#ifndef ZENITH_MIOPEN_OPS_HPP
#define ZENITH_MIOPEN_OPS_HPP

#include "types.hpp"
#include <string>

#ifdef ZENITH_HAS_MIOPEN
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#endif

namespace zenith {
namespace miopen {

/**
 * @brief Check if MIOpen is available.
 */
inline bool is_miopen_available() {
#ifdef ZENITH_HAS_MIOPEN
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  return (err == hipSuccess && device_count > 0);
#else
  return false;
#endif
}

/**
 * @brief Get MIOpen version.
 */
inline int get_miopen_version() {
#ifdef ZENITH_HAS_MIOPEN
  size_t major = 0, minor = 0, patch = 0;
  miopenGetVersion(&major, &minor, &patch);
  return static_cast<int>(major * 1000 + minor * 10 + patch);
#else
  return 0;
#endif
}

#ifdef ZENITH_HAS_MIOPEN

/**
 * @brief RAII wrapper for MIOpen handle.
 */
class MIOpenHandle {
public:
  MIOpenHandle() : handle_(nullptr), valid_(false) {
    miopenStatus_t status = miopenCreate(&handle_);
    valid_ = (status == miopenStatusSuccess);
  }

  ~MIOpenHandle() {
    if (valid_ && handle_) {
      miopenDestroy(handle_);
    }
  }

  // Non-copyable
  MIOpenHandle(const MIOpenHandle &) = delete;
  MIOpenHandle &operator=(const MIOpenHandle &) = delete;

  // Movable
  MIOpenHandle(MIOpenHandle &&other) noexcept
      : handle_(other.handle_), valid_(other.valid_) {
    other.handle_ = nullptr;
    other.valid_ = false;
  }

  MIOpenHandle &operator=(MIOpenHandle &&other) noexcept {
    if (this != &other) {
      if (valid_ && handle_) {
        miopenDestroy(handle_);
      }
      handle_ = other.handle_;
      valid_ = other.valid_;
      other.handle_ = nullptr;
      other.valid_ = false;
    }
    return *this;
  }

  miopenHandle_t get() const { return handle_; }
  bool is_valid() const { return valid_; }

private:
  miopenHandle_t handle_;
  bool valid_;
};

/**
 * @brief Get or create the global MIOpen handle.
 */
inline MIOpenHandle &get_global_handle() {
  static MIOpenHandle handle;
  return handle;
}

/**
 * @brief Convolution descriptor wrapper.
 */
class ConvolutionDescriptor {
public:
  ConvolutionDescriptor(int pad_h, int pad_w, int stride_h, int stride_w,
                        int dilation_h = 1, int dilation_w = 1)
      : desc_(nullptr) {
    miopenCreateConvolutionDescriptor(&desc_);
    miopenInitConvolutionDescriptor(desc_, miopenConvolution, pad_h, pad_w,
                                    stride_h, stride_w, dilation_h, dilation_w);
  }

  ~ConvolutionDescriptor() {
    if (desc_) {
      miopenDestroyConvolutionDescriptor(desc_);
    }
  }

  miopenConvolutionDescriptor_t get() const { return desc_; }

private:
  miopenConvolutionDescriptor_t desc_;
};

/**
 * @brief Tensor descriptor wrapper.
 */
class TensorDescriptor {
public:
  TensorDescriptor(int n, int c, int h, int w) : desc_(nullptr) {
    miopenCreateTensorDescriptor(&desc_);
    miopenSet4dTensorDescriptor(desc_, miopenFloat, n, c, h, w);
  }

  TensorDescriptor(int n, int c, int h, int w, miopenDataType_t dtype)
      : desc_(nullptr) {
    miopenCreateTensorDescriptor(&desc_);
    miopenSet4dTensorDescriptor(desc_, dtype, n, c, h, w);
  }

  ~TensorDescriptor() {
    if (desc_) {
      miopenDestroyTensorDescriptor(desc_);
    }
  }

  miopenTensorDescriptor_t get() const { return desc_; }

private:
  miopenTensorDescriptor_t desc_;
};

/**
 * @brief Forward convolution using MIOpen.
 */
inline Status conv2d_forward(const float *input, const float *weight,
                             float *output, int N, int C_in, int H, int W,
                             int C_out, int K_h, int K_w, int pad_h, int pad_w,
                             int stride_h, int stride_w) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "MIOpen handle not initialized");
  }

  // Calculate output dimensions
  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  // Create descriptors
  TensorDescriptor x_desc(N, C_in, H, W);
  TensorDescriptor w_desc(C_out, C_in, K_h, K_w);
  TensorDescriptor y_desc(N, C_out, H_out, W_out);
  ConvolutionDescriptor conv_desc(pad_h, pad_w, stride_h, stride_w);

  // Find best algorithm
  miopenConvFwdAlgorithm_t algo;
  size_t ws_size = 0;

  int algo_count = 0;
  miopenConvAlgoPerf_t perf;
  miopenFindConvolutionForwardAlgorithm(
      handle.get(), x_desc.get(), input, w_desc.get(), weight, conv_desc.get(),
      y_desc.get(), output, 1, &algo_count, &perf, nullptr, 0, false);

  algo = perf.fwd_algo;
  ws_size = perf.memory;

  // Allocate workspace
  void *workspace = nullptr;
  if (ws_size > 0) {
    hipMalloc(&workspace, ws_size);
  }

  // Run convolution
  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t status = miopenConvolutionForward(
      handle.get(), &alpha, x_desc.get(), input, w_desc.get(), weight,
      conv_desc.get(), algo, &beta, y_desc.get(), output, workspace, ws_size);

  // Free workspace
  if (workspace) {
    hipFree(workspace);
  }

  if (status != miopenStatusSuccess) {
    return Status(StatusCode::InternalError, "miopenConvolutionForward failed");
  }

  return Status::ok();
}

/**
 * @brief ReLU activation using MIOpen.
 */
inline Status relu_forward(const float *input, float *output, size_t size) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "MIOpen handle not initialized");
  }

  // Create activation descriptor
  miopenActivationDescriptor_t act_desc;
  miopenCreateActivationDescriptor(&act_desc);
  miopenSetActivationDescriptor(act_desc, miopenActivationRELU, 0.0, 0.0, 1.0);

  // Treat as 1D tensor
  TensorDescriptor x_desc(1, 1, 1, static_cast<int>(size));

  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t status =
      miopenActivationForward(handle.get(), act_desc, &alpha, x_desc.get(),
                              input, &beta, x_desc.get(), output);

  miopenDestroyActivationDescriptor(act_desc);

  if (status != miopenStatusSuccess) {
    return Status(StatusCode::InternalError, "miopenActivationForward failed");
  }

  return Status::ok();
}

/**
 * @brief Batch normalization forward inference using MIOpen.
 */
inline Status batchnorm_forward(const float *input, float *output,
                                const float *gamma, const float *beta,
                                const float *running_mean,
                                const float *running_var, int N, int C, int H,
                                int W, float epsilon = 1e-5f) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "MIOpen handle not initialized");
  }

  TensorDescriptor x_desc(N, C, H, W);
  TensorDescriptor bn_desc(1, C, 1, 1);

  float alpha = 1.0f, beta_param = 0.0f;

  miopenStatus_t status = miopenBatchNormalizationForwardInference(
      handle.get(), miopenBNSpatial, &alpha, &beta_param, x_desc.get(), input,
      x_desc.get(), output, bn_desc.get(), const_cast<float *>(gamma),
      const_cast<float *>(beta), const_cast<float *>(running_mean),
      const_cast<float *>(running_var), epsilon);

  if (status != miopenStatusSuccess) {
    return Status(StatusCode::InternalError,
                  "miopenBatchNormalizationForwardInference failed");
  }

  return Status::ok();
}

/**
 * @brief Max pooling 2D forward using MIOpen.
 */
inline Status maxpool2d_forward(const float *input, float *output, int N, int C,
                                int H, int W, int kernel_h, int kernel_w,
                                int stride_h, int stride_w, int pad_h = 0,
                                int pad_w = 0) {
  auto &handle = get_global_handle();
  if (!handle.is_valid()) {
    return Status(StatusCode::InternalError, "MIOpen handle not initialized");
  }

  int H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;

  TensorDescriptor x_desc(N, C, H, W);
  TensorDescriptor y_desc(N, C, H_out, W_out);

  // Create pooling descriptor
  miopenPoolingDescriptor_t pool_desc;
  miopenCreatePoolingDescriptor(&pool_desc);
  miopenSet2dPoolingDescriptor(pool_desc, miopenPoolingMax, kernel_h, kernel_w,
                               pad_h, pad_w, stride_h, stride_w);

  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t status =
      miopenPoolingForward(handle.get(), pool_desc, &alpha, x_desc.get(), input,
                           &beta, y_desc.get(), output, false, nullptr, 0);

  miopenDestroyPoolingDescriptor(pool_desc);

  if (status != miopenStatusSuccess) {
    return Status(StatusCode::InternalError, "miopenPoolingForward failed");
  }

  return Status::ok();
}

#else // No MIOpen

// Stub implementations
inline Status conv2d_forward(const float *, const float *, float *, int, int,
                             int, int, int, int, int, int, int, int, int) {
  return Status(StatusCode::NotImplemented, "MIOpen not available");
}

inline Status relu_forward(const float *, float *, size_t) {
  return Status(StatusCode::NotImplemented, "MIOpen not available");
}

inline Status batchnorm_forward(const float *, float *, const float *,
                                const float *, const float *, const float *,
                                int, int, int, int, float = 1e-5f) {
  return Status(StatusCode::NotImplemented, "MIOpen not available");
}

inline Status maxpool2d_forward(const float *, float *, int, int, int, int, int,
                                int, int, int, int = 0, int = 0) {
  return Status(StatusCode::NotImplemented, "MIOpen not available");
}

#endif // ZENITH_HAS_MIOPEN

} // namespace miopen
} // namespace zenith

#endif // ZENITH_MIOPEN_OPS_HPP
