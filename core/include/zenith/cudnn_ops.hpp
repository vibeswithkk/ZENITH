// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// cuDNN Operations Wrapper for Zenith Framework
// Provides optimized deep learning operations using NVIDIA cuDNN library.
// Per CetakBiru Section 4.3: Use optimized primitives instead of custom
// kernels.

#ifndef ZENITH_CUDNN_OPS_HPP
#define ZENITH_CUDNN_OPS_HPP

#include "types.hpp"

#ifdef ZENITH_HAS_CUDA
#ifdef ZENITH_HAS_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace zenith {
namespace cudnn {

// ============================================================================
// cuDNN Handle Manager (Thread-Safe Singleton)
// ============================================================================

class CudnnHandle {
public:
  static CudnnHandle &instance() {
    static CudnnHandle handle;
    return handle;
  }

  cudnnHandle_t get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      cudnnStatus_t status = cudnnCreate(&handle_);
      if (status != CUDNN_STATUS_SUCCESS) {
        return nullptr;
      }
      initialized_ = true;
    }
    return handle_;
  }

  bool is_available() const { return initialized_ && handle_ != nullptr; }

  ~CudnnHandle() {
    if (initialized_ && handle_) {
      cudnnDestroy(handle_);
    }
  }

  CudnnHandle(const CudnnHandle &) = delete;
  CudnnHandle &operator=(const CudnnHandle &) = delete;

private:
  CudnnHandle() : handle_(nullptr), initialized_(false) {}

  cudnnHandle_t handle_;
  bool initialized_;
  mutable std::mutex mutex_;
};

// ============================================================================
// Error Handling
// ============================================================================

#define CUDNN_CHECK(call)                                                      \
  do {                                                                         \
    cudnnStatus_t status = call;                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      return Status::Error(StatusCode::InternalError,                          \
                           std::string("cuDNN error: ") +                      \
                               cudnnGetErrorString(status));                   \
    }                                                                          \
  } while (0)

// ============================================================================
// Tensor Descriptor Wrapper
// ============================================================================

class TensorDesc {
public:
  TensorDesc() : desc_(nullptr) { cudnnCreateTensorDescriptor(&desc_); }

  ~TensorDesc() {
    if (desc_) {
      cudnnDestroyTensorDescriptor(desc_);
    }
  }

  cudnnStatus_t set_4d(cudnnTensorFormat_t format, cudnnDataType_t dtype, int n,
                       int c, int h, int w) {
    return cudnnSetTensor4dDescriptor(desc_, format, dtype, n, c, h, w);
  }

  cudnnTensorDescriptor_t get() const { return desc_; }

  TensorDesc(const TensorDesc &) = delete;
  TensorDesc &operator=(const TensorDesc &) = delete;
  TensorDesc(TensorDesc &&other) noexcept : desc_(other.desc_) {
    other.desc_ = nullptr;
  }

private:
  cudnnTensorDescriptor_t desc_;
};

// ============================================================================
// Filter Descriptor Wrapper
// ============================================================================

class FilterDesc {
public:
  FilterDesc() : desc_(nullptr) { cudnnCreateFilterDescriptor(&desc_); }

  ~FilterDesc() {
    if (desc_) {
      cudnnDestroyFilterDescriptor(desc_);
    }
  }

  cudnnStatus_t set_4d(cudnnDataType_t dtype, cudnnTensorFormat_t format, int k,
                       int c, int h, int w) {
    return cudnnSetFilter4dDescriptor(desc_, dtype, format, k, c, h, w);
  }

  cudnnFilterDescriptor_t get() const { return desc_; }

private:
  cudnnFilterDescriptor_t desc_;
};

// ============================================================================
// Convolution Descriptor Wrapper
// ============================================================================

class ConvDesc {
public:
  ConvDesc() : desc_(nullptr) { cudnnCreateConvolutionDescriptor(&desc_); }

  ~ConvDesc() {
    if (desc_) {
      cudnnDestroyConvolutionDescriptor(desc_);
    }
  }

  cudnnStatus_t set_2d(int pad_h, int pad_w, int stride_h, int stride_w,
                       int dilation_h, int dilation_w,
                       cudnnConvolutionMode_t mode, cudnnDataType_t dtype) {
    return cudnnSetConvolution2dDescriptor(desc_, pad_h, pad_w, stride_h,
                                           stride_w, dilation_h, dilation_w,
                                           mode, dtype);
  }

  cudnnConvolutionDescriptor_t get() const { return desc_; }

private:
  cudnnConvolutionDescriptor_t desc_;
};

// ============================================================================
// Pooling Descriptor Wrapper
// ============================================================================

class PoolingDesc {
public:
  PoolingDesc() : desc_(nullptr) { cudnnCreatePoolingDescriptor(&desc_); }

  ~PoolingDesc() {
    if (desc_) {
      cudnnDestroyPoolingDescriptor(desc_);
    }
  }

  cudnnStatus_t set_2d(cudnnPoolingMode_t mode, cudnnNanPropagation_t nan_prop,
                       int h, int w, int pad_h, int pad_w, int stride_h,
                       int stride_w) {
    return cudnnSetPooling2dDescriptor(desc_, mode, nan_prop, h, w, pad_h,
                                       pad_w, stride_h, stride_w);
  }

  cudnnPoolingDescriptor_t get() const { return desc_; }

private:
  cudnnPoolingDescriptor_t desc_;
};

// ============================================================================
// Activation Descriptor Wrapper
// ============================================================================

class ActivationDesc {
public:
  ActivationDesc() : desc_(nullptr) { cudnnCreateActivationDescriptor(&desc_); }

  ~ActivationDesc() {
    if (desc_) {
      cudnnDestroyActivationDescriptor(desc_);
    }
  }

  cudnnStatus_t set(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_prop,
                    double coef) {
    return cudnnSetActivationDescriptor(desc_, mode, nan_prop, coef);
  }

  cudnnActivationDescriptor_t get() const { return desc_; }

private:
  cudnnActivationDescriptor_t desc_;
};

// ============================================================================
// Convolution Operations
// ============================================================================

/// 2D Convolution Forward using cuDNN
/// @param input Input tensor [N, C_in, H, W], device pointer
/// @param weight Filter tensor [C_out, C_in, K_h, K_w], device pointer
/// @param bias Bias tensor [C_out], device pointer (can be nullptr)
/// @param output Output tensor [N, C_out, H_out, W_out], device pointer
/// @param workspace Pre-allocated workspace, device pointer
/// @param workspace_size Size of workspace in bytes
inline Status conv2d_forward(const float *input, const float *weight,
                             const float *bias, float *output, int N, int C_in,
                             int H, int W, int C_out, int K_h, int K_w,
                             int stride_h, int stride_w, int pad_h, int pad_w,
                             void *workspace, size_t workspace_size) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  // Calculate output dimensions
  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  // Create descriptors
  TensorDesc input_desc, output_desc;
  FilterDesc filter_desc;
  ConvDesc conv_desc;

  CUDNN_CHECK(
      input_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W));
  CUDNN_CHECK(filter_desc.set_4d(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out,
                                 C_in, K_h, K_w));
  CUDNN_CHECK(conv_desc.set_2d(pad_h, pad_w, stride_h, stride_w, 1, 1,
                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(output_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out,
                                 H_out, W_out));

  // Find best algorithm
  cudnnConvolutionFwdAlgo_t algo;
  int returned_count;
  cudnnConvolutionFwdAlgoPerf_t perf_results;

  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
      output_desc.get(), 1, &returned_count, &perf_results));

  algo = perf_results.algo;

  // Perform convolution
  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      handle, &alpha, input_desc.get(), input, filter_desc.get(), weight,
      conv_desc.get(), algo, workspace, workspace_size, &beta,
      output_desc.get(), output));

  // Add bias if provided
  if (bias != nullptr) {
    TensorDesc bias_desc;
    CUDNN_CHECK(
        bias_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
    alpha = 1.0f;
    beta = 1.0f;
    CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc.get(), bias, &beta,
                               output_desc.get(), output));
  }

  return Status::Ok();
}

/// Get workspace size required for convolution
inline Status conv2d_get_workspace_size(int N, int C_in, int H, int W,
                                        int C_out, int K_h, int K_w,
                                        int stride_h, int stride_w, int pad_h,
                                        int pad_w, size_t *workspace_size) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  TensorDesc input_desc, output_desc;
  FilterDesc filter_desc;
  ConvDesc conv_desc;

  CUDNN_CHECK(
      input_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W));
  CUDNN_CHECK(filter_desc.set_4d(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out,
                                 C_in, K_h, K_w));
  CUDNN_CHECK(conv_desc.set_2d(pad_h, pad_w, stride_h, stride_w, 1, 1,
                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(output_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out,
                                 H_out, W_out));

  cudnnConvolutionFwdAlgo_t algo =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
      output_desc.get(), algo, workspace_size));

  return Status::Ok();
}

// ============================================================================
// Activation Operations
// ============================================================================

/// ReLU activation using cuDNN
inline Status relu_forward(const float *input, float *output, int N, int C,
                           int H, int W) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  TensorDesc desc;
  ActivationDesc act_desc;

  CUDNN_CHECK(desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CUDNN_CHECK(
      act_desc.set(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnActivationForward(handle, act_desc.get(), &alpha, desc.get(),
                                     input, &beta, desc.get(), output));

  return Status::Ok();
}

/// Sigmoid activation using cuDNN
inline Status sigmoid_forward(const float *input, float *output, int N, int C,
                              int H, int W) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  TensorDesc desc;
  ActivationDesc act_desc;

  CUDNN_CHECK(desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CUDNN_CHECK(
      act_desc.set(CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnActivationForward(handle, act_desc.get(), &alpha, desc.get(),
                                     input, &beta, desc.get(), output));

  return Status::Ok();
}

/// Tanh activation using cuDNN
inline Status tanh_forward(const float *input, float *output, int N, int C,
                           int H, int W) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  TensorDesc desc;
  ActivationDesc act_desc;

  CUDNN_CHECK(desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CUDNN_CHECK(
      act_desc.set(CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnActivationForward(handle, act_desc.get(), &alpha, desc.get(),
                                     input, &beta, desc.get(), output));

  return Status::Ok();
}

// ============================================================================
// Pooling Operations
// ============================================================================

/// Max Pooling 2D using cuDNN
inline Status maxpool2d_forward(const float *input, float *output, int N, int C,
                                int H, int W, int K_h, int K_w, int stride_h,
                                int stride_w, int pad_h, int pad_w) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  TensorDesc input_desc, output_desc;
  PoolingDesc pool_desc;

  CUDNN_CHECK(
      input_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CUDNN_CHECK(output_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C,
                                 H_out, W_out));
  CUDNN_CHECK(pool_desc.set_2d(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, K_h,
                               K_w, pad_h, pad_w, stride_h, stride_w));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc.get(), &alpha,
                                  input_desc.get(), input, &beta,
                                  output_desc.get(), output));

  return Status::Ok();
}

/// Average Pooling 2D using cuDNN
inline Status avgpool2d_forward(const float *input, float *output, int N, int C,
                                int H, int W, int K_h, int K_w, int stride_h,
                                int stride_w, int pad_h, int pad_w) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  TensorDesc input_desc, output_desc;
  PoolingDesc pool_desc;

  CUDNN_CHECK(
      input_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CUDNN_CHECK(output_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C,
                                 H_out, W_out));
  CUDNN_CHECK(pool_desc.set_2d(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                               CUDNN_NOT_PROPAGATE_NAN, K_h, K_w, pad_h, pad_w,
                               stride_h, stride_w));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc.get(), &alpha,
                                  input_desc.get(), input, &beta,
                                  output_desc.get(), output));

  return Status::Ok();
}

// ============================================================================
// Softmax Operations
// ============================================================================

/// Softmax forward using cuDNN
inline Status softmax_forward(const float *input, float *output, int N, int C,
                              int H, int W, bool log_softmax = false) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  TensorDesc desc;
  CUDNN_CHECK(desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

  cudnnSoftmaxAlgorithm_t algo =
      log_softmax ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnSoftmaxForward(handle, algo, CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha, desc.get(), input, &beta, desc.get(),
                                  output));

  return Status::Ok();
}

// ============================================================================
// BatchNorm Operations
// ============================================================================

/// Batch Normalization forward (inference mode) using cuDNN
inline Status batchnorm_forward_inference(const float *input, float *output,
                                          const float *scale, const float *bias,
                                          const float *mean, const float *var,
                                          int N, int C, int H, int W,
                                          double epsilon) {
  cudnnHandle_t handle = CudnnHandle::instance().get();
  if (!handle) {
    return Status::Error(StatusCode::NotFound, "cuDNN not initialized");
  }

  TensorDesc input_desc, bn_desc;
  CUDNN_CHECK(
      input_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CUDNN_CHECK(bn_desc.set_4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, input_desc.get(), input,
      input_desc.get(), output, bn_desc.get(), scale, bias, mean, var,
      epsilon));

  return Status::Ok();
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if cuDNN is available
inline bool is_cudnn_available() {
  return CudnnHandle::instance().get() != nullptr;
}

/// Get cuDNN version
inline int get_cudnn_version() { return static_cast<int>(cudnnGetVersion()); }

} // namespace cudnn
} // namespace zenith

#else // !ZENITH_HAS_CUDNN

namespace zenith {
namespace cudnn {

inline bool is_cudnn_available() { return false; }
inline int get_cudnn_version() { return 0; }

inline Status conv2d_forward(const float *, const float *, const float *,
                             float *, int, int, int, int, int, int, int, int,
                             int, int, int, void *, size_t) {
  return Status::Error(StatusCode::NotImplemented,
                       "cuDNN not compiled - enable ZENITH_HAS_CUDNN");
}

inline Status relu_forward(const float *, float *, int, int, int, int) {
  return Status::Error(StatusCode::NotImplemented,
                       "cuDNN not compiled - enable ZENITH_HAS_CUDNN");
}

} // namespace cudnn
} // namespace zenith

#endif // ZENITH_HAS_CUDNN

#else // !ZENITH_HAS_CUDA

namespace zenith {
namespace cudnn {

inline bool is_cudnn_available() { return false; }
inline int get_cudnn_version() { return 0; }

} // namespace cudnn
} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_CUDNN_OPS_HPP
