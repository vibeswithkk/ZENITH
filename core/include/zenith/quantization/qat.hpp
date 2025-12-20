// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Quantization-Aware Training (QAT) Kernels
// Berdasarkan CetakBiru.md Fase 3: Pipeline Kuantisasi INT8 Penuh
// Referensi: PyTorch FakeQuantize, TensorFlow QAT, Jacob et al. 2018

#ifndef ZENITH_QUANTIZATION_QAT_HPP
#define ZENITH_QUANTIZATION_QAT_HPP

#include "../types.hpp"
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

// CUDA support
#ifdef ZENITH_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace zenith {
namespace quantization {

// ============================================================================
// CUDA Function Declarations
// ============================================================================

#ifdef ZENITH_HAS_CUDA
extern "C" {

/// Fake quantize forward (per-tensor) - GPU
void fake_quantize_forward_cuda(const float *input, float *output, int size,
                                float scale, int32_t zero_point, int64_t qmin,
                                int64_t qmax, cudaStream_t stream = 0);

/// Fake quantize backward with STE (per-tensor) - GPU
void fake_quantize_backward_ste_cuda(const float *input,
                                     const float *grad_output,
                                     float *grad_input, int size, float scale,
                                     int32_t zero_point, int64_t qmin,
                                     int64_t qmax, cudaStream_t stream = 0);

/// Fake quantize forward (per-channel) - GPU
void fake_quantize_per_channel_forward_cuda(const float *input, float *output,
                                            const float *scales,
                                            const int32_t *zero_points,
                                            int num_channels, int channel_size,
                                            int64_t qmin, int64_t qmax,
                                            cudaStream_t stream = 0);

/// Fake quantize backward with STE (per-channel) - GPU
void fake_quantize_per_channel_backward_ste_cuda(
    const float *input, const float *grad_output, float *grad_input,
    const float *scales, const int32_t *zero_points, int num_channels,
    int channel_size, int64_t qmin, int64_t qmax, cudaStream_t stream = 0);

/// Min/max observation (per-tensor) - GPU
void observe_minmax_cuda(const float *input, float *min_out, float *max_out,
                         int size, cudaStream_t stream = 0);

/// Min/max observation (per-channel) - GPU
void observe_minmax_per_channel_cuda(const float *input, float *min_out,
                                     float *max_out, int num_channels,
                                     int channel_size, cudaStream_t stream = 0);

/// Batch normalization folding - GPU
void bn_fold_cuda(const float *weight, const float *bias, const float *bn_mean,
                  const float *bn_var, const float *bn_gamma,
                  const float *bn_beta, float *weight_out, float *bias_out,
                  int out_channels, int weight_per_channel, float epsilon,
                  cudaStream_t stream = 0);

/// Compute symmetric quantization parameters - GPU
void compute_symmetric_qparams_cuda(const float *min_vals,
                                    const float *max_vals, float *scales,
                                    int32_t *zero_points, int num_channels,
                                    int64_t qmax, cudaStream_t stream = 0);

/// Compute asymmetric quantization parameters - GPU
void compute_asymmetric_qparams_cuda(const float *min_vals,
                                     const float *max_vals, float *scales,
                                     int32_t *zero_points, int num_channels,
                                     int64_t qmin, int64_t qmax,
                                     cudaStream_t stream = 0);

} // extern "C"
#endif // ZENITH_HAS_CUDA

// ============================================================================
// QAT Configuration
// ============================================================================

/// Configuration for QAT training
struct QATConfig {
  int weight_bits = 8;
  int activation_bits = 8;
  bool symmetric_weights = true;
  bool symmetric_activations = true;
  bool per_channel_weights = true;
  float averaging_constant = 0.01f;
  int freeze_observers_after_epochs = 0;
  int start_qat_after_epochs = 0;
};

// ============================================================================
// Fake Quantization Operations
// ============================================================================

/// Compute quantization bounds for given bit width
inline void compute_qbounds(int num_bits, bool symmetric, int64_t &qmin,
                            int64_t &qmax) {
  if (symmetric) {
    qmin = -(1LL << (num_bits - 1));
    qmax = (1LL << (num_bits - 1)) - 1;
  } else {
    qmin = 0;
    qmax = (1LL << num_bits) - 1;
  }
}

/// Compute scale and zero_point from min/max values (symmetric)
inline void compute_symmetric_qparams(float min_val, float max_val,
                                      int num_bits, float &scale,
                                      int32_t &zero_point) {
  int64_t qmin, qmax;
  compute_qbounds(num_bits, true, qmin, qmax);

  float abs_max = std::max(std::abs(min_val), std::abs(max_val));
  scale = abs_max / static_cast<float>(qmax);
  scale = std::max(scale, 1e-8f); // Prevent division by zero
  zero_point = 0;
}

/// Compute scale and zero_point from min/max values (asymmetric)
inline void compute_asymmetric_qparams(float min_val, float max_val,
                                       int num_bits, float &scale,
                                       int32_t &zero_point) {
  int64_t qmin, qmax;
  compute_qbounds(num_bits, false, qmin, qmax);

  scale = (max_val - min_val) / static_cast<float>(qmax - qmin);
  scale = std::max(scale, 1e-8f);
  zero_point = static_cast<int32_t>(std::round(qmin - min_val / scale));
  zero_point = std::clamp(zero_point, static_cast<int32_t>(qmin),
                          static_cast<int32_t>(qmax));
}

// ============================================================================
// Fake Quantize Forward/Backward CPU Kernels
// ============================================================================

/// Fake quantize forward pass (CPU)
/// Implements: x_dq = dequant(quant(x, scale, zp), scale, zp)
inline void fake_quantize_forward_cpu(const float *input, float *output,
                                      size_t size, float scale,
                                      int32_t zero_point, int64_t qmin,
                                      int64_t qmax) {
  for (size_t i = 0; i < size; ++i) {
    // Quantize
    float scaled = input[i] / scale + static_cast<float>(zero_point);
    int64_t quantized = static_cast<int64_t>(std::round(scaled));
    quantized = std::clamp(quantized, qmin, qmax);

    // Dequantize
    output[i] =
        (static_cast<float>(quantized) - static_cast<float>(zero_point)) *
        scale;
  }
}

/// Fake quantize backward pass with STE (CPU)
/// Implements straight-through estimator: grad_input = grad_output * mask
/// where mask = 1 if x is within quantization range, 0 otherwise
inline void fake_quantize_backward_ste_cpu(const float *input,
                                           const float *grad_output,
                                           float *grad_input, size_t size,
                                           float scale, int32_t zero_point,
                                           int64_t qmin, int64_t qmax) {
  for (size_t i = 0; i < size; ++i) {
    // Compute raw quantized value (before clipping)
    float scaled = input[i] / scale + static_cast<float>(zero_point);
    int64_t quantized_raw = static_cast<int64_t>(std::round(scaled));

    // STE mask: 1 if within range, 0 otherwise
    float mask = (quantized_raw >= qmin && quantized_raw <= qmax) ? 1.0f : 0.0f;

    // Pass gradient through with mask
    grad_input[i] = grad_output[i] * mask;
  }
}

/// Per-channel fake quantize forward (CPU)
inline void fake_quantize_per_channel_forward_cpu(
    const float *input, float *output, const float *scales,
    const int32_t *zero_points, size_t num_channels, size_t channel_size,
    int64_t qmin, int64_t qmax) {
  for (size_t c = 0; c < num_channels; ++c) {
    float scale = scales[c];
    int32_t zp = zero_points[c];

    for (size_t i = 0; i < channel_size; ++i) {
      size_t idx = c * channel_size + i;

      // Quantize
      float scaled = input[idx] / scale + static_cast<float>(zp);
      int64_t quantized = static_cast<int64_t>(std::round(scaled));
      quantized = std::clamp(quantized, qmin, qmax);

      // Dequantize
      output[idx] =
          (static_cast<float>(quantized) - static_cast<float>(zp)) * scale;
    }
  }
}

// ============================================================================
// Observer for Statistics Collection
// ============================================================================

/// Observer that tracks min/max statistics for calibration
class MinMaxObserver {
public:
  MinMaxObserver(bool per_channel = false, int channel_axis = 0)
      : per_channel_(per_channel), channel_axis_(channel_axis),
        initialized_(false) {}

  /// Update statistics with new tensor
  void observe(const float *data, const Shape &shape) {
    if (per_channel_) {
      observe_per_channel(data, shape);
    } else {
      observe_per_tensor(data, shape);
    }
  }

  /// Get computed scale and zero_point (symmetric)
  void get_symmetric_qparams(std::vector<float> &scales,
                             std::vector<int32_t> &zero_points,
                             int num_bits = 8) const {
    scales.resize(min_vals_.size());
    zero_points.resize(min_vals_.size());

    for (size_t i = 0; i < min_vals_.size(); ++i) {
      compute_symmetric_qparams(min_vals_[i], max_vals_[i], num_bits, scales[i],
                                zero_points[i]);
    }
  }

  /// Get computed scale and zero_point (asymmetric)
  void get_asymmetric_qparams(std::vector<float> &scales,
                              std::vector<int32_t> &zero_points,
                              int num_bits = 8) const {
    scales.resize(min_vals_.size());
    zero_points.resize(min_vals_.size());

    for (size_t i = 0; i < min_vals_.size(); ++i) {
      compute_asymmetric_qparams(min_vals_[i], max_vals_[i], num_bits,
                                 scales[i], zero_points[i]);
    }
  }

  /// Reset observer state
  void reset() {
    min_vals_.clear();
    max_vals_.clear();
    initialized_ = false;
  }

  [[nodiscard]] bool is_initialized() const { return initialized_; }

private:
  bool per_channel_;
  int channel_axis_;
  bool initialized_;
  std::vector<float> min_vals_;
  std::vector<float> max_vals_;

  void observe_per_tensor(const float *data, const Shape &shape) {
    size_t numel = static_cast<size_t>(shape.numel());
    if (numel == 0)
      return;

    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 1; i < numel; ++i) {
      min_val = std::min(min_val, data[i]);
      max_val = std::max(max_val, data[i]);
    }

    if (!initialized_) {
      min_vals_ = {min_val};
      max_vals_ = {max_val};
      initialized_ = true;
    } else {
      min_vals_[0] = std::min(min_vals_[0], min_val);
      max_vals_[0] = std::max(max_vals_[0], max_val);
    }
  }

  void observe_per_channel(const float *data, const Shape &shape) {
    if (shape.rank() == 0)
      return;

    size_t num_channels = static_cast<size_t>(shape[channel_axis_]);
    size_t total_elems = static_cast<size_t>(shape.numel());
    size_t elems_per_channel = total_elems / num_channels;

    if (!initialized_) {
      min_vals_.resize(num_channels, std::numeric_limits<float>::max());
      max_vals_.resize(num_channels, std::numeric_limits<float>::lowest());
      initialized_ = true;
    }

    // Simplified: assumes channel_axis_ == 0
    for (size_t c = 0; c < num_channels; ++c) {
      for (size_t i = 0; i < elems_per_channel; ++i) {
        size_t idx = c * elems_per_channel + i;
        min_vals_[c] = std::min(min_vals_[c], data[idx]);
        max_vals_[c] = std::max(max_vals_[c], data[idx]);
      }
    }
  }
};

/// Moving average observer with exponential moving average
class MovingAverageObserver {
public:
  explicit MovingAverageObserver(float averaging_constant = 0.01f,
                                 bool per_channel = false)
      : averaging_constant_(averaging_constant), per_channel_(per_channel),
        initialized_(false) {}

  void observe(const float *data, const Shape &shape) {
    size_t numel = static_cast<size_t>(shape.numel());
    if (numel == 0)
      return;

    // Find current min/max
    float new_min = data[0];
    float new_max = data[0];
    for (size_t i = 1; i < numel; ++i) {
      new_min = std::min(new_min, data[i]);
      new_max = std::max(new_max, data[i]);
    }

    // Update with EMA
    if (!initialized_) {
      min_val_ = new_min;
      max_val_ = new_max;
      initialized_ = true;
    } else {
      min_val_ = (1.0f - averaging_constant_) * min_val_ +
                 averaging_constant_ * new_min;
      max_val_ = (1.0f - averaging_constant_) * max_val_ +
                 averaging_constant_ * new_max;
    }
  }

  void get_qparams(float &scale, int32_t &zero_point, bool symmetric = true,
                   int num_bits = 8) const {
    if (symmetric) {
      compute_symmetric_qparams(min_val_, max_val_, num_bits, scale,
                                zero_point);
    } else {
      compute_asymmetric_qparams(min_val_, max_val_, num_bits, scale,
                                 zero_point);
    }
  }

  void reset() {
    min_val_ = 0.0f;
    max_val_ = 0.0f;
    initialized_ = false;
  }

  [[nodiscard]] bool is_initialized() const { return initialized_; }
  [[nodiscard]] float min_val() const { return min_val_; }
  [[nodiscard]] float max_val() const { return max_val_; }

private:
  float averaging_constant_;
  bool per_channel_;
  bool initialized_;
  float min_val_ = 0.0f;
  float max_val_ = 0.0f;
};

// ============================================================================
// Batch Normalization Folding
// ============================================================================

/// Fold batch normalization into preceding convolution/linear layer
/// W_folded = gamma * W / sqrt(var + eps)
/// b_folded = gamma * (b - mean) / sqrt(var + eps) + beta
inline void fold_bn_into_conv(const float *weight, const float *bias,
                              const float *bn_mean, const float *bn_var,
                              const float *bn_gamma, const float *bn_beta,
                              float *weight_out, float *bias_out,
                              size_t out_channels, size_t weight_per_channel,
                              float epsilon = 1e-5f) {
  for (size_t c = 0; c < out_channels; ++c) {
    float std_val = std::sqrt(bn_var[c] + epsilon);
    float scale = bn_gamma[c] / std_val;

    // Fold weight
    for (size_t i = 0; i < weight_per_channel; ++i) {
      size_t idx = c * weight_per_channel + i;
      weight_out[idx] = weight[idx] * scale;
    }

    // Fold bias
    float orig_bias = (bias != nullptr) ? bias[c] : 0.0f;
    bias_out[c] = bn_gamma[c] * (orig_bias - bn_mean[c]) / std_val + bn_beta[c];
  }
}

// ============================================================================
// FakeQuantize Operator Class
// ============================================================================

/// Complete fake quantization operator with observer
class FakeQuantizeOp {
public:
  FakeQuantizeOp(int num_bits = 8, bool symmetric = true,
                 bool per_channel = false, float averaging_constant = 0.01f)
      : num_bits_(num_bits), symmetric_(symmetric), per_channel_(per_channel),
        observer_(averaging_constant, per_channel), fake_quant_enabled_(true),
        observer_enabled_(true) {
    compute_qbounds(num_bits, symmetric, qmin_, qmax_);
  }

  /// Forward pass with observation
  void forward(const float *input, float *output, const Shape &shape) {
    size_t numel = static_cast<size_t>(shape.numel());

    // Observe if enabled
    if (observer_enabled_) {
      observer_.observe(input, shape);
    }

    // Pass through if fake quant disabled or not calibrated
    if (!fake_quant_enabled_ || !observer_.is_initialized()) {
      std::copy(input, input + numel, output);
      return;
    }

    // Get quantization parameters
    float scale;
    int32_t zero_point;
    observer_.get_qparams(scale, zero_point, symmetric_, num_bits_);

    // Apply fake quantization
    fake_quantize_forward_cpu(input, output, numel, scale, zero_point, qmin_,
                              qmax_);
  }

  /// Backward pass with STE
  void backward(const float *input, const float *grad_output, float *grad_input,
                const Shape &shape) {
    size_t numel = static_cast<size_t>(shape.numel());

    if (!observer_.is_initialized()) {
      std::copy(grad_output, grad_output + numel, grad_input);
      return;
    }

    float scale;
    int32_t zero_point;
    observer_.get_qparams(scale, zero_point, symmetric_, num_bits_);

    fake_quantize_backward_ste_cpu(input, grad_output, grad_input, numel, scale,
                                   zero_point, qmin_, qmax_);
  }

  void enable_fake_quant() { fake_quant_enabled_ = true; }
  void disable_fake_quant() { fake_quant_enabled_ = false; }
  void enable_observer() { observer_enabled_ = true; }
  void disable_observer() { observer_enabled_ = false; }
  void reset() { observer_.reset(); }

  [[nodiscard]] bool is_calibrated() const {
    return observer_.is_initialized();
  }

  void get_qparams(float &scale, int32_t &zero_point) const {
    observer_.get_qparams(scale, zero_point, symmetric_, num_bits_);
  }

private:
  int num_bits_;
  bool symmetric_;
  bool per_channel_;
  int64_t qmin_;
  int64_t qmax_;
  MovingAverageObserver observer_;
  bool fake_quant_enabled_;
  bool observer_enabled_;
};

} // namespace quantization
} // namespace zenith

#endif // ZENITH_QUANTIZATION_QAT_HPP
