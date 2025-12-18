// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Quantization Types and Utilities
// Berdasarkan CetakBiru.md Bab 4.1 tentang kuantisasi dengan kalibrasi
// Referensi: TensorRT INT8 Quantization, ONNX Quantization

#ifndef ZENITH_QUANTIZATION_TYPES_HPP
#define ZENITH_QUANTIZATION_TYPES_HPP

#include "../types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace zenith {
namespace quantization {

// ============================================================================
// Quantization Parameters
// ============================================================================

/// Parameter kuantisasi untuk satu tensor
/// Mengikuti formula dari CetakBiru:
/// Q = round(X / scale + zero_point)
/// X' = scale * (Q - zero_point)
struct QuantizationParams {
  float scale = 1.0f;     // Scaling factor
  int32_t zero_point = 0; // Zero point offset

  // Range untuk INT8
  int32_t qmin = -128;
  int32_t qmax = 127;

  /// Quantize nilai float ke int8
  [[nodiscard]] int8_t quantize(float value) const {
    int32_t q = static_cast<int32_t>(std::round(value / scale)) + zero_point;
    q = std::clamp(q, qmin, qmax);
    return static_cast<int8_t>(q);
  }

  /// Dequantize nilai int8 ke float
  [[nodiscard]] float dequantize(int8_t quantized) const {
    return scale *
           (static_cast<float>(quantized) - static_cast<float>(zero_point));
  }

  /// Quantize array float ke array int8
  void quantize_array(const float *input, int8_t *output, size_t size) const {
    for (size_t i = 0; i < size; ++i) {
      output[i] = quantize(input[i]);
    }
  }

  /// Dequantize array int8 ke array float
  void dequantize_array(const int8_t *input, float *output, size_t size) const {
    for (size_t i = 0; i < size; ++i) {
      output[i] = dequantize(input[i]);
    }
  }

  /// Hitung quantization error
  [[nodiscard]] float compute_error(const float *original, size_t size) const {
    float total_error = 0.0f;
    for (size_t i = 0; i < size; ++i) {
      int8_t q = quantize(original[i]);
      float reconstructed = dequantize(q);
      float diff = original[i] - reconstructed;
      total_error += diff * diff;
    }
    return std::sqrt(total_error / static_cast<float>(size));
  }
};

// ============================================================================
// Calibration Method Enum
// ============================================================================

/// Metode kalibrasi untuk menentukan parameter kuantisasi
/// Berdasarkan CetakBiru: entropy minimization dan percentile matching
enum class CalibrationMethod {
  MinMax,     // Simple min-max (cepat tapi sensitif terhadap outliers)
  Entropy,    // KL divergence minimization (akurat, lambat)
  Percentile, // Percentile-based (99.99th percentile, robust)
  MSE,        // Mean Squared Error minimization
};

// ============================================================================
// Quantization Mode Enum
// ============================================================================

/// Mode kuantisasi
enum class QuantizationMode {
  PerTensor,  // Satu scale/zero_point per tensor
  PerChannel, // Satu scale/zero_point per output channel
};

// ============================================================================
// Quantization Info untuk satu layer
// ============================================================================

/// Informasi kuantisasi lengkap untuk satu layer/tensor
struct QuantizationInfo {
  std::string tensor_name;
  QuantizationParams params;
  QuantizationMode mode = QuantizationMode::PerTensor;
  CalibrationMethod method_used = CalibrationMethod::MinMax;

  // Untuk per-channel quantization
  std::vector<QuantizationParams> channel_params;

  // Statistik dari kalibrasi
  float min_value = 0.0f;
  float max_value = 0.0f;
  float mean_value = 0.0f;
  float std_value = 0.0f;

  // Apakah ini weight atau activation
  bool is_weight = false;
};

// ============================================================================
// Histogram untuk Kalibrasi Entropy
// ============================================================================

/// Histogram untuk mengumpulkan distribusi nilai
class Histogram {
public:
  explicit Histogram(int num_bins = 2048, float min_val = -10.0f,
                     float max_val = 10.0f)
      : num_bins_(num_bins), min_val_(min_val), max_val_(max_val),
        bin_width_((max_val - min_val) / static_cast<float>(num_bins)),
        bins_(static_cast<size_t>(num_bins), 0) {}

  /// Reset histogram
  void reset() {
    std::fill(bins_.begin(), bins_.end(), 0);
    total_count_ = 0;
  }

  /// Tambahkan satu nilai ke histogram
  void add(float value) {
    int bin = get_bin(value);
    if (bin >= 0 && bin < num_bins_) {
      bins_[static_cast<size_t>(bin)]++;
      total_count_++;
    }
  }

  /// Tambahkan array nilai ke histogram
  void add_array(const float *values, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      add(values[i]);
    }
  }

  /// Dapatkan bin untuk nilai tertentu
  [[nodiscard]] int get_bin(float value) const {
    if (value < min_val_)
      return 0;
    if (value >= max_val_)
      return num_bins_ - 1;
    return static_cast<int>((value - min_val_) / bin_width_);
  }

  /// Dapatkan nilai tengah bin
  [[nodiscard]] float get_bin_center(int bin) const {
    return min_val_ + (static_cast<float>(bin) + 0.5f) * bin_width_;
  }

  /// Dapatkan jumlah di bin tertentu
  [[nodiscard]] int64_t count(int bin) const {
    if (bin < 0 || bin >= num_bins_)
      return 0;
    return bins_[static_cast<size_t>(bin)];
  }

  [[nodiscard]] int num_bins() const { return num_bins_; }
  [[nodiscard]] float min_val() const { return min_val_; }
  [[nodiscard]] float max_val() const { return max_val_; }
  [[nodiscard]] int64_t total_count() const { return total_count_; }
  [[nodiscard]] const std::vector<int64_t> &bins() const { return bins_; }

  /// Hitung KL divergence antara histogram ini dan referensi
  [[nodiscard]] double kl_divergence(const Histogram &reference) const {
    double kl = 0.0;
    for (int i = 0; i < num_bins_; ++i) {
      double p =
          static_cast<double>(count(i)) / static_cast<double>(total_count_ + 1);
      double q = static_cast<double>(reference.count(i)) /
                 static_cast<double>(reference.total_count() + 1);
      if (p > 1e-10 && q > 1e-10) {
        kl += p * std::log(p / q);
      }
    }
    return kl;
  }

private:
  int num_bins_;
  float min_val_;
  float max_val_;
  float bin_width_;
  std::vector<int64_t> bins_;
  int64_t total_count_ = 0;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Hitung parameter kuantisasi dengan metode MinMax
inline QuantizationParams compute_minmax_params(float min_val, float max_val,
                                                bool symmetric = true) {
  QuantizationParams params;

  if (symmetric) {
    // Symmetric quantization: zero_point = 0
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    params.scale = abs_max / 127.0f;
    params.zero_point = 0;
  } else {
    // Asymmetric quantization
    params.scale = (max_val - min_val) / 255.0f;
    params.zero_point =
        static_cast<int32_t>(std::round(-128.0f - min_val / params.scale));
    params.zero_point = std::clamp(params.zero_point, -128, 127);
  }

  // Hindari scale = 0
  if (params.scale < 1e-10f) {
    params.scale = 1e-10f;
  }

  return params;
}

/// Hitung min dan max dari array
inline std::pair<float, float> compute_minmax(const float *data, size_t size) {
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < size; ++i) {
    min_val = std::min(min_val, data[i]);
    max_val = std::max(max_val, data[i]);
  }

  return {min_val, max_val};
}

/// Hitung percentile dari sorted array
inline float compute_percentile(const float *sorted_data, size_t size,
                                float percentile) {
  if (size == 0)
    return 0.0f;

  float idx = percentile * static_cast<float>(size - 1) / 100.0f;
  size_t lower_idx = static_cast<size_t>(std::floor(idx));
  size_t upper_idx = static_cast<size_t>(std::ceil(idx));

  if (lower_idx == upper_idx) {
    return sorted_data[lower_idx];
  }

  float frac = idx - static_cast<float>(lower_idx);
  return sorted_data[lower_idx] * (1.0f - frac) + sorted_data[upper_idx] * frac;
}

} // namespace quantization
} // namespace zenith

#endif // ZENITH_QUANTIZATION_TYPES_HPP
