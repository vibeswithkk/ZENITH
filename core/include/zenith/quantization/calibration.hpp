// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Calibration Utilities untuk INT8 Quantization
// Berdasarkan CetakBiru.md: Kalibrasi berbasis distribusi dengan KL divergence
// Referensi: TensorRT Entropy Calibrator, NVIDIA TensorFlow Quantization
// Toolkit

#ifndef ZENITH_QUANTIZATION_CALIBRATION_HPP
#define ZENITH_QUANTIZATION_CALIBRATION_HPP

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace zenith {
namespace quantization {

// ============================================================================
// Base Calibrator Class
// ============================================================================

/// Base class untuk semua calibrator
class Calibrator {
public:
  virtual ~Calibrator() = default;

  /// Nama calibrator untuk logging
  [[nodiscard]] virtual std::string name() const = 0;

  /// Metode kalibrasi yang digunakan
  [[nodiscard]] virtual CalibrationMethod method() const = 0;

  /// Hitung parameter kuantisasi dari data
  [[nodiscard]] virtual QuantizationParams compute_params(const float *data,
                                                          size_t size) = 0;

  /// Reset state internal
  virtual void reset() {}
};

// ============================================================================
// MinMax Calibrator
// ============================================================================

/// Calibrator MinMax - metode paling sederhana
/// Menggunakan min dan max absolut dari data
class MinMaxCalibrator : public Calibrator {
public:
  explicit MinMaxCalibrator(bool symmetric = true) : symmetric_(symmetric) {}

  [[nodiscard]] std::string name() const override { return "MinMax"; }

  [[nodiscard]] CalibrationMethod method() const override {
    return CalibrationMethod::MinMax;
  }

  [[nodiscard]] QuantizationParams compute_params(const float *data,
                                                  size_t size) override {
    auto [min_val, max_val] = compute_minmax(data, size);
    return compute_minmax_params(min_val, max_val, symmetric_);
  }

private:
  bool symmetric_;
};

// ============================================================================
// Percentile Calibrator
// ============================================================================

/// Calibrator Percentile - robust terhadap outliers
/// Menggunakan percentile tertentu (default 99.99%) sebagai range
class PercentileCalibrator : public Calibrator {
public:
  explicit PercentileCalibrator(float percentile = 99.99f,
                                bool symmetric = true)
      : percentile_(percentile), symmetric_(symmetric) {}

  [[nodiscard]] std::string name() const override { return "Percentile"; }

  [[nodiscard]] CalibrationMethod method() const override {
    return CalibrationMethod::Percentile;
  }

  [[nodiscard]] QuantizationParams compute_params(const float *data,
                                                  size_t size) override {
    // Copy dan sort data
    std::vector<float> sorted(data, data + size);
    std::sort(sorted.begin(), sorted.end());

    // Hitung percentile
    float low_percentile = 100.0f - percentile_;
    float min_val = compute_percentile(sorted.data(), size, low_percentile);
    float max_val = compute_percentile(sorted.data(), size, percentile_);

    return compute_minmax_params(min_val, max_val, symmetric_);
  }

private:
  float percentile_;
  bool symmetric_;
};

// ============================================================================
// Entropy Calibrator (KL Divergence)
// ============================================================================

/// Calibrator Entropy - menggunakan KL divergence minimization
/// Berdasarkan CetakBiru: meminimalkan divergensi KL antara distribusi
/// float dan kuantisasi.
/// Referensi: TensorRT IInt8EntropyCalibrator
class EntropyCalibrator : public Calibrator {
public:
  explicit EntropyCalibrator(int num_bins = 2048, int num_iterations = 128)
      : num_bins_(num_bins), num_iterations_(num_iterations) {}

  [[nodiscard]] std::string name() const override { return "Entropy"; }

  [[nodiscard]] CalibrationMethod method() const override {
    return CalibrationMethod::Entropy;
  }

  [[nodiscard]] QuantizationParams compute_params(const float *data,
                                                  size_t size) override {
    if (size == 0) {
      return QuantizationParams{};
    }

    // Hitung min/max untuk range histogram
    auto [data_min, data_max] = compute_minmax(data, size);
    float abs_max = std::max(std::abs(data_min), std::abs(data_max));

    // Buat histogram dari data asli
    Histogram original_hist(num_bins_, -abs_max, abs_max);
    original_hist.add_array(data, size);

    // Cari threshold optimal dengan minimasi KL divergence
    float best_threshold = abs_max;
    double min_kl = std::numeric_limits<double>::max();

    // Iterasi melalui berbagai threshold (dari 128 bin ke num_bins)
    for (int num_quantized_bins = 128; num_quantized_bins <= num_bins_;
         num_quantized_bins += num_bins_ / num_iterations_) {

      float threshold = abs_max * static_cast<float>(num_quantized_bins) /
                        static_cast<float>(num_bins_);

      // Simulasi kuantisasi dengan threshold ini
      QuantizationParams test_params;
      test_params.scale = threshold / 127.0f;
      test_params.zero_point = 0;

      // Buat histogram dari data yang dikuantisasi lalu didekuantisasi
      Histogram quantized_hist(num_bins_, -abs_max, abs_max);

      for (size_t i = 0; i < size; ++i) {
        float original = data[i];
        // Clamp ke threshold
        float clamped = std::clamp(original, -threshold, threshold);
        // Quantize dan dequantize
        int8_t q = test_params.quantize(clamped);
        float reconstructed = test_params.dequantize(q);
        quantized_hist.add(reconstructed);
      }

      // Hitung KL divergence
      double kl = compute_kl_divergence(original_hist, quantized_hist);

      if (kl < min_kl) {
        min_kl = kl;
        best_threshold = threshold;
      }
    }

    // Kembalikan parameter dengan threshold optimal
    QuantizationParams params;
    params.scale = best_threshold / 127.0f;
    params.zero_point = 0; // Symmetric quantization

    // Hindari scale = 0
    if (params.scale < 1e-10f) {
      params.scale = 1e-10f;
    }

    return params;
  }

private:
  int num_bins_;
  int num_iterations_;

  /// Hitung KL divergence antara dua histogram
  double compute_kl_divergence(const Histogram &p, const Histogram &q) {
    double kl = 0.0;
    double p_total = static_cast<double>(p.total_count()) + 1e-10;
    double q_total = static_cast<double>(q.total_count()) + 1e-10;

    for (int i = 0; i < p.num_bins(); ++i) {
      double p_prob = static_cast<double>(p.count(i)) / p_total;
      double q_prob = static_cast<double>(q.count(i)) / q_total;

      // Avoid log(0) dengan epsilon
      p_prob = std::max(p_prob, 1e-10);
      q_prob = std::max(q_prob, 1e-10);

      if (p.count(i) > 0) {
        kl += p_prob * std::log(p_prob / q_prob);
      }
    }

    return kl;
  }
};

// ============================================================================
// MSE Calibrator
// ============================================================================

/// Calibrator MSE - meminimalkan mean squared error
class MSECalibrator : public Calibrator {
public:
  explicit MSECalibrator(int num_iterations = 100)
      : num_iterations_(num_iterations) {}

  [[nodiscard]] std::string name() const override { return "MSE"; }

  [[nodiscard]] CalibrationMethod method() const override {
    return CalibrationMethod::MSE;
  }

  [[nodiscard]] QuantizationParams compute_params(const float *data,
                                                  size_t size) override {
    if (size == 0) {
      return QuantizationParams{};
    }

    auto [data_min, data_max] = compute_minmax(data, size);
    float abs_max = std::max(std::abs(data_min), std::abs(data_max));

    // Search untuk scale optimal
    float best_scale = abs_max / 127.0f;
    float min_mse = std::numeric_limits<float>::max();

    for (int i = 1; i <= num_iterations_; ++i) {
      float ratio = static_cast<float>(i) / static_cast<float>(num_iterations_);
      float threshold = abs_max * ratio;
      float test_scale = threshold / 127.0f;

      if (test_scale < 1e-10f)
        continue;

      QuantizationParams test_params;
      test_params.scale = test_scale;
      test_params.zero_point = 0;

      // Hitung MSE
      float mse = test_params.compute_error(data, size);

      if (mse < min_mse) {
        min_mse = mse;
        best_scale = test_scale;
      }
    }

    QuantizationParams params;
    params.scale = best_scale;
    params.zero_point = 0;

    if (params.scale < 1e-10f) {
      params.scale = 1e-10f;
    }

    return params;
  }

private:
  int num_iterations_;
};

// ============================================================================
// Calibrator Factory
// ============================================================================

/// Factory untuk membuat calibrator berdasarkan metode
inline std::unique_ptr<Calibrator> create_calibrator(CalibrationMethod method) {
  switch (method) {
  case CalibrationMethod::MinMax:
    return std::make_unique<MinMaxCalibrator>();
  case CalibrationMethod::Percentile:
    return std::make_unique<PercentileCalibrator>();
  case CalibrationMethod::Entropy:
    return std::make_unique<EntropyCalibrator>();
  case CalibrationMethod::MSE:
    return std::make_unique<MSECalibrator>();
  default:
    return std::make_unique<MinMaxCalibrator>();
  }
}

} // namespace quantization
} // namespace zenith

#endif // ZENITH_QUANTIZATION_CALIBRATION_HPP
