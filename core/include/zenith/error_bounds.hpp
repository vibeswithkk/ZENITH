// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Error Bounds Verification untuk Numerical Stability
// Berdasarkan CetakBiru.md Bab 4.1: Jaminan Stabilitas Numerik
// Referensi: TensorRT precision validation, ONNX Runtime accuracy verification

#ifndef ZENITH_ERROR_BOUNDS_HPP
#define ZENITH_ERROR_BOUNDS_HPP

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace zenith {
namespace verification {

// ============================================================================
// Error Tolerance Configuration
// ============================================================================

/// Konfigurasi toleransi error berdasarkan CetakBiru formula
struct ErrorTolerance {
  /// Relative error tolerance (δ) - default dari CetakBiru
  float delta_fp32 = 1e-6f; // FP32 tolerance
  float delta_fp16 = 1e-3f; // FP16 tolerance
  float delta_int8 = 5e-2f; // INT8 tolerance (lebih besar)

  /// Epsilon untuk menghindari division by zero
  float epsilon = 1e-10f;

  /// Gradient fidelity coefficient (γ) - dari CetakBiru
  float gamma = 0.001f;

  /// Dynamic range safety margin (α) - dari CetakBiru
  float alpha = 0.95f;

  /// Get tolerance berdasarkan data type
  [[nodiscard]] float get_tolerance(DataType dtype) const {
    switch (dtype) {
    case DataType::Float32:
      return delta_fp32;
    case DataType::Float16:
    case DataType::BFloat16:
      return delta_fp16;
    case DataType::Int8:
    case DataType::UInt8:
      return delta_int8;
    default:
      return delta_fp32;
    }
  }
};

// ============================================================================
// Error Metrics
// ============================================================================

/// Metrik error untuk satu perbandingan
struct ErrorMetrics {
  /// Maximum absolute error
  float max_abs_error = 0.0f;

  /// Mean absolute error
  float mean_abs_error = 0.0f;

  /// Maximum relative error (formula dari CetakBiru)
  float max_rel_error = 0.0f;

  /// Mean relative error
  float mean_rel_error = 0.0f;

  /// Root mean square error
  float rmse = 0.0f;

  /// Signal-to-noise ratio (dB)
  float snr_db = 0.0f;

  /// Peak signal-to-noise ratio (dB)
  float psnr_db = 0.0f;

  /// Cosine similarity
  float cosine_similarity = 1.0f;

  /// Jumlah elemen yang melebihi toleransi
  size_t num_violations = 0;

  /// Total elemen yang dibandingkan
  size_t total_elements = 0;

  /// Apakah semua elemen dalam toleransi
  bool within_tolerance = true;

  /// Index elemen dengan error terbesar
  size_t max_error_index = 0;

  /// Nilai reference di max error
  float max_error_reference = 0.0f;

  /// Nilai optimized di max error
  float max_error_optimized = 0.0f;
};

// ============================================================================
// Verification Result
// ============================================================================

/// Hasil verifikasi lengkap
struct VerificationResult {
  bool passed = false;
  std::string message;

  /// Tolerance yang digunakan
  float tolerance_used = 0.0f;

  /// Metrics detail
  ErrorMetrics metrics;

  /// Summary string
  [[nodiscard]] std::string summary() const {
    std::ostringstream ss;
    ss << (passed ? "[PASS]" : "[FAIL]") << " ";
    ss << "max_rel_err=" << metrics.max_rel_error;
    ss << " (tol=" << tolerance_used << ")";
    ss << " violations=" << metrics.num_violations;
    ss << "/" << metrics.total_elements;
    return ss.str();
  }
};

// ============================================================================
// Error Bounds Verifier
// ============================================================================

/// Verifier utama untuk error bounds
/// Implementasi formula dari CetakBiru:
/// |T(F)(x) - F(x)| / (|F(x)| + ε) ≤ δ
class ErrorBoundsVerifier {
public:
  explicit ErrorBoundsVerifier(ErrorTolerance config = {})
      : config_(std::move(config)) {}

  /// Set konfigurasi
  void set_config(ErrorTolerance config) { config_ = std::move(config); }

  /// Verifikasi dua tensor float
  VerificationResult verify(const float *reference, const float *optimized,
                            size_t size, DataType dtype = DataType::Float32) {
    VerificationResult result;
    result.tolerance_used = config_.get_tolerance(dtype);

    if (size == 0) {
      result.passed = true;
      result.message = "Empty tensor - trivially passed";
      return result;
    }

    ErrorMetrics &m = result.metrics;
    m.total_elements = size;

    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;
    double sum_sq_error = 0.0;
    double sum_ref_sq = 0.0;
    double sum_opt_sq = 0.0;
    double dot_product = 0.0;

    float ref_max = std::numeric_limits<float>::lowest();
    float ref_min = std::numeric_limits<float>::max();

    for (size_t i = 0; i < size; ++i) {
      float ref = reference[i];
      float opt = optimized[i];

      // Update range
      ref_max = std::max(ref_max, ref);
      ref_min = std::min(ref_min, ref);

      // Absolute error
      float abs_err = std::abs(opt - ref);

      // Relative error (formula CetakBiru)
      // |T(F)(x) - F(x)| / (|F(x)| + ε)
      float rel_err = abs_err / (std::abs(ref) + config_.epsilon);

      // Update metrics
      sum_abs_error += abs_err;
      sum_rel_error += rel_err;
      sum_sq_error += static_cast<double>(abs_err) * abs_err;
      sum_ref_sq += static_cast<double>(ref) * ref;
      sum_opt_sq += static_cast<double>(opt) * opt;
      dot_product += static_cast<double>(ref) * opt;

      // Track max absolute error
      if (abs_err > m.max_abs_error) {
        m.max_abs_error = abs_err;
        m.max_error_index = i;
        m.max_error_reference = ref;
        m.max_error_optimized = opt;
      }

      // Track max relative error
      if (rel_err > m.max_rel_error) {
        m.max_rel_error = rel_err;
      }

      // Count violations
      if (rel_err > result.tolerance_used) {
        m.num_violations++;
      }
    }

    // Compute averages
    m.mean_abs_error = static_cast<float>(sum_abs_error / size);
    m.mean_rel_error = static_cast<float>(sum_rel_error / size);
    m.rmse = static_cast<float>(std::sqrt(sum_sq_error / size));

    // SNR (signal-to-noise ratio)
    if (sum_sq_error > 1e-20) {
      m.snr_db =
          static_cast<float>(10.0 * std::log10(sum_ref_sq / sum_sq_error));
    } else {
      m.snr_db = std::numeric_limits<float>::infinity();
    }

    // PSNR (peak SNR)
    float peak = std::max(std::abs(ref_max), std::abs(ref_min));
    if (m.rmse > 1e-10f && peak > 1e-10f) {
      m.psnr_db = 20.0f * std::log10(peak / m.rmse);
    } else {
      m.psnr_db = std::numeric_limits<float>::infinity();
    }

    // Cosine similarity
    double norm_ref = std::sqrt(sum_ref_sq);
    double norm_opt = std::sqrt(sum_opt_sq);
    if (norm_ref > 1e-10 && norm_opt > 1e-10) {
      m.cosine_similarity =
          static_cast<float>(dot_product / (norm_ref * norm_opt));
    }

    // Final judgment
    m.within_tolerance = (m.num_violations == 0);
    result.passed = m.within_tolerance;

    if (result.passed) {
      result.message = "All elements within tolerance";
    } else {
      std::ostringstream ss;
      ss << m.num_violations << " of " << size
         << " elements exceed tolerance. Max rel error: " << m.max_rel_error
         << " at index " << m.max_error_index;
      result.message = ss.str();
    }

    return result;
  }

  /// Verifikasi dengan toleransi absolut (alternatif)
  VerificationResult verify_absolute(const float *reference,
                                     const float *optimized, size_t size,
                                     float abs_tolerance) {
    VerificationResult result;
    result.tolerance_used = abs_tolerance;

    if (size == 0) {
      result.passed = true;
      return result;
    }

    ErrorMetrics &m = result.metrics;
    m.total_elements = size;

    for (size_t i = 0; i < size; ++i) {
      float abs_err = std::abs(optimized[i] - reference[i]);

      if (abs_err > m.max_abs_error) {
        m.max_abs_error = abs_err;
        m.max_error_index = i;
        m.max_error_reference = reference[i];
        m.max_error_optimized = optimized[i];
      }

      if (abs_err > abs_tolerance) {
        m.num_violations++;
      }
    }

    m.within_tolerance = (m.num_violations == 0);
    result.passed = m.within_tolerance;
    result.message = result.passed ? "All elements within absolute tolerance"
                                   : "Elements exceed absolute tolerance";

    return result;
  }

  /// Verifikasi gradient fidelity (formula dari CetakBiru)
  /// ||∇̂L(θ) - ∇L(θ)||₂ ≤ γ · ||∇L(θ)||₂
  VerificationResult verify_gradient(const float *reference_grad,
                                     const float *optimized_grad, size_t size) {
    VerificationResult result;
    result.tolerance_used = config_.gamma;

    if (size == 0) {
      result.passed = true;
      return result;
    }

    // Compute L2 norm of difference
    double diff_sq_sum = 0.0;
    double ref_sq_sum = 0.0;

    for (size_t i = 0; i < size; ++i) {
      float diff = optimized_grad[i] - reference_grad[i];
      diff_sq_sum += static_cast<double>(diff) * diff;
      ref_sq_sum += static_cast<double>(reference_grad[i]) * reference_grad[i];
    }

    double diff_norm = std::sqrt(diff_sq_sum);
    double ref_norm = std::sqrt(ref_sq_sum);

    result.metrics.total_elements = size;
    result.metrics.rmse = static_cast<float>(diff_norm / std::sqrt(size));

    // Check: ||∇̂L - ∇L||₂ ≤ γ · ||∇L||₂
    if (ref_norm < 1e-10) {
      // Reference gradient is near zero
      result.passed = diff_norm < 1e-10;
    } else {
      double relative_grad_error = diff_norm / ref_norm;
      result.metrics.max_rel_error = static_cast<float>(relative_grad_error);
      result.passed = relative_grad_error <= config_.gamma;
    }

    result.message = result.passed ? "Gradient fidelity check passed"
                                   : "Gradient deviation exceeds gamma";

    return result;
  }

private:
  ErrorTolerance config_;
};

// ============================================================================
// Dynamic Range Analyzer (untuk Mixed Precision Safety)
// ============================================================================

/// Analyzer untuk menentukan apakah data aman untuk lower precision
/// Berdasarkan CetakBiru formula:
/// max(|x|) < MaxBound(P) · α dan min(|x|) > MinBound(P) / α
class DynamicRangeAnalyzer {
public:
  struct RangeAnalysisResult {
    bool safe_for_fp16 = false;
    bool safe_for_bf16 = false;
    bool safe_for_int8 = false;

    float data_min = 0.0f;
    float data_max = 0.0f;
    float data_abs_max = 0.0f;
    float data_abs_min = 0.0f; // Non-zero minimum
    float dynamic_range = 0.0f;

    std::string recommendation;
  };

  explicit DynamicRangeAnalyzer(float safety_margin = 0.95f)
      : alpha_(safety_margin) {}

  RangeAnalysisResult analyze(const float *data, size_t size) {
    RangeAnalysisResult result;

    if (size == 0) {
      result.recommendation = "Empty data";
      return result;
    }

    // Find range
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float abs_max = 0.0f;
    float abs_min_nonzero = std::numeric_limits<float>::max();

    for (size_t i = 0; i < size; ++i) {
      float v = data[i];
      min_val = std::min(min_val, v);
      max_val = std::max(max_val, v);

      float abs_v = std::abs(v);
      abs_max = std::max(abs_max, abs_v);
      if (abs_v > 1e-30f) {
        abs_min_nonzero = std::min(abs_min_nonzero, abs_v);
      }
    }

    result.data_min = min_val;
    result.data_max = max_val;
    result.data_abs_max = abs_max;
    result.data_abs_min = abs_min_nonzero;

    // Dynamic range in dB
    if (abs_min_nonzero > 0 && abs_max > 0) {
      result.dynamic_range = 20.0f * std::log10(abs_max / abs_min_nonzero);
    }

    // FP16 bounds: max ~65504, min subnormal ~5.96e-8
    constexpr float FP16_MAX = 65504.0f;
    constexpr float FP16_MIN_SUBNORMAL = 5.96e-8f;

    // BF16 bounds: max ~3.39e38, min subnormal ~1.18e-38 (same as FP32)
    constexpr float BF16_MAX = 3.39e38f;
    constexpr float BF16_MIN = 1.18e-38f;

    // INT8: need to check if data fits well
    // Typically requires abs_max < ~127 after scaling

    // Check FP16 safety (formula dari CetakBiru)
    // max(|x|) < MaxBound(FP16) · α dan min(|x|) > MinBound(FP16) / α
    bool fp16_max_ok = abs_max < FP16_MAX * alpha_;
    bool fp16_min_ok = abs_min_nonzero > FP16_MIN_SUBNORMAL / alpha_ ||
                       abs_min_nonzero == std::numeric_limits<float>::max();
    result.safe_for_fp16 = fp16_max_ok && fp16_min_ok;

    // Check BF16 safety
    bool bf16_max_ok = abs_max < BF16_MAX * alpha_;
    bool bf16_min_ok = abs_min_nonzero > BF16_MIN / alpha_ ||
                       abs_min_nonzero == std::numeric_limits<float>::max();
    result.safe_for_bf16 = bf16_max_ok && bf16_min_ok;

    // INT8 safety: dynamic range < 127/1 = ~42 dB typical
    result.safe_for_int8 = result.dynamic_range < 40.0f && abs_max < 1e6f;

    // Recommendation
    std::ostringstream ss;
    ss << "Range: [" << min_val << ", " << max_val << "] ";
    ss << "DR: " << result.dynamic_range << "dB. ";

    if (result.safe_for_int8) {
      ss << "Recommended: INT8";
    } else if (result.safe_for_fp16) {
      ss << "Recommended: FP16";
    } else if (result.safe_for_bf16) {
      ss << "Recommended: BF16";
    } else {
      ss << "Recommended: FP32 (data has extreme range)";
    }

    result.recommendation = ss.str();
    return result;
  }

private:
  float alpha_; // Safety margin
};

// ============================================================================
// Batch Verifier (untuk multiple tensors)
// ============================================================================

/// Verifier untuk verifikasi batch
class BatchVerifier {
public:
  struct TensorVerification {
    std::string tensor_name;
    VerificationResult result;
  };

  struct BatchResult {
    bool all_passed = true;
    size_t num_passed = 0;
    size_t num_failed = 0;
    std::vector<TensorVerification> tensor_results;
    std::string summary;
  };

  explicit BatchVerifier(ErrorTolerance config = {}) : verifier_(config) {}

  /// Tambah tensor untuk verifikasi
  void add_tensor(const std::string &name, const float *reference,
                  const float *optimized, size_t size,
                  DataType dtype = DataType::Float32) {
    pending_.push_back({name, reference, optimized, size, dtype});
  }

  /// Jalankan semua verifikasi
  BatchResult verify_all() {
    BatchResult result;

    for (const auto &item : pending_) {
      TensorVerification tv;
      tv.tensor_name = item.name;
      tv.result = verifier_.verify(item.reference, item.optimized, item.size,
                                   item.dtype);

      if (tv.result.passed) {
        result.num_passed++;
      } else {
        result.num_failed++;
        result.all_passed = false;
      }

      result.tensor_results.push_back(tv);
    }

    // Summary
    std::ostringstream ss;
    ss << result.num_passed << "/" << (result.num_passed + result.num_failed)
       << " tensors passed verification";
    result.summary = ss.str();

    pending_.clear();
    return result;
  }

private:
  struct PendingVerification {
    std::string name;
    const float *reference;
    const float *optimized;
    size_t size;
    DataType dtype;
  };

  ErrorBoundsVerifier verifier_;
  std::vector<PendingVerification> pending_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Quick verification dengan default tolerances
inline bool quick_verify(const float *reference, const float *optimized,
                         size_t size, DataType dtype = DataType::Float32) {
  ErrorBoundsVerifier verifier;
  return verifier.verify(reference, optimized, size, dtype).passed;
}

/// Check if precision conversion is safe
inline bool is_safe_for_precision(const float *data, size_t size,
                                  DataType target_dtype) {
  DynamicRangeAnalyzer analyzer;
  auto result = analyzer.analyze(data, size);

  switch (target_dtype) {
  case DataType::Float16:
    return result.safe_for_fp16;
  case DataType::BFloat16:
    return result.safe_for_bf16;
  case DataType::Int8:
  case DataType::UInt8:
    return result.safe_for_int8;
  default:
    return true;
  }
}

} // namespace verification
} // namespace zenith

#endif // ZENITH_ERROR_BOUNDS_HPP
