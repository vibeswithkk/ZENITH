// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Python Bindings untuk Features Baru
// Kernel Fusion, Auto-tuning, Pruning, Error Bounds
// Berdasarkan CetakBiru.md: Python User Interface (Thin Wrapper)

#ifndef ZENITH_PYTHON_FEATURES_BINDINGS_HPP
#define ZENITH_PYTHON_FEATURES_BINDINGS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include core headers
#include <zenith/types.hpp>

namespace py = pybind11;

namespace zenith {
namespace python {

// ============================================================================
// Fused Operations Bindings (CPU fallback implementations)
// GPU implementations akan call CUDA kernels melalui #ifdef ZENITH_HAS_CUDA
// ============================================================================

/// Fused Bias + ReLU: Y = max(0, X + bias)
inline py::array_t<float> fused_bias_relu_cpu(py::array_t<float> input,
                                              py::array_t<float> bias) {
  auto input_buf = input.request();
  auto bias_buf = bias.request();

  if (input_buf.ndim < 1 || bias_buf.ndim != 1) {
    throw std::runtime_error("Invalid input dimensions");
  }

  size_t features = static_cast<size_t>(bias_buf.shape[0]);
  size_t total = static_cast<size_t>(input_buf.size);
  size_t batch = total / features;

  auto result = py::array_t<float>(input_buf.shape);
  auto result_buf = result.request();

  float *in_ptr = static_cast<float *>(input_buf.ptr);
  float *bias_ptr = static_cast<float *>(bias_buf.ptr);
  float *out_ptr = static_cast<float *>(result_buf.ptr);

  for (size_t i = 0; i < total; ++i) {
    size_t feature_idx = i % features;
    float val = in_ptr[i] + bias_ptr[feature_idx];
    out_ptr[i] = val > 0.0f ? val : 0.0f;
  }

  return result;
}

/// Fused Bias + GELU
inline py::array_t<float> fused_bias_gelu_cpu(py::array_t<float> input,
                                              py::array_t<float> bias) {
  auto input_buf = input.request();
  auto bias_buf = bias.request();

  size_t features = static_cast<size_t>(bias_buf.shape[0]);
  size_t total = static_cast<size_t>(input_buf.size);

  auto result = py::array_t<float>(input_buf.shape);
  auto result_buf = result.request();

  float *in_ptr = static_cast<float *>(input_buf.ptr);
  float *bias_ptr = static_cast<float *>(bias_buf.ptr);
  float *out_ptr = static_cast<float *>(result_buf.ptr);

  constexpr float sqrt_2_over_pi = 0.7978845608028654f;
  constexpr float gelu_coef = 0.044715f;

  for (size_t i = 0; i < total; ++i) {
    size_t feature_idx = i % features;
    float x = in_ptr[i] + bias_ptr[feature_idx];
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + gelu_coef * x_cubed);
    out_ptr[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }

  return result;
}

/// Fused Add + ReLU: Y = max(0, X + residual)
inline py::array_t<float> fused_add_relu_cpu(py::array_t<float> x,
                                             py::array_t<float> residual) {
  auto x_buf = x.request();
  auto res_buf = residual.request();

  if (x_buf.size != res_buf.size) {
    throw std::runtime_error("Size mismatch");
  }

  size_t total = static_cast<size_t>(x_buf.size);

  auto result = py::array_t<float>(x_buf.shape);
  auto result_buf = result.request();

  float *x_ptr = static_cast<float *>(x_buf.ptr);
  float *res_ptr = static_cast<float *>(res_buf.ptr);
  float *out_ptr = static_cast<float *>(result_buf.ptr);

  for (size_t i = 0; i < total; ++i) {
    float val = x_ptr[i] + res_ptr[i];
    out_ptr[i] = val > 0.0f ? val : 0.0f;
  }

  return result;
}

/// Fused Add + LayerNorm
inline py::array_t<float> fused_add_layernorm_cpu(py::array_t<float> x,
                                                  py::array_t<float> residual,
                                                  py::array_t<float> gamma,
                                                  py::array_t<float> beta,
                                                  float eps = 1e-5f) {
  auto x_buf = x.request();
  auto res_buf = residual.request();
  auto gamma_buf = gamma.request();
  auto beta_buf = beta.request();

  if (x_buf.ndim != 2) {
    throw std::runtime_error("Expected 2D input");
  }

  int batch = static_cast<int>(x_buf.shape[0]);
  int hidden = static_cast<int>(x_buf.shape[1]);

  auto result = py::array_t<float>(x_buf.shape);
  auto result_buf = result.request();

  float *x_ptr = static_cast<float *>(x_buf.ptr);
  float *res_ptr = static_cast<float *>(res_buf.ptr);
  float *gamma_ptr = static_cast<float *>(gamma_buf.ptr);
  float *beta_ptr = static_cast<float *>(beta_buf.ptr);
  float *out_ptr = static_cast<float *>(result_buf.ptr);

  for (int b = 0; b < batch; ++b) {
    float *x_row = x_ptr + b * hidden;
    float *res_row = res_ptr + b * hidden;
    float *out_row = out_ptr + b * hidden;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < hidden; ++i) {
      sum += x_row[i] + res_row[i];
    }
    float mean = sum / static_cast<float>(hidden);

    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < hidden; ++i) {
      float val = x_row[i] + res_row[i] - mean;
      var_sum += val * val;
    }
    float var = var_sum / static_cast<float>(hidden);
    float inv_std = 1.0f / std::sqrt(var + eps);

    // Normalize
    for (int i = 0; i < hidden; ++i) {
      float normalized = (x_row[i] + res_row[i] - mean) * inv_std;
      out_row[i] = normalized * gamma_ptr[i] + beta_ptr[i];
    }
  }

  return result;
}

// ============================================================================
// Error Bounds Verification Bindings
// ============================================================================

/// Compute relative error
inline py::dict compute_error_metrics(py::array_t<float> reference,
                                      py::array_t<float> optimized,
                                      float epsilon = 1e-10f) {
  auto ref_buf = reference.request();
  auto opt_buf = optimized.request();

  if (ref_buf.size != opt_buf.size) {
    throw std::runtime_error("Size mismatch");
  }

  size_t size = static_cast<size_t>(ref_buf.size);
  float *ref_ptr = static_cast<float *>(ref_buf.ptr);
  float *opt_ptr = static_cast<float *>(opt_buf.ptr);

  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  double sum_abs_error = 0.0;
  double sum_sq_error = 0.0;
  size_t violations_1e3 = 0;
  size_t violations_1e6 = 0;

  for (size_t i = 0; i < size; ++i) {
    float abs_err = std::abs(opt_ptr[i] - ref_ptr[i]);
    float rel_err = abs_err / (std::abs(ref_ptr[i]) + epsilon);

    max_abs_error = std::max(max_abs_error, abs_err);
    max_rel_error = std::max(max_rel_error, rel_err);
    sum_abs_error += abs_err;
    sum_sq_error += static_cast<double>(abs_err) * abs_err;

    if (rel_err > 1e-3f)
      violations_1e3++;
    if (rel_err > 1e-6f)
      violations_1e6++;
  }

  float mean_abs_error = static_cast<float>(sum_abs_error / size);
  float rmse = static_cast<float>(std::sqrt(sum_sq_error / size));

  py::dict result;
  result["max_abs_error"] = max_abs_error;
  result["max_rel_error"] = max_rel_error;
  result["mean_abs_error"] = mean_abs_error;
  result["rmse"] = rmse;
  result["violations_1e3"] = violations_1e3;
  result["violations_1e6"] = violations_1e6;
  result["total_elements"] = size;
  result["passed_fp16_tolerance"] = (max_rel_error <= 1e-3f);
  result["passed_fp32_tolerance"] = (max_rel_error <= 1e-6f);

  return result;
}

/// Check if data is safe for FP16
inline py::dict check_fp16_safety(py::array_t<float> data,
                                  float safety_margin = 0.95f) {
  auto buf = data.request();
  size_t size = static_cast<size_t>(buf.size);
  float *ptr = static_cast<float *>(buf.ptr);

  constexpr float FP16_MAX = 65504.0f;
  constexpr float FP16_MIN_SUBNORMAL = 5.96e-8f;

  float abs_max = 0.0f;
  float abs_min_nonzero = std::numeric_limits<float>::max();
  float data_min = std::numeric_limits<float>::max();
  float data_max = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < size; ++i) {
    float v = ptr[i];
    data_min = std::min(data_min, v);
    data_max = std::max(data_max, v);

    float abs_v = std::abs(v);
    abs_max = std::max(abs_max, abs_v);
    if (abs_v > 1e-30f) {
      abs_min_nonzero = std::min(abs_min_nonzero, abs_v);
    }
  }

  bool max_ok = abs_max < FP16_MAX * safety_margin;
  bool min_ok = abs_min_nonzero > FP16_MIN_SUBNORMAL / safety_margin ||
                abs_min_nonzero == std::numeric_limits<float>::max();

  float dynamic_range = 0.0f;
  if (abs_min_nonzero > 0 && abs_max > 0 &&
      abs_min_nonzero != std::numeric_limits<float>::max()) {
    dynamic_range = 20.0f * std::log10(abs_max / abs_min_nonzero);
  }

  py::dict result;
  result["safe_for_fp16"] = max_ok && min_ok;
  result["max_ok"] = max_ok;
  result["min_ok"] = min_ok;
  result["abs_max"] = abs_max;
  result["abs_min_nonzero"] = abs_min_nonzero;
  result["data_min"] = data_min;
  result["data_max"] = data_max;
  result["dynamic_range_db"] = dynamic_range;

  return result;
}

// ============================================================================
// Pruning Bindings
// ============================================================================

/// Magnitude-based pruning
inline py::array_t<float> magnitude_prune(py::array_t<float> weights,
                                          float target_sparsity) {
  auto buf = weights.request();
  size_t size = static_cast<size_t>(buf.size);

  auto result = py::array_t<float>(buf.shape);
  auto result_buf = result.request();

  float *in_ptr = static_cast<float *>(buf.ptr);
  float *out_ptr = static_cast<float *>(result_buf.ptr);

  // Copy and compute magnitudes
  std::vector<std::pair<float, size_t>> magnitudes(size);
  for (size_t i = 0; i < size; ++i) {
    magnitudes[i] = {std::abs(in_ptr[i]), i};
    out_ptr[i] = in_ptr[i];
  }

  // Sort by magnitude
  std::sort(magnitudes.begin(), magnitudes.end());

  // Prune lowest magnitudes
  size_t num_to_prune =
      static_cast<size_t>(static_cast<float>(size) * target_sparsity);
  for (size_t i = 0; i < num_to_prune; ++i) {
    out_ptr[magnitudes[i].second] = 0.0f;
  }

  return result;
}

/// Compute sparsity
inline float compute_sparsity(py::array_t<float> weights) {
  auto buf = weights.request();
  size_t size = static_cast<size_t>(buf.size);
  float *ptr = static_cast<float *>(buf.ptr);

  size_t zeros = 0;
  for (size_t i = 0; i < size; ++i) {
    if (ptr[i] == 0.0f)
      zeros++;
  }

  return static_cast<float>(zeros) / static_cast<float>(size);
}

// ============================================================================
// Module Registration Helper
// ============================================================================

inline void register_feature_bindings(py::module_ &m) {
  // ========================================================================
  // Fused Operations Submodule
  // ========================================================================
  auto fused = m.def_submodule("fused", "Fused kernel operations");

  fused.def("bias_relu", &fused_bias_relu_cpu, py::arg("input"),
            py::arg("bias"),
            "Fused bias addition + ReLU: Y = max(0, X + bias)");

  fused.def("bias_gelu", &fused_bias_gelu_cpu, py::arg("input"),
            py::arg("bias"), "Fused bias addition + GELU: Y = GELU(X + bias)");

  fused.def("add_relu", &fused_add_relu_cpu, py::arg("x"), py::arg("residual"),
            "Fused element-wise add + ReLU: Y = max(0, X + residual)");

  fused.def("add_layernorm", &fused_add_layernorm_cpu, py::arg("x"),
            py::arg("residual"), py::arg("gamma"), py::arg("beta"),
            py::arg("eps") = 1e-5f,
            "Fused residual add + LayerNorm: Y = LayerNorm(X + residual)");

  // ========================================================================
  // Error Verification Submodule
  // ========================================================================
  auto verify = m.def_submodule("verify", "Error bounds verification");

  verify.def("compute_error_metrics", &compute_error_metrics,
             py::arg("reference"), py::arg("optimized"),
             py::arg("epsilon") = 1e-10f,
             "Compute error metrics between reference and optimized tensors");

  verify.def("check_fp16_safety", &check_fp16_safety, py::arg("data"),
             py::arg("safety_margin") = 0.95f,
             "Check if data is safe for FP16 conversion");

  // ========================================================================
  // Pruning Submodule
  // ========================================================================
  auto prune = m.def_submodule("prune", "Model pruning utilities");

  prune.def("magnitude_prune", &magnitude_prune, py::arg("weights"),
            py::arg("target_sparsity"),
            "Prune weights by magnitude with target sparsity");

  prune.def("compute_sparsity", &compute_sparsity, py::arg("weights"),
            "Compute sparsity ratio of a tensor");
}

} // namespace python
} // namespace zenith

#endif // ZENITH_PYTHON_FEATURES_BINDINGS_HPP
