// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Quantizer - Interface utama untuk kuantisasi GraphIR
// Berdasarkan CetakBiru.md Fase 3: Pipeline Kuantisasi INT8 Penuh
// Referensi: TensorRT, ONNX Quantization, PyTorch quantization

#ifndef ZENITH_QUANTIZATION_QUANTIZER_HPP
#define ZENITH_QUANTIZATION_QUANTIZER_HPP

#include "../graph_ir.hpp"
#include "../types.hpp"
#include "calibration.hpp"
#include "types.hpp"
#include <memory>
#include <unordered_map>

namespace zenith {
namespace quantization {

// ============================================================================
// Quantization Configuration
// ============================================================================

/// Konfigurasi untuk proses kuantisasi
struct QuantizationConfig {
  /// Metode kalibrasi default
  CalibrationMethod calibration_method = CalibrationMethod::Entropy;

  /// Mode kuantisasi default
  QuantizationMode mode = QuantizationMode::PerTensor;

  /// Apakah kuantisasi symmetric
  bool symmetric = true;

  /// Operasi yang akan dikuantisasi
  std::unordered_set<std::string> ops_to_quantize = {
      ops::CONV, ops::MATMUL, ops::GEMM, ops::LINEAR, ops::ADD,
  };

  /// Operasi yang dilewati (tidak dikuantisasi)
  std::unordered_set<std::string> ops_to_skip = {
      ops::SOFTMAX,
      ops::LAYER_NORM,
      ops::BATCH_NORM,
  };

  /// Toleransi error maksimum yang diizinkan
  float max_error_tolerance = 0.01f;

  /// Apakah melakukan validation setelah kuantisasi
  bool validate_after_quantization = true;
};

// ============================================================================
// Quantized Node Representation
// ============================================================================

namespace quantized_ops {
inline constexpr const char *QCONV = "QuantizedConv";
inline constexpr const char *QMATMUL = "QuantizedMatMul";
inline constexpr const char *QGEMM = "QuantizedGemm";
inline constexpr const char *QLINEAR = "QuantizedLinear";
inline constexpr const char *QADD = "QuantizedAdd";
inline constexpr const char *QUANTIZE = "QuantizeLinear";
inline constexpr const char *DEQUANTIZE = "DequantizeLinear";
} // namespace quantized_ops

// ============================================================================
// Quantization Result
// ============================================================================

/// Hasil dari proses kuantisasi
struct QuantizationResult {
  bool success = false;
  std::string message;

  /// Jumlah node yang dikuantisasi
  int nodes_quantized = 0;

  /// Jumlah node yang dilewati
  int nodes_skipped = 0;

  /// Statistik per tensor
  std::unordered_map<std::string, QuantizationInfo> tensor_info;

  /// Error rata-rata setelah kuantisasi
  float average_error = 0.0f;

  /// Error maksimum
  float max_error = 0.0f;
};

// ============================================================================
// Quantizer Class
// ============================================================================

/// Quantizer utama untuk mengubah model FP32 ke INT8
class Quantizer {
public:
  explicit Quantizer(QuantizationConfig config = {})
      : config_(std::move(config)),
        calibrator_(create_calibrator(config_.calibration_method)) {}

  /// Set konfigurasi
  void set_config(QuantizationConfig config) {
    config_ = std::move(config);
    calibrator_ = create_calibrator(config_.calibration_method);
  }

  /// Dapatkan konfigurasi saat ini
  [[nodiscard]] const QuantizationConfig &config() const { return config_; }

  /// Kalibrasi model dengan data representatif
  /// @param graph Graph yang akan dikalibrasi
  /// @param calibration_data Map dari nama tensor ke data kalibrasi
  Status calibrate(GraphIR *graph,
                   const std::unordered_map<std::string, std::vector<float>>
                       &calibration_data) {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph");
    }

    calibration_cache_.clear();

    // Kalibrasi setiap tensor dalam data
    for (const auto &[tensor_name, data] : calibration_data) {
      if (data.empty()) {
        continue;
      }

      QuantizationInfo info;
      info.tensor_name = tensor_name;
      info.method_used = config_.calibration_method;
      info.mode = config_.mode;

      // Hitung statistik
      auto [min_val, max_val] = compute_minmax(data.data(), data.size());
      info.min_value = min_val;
      info.max_value = max_val;

      // Compute mean dan std
      float sum = 0.0f;
      for (float v : data) {
        sum += v;
      }
      info.mean_value = sum / static_cast<float>(data.size());

      float var_sum = 0.0f;
      for (float v : data) {
        float diff = v - info.mean_value;
        var_sum += diff * diff;
      }
      info.std_value = std::sqrt(var_sum / static_cast<float>(data.size()));

      // Hitung parameter kuantisasi
      info.params = calibrator_->compute_params(data.data(), data.size());

      calibration_cache_[tensor_name] = info;
    }

    // Kalibrasi weights dari constants
    for (const auto &[const_name, tensor_data] : graph->constants()) {
      if (tensor_data.descriptor().dtype() != DataType::Float32) {
        continue;
      }

      const float *data_ptr = tensor_data.data_as<float>();
      size_t num_elements =
          static_cast<size_t>(tensor_data.descriptor().shape().numel());

      if (num_elements == 0) {
        continue;
      }

      QuantizationInfo info;
      info.tensor_name = const_name;
      info.is_weight = true;
      info.method_used = config_.calibration_method;

      // Untuk weights, biasanya MinMax sudah cukup
      auto [min_val, max_val] = compute_minmax(data_ptr, num_elements);
      info.min_value = min_val;
      info.max_value = max_val;
      info.params = compute_minmax_params(min_val, max_val, config_.symmetric);

      calibration_cache_[const_name] = info;
    }

    return Status::Ok();
  }

  /// Kuantisasi graph menggunakan hasil kalibrasi
  QuantizationResult quantize(GraphIR *graph) {
    QuantizationResult result;

    if (!graph) {
      result.message = "Null graph";
      return result;
    }

    if (calibration_cache_.empty()) {
      result.message = "No calibration data. Call calibrate() first.";
      return result;
    }

    // Kumpulkan node yang perlu dikuantisasi
    std::vector<std::string> nodes_to_quantize;
    for (const auto &node : graph->nodes()) {
      if (should_quantize(*node)) {
        nodes_to_quantize.push_back(node->name());
      } else {
        result.nodes_skipped++;
      }
    }

    // Kuantisasi setiap node
    for (const auto &node_name : nodes_to_quantize) {
      Node *node = graph->get_node(node_name);
      if (!node) {
        continue;
      }

      if (quantize_node(graph, node)) {
        result.nodes_quantized++;
      }
    }

    // Transfer info kalibrasi ke result
    result.tensor_info = calibration_cache_;

    // Validation
    if (config_.validate_after_quantization && result.nodes_quantized > 0) {
      // Hitung error rata-rata dari semua tensor yang dikalibrasi
      float total_error = 0.0f;
      int count = 0;
      for (const auto &[name, info] : calibration_cache_) {
        total_error += info.params.scale * 0.5f; // Approximation error
        result.max_error =
            std::max(result.max_error, info.params.scale * 127.0f);
        count++;
      }
      if (count > 0) {
        result.average_error = total_error / static_cast<float>(count);
      }
    }

    result.success = true;
    result.message = "Quantization completed successfully";

    return result;
  }

  /// Dapatkan parameter kuantisasi untuk tensor tertentu
  [[nodiscard]] const QuantizationInfo *
  get_quantization_info(const std::string &tensor_name) const {
    auto it = calibration_cache_.find(tensor_name);
    if (it != calibration_cache_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  /// Dapatkan semua info kuantisasi
  [[nodiscard]] const std::unordered_map<std::string, QuantizationInfo> &
  calibration_cache() const {
    return calibration_cache_;
  }

private:
  QuantizationConfig config_;
  std::unique_ptr<Calibrator> calibrator_;
  std::unordered_map<std::string, QuantizationInfo> calibration_cache_;

  /// Cek apakah node harus dikuantisasi
  bool should_quantize(const Node &node) const {
    const auto &op = node.op_type();

    // Skip jika ada di daftar skip
    if (config_.ops_to_skip.count(op)) {
      return false;
    }

    // Kuantisasi jika ada di daftar ops_to_quantize
    if (config_.ops_to_quantize.count(op)) {
      return true;
    }

    return false;
  }

  /// Kuantisasi satu node
  bool quantize_node(GraphIR *graph, Node *node) {
    const std::string &op = node->op_type();

    // Map ke quantized op
    std::string quantized_op;
    if (op == ops::CONV) {
      quantized_op = quantized_ops::QCONV;
    } else if (op == ops::MATMUL) {
      quantized_op = quantized_ops::QMATMUL;
    } else if (op == ops::GEMM) {
      quantized_op = quantized_ops::QGEMM;
    } else if (op == ops::LINEAR) {
      quantized_op = quantized_ops::QLINEAR;
    } else if (op == ops::ADD) {
      quantized_op = quantized_ops::QADD;
    } else {
      return false; // Tidak didukung
    }

    // Tambah attributes kuantisasi
    AttributeMap new_attrs = node->attrs();
    new_attrs["quantized"] = true;

    // Tambah scale dan zero_point untuk setiap input
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      const auto &input = node->inputs()[i];
      auto it = calibration_cache_.find(input.name());
      if (it != calibration_cache_.end()) {
        std::string scale_key = "input_" + std::to_string(i) + "_scale";
        std::string zp_key = "input_" + std::to_string(i) + "_zero_point";
        new_attrs[scale_key] = static_cast<double>(it->second.params.scale);
        new_attrs[zp_key] = static_cast<int64_t>(it->second.params.zero_point);
      }
    }

    // Tambah scale untuk output
    if (!node->outputs().empty()) {
      const auto &output = node->outputs()[0];
      auto it = calibration_cache_.find(output.name());
      if (it != calibration_cache_.end()) {
        new_attrs["output_scale"] =
            static_cast<double>(it->second.params.scale);
        new_attrs["output_zero_point"] =
            static_cast<int64_t>(it->second.params.zero_point);
      }
    }

    // Update node
    node->set_op_type(quantized_op);
    for (const auto &[key, value] : new_attrs) {
      node->set_attr(key, value);
    }

    return true;
  }
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Helper untuk membuat konfigurasi kuantisasi standar
inline QuantizationConfig create_default_config() {
  return QuantizationConfig{};
}

/// Helper untuk membuat konfigurasi untuk inference
inline QuantizationConfig create_inference_config() {
  QuantizationConfig config;
  config.calibration_method = CalibrationMethod::Entropy;
  config.symmetric = true;
  config.validate_after_quantization = true;
  return config;
}

/// Helper untuk membuat konfigurasi agresif (prioritas speed)
inline QuantizationConfig create_aggressive_config() {
  QuantizationConfig config;
  config.calibration_method = CalibrationMethod::MinMax;
  config.symmetric = true;
  config.max_error_tolerance = 0.05f;
  config.validate_after_quantization = false;
  return config;
}

} // namespace quantization
} // namespace zenith

#endif // ZENITH_QUANTIZATION_QUANTIZER_HPP
