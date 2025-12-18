// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Constant Folding Pass
// Evaluasi operasi dengan input konstan saat compile-time.
// Referensi: TensorFlow Grappler Constant Folding Optimizer

#ifndef ZENITH_OPTIMIZATION_CONSTANT_FOLDING_HPP
#define ZENITH_OPTIMIZATION_CONSTANT_FOLDING_HPP

#include "../graph_ir.hpp"
#include "../graph_optimizer.hpp"
#include <cmath>
#include <functional>
#include <unordered_map>

namespace zenith {
namespace optimizer {

/// Constant Folding Pass
/// Identifikasi node yang semua inputnya adalah konstanta,
/// evaluasi hasilnya, dan ganti dengan node Constant.
class ConstantFoldingPass : public OptimizationPass {
public:
  [[nodiscard]] std::string name() const override { return "ConstantFolding"; }

  [[nodiscard]] std::string description() const override {
    return "Evaluasi operasi dengan input konstan saat compile-time";
  }

  [[nodiscard]] int optimization_level() const override {
    return 1; // O1 - optimisasi dasar
  }

  Status run(GraphIR *graph) override {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph");
    }

    folded_count_ = 0;
    bool changed = true;

    // Iterasi sampai tidak ada perubahan (fixed-point)
    while (changed) {
      changed = false;

      // Kumpulkan node yang bisa di-fold
      std::vector<std::string> foldable_nodes;

      for (const auto &node : graph->nodes()) {
        if (can_fold(*node, *graph)) {
          foldable_nodes.push_back(node->name());
        }
      }

      // Fold setiap node yang bisa di-fold
      for (const auto &node_name : foldable_nodes) {
        Node *node = graph->get_node(node_name);
        if (node && fold_node(node, graph)) {
          changed = true;
          folded_count_++;
        }
      }
    }

    return Status::Ok();
  }

  /// Jumlah node yang di-fold di run terakhir
  [[nodiscard]] size_t folded_count() const { return folded_count_; }

private:
  size_t folded_count_ = 0;

  /// Cek apakah node bisa di-fold
  bool can_fold(const Node &node, const GraphIR &graph) {
    // Skip jika sudah constant
    if (node.op_type() == ops::CONSTANT) {
      return false;
    }

    // Node harus memiliki input
    if (node.inputs().empty()) {
      return false;
    }

    // Semua input harus berupa konstanta atau output dari Constant node
    for (const auto &input : node.inputs()) {
      const TensorData *constant = graph.get_constant(input.name());
      if (!constant) {
        // Cek apakah ada Constant node yang memproduksi input ini
        bool found_constant = false;
        for (const auto &n : graph.nodes()) {
          if (n->op_type() == ops::CONSTANT) {
            for (const auto &out : n->outputs()) {
              if (out.name() == input.name()) {
                found_constant = true;
                break;
              }
            }
          }
          if (found_constant)
            break;
        }
        if (!found_constant) {
          return false;
        }
      }
    }

    // Operasi harus mendukung constant folding
    return is_foldable_op(node.op_type());
  }

  /// Cek apakah operasi mendukung constant folding
  bool is_foldable_op(const std::string &op_type) {
    static const std::unordered_set<std::string> foldable_ops = {
        ops::ADD,       ops::SUB,      ops::MUL,    ops::DIV,
        ops::RELU,      ops::SIGMOID,  ops::TANH,   ops::RESHAPE,
        ops::TRANSPOSE, ops::IDENTITY, ops::CONCAT,
    };
    return foldable_ops.count(op_type) > 0;
  }

  /// Fold satu node menjadi konstanta
  bool fold_node(Node *node, GraphIR *graph) {
    // Dapatkan data input
    std::vector<const TensorData *> input_data;
    for (const auto &input : node->inputs()) {
      const TensorData *data = graph->get_constant(input.name());
      if (!data) {
        return false; // Tidak bisa fold tanpa data
      }
      input_data.push_back(data);
    }

    // Evaluasi berdasarkan tipe operasi
    std::vector<uint8_t> result_data;
    TensorDescriptor result_desc;

    const std::string &op = node->op_type();

    if (op == ops::ADD && input_data.size() == 2) {
      if (!evaluate_binary_op(input_data[0], input_data[1], result_data,
                              result_desc,
                              [](float a, float b) { return a + b; })) {
        return false;
      }
    } else if (op == ops::SUB && input_data.size() == 2) {
      if (!evaluate_binary_op(input_data[0], input_data[1], result_data,
                              result_desc,
                              [](float a, float b) { return a - b; })) {
        return false;
      }
    } else if (op == ops::MUL && input_data.size() == 2) {
      if (!evaluate_binary_op(input_data[0], input_data[1], result_data,
                              result_desc,
                              [](float a, float b) { return a * b; })) {
        return false;
      }
    } else if (op == ops::DIV && input_data.size() == 2) {
      if (!evaluate_binary_op(
              input_data[0], input_data[1], result_data, result_desc,
              [](float a, float b) { return b != 0.0f ? a / b : 0.0f; })) {
        return false;
      }
    } else if (op == ops::RELU && input_data.size() == 1) {
      if (!evaluate_unary_op(input_data[0], result_data, result_desc,
                             [](float x) { return x > 0.0f ? x : 0.0f; })) {
        return false;
      }
    } else if (op == ops::SIGMOID && input_data.size() == 1) {
      if (!evaluate_unary_op(
              input_data[0], result_data, result_desc,
              [](float x) { return 1.0f / (1.0f + std::exp(-x)); })) {
        return false;
      }
    } else if (op == ops::TANH && input_data.size() == 1) {
      if (!evaluate_unary_op(input_data[0], result_data, result_desc,
                             [](float x) { return std::tanh(x); })) {
        return false;
      }
    } else if (op == ops::IDENTITY && input_data.size() == 1) {
      // Identity: langsung copy
      result_data = input_data[0]->data();
      result_desc = input_data[0]->descriptor();
    } else {
      return false; // Operasi tidak didukung
    }

    // Buat nama untuk konstanta baru
    std::string const_name = node->name() + "_folded";
    if (!node->outputs().empty()) {
      const_name = node->outputs()[0].name();
    }
    result_desc.set_name(const_name);

    // Tambahkan konstanta ke graph
    TensorData folded_data(result_desc, std::move(result_data));
    graph->add_constant(const_name, std::move(folded_data));

    // Hapus node asli
    graph->remove_node(node->name());

    return true;
  }

  /// Evaluasi operasi unary element-wise
  bool evaluate_unary_op(const TensorData *input, std::vector<uint8_t> &result,
                         TensorDescriptor &result_desc,
                         std::function<float(float)> op) {
    if (input->descriptor().dtype() != DataType::Float32) {
      return false; // Hanya mendukung FP32 untuk saat ini
    }

    const auto &desc = input->descriptor();
    int64_t num_elements = desc.shape().numel();
    if (num_elements <= 0)
      return false;

    result.resize(static_cast<size_t>(num_elements) * sizeof(float));
    const float *in_ptr = input->data_as<float>();
    float *out_ptr = reinterpret_cast<float *>(result.data());

    for (int64_t i = 0; i < num_elements; ++i) {
      out_ptr[i] = op(in_ptr[i]);
    }

    result_desc = desc;
    return true;
  }

  /// Evaluasi operasi binary element-wise
  bool evaluate_binary_op(const TensorData *a, const TensorData *b,
                          std::vector<uint8_t> &result,
                          TensorDescriptor &result_desc,
                          std::function<float(float, float)> op) {
    if (a->descriptor().dtype() != DataType::Float32 ||
        b->descriptor().dtype() != DataType::Float32) {
      return false;
    }

    // Cek shape compatibility (sederhana: harus sama)
    if (a->descriptor().shape() != b->descriptor().shape()) {
      // TODO: Implementasi broadcasting
      return false;
    }

    const auto &desc = a->descriptor();
    int64_t num_elements = desc.shape().numel();
    if (num_elements <= 0)
      return false;

    result.resize(static_cast<size_t>(num_elements) * sizeof(float));
    const float *a_ptr = a->data_as<float>();
    const float *b_ptr = b->data_as<float>();
    float *out_ptr = reinterpret_cast<float *>(result.data());

    for (int64_t i = 0; i < num_elements; ++i) {
      out_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }

    result_desc = desc;
    return true;
  }
};

} // namespace optimizer
} // namespace zenith

#endif // ZENITH_OPTIMIZATION_CONSTANT_FOLDING_HPP
