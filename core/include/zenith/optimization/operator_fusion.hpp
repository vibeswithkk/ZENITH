// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Operator Fusion Pass
// Menggabungkan operasi berurutan menjadi satu kernel fused.
// Referensi: TensorFlow XLA Fusion, TensorRT operator fusion

#ifndef ZENITH_OPTIMIZATION_OPERATOR_FUSION_HPP
#define ZENITH_OPTIMIZATION_OPERATOR_FUSION_HPP

#include "../graph_ir.hpp"
#include "../graph_optimizer.hpp"
#include <unordered_set>
#include <vector>

namespace zenith {
namespace optimizer {

// ============================================================================
// Fused Operation Types
// ============================================================================

namespace fused_ops {
inline constexpr const char *CONV_BN_RELU = "FusedConvBNRelu";
inline constexpr const char *MATMUL_ADD = "FusedMatMulAdd";
inline constexpr const char *LINEAR_RELU = "FusedLinearRelu";
inline constexpr const char *LINEAR_GELU = "FusedLinearGelu";
inline constexpr const char *LAYERNORM_ADD = "FusedLayerNormAdd";
inline constexpr const char *ADD_RELU = "FusedAddRelu";
} // namespace fused_ops

// ============================================================================
// Fusion Pattern
// ============================================================================

/// Representasi pola fusion
struct FusionPattern {
  std::string name;
  std::vector<std::string> ops; // Urutan operasi yang di-fuse
  std::string fused_op;         // Nama operasi hasil fusion
  bool requires_same_dtype = true;
};

// ============================================================================
// Operator Fusion Pass
// ============================================================================

/// Operator Fusion Pass
/// Pattern matching untuk pola-pola umum dan menggabungkannya
/// menjadi operasi fused yang lebih efisien.
class OperatorFusionPass : public OptimizationPass {
public:
  OperatorFusionPass() { register_default_patterns(); }

  [[nodiscard]] std::string name() const override { return "OperatorFusion"; }

  [[nodiscard]] std::string description() const override {
    return "Menggabungkan operasi berurutan menjadi kernel fused";
  }

  [[nodiscard]] int optimization_level() const override {
    return 2; // O2 - optimisasi medium
  }

  /// Tambah pola fusion custom
  void add_pattern(FusionPattern pattern) {
    patterns_.push_back(std::move(pattern));
  }

  Status run(GraphIR *graph) override {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph");
    }

    fused_count_ = 0;
    bool changed = true;

    // Iterasi sampai tidak ada perubahan
    while (changed) {
      changed = false;

      for (const auto &pattern : patterns_) {
        if (apply_pattern(graph, pattern)) {
          changed = true;
          fused_count_++;
        }
      }
    }

    return Status::Ok();
  }

  /// Jumlah fusion yang dilakukan di run terakhir
  [[nodiscard]] size_t fused_count() const { return fused_count_; }

private:
  std::vector<FusionPattern> patterns_;
  size_t fused_count_ = 0;

  /// Daftarkan pola-pola fusion default
  void register_default_patterns() {
    // Conv + BatchNorm + ReLU
    patterns_.push_back(FusionPattern{
        "ConvBNRelu",
        {ops::CONV, ops::BATCH_NORM, ops::RELU},
        fused_ops::CONV_BN_RELU,
    });

    // MatMul + Add (bias)
    patterns_.push_back(FusionPattern{
        "MatMulAdd",
        {ops::MATMUL, ops::ADD},
        fused_ops::MATMUL_ADD,
    });

    // Linear + ReLU
    patterns_.push_back(FusionPattern{
        "LinearRelu",
        {ops::LINEAR, ops::RELU},
        fused_ops::LINEAR_RELU,
    });

    // Linear + GELU
    patterns_.push_back(FusionPattern{
        "LinearGelu",
        {ops::LINEAR, ops::GELU},
        fused_ops::LINEAR_GELU,
    });

    // Add + ReLU
    patterns_.push_back(FusionPattern{
        "AddRelu",
        {ops::ADD, ops::RELU},
        fused_ops::ADD_RELU,
    });

    // LayerNorm + Add (residual)
    patterns_.push_back(FusionPattern{
        "LayerNormAdd",
        {ops::LAYER_NORM, ops::ADD},
        fused_ops::LAYERNORM_ADD,
    });
  }

  /// Coba apply satu pola ke graph
  bool apply_pattern(GraphIR *graph, const FusionPattern &pattern) {
    if (pattern.ops.size() < 2) {
      return false;
    }

    // Temukan kandidat - node pertama dalam pola
    for (const auto &node : graph->nodes()) {
      if (node->op_type() != pattern.ops[0]) {
        continue;
      }

      // Coba match pola lengkap dari node ini
      std::vector<Node *> matched_nodes;
      if (match_pattern(graph, node.get(), pattern, matched_nodes)) {
        // Apply fusion
        if (fuse_nodes(graph, matched_nodes, pattern)) {
          return true;
        }
      }
    }

    return false;
  }

  /// Match pola mulai dari node tertentu
  bool match_pattern(GraphIR *graph, Node *start_node,
                     const FusionPattern &pattern,
                     std::vector<Node *> &matched) {
    matched.clear();
    matched.push_back(start_node);

    Node *current = start_node;

    for (size_t i = 1; i < pattern.ops.size(); ++i) {
      // Cari consumer dari output current node
      Node *next = find_single_consumer(graph, current);
      if (!next) {
        return false;
      }

      // Cek apakah op type cocok
      if (next->op_type() != pattern.ops[i]) {
        return false;
      }

      // Cek bahwa current node hanya punya satu consumer
      if (count_consumers(graph, current) != 1) {
        return false;
      }

      matched.push_back(next);
      current = next;
    }

    return true;
  }

  /// Temukan consumer tunggal dari node
  Node *find_single_consumer(GraphIR *graph, Node *producer) {
    if (producer->outputs().empty()) {
      return nullptr;
    }

    const std::string &output_name = producer->outputs()[0].name();
    Node *consumer = nullptr;

    for (const auto &node : graph->nodes()) {
      if (node.get() == producer)
        continue;

      for (const auto &input : node->inputs()) {
        if (input.name() == output_name) {
          if (consumer) {
            return nullptr; // Multiple consumers
          }
          consumer = node.get();
        }
      }
    }

    return consumer;
  }

  /// Hitung jumlah consumer dari node
  int count_consumers(GraphIR *graph, Node *producer) {
    if (producer->outputs().empty()) {
      return 0;
    }

    const std::string &output_name = producer->outputs()[0].name();
    int count = 0;

    for (const auto &node : graph->nodes()) {
      if (node.get() == producer)
        continue;

      for (const auto &input : node->inputs()) {
        if (input.name() == output_name) {
          count++;
          break; // Count each consumer once
        }
      }
    }

    return count;
  }

  /// Fuse nodes menjadi satu node baru
  bool fuse_nodes(GraphIR *graph, const std::vector<Node *> &nodes,
                  const FusionPattern &pattern) {
    if (nodes.size() < 2) {
      return false;
    }

    // Node pertama: ambil inputs
    // Node terakhir: ambil outputs
    Node *first = nodes.front();
    Node *last = nodes.back();

    // Kumpulkan semua attributes dari nodes
    AttributeMap fused_attrs;
    for (size_t i = 0; i < nodes.size(); ++i) {
      for (const auto &[key, value] : nodes[i]->attrs()) {
        std::string prefixed_key = "fused_" + std::to_string(i) + "_" + key;
        fused_attrs[prefixed_key] = value;
      }
    }

    // Tandai jenis fusion
    fused_attrs["fusion_type"] = pattern.name;

    // Buat nama untuk fused node
    std::string fused_name = first->name() + "_fused";

    // Buat fused node
    graph->add_node(pattern.fused_op, fused_name, first->inputs(),
                    last->outputs(), std::move(fused_attrs));

    // Hapus nodes lama (dari belakang ke depan untuk menghindari invalidation)
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      graph->remove_node((*it)->name());
    }

    return true;
  }
};

} // namespace optimizer
} // namespace zenith

#endif // ZENITH_OPTIMIZATION_OPERATOR_FUSION_HPP
