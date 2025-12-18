// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Fused Kernel Definitions dan Registry
// Berdasarkan CetakBiru.md: Kernel fusion untuk percepatan inferensi
// Referensi: TensorRT layer fusion, cuDNN Graph API, NVIDIA APEX

#ifndef ZENITH_KERNEL_FUSION_HPP
#define ZENITH_KERNEL_FUSION_HPP

#include "graph_ir.hpp"
#include "types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace zenith {
namespace fusion {

// ============================================================================
// Fused Kernel Types
// ============================================================================

/// Enumerasi untuk jenis kernel fused yang tersedia
enum class FusedKernelType {
  // Element-wise fusions
  BiasRelu,    // Y = ReLU(X + bias)
  BiasGelu,    // Y = GELU(X + bias)
  BiasSigmoid, // Y = Sigmoid(X + bias)
  BiasTanh,    // Y = Tanh(X + bias)

  // Normalization fusions
  LayerNormAdd,  // Y = LayerNorm(X) + residual
  AddLayerNorm,  // Y = LayerNorm(X + residual)
  LayerNormRelu, // Y = ReLU(LayerNorm(X))
  BatchNormRelu, // Y = ReLU(BatchNorm(X))

  // Convolution fusions
  ConvBiasRelu,      // Y = ReLU(Conv(X) + bias)
  ConvBias,          // Y = Conv(X) + bias
  ConvBatchNormRelu, // Y = ReLU(BatchNorm(Conv(X)))

  // Linear/GEMM fusions
  LinearBiasRelu, // Y = ReLU(Linear(X) + bias)
  LinearBiasGelu, // Y = GELU(Linear(X) + bias)
  MatMulAdd,      // Y = MatMul(A, B) + bias

  // Attention-specific fusions
  ScaledDotProduct, // Fused scaled dot-product attention
  SoftmaxMask,      // Softmax with mask application

  // Residual fusions
  AddRelu, // Y = ReLU(X + Y)
  AddGelu, // Y = GELU(X + Y)
  AddSilu, // Y = SiLU(X + Y)

  // Custom
  Custom,
};

// ============================================================================
// Fused Kernel Descriptor
// ============================================================================

/// Deskripsi lengkap dari sebuah fused kernel
struct FusedKernelDescriptor {
  FusedKernelType type;
  std::string name;
  std::string description;

  // Operasi yang di-fuse
  std::vector<std::string> fused_ops;

  // Input/output signatures
  int num_inputs = 2;
  int num_outputs = 1;

  // Requirements
  bool requires_bias = false;
  bool requires_scale = false;
  bool supports_fp16 = true;
  bool supports_int8 = false;

  // Performance hints
  float expected_speedup = 1.5f; // Dibandingkan operasi terpisah
};

// ============================================================================
// Fused Kernel Registry
// ============================================================================

/// Registry untuk semua fused kernels yang tersedia
class FusedKernelRegistry {
public:
  static FusedKernelRegistry &instance() {
    static FusedKernelRegistry registry;
    return registry;
  }

  /// Daftarkan fused kernel baru
  void register_kernel(FusedKernelType type,
                       const FusedKernelDescriptor &desc) {
    kernels_[type] = desc;
    name_to_type_[desc.name] = type;
  }

  /// Dapatkan deskriptor kernel
  const FusedKernelDescriptor *get_descriptor(FusedKernelType type) const {
    auto it = kernels_.find(type);
    return it != kernels_.end() ? &it->second : nullptr;
  }

  /// Cek apakah kernel terdaftar
  bool has_kernel(FusedKernelType type) const {
    return kernels_.count(type) > 0;
  }

  /// Dapatkan type dari nama
  FusedKernelType get_type_by_name(const std::string &name) const {
    auto it = name_to_type_.find(name);
    return it != name_to_type_.end() ? it->second : FusedKernelType::Custom;
  }

  /// Dapatkan semua kernel yang terdaftar
  std::vector<FusedKernelType> get_all_types() const {
    std::vector<FusedKernelType> types;
    types.reserve(kernels_.size());
    for (const auto &[type, _] : kernels_) {
      types.push_back(type);
    }
    return types;
  }

private:
  FusedKernelRegistry() { register_builtin_kernels(); }

  void register_builtin_kernels() {
    // Bias + Activation fusions
    register_kernel(FusedKernelType::BiasRelu,
                    {FusedKernelType::BiasRelu,
                     "FusedBiasRelu",
                     "Fused bias addition with ReLU activation",
                     {ops::ADD, ops::RELU},
                     2,
                     1,
                     true,
                     false,
                     true,
                     true,
                     1.8f});

    register_kernel(FusedKernelType::BiasGelu,
                    {FusedKernelType::BiasGelu,
                     "FusedBiasGelu",
                     "Fused bias addition with GELU activation",
                     {ops::ADD, ops::GELU},
                     2,
                     1,
                     true,
                     false,
                     true,
                     false,
                     2.0f});

    register_kernel(FusedKernelType::BiasSigmoid,
                    {FusedKernelType::BiasSigmoid,
                     "FusedBiasSigmoid",
                     "Fused bias addition with Sigmoid activation",
                     {ops::ADD, ops::SIGMOID},
                     2,
                     1,
                     true,
                     false,
                     true,
                     false,
                     1.7f});

    // LayerNorm fusions
    register_kernel(FusedKernelType::LayerNormAdd,
                    {FusedKernelType::LayerNormAdd,
                     "FusedLayerNormAdd",
                     "Fused LayerNorm followed by residual addition",
                     {ops::LAYER_NORM, ops::ADD},
                     3,
                     1,
                     false,
                     true,
                     true,
                     false,
                     1.5f});

    register_kernel(FusedKernelType::AddLayerNorm,
                    {FusedKernelType::AddLayerNorm,
                     "FusedAddLayerNorm",
                     "Residual addition followed by LayerNorm",
                     {ops::ADD, ops::LAYER_NORM},
                     3,
                     1,
                     false,
                     true,
                     true,
                     false,
                     1.6f});

    // Convolution fusions
    register_kernel(FusedKernelType::ConvBiasRelu,
                    {FusedKernelType::ConvBiasRelu,
                     "FusedConvBiasRelu",
                     "Fused Conv + Bias + ReLU",
                     {ops::CONV, ops::ADD, ops::RELU},
                     3,
                     1,
                     true,
                     false,
                     true,
                     true,
                     2.5f});

    register_kernel(FusedKernelType::ConvBatchNormRelu,
                    {FusedKernelType::ConvBatchNormRelu,
                     "FusedConvBNRelu",
                     "Fused Conv + BatchNorm + ReLU",
                     {ops::CONV, ops::BATCH_NORM, ops::RELU},
                     2,
                     1,
                     false,
                     true,
                     true,
                     true,
                     3.0f});

    // Linear fusions
    register_kernel(FusedKernelType::LinearBiasRelu,
                    {FusedKernelType::LinearBiasRelu,
                     "FusedLinearBiasRelu",
                     "Fused Linear + Bias + ReLU",
                     {ops::LINEAR, ops::ADD, ops::RELU},
                     3,
                     1,
                     true,
                     false,
                     true,
                     true,
                     2.2f});

    register_kernel(FusedKernelType::LinearBiasGelu,
                    {FusedKernelType::LinearBiasGelu,
                     "FusedLinearBiasGelu",
                     "Fused Linear + Bias + GELU",
                     {ops::LINEAR, ops::ADD, ops::GELU},
                     3,
                     1,
                     true,
                     false,
                     true,
                     false,
                     2.5f});

    register_kernel(FusedKernelType::MatMulAdd, {FusedKernelType::MatMulAdd,
                                                 "FusedMatMulAdd",
                                                 "Fused MatMul + Add (GEMM)",
                                                 {ops::MATMUL, ops::ADD},
                                                 3,
                                                 1,
                                                 true,
                                                 false,
                                                 true,
                                                 true,
                                                 1.8f});

    // Residual fusions
    register_kernel(FusedKernelType::AddRelu, {FusedKernelType::AddRelu,
                                               "FusedAddRelu",
                                               "Fused element-wise Add + ReLU",
                                               {ops::ADD, ops::RELU},
                                               2,
                                               1,
                                               false,
                                               false,
                                               true,
                                               true,
                                               1.9f});

    register_kernel(FusedKernelType::AddGelu, {FusedKernelType::AddGelu,
                                               "FusedAddGelu",
                                               "Fused element-wise Add + GELU",
                                               {ops::ADD, ops::GELU},
                                               2,
                                               1,
                                               false,
                                               false,
                                               true,
                                               false,
                                               2.0f});
  }

  std::unordered_map<FusedKernelType, FusedKernelDescriptor> kernels_;
  std::unordered_map<std::string, FusedKernelType> name_to_type_;
};

// ============================================================================
// Fusion Pattern Matcher
// ============================================================================

/// Mencocokkan pola operasi dalam graph dengan fused kernels yang tersedia
class FusionPatternMatcher {
public:
  struct MatchResult {
    bool matched = false;
    FusedKernelType kernel_type = FusedKernelType::Custom;
    std::vector<std::string> matched_node_names;
    float estimated_speedup = 1.0f;
  };

  /// Cari pola fusion mulai dari node tertentu
  MatchResult find_fusion_pattern(const GraphIR &graph,
                                  const Node &start_node) {
    MatchResult result;

    const auto &registry = FusedKernelRegistry::instance();

    // Coba semua pola yang terdaftar
    for (FusedKernelType type : registry.get_all_types()) {
      const auto *desc = registry.get_descriptor(type);
      if (!desc || desc->fused_ops.empty()) {
        continue;
      }

      // Cek apakah start_node cocok dengan op pertama
      if (start_node.op_type() != desc->fused_ops[0]) {
        continue;
      }

      // Coba match pola lengkap
      std::vector<std::string> matched_names;
      if (match_pattern_from_node(graph, start_node, desc->fused_ops,
                                  matched_names)) {
        result.matched = true;
        result.kernel_type = type;
        result.matched_node_names = std::move(matched_names);
        result.estimated_speedup = desc->expected_speedup;
        return result; // Return first match
      }
    }

    return result;
  }

private:
  /// Match pola operasi dari node start
  bool match_pattern_from_node(const GraphIR &graph, const Node &start,
                               const std::vector<std::string> &pattern,
                               std::vector<std::string> &matched_names) {
    matched_names.clear();
    matched_names.push_back(start.name());

    const Node *current = &start;

    for (size_t i = 1; i < pattern.size(); ++i) {
      // Cari successor (consumer) dari current node
      const Node *next = find_single_successor(graph, *current);
      if (!next) {
        return false;
      }

      // Cek op type
      if (next->op_type() != pattern[i]) {
        return false;
      }

      // Pastikan current hanya punya satu consumer
      if (count_successors(graph, *current) != 1) {
        return false;
      }

      matched_names.push_back(next->name());
      current = next;
    }

    return true;
  }

  /// Cari successor tunggal dari node
  const Node *find_single_successor(const GraphIR &graph, const Node &node) {
    if (node.outputs().empty()) {
      return nullptr;
    }

    const std::string &output_name = node.outputs()[0].name();
    const Node *successor = nullptr;

    for (const auto &n : graph.nodes()) {
      if (n.get() == &node)
        continue;

      for (const auto &input : n->inputs()) {
        if (input.name() == output_name) {
          if (successor) {
            return nullptr; // Multiple successors
          }
          successor = n.get();
        }
      }
    }

    return successor;
  }

  /// Hitung jumlah successor
  int count_successors(const GraphIR &graph, const Node &node) {
    if (node.outputs().empty()) {
      return 0;
    }

    const std::string &output_name = node.outputs()[0].name();
    int count = 0;

    for (const auto &n : graph.nodes()) {
      if (n.get() == &node)
        continue;

      for (const auto &input : n->inputs()) {
        if (input.name() == output_name) {
          count++;
          break;
        }
      }
    }

    return count;
  }
};

// ============================================================================
// Kernel Fusion Pass
// ============================================================================

/// Optimization pass yang melakukan kernel fusion
class KernelFusionPass : public optimizer::OptimizationPass {
public:
  [[nodiscard]] std::string name() const override { return "KernelFusion"; }

  [[nodiscard]] std::string description() const override {
    return "Menggabungkan operasi berurutan menjadi fused kernels";
  }

  [[nodiscard]] int optimization_level() const override { return 2; }

  Status run(GraphIR *graph) override {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph");
    }

    fused_count_ = 0;
    bool changed = true;

    while (changed) {
      changed = false;

      for (const auto &node : graph->nodes()) {
        auto match = matcher_.find_fusion_pattern(*graph, *node);
        if (match.matched) {
          if (apply_fusion(graph, match)) {
            changed = true;
            fused_count_++;
          }
        }
      }
    }

    return Status::Ok();
  }

  [[nodiscard]] int fused_count() const { return fused_count_; }

private:
  FusionPatternMatcher matcher_;
  int fused_count_ = 0;

  bool apply_fusion(GraphIR *graph,
                    const FusionPatternMatcher::MatchResult &match) {
    if (match.matched_node_names.size() < 2) {
      return false;
    }

    // Dapatkan nodes
    std::vector<Node *> nodes;
    for (const auto &name : match.matched_node_names) {
      Node *n = graph->get_node(name);
      if (!n)
        return false;
      nodes.push_back(n);
    }

    Node *first = nodes.front();
    Node *last = nodes.back();

    // Dapatkan descriptor
    const auto *desc =
        FusedKernelRegistry::instance().get_descriptor(match.kernel_type);
    if (!desc) {
      return false;
    }

    // Kumpulkan semua inputs dari semua nodes
    std::vector<TensorDescriptor> all_inputs;
    for (const auto &input : first->inputs()) {
      all_inputs.push_back(input);
    }
    // Tambahkan inputs tambahan dari nodes lain (misalnya bias)
    for (size_t i = 1; i < nodes.size(); ++i) {
      for (const auto &input : nodes[i]->inputs()) {
        // Skip jika input adalah output dari node sebelumnya
        bool is_internal = false;
        for (size_t j = 0; j < i; ++j) {
          for (const auto &out : nodes[j]->outputs()) {
            if (out.name() == input.name()) {
              is_internal = true;
              break;
            }
          }
          if (is_internal)
            break;
        }
        if (!is_internal) {
          all_inputs.push_back(input);
        }
      }
    }

    // Buat attributes untuk fused node
    AttributeMap attrs;
    attrs["fused_kernel_type"] = desc->name;
    attrs["num_fused_ops"] =
        static_cast<int64_t>(match.matched_node_names.size());

    // Buat fused node
    std::string fused_name = first->name() + "_fused";
    graph->add_node(desc->name, fused_name, all_inputs, last->outputs(),
                    std::move(attrs));

    // Hapus nodes lama (dari belakang ke depan)
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      graph->remove_node((*it)->name());
    }

    return true;
  }
};

} // namespace fusion
} // namespace zenith

#endif // ZENITH_KERNEL_FUSION_HPP
