// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Neural Network Pruning System
// Berdasarkan riset: Magnitude-based, Structured, Unstructured pruning
// Referensi: PyTorch torch.nn.utils.prune, TensorFlow Model Optimization

#ifndef ZENITH_PRUNING_HPP
#define ZENITH_PRUNING_HPP

#include "graph_ir.hpp"
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace zenith {
namespace pruning {

// ============================================================================
// Pruning Method Enum
// ============================================================================

/// Metode pruning yang tersedia
enum class PruningMethod {
  Magnitude, // Hapus weights dengan magnitude kecil (L1 norm)
  Random,    // Hapus weights secara acak
  Gradient,  // Berdasarkan gradient importance
  Taylor,    // Taylor expansion importance scoring
  Movement,  // Movement pruning (for fine-tuning)
};

/// Struktur pruning
enum class PruningStructure {
  Unstructured,   // Prune individual weights
  Structured,     // Prune entire neurons/filters/channels
  SemiStructured, // N:M sparsity (e.g., 2:4)
};

// ============================================================================
// Pruning Configuration
// ============================================================================

/// Konfigurasi untuk pruning
struct PruningConfig {
  PruningMethod method = PruningMethod::Magnitude;
  PruningStructure structure = PruningStructure::Unstructured;

  /// Target sparsity (0.0 - 1.0)
  /// Contoh: 0.9 berarti 90% weights akan di-prune (tinggal 10%)
  float target_sparsity = 0.5f;

  /// Untuk semi-structured: N dari M values di-prune
  int n_of_m_n = 2;
  int n_of_m_m = 4;

  /// Threshold untuk magnitude pruning (alternatif dari sparsity)
  float magnitude_threshold = 0.0f;
  bool use_threshold = false;

  /// Layer types yang akan di-prune
  std::vector<std::string> layers_to_prune = {
      ops::CONV,
      ops::LINEAR,
      ops::MATMUL,
      ops::GEMM,
  };

  /// Layers yang dilewati
  std::vector<std::string> layers_to_skip = {};

  /// Apakah pruning per-layer atau global
  bool global_pruning = false;
};

// ============================================================================
// Pruning Mask
// ============================================================================

/// Mask untuk pruning (1 = keep, 0 = prune)
class PruningMask {
public:
  PruningMask() = default;

  explicit PruningMask(size_t size, bool initial_value = true)
      : mask_(size, initial_value ? 1.0f : 0.0f) {}

  /// Set nilai mask
  void set(size_t idx, bool keep) {
    if (idx < mask_.size()) {
      mask_[idx] = keep ? 1.0f : 0.0f;
    }
  }

  /// Get nilai mask
  [[nodiscard]] bool get(size_t idx) const {
    return idx < mask_.size() && mask_[idx] > 0.5f;
  }

  /// Apply mask ke tensor
  void apply(float *data, size_t size) const {
    size_t n = std::min(size, mask_.size());
    for (size_t i = 0; i < n; ++i) {
      data[i] *= mask_[i];
    }
  }

  /// Apply mask ke int8 tensor
  void apply(int8_t *data, size_t size) const {
    size_t n = std::min(size, mask_.size());
    for (size_t i = 0; i < n; ++i) {
      if (mask_[i] < 0.5f) {
        data[i] = 0;
      }
    }
  }

  /// Hitung sparsity aktual
  [[nodiscard]] float sparsity() const {
    if (mask_.empty())
      return 0.0f;
    size_t zeros = 0;
    for (float v : mask_) {
      if (v < 0.5f)
        zeros++;
    }
    return static_cast<float>(zeros) / static_cast<float>(mask_.size());
  }

  /// Jumlah non-zero
  [[nodiscard]] size_t nnz() const {
    size_t count = 0;
    for (float v : mask_) {
      if (v >= 0.5f)
        count++;
    }
    return count;
  }

  [[nodiscard]] size_t size() const { return mask_.size(); }

  [[nodiscard]] const std::vector<float> &data() const { return mask_; }
  std::vector<float> &data() { return mask_; }

private:
  std::vector<float> mask_;
};

// ============================================================================
// Importance Scorer
// ============================================================================

/// Base class untuk menghitung importance score
class ImportanceScorer {
public:
  virtual ~ImportanceScorer() = default;

  /// Hitung importance untuk setiap weight
  virtual std::vector<float> score(const float *weights, size_t size) const = 0;
};

/// Magnitude-based scorer (L1 norm)
class MagnitudeScorer : public ImportanceScorer {
public:
  std::vector<float> score(const float *weights, size_t size) const override {
    std::vector<float> scores(size);
    for (size_t i = 0; i < size; ++i) {
      scores[i] = std::abs(weights[i]);
    }
    return scores;
  }
};

/// L2 norm scorer
class L2NormScorer : public ImportanceScorer {
public:
  std::vector<float> score(const float *weights, size_t size) const override {
    std::vector<float> scores(size);
    for (size_t i = 0; i < size; ++i) {
      scores[i] = weights[i] * weights[i];
    }
    return scores;
  }
};

/// Random scorer (untuk random pruning)
class RandomScorer : public ImportanceScorer {
public:
  explicit RandomScorer(unsigned int seed = 42) : rng_(seed) {}

  std::vector<float> score(const float *weights, size_t size) const override {
    std::vector<float> scores(size);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
      scores[i] = dist(rng_);
    }
    return scores;
  }

private:
  mutable std::mt19937 rng_;
};

// ============================================================================
// Pruning Result
// ============================================================================

/// Hasil dari proses pruning
struct PruningResult {
  bool success = false;
  std::string message;

  /// Statistik per layer
  struct LayerStats {
    std::string layer_name;
    size_t original_params = 0;
    size_t pruned_params = 0;
    float sparsity = 0.0f;
  };
  std::vector<LayerStats> layer_stats;

  /// Statistik global
  size_t total_original_params = 0;
  size_t total_pruned_params = 0;
  float global_sparsity = 0.0f;

  /// Compression ratio
  float compression_ratio = 1.0f;
};

// ============================================================================
// Unstructured Pruner
// ============================================================================

/// Pruner untuk unstructured pruning (individual weights)
class UnstructuredPruner {
public:
  explicit UnstructuredPruner(const PruningConfig &config)
      : config_(config), scorer_(create_scorer(config.method)) {}

  /// Generate pruning mask untuk satu tensor
  PruningMask create_mask(const float *weights, size_t size) {
    PruningMask mask(size, true);

    // Hitung importance scores
    auto scores = scorer_->score(weights, size);

    if (config_.use_threshold) {
      // Threshold-based pruning
      for (size_t i = 0; i < size; ++i) {
        if (scores[i] < config_.magnitude_threshold) {
          mask.set(i, false);
        }
      }
    } else {
      // Sparsity-based pruning
      size_t num_to_prune = static_cast<size_t>(static_cast<float>(size) *
                                                config_.target_sparsity);

      // Sort indices by score
      std::vector<size_t> indices(size);
      for (size_t i = 0; i < size; ++i)
        indices[i] = i;
      std::sort(indices.begin(), indices.end(), [&scores](size_t a, size_t b) {
        return scores[a] < scores[b];
      });

      // Prune lowest scores
      for (size_t i = 0; i < num_to_prune && i < size; ++i) {
        mask.set(indices[i], false);
      }
    }

    return mask;
  }

  /// Apply pruning langsung ke tensor
  void prune(float *weights, size_t size) {
    auto mask = create_mask(weights, size);
    mask.apply(weights, size);
  }

private:
  PruningConfig config_;
  std::unique_ptr<ImportanceScorer> scorer_;

  std::unique_ptr<ImportanceScorer> create_scorer(PruningMethod method) {
    switch (method) {
    case PruningMethod::Magnitude:
      return std::make_unique<MagnitudeScorer>();
    case PruningMethod::Random:
      return std::make_unique<RandomScorer>();
    default:
      return std::make_unique<MagnitudeScorer>();
    }
  }
};

// ============================================================================
// Structured Pruner
// ============================================================================

/// Pruner untuk structured pruning (filters/channels)
class StructuredPruner {
public:
  explicit StructuredPruner(const PruningConfig &config) : config_(config) {}

  /// Prune filters dari tensor 4D [out_channels, in_channels, H, W]
  PruningResult prune_filters(float *weights, const Shape &shape) {
    PruningResult result;

    if (shape.rank() < 2) {
      result.message = "Shape must have at least 2 dimensions";
      return result;
    }

    int64_t num_filters = shape[0];
    int64_t filter_size = shape.numel() / num_filters;

    // Hitung L1 norm per filter
    std::vector<float> filter_norms(static_cast<size_t>(num_filters));
    for (int64_t f = 0; f < num_filters; ++f) {
      float norm = 0.0f;
      for (int64_t i = 0; i < filter_size; ++i) {
        norm += std::abs(weights[f * filter_size + i]);
      }
      filter_norms[static_cast<size_t>(f)] = norm;
    }

    // Sort filters by norm
    std::vector<size_t> indices(static_cast<size_t>(num_filters));
    for (size_t i = 0; i < static_cast<size_t>(num_filters); ++i)
      indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&filter_norms](size_t a, size_t b) {
                return filter_norms[a] < filter_norms[b];
              });

    // Prune lowest-norm filters
    size_t num_to_prune = static_cast<size_t>(static_cast<float>(num_filters) *
                                              config_.target_sparsity);

    for (size_t i = 0; i < num_to_prune; ++i) {
      size_t filter_idx = indices[i];
      // Zero out entire filter
      for (int64_t j = 0; j < filter_size; ++j) {
        weights[static_cast<int64_t>(filter_idx) * filter_size + j] = 0.0f;
      }
    }

    result.success = true;
    result.total_original_params = static_cast<size_t>(shape.numel());
    result.total_pruned_params =
        num_to_prune * static_cast<size_t>(filter_size);
    result.global_sparsity = static_cast<float>(result.total_pruned_params) /
                             static_cast<float>(result.total_original_params);
    result.compression_ratio = 1.0f / (1.0f - result.global_sparsity);

    return result;
  }

  /// Prune neurons dari tensor 2D [out_features, in_features]
  PruningResult prune_neurons(float *weights, const Shape &shape) {
    PruningResult result;

    if (shape.rank() != 2) {
      result.message = "Shape must be 2D for neuron pruning";
      return result;
    }

    int64_t num_neurons = shape[0];
    int64_t neuron_size = shape[1];

    // Hitung L1 norm per neuron
    std::vector<float> neuron_norms(static_cast<size_t>(num_neurons));
    for (int64_t n = 0; n < num_neurons; ++n) {
      float norm = 0.0f;
      for (int64_t i = 0; i < neuron_size; ++i) {
        norm += std::abs(weights[n * neuron_size + i]);
      }
      neuron_norms[static_cast<size_t>(n)] = norm;
    }

    // Sort neurons by norm
    std::vector<size_t> indices(static_cast<size_t>(num_neurons));
    for (size_t i = 0; i < static_cast<size_t>(num_neurons); ++i)
      indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&neuron_norms](size_t a, size_t b) {
                return neuron_norms[a] < neuron_norms[b];
              });

    // Prune lowest-norm neurons
    size_t num_to_prune = static_cast<size_t>(static_cast<float>(num_neurons) *
                                              config_.target_sparsity);

    for (size_t i = 0; i < num_to_prune; ++i) {
      size_t neuron_idx = indices[i];
      for (int64_t j = 0; j < neuron_size; ++j) {
        weights[static_cast<int64_t>(neuron_idx) * neuron_size + j] = 0.0f;
      }
    }

    result.success = true;
    result.total_original_params = static_cast<size_t>(shape.numel());
    result.total_pruned_params =
        num_to_prune * static_cast<size_t>(neuron_size);
    result.global_sparsity = static_cast<float>(result.total_pruned_params) /
                             static_cast<float>(result.total_original_params);
    result.compression_ratio = 1.0f / (1.0f - result.global_sparsity);

    return result;
  }

private:
  PruningConfig config_;
};

// ============================================================================
// Graph Pruner (untuk GraphIR)
// ============================================================================

/// Pruner untuk mengaplikasikan pruning ke seluruh graph
class GraphPruner {
public:
  explicit GraphPruner(PruningConfig config) : config_(std::move(config)) {}

  /// Prune semua weights dalam graph
  PruningResult prune(GraphIR *graph) {
    PruningResult result;

    if (!graph) {
      result.message = "Null graph";
      return result;
    }

    // Prune semua constants (weights)
    for (auto &[name, tensor_data] : graph->constants()) {
      const auto &desc = tensor_data.descriptor();

      // Skip jika bukan float
      if (desc.dtype() != DataType::Float32) {
        continue;
      }

      // Check if this should be pruned
      if (!should_prune_tensor(name)) {
        continue;
      }

      // Get mutable data
      std::vector<uint8_t> &data =
          const_cast<std::vector<uint8_t> &>(tensor_data.data());
      float *weights = reinterpret_cast<float *>(data.data());
      size_t num_elements = static_cast<size_t>(desc.shape().numel());

      if (num_elements == 0) {
        continue;
      }

      PruningResult::LayerStats stats;
      stats.layer_name = name;
      stats.original_params = num_elements;

      if (config_.structure == PruningStructure::Unstructured) {
        UnstructuredPruner pruner(config_);
        pruner.prune(weights, num_elements);

        // Count zeros
        size_t zeros = 0;
        for (size_t i = 0; i < num_elements; ++i) {
          if (weights[i] == 0.0f)
            zeros++;
        }
        stats.pruned_params = zeros;
      } else if (config_.structure == PruningStructure::Structured) {
        StructuredPruner pruner(config_);
        if (desc.shape().rank() == 4) {
          auto layer_result = pruner.prune_filters(weights, desc.shape());
          stats.pruned_params = layer_result.total_pruned_params;
        } else if (desc.shape().rank() == 2) {
          auto layer_result = pruner.prune_neurons(weights, desc.shape());
          stats.pruned_params = layer_result.total_pruned_params;
        }
      }

      stats.sparsity = static_cast<float>(stats.pruned_params) /
                       static_cast<float>(stats.original_params);
      result.layer_stats.push_back(stats);

      result.total_original_params += stats.original_params;
      result.total_pruned_params += stats.pruned_params;
    }

    // Calculate global stats
    if (result.total_original_params > 0) {
      result.global_sparsity = static_cast<float>(result.total_pruned_params) /
                               static_cast<float>(result.total_original_params);
      float density = 1.0f - result.global_sparsity;
      result.compression_ratio = density > 0.0f ? 1.0f / density : 1.0f;
    }

    result.success = true;
    result.message = "Pruning completed successfully";

    return result;
  }

private:
  PruningConfig config_;

  bool should_prune_tensor(const std::string &name) {
    // Skip jika ada di skip list
    for (const auto &skip : config_.layers_to_skip) {
      if (name.find(skip) != std::string::npos) {
        return false;
      }
    }

    // Prune bias? Biasanya tidak
    if (name.find("bias") != std::string::npos) {
      return false;
    }

    return true;
  }
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Create default config untuk target sparsity
inline PruningConfig
create_config(float target_sparsity,
              PruningMethod method = PruningMethod::Magnitude,
              PruningStructure structure = PruningStructure::Unstructured) {
  PruningConfig config;
  config.target_sparsity = target_sparsity;
  config.method = method;
  config.structure = structure;
  return config;
}

} // namespace pruning
} // namespace zenith

#endif // ZENITH_PRUNING_HPP
