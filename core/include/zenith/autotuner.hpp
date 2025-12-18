// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Auto-tuning System untuk Kernel Selection
// Berdasarkan CetakBiru.md Bab 3.2: Kernel Scheduler & Auto-Tuner
// Referensi: TVM AutoTVM, XLA XTAT, Halide Autoscheduler

#ifndef ZENITH_AUTOTUNER_HPP
#define ZENITH_AUTOTUNER_HPP

#include "types.hpp"
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace zenith {
namespace autotuner {

// ============================================================================
// Tuning Configuration
// ============================================================================

/// Konfigurasi untuk satu parameter yang dapat di-tune
struct TunableParameter {
  std::string name;
  std::vector<int64_t> candidates; // Nilai-nilai yang mungkin
  int64_t default_value = 0;
  int64_t current_value = 0;

  TunableParameter() = default;
  TunableParameter(std::string n, std::vector<int64_t> cands, int64_t def = 0)
      : name(std::move(n)), candidates(std::move(cands)), default_value(def),
        current_value(def) {}
};

/// Konfigurasi lengkap untuk tuning satu kernel
struct TuningConfig {
  std::string kernel_name;
  std::vector<TunableParameter> parameters;

  // Constraints
  size_t max_iterations = 100;
  double timeout_seconds = 60.0;
  int warmup_runs = 3;
  int measurement_runs = 5;

  /// Hitung total kombinasi
  [[nodiscard]] size_t total_combinations() const {
    size_t total = 1;
    for (const auto &param : parameters) {
      if (!param.candidates.empty()) {
        total *= param.candidates.size();
      }
    }
    return total;
  }
};

// ============================================================================
// Tuning Result
// ============================================================================

/// Hasil dari satu pengukuran
struct MeasurementResult {
  std::vector<int64_t> config_values;
  double latency_ms = 0.0;  // Latensi rata-rata
  double latency_std = 0.0; // Standar deviasi
  double throughput_gflops = 0.0;
  bool valid = false;
  std::string error_message;
};

/// Hasil tuning lengkap
struct TuningResult {
  std::string kernel_name;
  std::vector<int64_t> best_config;
  double best_latency_ms = std::numeric_limits<double>::max();
  double baseline_latency_ms = 0.0;
  double speedup = 1.0;

  std::vector<MeasurementResult> all_measurements;
  size_t total_trials = 0;
  size_t valid_trials = 0;
  double tuning_time_seconds = 0.0;

  [[nodiscard]] bool success() const { return valid_trials > 0; }
};

// ============================================================================
// Search Strategy
// ============================================================================

/// Strategi pencarian untuk auto-tuning
enum class SearchStrategy {
  GridSearch,   // Exhaustive search
  RandomSearch, // Random sampling
  SimulatedAnnealing,
  GeneticAlgorithm,
  Bayesian, // Model-based (simplified)
};

// ============================================================================
// Cost Model (untuk guided search)
// ============================================================================

/// Model biaya sederhana untuk prediksi performa
class CostModel {
public:
  virtual ~CostModel() = default;

  /// Prediksi biaya untuk konfigurasi tertentu
  virtual double predict(const std::vector<int64_t> &config) = 0;

  /// Update model dengan hasil pengukuran
  virtual void update(const std::vector<int64_t> &config, double actual) = 0;

  /// Reset model
  virtual void reset() {}
};

/// Cost model berbasis history (simple lookup)
class HistoryCostModel : public CostModel {
public:
  double predict(const std::vector<int64_t> &config) override {
    auto key = make_key(config);
    auto it = history_.find(key);
    if (it != history_.end()) {
      return it->second;
    }
    // Return average if no history
    if (!history_.empty()) {
      double sum = 0.0;
      for (const auto &[_, v] : history_) {
        sum += v;
      }
      return sum / static_cast<double>(history_.size());
    }
    return 1e9; // Large default
  }

  void update(const std::vector<int64_t> &config, double actual) override {
    history_[make_key(config)] = actual;
  }

  void reset() override { history_.clear(); }

private:
  std::unordered_map<std::string, double> history_;

  std::string make_key(const std::vector<int64_t> &config) {
    std::string key;
    for (auto v : config) {
      key += std::to_string(v) + "_";
    }
    return key;
  }
};

// ============================================================================
// Kernel Variant
// ============================================================================

/// Representasi satu varian kernel dengan parameter tertentu
struct KernelVariant {
  std::string name;
  std::vector<int64_t> parameters;
  std::function<void()> execute_fn; // Fungsi eksekusi
};

// ============================================================================
// Auto-Tuner
// ============================================================================

/// Auto-tuner utama untuk kernel selection
class AutoTuner {
public:
  explicit AutoTuner(SearchStrategy strategy = SearchStrategy::RandomSearch,
                     int seed = 42)
      : strategy_(strategy), rng_(static_cast<unsigned int>(seed)),
        cost_model_(std::make_unique<HistoryCostModel>()) {}

  /// Set search strategy
  void set_strategy(SearchStrategy strategy) { strategy_ = strategy; }

  /// Set cost model
  void set_cost_model(std::unique_ptr<CostModel> model) {
    cost_model_ = std::move(model);
  }

  /// Tune kernel dengan konfigurasi dan benchmark function
  TuningResult
  tune(const TuningConfig &config,
       std::function<double(const std::vector<int64_t> &)> benchmark_fn) {
    TuningResult result;
    result.kernel_name = config.kernel_name;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get default config
    std::vector<int64_t> default_config;
    for (const auto &param : config.parameters) {
      default_config.push_back(param.default_value);
    }

    // Measure baseline
    result.baseline_latency_ms = benchmark_fn(default_config);

    // Generate candidates based on strategy
    std::vector<std::vector<int64_t>> candidates;
    generate_candidates(config, candidates);

    // Evaluate candidates
    for (const auto &candidate : candidates) {
      MeasurementResult measurement;
      measurement.config_values = candidate;

      try {
        double latency = benchmark_fn(candidate);
        measurement.latency_ms = latency;
        measurement.valid = true;
        result.valid_trials++;

        // Update cost model
        cost_model_->update(candidate, latency);

        // Update best
        if (latency < result.best_latency_ms) {
          result.best_latency_ms = latency;
          result.best_config = candidate;
        }
      } catch (const std::exception &e) {
        measurement.valid = false;
        measurement.error_message = e.what();
      }

      result.all_measurements.push_back(measurement);
      result.total_trials++;

      // Check timeout
      auto now = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(now - start_time).count();
      if (elapsed >= config.timeout_seconds) {
        break;
      }

      if (result.total_trials >= config.max_iterations) {
        break;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.tuning_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    // Calculate speedup
    if (result.best_latency_ms > 0 && result.baseline_latency_ms > 0) {
      result.speedup = result.baseline_latency_ms / result.best_latency_ms;
    }

    return result;
  }

  /// Tune dengan multiple measurement untuk mengurangi variance
  TuningResult tune_with_repeat(
      const TuningConfig &config,
      std::function<double(const std::vector<int64_t> &)> benchmark_fn,
      int num_repeats = 5) {
    auto wrapped_fn = [&](const std::vector<int64_t> &cfg) -> double {
      double total = 0.0;
      for (int i = 0; i < num_repeats; ++i) {
        total += benchmark_fn(cfg);
      }
      return total / num_repeats;
    };
    return tune(config, wrapped_fn);
  }

private:
  SearchStrategy strategy_;
  std::mt19937 rng_;
  std::unique_ptr<CostModel> cost_model_;

  /// Generate kandidat berdasarkan strategy
  void generate_candidates(const TuningConfig &config,
                           std::vector<std::vector<int64_t>> &candidates) {
    switch (strategy_) {
    case SearchStrategy::GridSearch:
      generate_grid_candidates(config, candidates);
      break;
    case SearchStrategy::RandomSearch:
      generate_random_candidates(config, candidates, config.max_iterations);
      break;
    case SearchStrategy::SimulatedAnnealing:
    case SearchStrategy::GeneticAlgorithm:
    case SearchStrategy::Bayesian:
      // Fallback to random for now
      generate_random_candidates(config, candidates, config.max_iterations);
      break;
    }
  }

  /// Grid search: semua kombinasi
  void generate_grid_candidates(const TuningConfig &config,
                                std::vector<std::vector<int64_t>> &candidates) {
    if (config.parameters.empty()) {
      return;
    }

    // Start with first parameter
    for (auto v : config.parameters[0].candidates) {
      candidates.push_back({v});
    }

    // Add remaining parameters
    for (size_t i = 1; i < config.parameters.size(); ++i) {
      std::vector<std::vector<int64_t>> new_candidates;
      for (const auto &existing : candidates) {
        for (auto v : config.parameters[i].candidates) {
          auto extended = existing;
          extended.push_back(v);
          new_candidates.push_back(extended);
        }
      }
      candidates = std::move(new_candidates);

      // Limit untuk mencegah explosion
      if (candidates.size() > config.max_iterations * 10) {
        break;
      }
    }

    // Shuffle dan limit
    std::shuffle(candidates.begin(), candidates.end(), rng_);
    if (candidates.size() > config.max_iterations) {
      candidates.resize(config.max_iterations);
    }
  }

  /// Random search: sampling acak
  void generate_random_candidates(const TuningConfig &config,
                                  std::vector<std::vector<int64_t>> &candidates,
                                  size_t num_samples) {
    for (size_t i = 0; i < num_samples; ++i) {
      std::vector<int64_t> sample;
      for (const auto &param : config.parameters) {
        if (param.candidates.empty()) {
          sample.push_back(param.default_value);
        } else {
          std::uniform_int_distribution<size_t> dist(
              0, param.candidates.size() - 1);
          sample.push_back(param.candidates[dist(rng_)]);
        }
      }
      candidates.push_back(sample);
    }
  }
};

// ============================================================================
// Tuning Cache
// ============================================================================

/// Cache untuk menyimpan hasil tuning
class TuningCache {
public:
  /// Simpan hasil tuning
  void save(const std::string &kernel_name, const Shape &input_shape,
            const std::vector<int64_t> &best_config) {
    std::string key = make_key(kernel_name, input_shape);
    cache_[key] = best_config;
  }

  /// Load hasil tuning (return true jika ditemukan)
  bool load(const std::string &kernel_name, const Shape &input_shape,
            std::vector<int64_t> &config) const {
    std::string key = make_key(kernel_name, input_shape);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      config = it->second;
      return true;
    }
    return false;
  }

  /// Cek apakah ada di cache
  [[nodiscard]] bool has(const std::string &kernel_name,
                         const Shape &input_shape) const {
    return cache_.count(make_key(kernel_name, input_shape)) > 0;
  }

  /// Clear cache
  void clear() { cache_.clear(); }

  /// Jumlah entry di cache
  [[nodiscard]] size_t size() const { return cache_.size(); }

private:
  std::unordered_map<std::string, std::vector<int64_t>> cache_;

  std::string make_key(const std::string &kernel_name,
                       const Shape &shape) const {
    std::string key = kernel_name + "_";
    for (size_t i = 0; i < shape.rank(); ++i) {
      key += std::to_string(shape[i]) + "_";
    }
    return key;
  }
};

// ============================================================================
// Common Tunable Parameters
// ============================================================================

namespace tunable {

/// Block sizes untuk CUDA kernels
inline TunableParameter block_size_x() {
  return {"block_size_x", {32, 64, 128, 256, 512, 1024}, 256};
}

inline TunableParameter block_size_y() {
  return {"block_size_y", {1, 2, 4, 8, 16, 32}, 1};
}

/// Tile sizes untuk GEMM
inline TunableParameter tile_m() {
  return {"tile_m", {16, 32, 64, 128, 256}, 64};
}

inline TunableParameter tile_n() {
  return {"tile_n", {16, 32, 64, 128, 256}, 64};
}

inline TunableParameter tile_k() { return {"tile_k", {8, 16, 32, 64}, 16}; }

/// Unroll factors
inline TunableParameter unroll_factor() {
  return {"unroll_factor", {1, 2, 4, 8}, 4};
}

/// Vector widths
inline TunableParameter vector_width() {
  return {"vector_width", {1, 2, 4, 8}, 4};
}

/// Use shared memory
inline TunableParameter use_shared_memory() {
  return {"use_shared_memory", {0, 1}, 1};
}

} // namespace tunable

} // namespace autotuner
} // namespace zenith

#endif // ZENITH_AUTOTUNER_HPP
