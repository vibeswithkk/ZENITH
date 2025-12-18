// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Graph Optimizer Framework untuk Zenith
// Berdasarkan CetakBiru.md Bab 3.2: High-Level Graph Optimizer
// Referensi: TensorFlow XLA, ONNX Runtime optimizers

#ifndef ZENITH_GRAPH_OPTIMIZER_HPP
#define ZENITH_GRAPH_OPTIMIZER_HPP

#include "graph_ir.hpp"
#include "types.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace zenith {
namespace optimizer {

// ============================================================================
// Optimization Pass Base Class
// ============================================================================

/// Base class untuk semua optimization passes.
/// Setiap pass mengimplementasi metode run() yang memodifikasi GraphIR.
class OptimizationPass {
public:
  virtual ~OptimizationPass() = default;

  /// Nama pass untuk logging dan debugging
  [[nodiscard]] virtual std::string name() const = 0;

  /// Deskripsi singkat tentang apa yang dilakukan pass ini
  [[nodiscard]] virtual std::string description() const = 0;

  /// Jalankan optimization pass pada graph
  /// @param graph Graph yang akan dioptimasi
  /// @return Status sukses atau error
  virtual Status run(GraphIR *graph) = 0;

  /// Apakah pass ini enabled secara default
  [[nodiscard]] virtual bool enabled_by_default() const { return true; }

  /// Level optimisasi minimum untuk pass ini (0=always, 1=O1, 2=O2, 3=O3)
  [[nodiscard]] virtual int optimization_level() const { return 1; }
};

// ============================================================================
// Pass Statistics
// ============================================================================

/// Statistik hasil dari satu pass
struct PassStatistics {
  std::string pass_name;
  bool success = false;
  int nodes_before = 0;
  int nodes_after = 0;
  int nodes_removed = 0;
  int nodes_added = 0;
  int nodes_modified = 0;
  double duration_ms = 0.0;
  std::string message;
};

// ============================================================================
// Pass Manager
// ============================================================================

/// Mengelola dan menjalankan optimization passes secara berurutan.
/// Pass manager mengatur urutan eksekusi dan mengumpulkan statistik.
class PassManager {
public:
  /// Optimization level (seperti compiler flags -O0, -O1, -O2, -O3)
  enum class Level {
    O0 = 0, // Tidak ada optimisasi
    O1 = 1, // Optimisasi dasar (constant folding, DCE)
    O2 = 2, // Optimisasi medium (O1 + fusion sederhana)
    O3 = 3, // Optimisasi agresif (O2 + fusion kompleks)
  };

  PassManager() = default;

  /// Set optimization level
  void set_level(Level level) { level_ = level; }
  [[nodiscard]] Level level() const { return level_; }

  /// Tambahkan pass ke pipeline
  void add_pass(std::unique_ptr<OptimizationPass> pass) {
    passes_.push_back(std::move(pass));
  }

  /// Tambahkan pass dengan konstruksi in-place
  template <typename PassType, typename... Args> void add_pass(Args &&...args) {
    passes_.push_back(std::make_unique<PassType>(std::forward<Args>(args)...));
  }

  /// Jalankan semua passes pada graph
  Status run(GraphIR *graph) {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph");
    }

    statistics_.clear();

    for (const auto &pass : passes_) {
      // Skip pass jika level-nya lebih tinggi dari yang diminta
      if (pass->optimization_level() > static_cast<int>(level_)) {
        continue;
      }

      // Skip pass yang disabled
      if (!pass->enabled_by_default() && level_ == Level::O0) {
        continue;
      }

      PassStatistics stats;
      stats.pass_name = pass->name();
      stats.nodes_before = static_cast<int>(graph->num_nodes());

      auto start = std::chrono::high_resolution_clock::now();
      auto status = pass->run(graph);
      auto end = std::chrono::high_resolution_clock::now();

      stats.duration_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      stats.nodes_after = static_cast<int>(graph->num_nodes());
      stats.nodes_removed = stats.nodes_before - stats.nodes_after;
      stats.success = status.ok();
      stats.message = status.message();

      statistics_.push_back(stats);

      if (!status.ok()) {
        return status;
      }
    }

    return Status::Ok();
  }

  /// Dapatkan statistik dari run terakhir
  [[nodiscard]] const std::vector<PassStatistics> &statistics() const {
    return statistics_;
  }

  /// Cetak ringkasan statistik
  [[nodiscard]] std::string summary() const {
    std::string result = "=== Optimization Summary ===\n";
    int total_removed = 0;
    double total_time = 0.0;

    for (const auto &stat : statistics_) {
      result += "  " + stat.pass_name + ": ";
      if (stat.success) {
        result += "OK";
        if (stat.nodes_removed > 0) {
          result += " (-" + std::to_string(stat.nodes_removed) + " nodes)";
        }
        if (stat.nodes_removed < 0) {
          result += " (+" + std::to_string(-stat.nodes_removed) + " nodes)";
        }
      } else {
        result += "FAILED: " + stat.message;
      }
      result += " [" + std::to_string(stat.duration_ms) + "ms]\n";
      total_removed += stat.nodes_removed;
      total_time += stat.duration_ms;
    }

    result += "Total: " + std::to_string(total_removed) + " nodes removed in " +
              std::to_string(total_time) + "ms\n";
    return result;
  }

  /// Hapus semua passes
  void clear() {
    passes_.clear();
    statistics_.clear();
  }

  /// Jumlah passes yang terdaftar
  [[nodiscard]] size_t num_passes() const { return passes_.size(); }

private:
  std::vector<std::unique_ptr<OptimizationPass>> passes_;
  std::vector<PassStatistics> statistics_;
  Level level_ = Level::O2;
};

// ============================================================================
// Standard Pass Factory
// ============================================================================

/// Factory untuk membuat PassManager dengan konfigurasi standar
class StandardPassFactory {
public:
  /// Buat PassManager dengan passes standar untuk level tertentu
  static std::unique_ptr<PassManager> create_standard(PassManager::Level level);

  /// Buat PassManager untuk inferensi (optimisasi agresif)
  static std::unique_ptr<PassManager> create_for_inference();

  /// Buat PassManager untuk training (konservatif, jaga gradients)
  static std::unique_ptr<PassManager> create_for_training();
};

} // namespace optimizer
} // namespace zenith

#endif // ZENITH_GRAPH_OPTIMIZER_HPP
