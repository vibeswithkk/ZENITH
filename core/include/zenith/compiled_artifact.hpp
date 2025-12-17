// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_COMPILED_ARTIFACT_HPP
#define ZENITH_COMPILED_ARTIFACT_HPP

#include "backend.hpp"
#include "graph_ir.hpp"
#include "target_descriptor.hpp"
#include "types.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

namespace zenith {

/// Execution statistics
struct ExecutionStats {
  size_t num_runs = 0;
  double total_time_ms = 0.0;
  double min_time_ms = std::numeric_limits<double>::max();
  double max_time_ms = 0.0;
  size_t peak_memory_bytes = 0;

  [[nodiscard]] double avg_time_ms() const {
    return num_runs > 0 ? total_time_ms / static_cast<double>(num_runs) : 0.0;
  }

  void record(double time_ms, size_t memory = 0) {
    num_runs++;
    total_time_ms += time_ms;
    min_time_ms = std::min(min_time_ms, time_ms);
    max_time_ms = std::max(max_time_ms, time_ms);
    peak_memory_bytes = std::max(peak_memory_bytes, memory);
  }
};

/// Input map type for execution
using InputMap = std::unordered_map<std::string, const void *>;

/// Output map type for execution
using OutputMap = std::unordered_map<std::string, void *>;

/// Compiled model artifact ready for execution.
/// Based on CetakBiru 4.4 - CompilationSession output.
class CompiledArtifact {
public:
  CompiledArtifact() = default;

  CompiledArtifact(std::unique_ptr<GraphIR> optimized_graph,
                   std::shared_ptr<Backend> backend, TargetDescriptor target)
      : optimized_graph_(std::move(optimized_graph)),
        backend_(std::move(backend)), target_(std::move(target)) {}

  /// Execute the compiled model
  /// @param inputs Map of input tensor names to data pointers
  /// @param outputs Map of output tensor names to data pointers (pre-allocated)
  /// @return Status indicating success or failure
  Status run(const InputMap &inputs, OutputMap &outputs) {
    if (!optimized_graph_) {
      return Status::Error(StatusCode::InvalidGraph,
                           "No graph loaded in artifact");
    }
    if (!backend_) {
      return Status::Error(StatusCode::InvalidArgument, "No backend available");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Execute through backend
    auto status = backend_->execute(*optimized_graph_, inputs, outputs);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    stats_.record(elapsed_ms);

    return status;
  }

  /// Get the optimized graph
  [[nodiscard]] const GraphIR *graph() const { return optimized_graph_.get(); }

  /// Get the target backend
  [[nodiscard]] Backend *backend() const { return backend_.get(); }

  /// Get the target descriptor
  [[nodiscard]] const TargetDescriptor &target() const { return target_; }

  /// Get execution statistics
  [[nodiscard]] const ExecutionStats &stats() const { return stats_; }

  /// Check if artifact is valid
  [[nodiscard]] bool is_valid() const {
    return optimized_graph_ != nullptr && backend_ != nullptr;
  }

  /// Get graph input tensor descriptors
  [[nodiscard]] const std::vector<TensorDescriptor> &inputs() const {
    static std::vector<TensorDescriptor> empty;
    return optimized_graph_ ? optimized_graph_->inputs() : empty;
  }

  /// Get graph output tensor descriptors
  [[nodiscard]] const std::vector<TensorDescriptor> &outputs() const {
    static std::vector<TensorDescriptor> empty;
    return optimized_graph_ ? optimized_graph_->outputs() : empty;
  }

  /// Generate summary string
  [[nodiscard]] std::string summary() const {
    if (!optimized_graph_) {
      return "CompiledArtifact: (empty)";
    }

    std::string result = "CompiledArtifact:\n";
    result += "  Graph: " + optimized_graph_->name() + "\n";
    result +=
        "  Nodes: " + std::to_string(optimized_graph_->num_nodes()) + "\n";
    result += "  Target: " + target_.to_string() + "\n";
    result +=
        "  Inputs: " + std::to_string(optimized_graph_->inputs().size()) + "\n";
    result +=
        "  Outputs: " + std::to_string(optimized_graph_->outputs().size()) +
        "\n";

    if (stats_.num_runs > 0) {
      result += "  Avg Time: " + std::to_string(stats_.avg_time_ms()) + " ms\n";
    }

    return result;
  }

private:
  std::unique_ptr<GraphIR> optimized_graph_;
  std::shared_ptr<Backend> backend_;
  TargetDescriptor target_;
  ExecutionStats stats_;
};

} // namespace zenith

#endif // ZENITH_COMPILED_ARTIFACT_HPP
