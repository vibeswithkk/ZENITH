// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_COMPILATION_SESSION_HPP
#define ZENITH_COMPILATION_SESSION_HPP

#include "backend.hpp"
#include "compiled_artifact.hpp"
#include "graph_ir.hpp"
#include "kernel_registry.hpp"
#include "target_descriptor.hpp"
#include "types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace zenith {

/// Optimization pass function type
using OptimizationPass = std::function<Status(GraphIR &)>;

/// Compilation session for optimizing and compiling models.
/// Based on CetakBiru 4.4 - CompilationSession class diagram.
/// Follows ONNX Runtime session pattern.
class CompilationSession {
public:
  /// Create a new compilation session
  CompilationSession() = default;

  /// Create session with target
  explicit CompilationSession(TargetDescriptor target)
      : target_(std::move(target)) {}

  // ========================================================================
  // Model Loading
  // ========================================================================

  /// Load model from GraphIR
  Status load(std::unique_ptr<GraphIR> graph) {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph provided");
    }
    model_ir_ = std::move(graph);
    is_compiled_ = false;
    return Status::Ok();
  }

  /// Load model from ONNX file path
  Status load_onnx(const std::string &path);

  /// Load model from ONNX bytes
  Status load_onnx_bytes(const std::vector<uint8_t> &data);

  // ========================================================================
  // Configuration
  // ========================================================================

  /// Set target descriptor
  void set_target(TargetDescriptor target) {
    target_ = std::move(target);
    is_compiled_ = false;
  }

  /// Get current target
  [[nodiscard]] const TargetDescriptor &target() const { return target_; }

  /// Add an optimization pass
  void add_pass(const std::string &name, OptimizationPass pass) {
    passes_.push_back({name, std::move(pass)});
  }

  /// Clear all optimization passes
  void clear_passes() { passes_.clear(); }

  /// Add standard optimization passes based on opt level
  void add_standard_passes();

  // ========================================================================
  // Compilation
  // ========================================================================

  /// Compile the model
  /// @return Status indicating success or failure
  Status compile() {
    if (!model_ir_) {
      return Status::Error(StatusCode::InvalidGraph, "No model loaded");
    }

    // Validate graph first
    auto validate_status = model_ir_->validate();
    if (!validate_status.ok()) {
      return validate_status;
    }

    // Run optimization passes
    auto opt_status = optimize();
    if (!opt_status.ok()) {
      return opt_status;
    }

    // Get backend
    auto backend = get_or_create_backend();
    if (!backend) {
      return Status::Error(StatusCode::NotFound,
                           "Backend not available: " + target_.backend);
    }

    // Check backend availability
    if (!backend->is_available()) {
      return Status::Error(StatusCode::NotFound,
                           "Backend not available on system: " +
                               target_.backend);
    }

    // Create compiled artifact
    compiled_artifact_ = std::make_unique<CompiledArtifact>(model_ir_->clone(),
                                                            backend, target_);

    is_compiled_ = true;
    return Status::Ok();
  }

  /// Run optimization passes
  Status optimize() {
    if (!model_ir_) {
      return Status::Error(StatusCode::InvalidGraph,
                           "No model loaded for optimization");
    }

    for (const auto &[name, pass] : passes_) {
      auto status = pass(*model_ir_);
      if (!status.ok()) {
        return Status::Error(status.code(),
                             "Optimization pass '" + name +
                                 "' failed: " + status.message());
      }
    }

    return Status::Ok();
  }

  /// Get compiled artifact
  /// @return Pointer to compiled artifact, or nullptr if not compiled
  [[nodiscard]] CompiledArtifact *get_compiled_artifact() const {
    return compiled_artifact_.get();
  }

  /// Transfer ownership of compiled artifact
  [[nodiscard]] std::unique_ptr<CompiledArtifact> release_artifact() {
    is_compiled_ = false;
    return std::move(compiled_artifact_);
  }

  // ========================================================================
  // Query Methods
  // ========================================================================

  /// Check if model is loaded
  [[nodiscard]] bool has_model() const { return model_ir_ != nullptr; }

  /// Check if model is compiled
  [[nodiscard]] bool is_compiled() const { return is_compiled_; }

  /// Get current graph (before or after optimization)
  [[nodiscard]] const GraphIR *graph() const { return model_ir_.get(); }

  /// Get mutable graph for advanced manipulation
  [[nodiscard]] GraphIR *mutable_graph() {
    is_compiled_ = false;
    return model_ir_.get();
  }

  /// Get list of registered optimization passes
  [[nodiscard]] std::vector<std::string> list_passes() const {
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto &[name, _] : passes_) {
      names.push_back(name);
    }
    return names;
  }

  /// Generate session summary
  [[nodiscard]] std::string summary() const {
    std::string result = "CompilationSession:\n";
    result += "  Target: " + target_.to_string() + "\n";
    result += "  Model: " + (model_ir_ ? model_ir_->name() : "(none)") + "\n";
    result += "  Compiled: " + std::string(is_compiled_ ? "yes" : "no") + "\n";
    result += "  Passes: " + std::to_string(passes_.size()) + "\n";

    if (model_ir_) {
      result += "  Nodes: " + std::to_string(model_ir_->num_nodes()) + "\n";
    }

    return result;
  }

private:
  /// Get or create backend for current target
  std::shared_ptr<Backend> get_or_create_backend();

  TargetDescriptor target_;
  std::unique_ptr<GraphIR> model_ir_;
  std::unique_ptr<CompiledArtifact> compiled_artifact_;
  std::vector<std::pair<std::string, OptimizationPass>> passes_;
  bool is_compiled_ = false;
};

} // namespace zenith

#endif // ZENITH_COMPILATION_SESSION_HPP
