// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_BACKEND_HPP
#define ZENITH_BACKEND_HPP

#include "graph_ir.hpp"
#include "types.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace zenith {

// ============================================================================
// Backend - Hardware Abstraction Layer Base Class
// ============================================================================

/// Abstract base class for all hardware backends.
/// Provides a uniform interface for executing computation graphs on
/// different hardware targets (CPU, CUDA, ROCm, etc.)
///
/// Based on CetakBiru Section 4.2 - Hardware Abstraction Layer
class Backend {
public:
  virtual ~Backend() = default;

  // ========================================================================
  // Backend Identity
  // ========================================================================

  /// Get the name of this backend (e.g., "cpu", "cuda", "rocm")
  [[nodiscard]] virtual std::string name() const = 0;

  /// Get a human-readable description
  [[nodiscard]] virtual std::string description() const = 0;

  // ========================================================================
  // Availability and Capabilities
  // ========================================================================

  /// Check if this backend is available on the current system
  [[nodiscard]] virtual bool is_available() const = 0;

  /// Get the compute capability or version string
  [[nodiscard]] virtual std::string version() const { return "1.0"; }

  /// Check if backend supports a specific data type
  [[nodiscard]] virtual bool supports_dtype(DataType dtype) const {
    // Default: support common types
    return dtype == DataType::Float32 || dtype == DataType::Float64 ||
           dtype == DataType::Int32 || dtype == DataType::Int64;
  }

  /// Check if backend supports a specific operation
  [[nodiscard]] virtual bool supports_op(const std::string &op_type) const {
    // Default: check against known supported ops
    static const std::vector<std::string> basic_ops = {
        "Add",  "Sub",     "Mul",     "Div",     "MatMul",
        "Gemm", "Conv",    "Conv2D",  "Relu",    "Sigmoid",
        "Tanh", "Softmax", "MaxPool", "AvgPool", "Flatten"};
    for (const auto &op : basic_ops) {
      if (op == op_type)
        return true;
    }
    return false;
  }

  // ========================================================================
  // Execution
  // ========================================================================

  /// Execute a computation graph
  /// @param graph The graph to execute
  /// @param inputs Map of input tensor names to data pointers
  /// @param outputs Map of output tensor names to data pointers (pre-allocated)
  /// @return Status indicating success or failure
  [[nodiscard]] virtual Status
  execute(const GraphIR &graph,
          const std::unordered_map<std::string, const void *> &inputs,
          std::unordered_map<std::string, void *> &outputs) = 0;

  // ========================================================================
  // Memory Management
  // ========================================================================

  /// Allocate memory on this backend's device
  /// @param size_bytes Number of bytes to allocate
  /// @return Pointer to allocated memory, or nullptr on failure
  [[nodiscard]] virtual void *allocate(size_t size_bytes) = 0;

  /// Free memory allocated by this backend
  virtual void deallocate(void *ptr) = 0;

  /// Copy data from host to device
  virtual Status copy_to_device(void *dst, const void *src, size_t size_bytes) {
    // Default: memcpy for CPU-like backends
    std::memcpy(dst, src, size_bytes);
    return Status::Ok();
  }

  /// Copy data from device to host
  virtual Status copy_to_host(void *dst, const void *src, size_t size_bytes) {
    // Default: memcpy for CPU-like backends
    std::memcpy(dst, src, size_bytes);
    return Status::Ok();
  }

  // ========================================================================
  // Synchronization
  // ========================================================================

  /// Wait for all pending operations to complete
  virtual void synchronize() {
    // Default: no-op for synchronous backends
  }
};

// ============================================================================
// Backend Registry
// ============================================================================

/// Singleton registry for available backends
class BackendRegistry {
public:
  static BackendRegistry &instance() {
    static BackendRegistry registry;
    return registry;
  }

  /// Register a backend
  void register_backend(std::shared_ptr<Backend> backend) {
    backends_[backend->name()] = std::move(backend);
  }

  /// Get a backend by name
  [[nodiscard]] Backend *get(const std::string &name) const {
    auto it = backends_.find(name);
    if (it != backends_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  /// Get default backend (first available)
  [[nodiscard]] Backend *get_default() const {
    for (const auto &[name, backend] : backends_) {
      if (backend->is_available()) {
        return backend.get();
      }
    }
    return nullptr;
  }

  /// List all registered backend names
  [[nodiscard]] std::vector<std::string> list_backends() const {
    std::vector<std::string> names;
    names.reserve(backends_.size());
    for (const auto &[name, _] : backends_) {
      names.push_back(name);
    }
    return names;
  }

private:
  BackendRegistry() = default;
  std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
};

} // namespace zenith

#endif // ZENITH_BACKEND_HPP
