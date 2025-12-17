// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Backend Dispatcher for Zenith Framework
// Implements intelligent operation routing based on blueprint Section 4.4.
// The dispatcher selects optimal backend/kernel for each operation.

#ifndef ZENITH_DISPATCHER_HPP
#define ZENITH_DISPATCHER_HPP

#include "backend.hpp"
#include "cublas_ops.hpp"
#include "cudnn_ops.hpp"
#include "graph_ir.hpp"
#include "kernel_registry.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace zenith {

// ============================================================================
// Operation Handler Type
// ============================================================================

/// Handler function signature for operation execution
using OpHandler = std::function<Status(
    const Node &, std::unordered_map<std::string, void *> &)>;

// ============================================================================
// Backend Preference
// ============================================================================

/// Preference level for backend selection
enum class BackendPref {
  Required,  // Must use this backend, fail if unavailable
  Preferred, // Use if available, fallback otherwise
  Optional,  // Use only if no other option
};

/// Backend selection criteria
struct BackendCriteria {
  std::string backend_name;
  BackendPref preference = BackendPref::Preferred;
  bool use_cudnn = true;  // Prefer cuDNN for supported ops
  bool use_cublas = true; // Prefer cuBLAS for matmul
};

// ============================================================================
// Dispatcher Class
// ============================================================================

/// Central dispatcher for routing operations to appropriate backends/kernels.
/// Per CetakBiru Section 4.4: "KernelRegistry adalah inti dari sistem dispatch"
class Dispatcher {
public:
  /// Get singleton instance
  static Dispatcher &instance() {
    static Dispatcher dispatcher;
    return dispatcher;
  }

  // ==========================================================================
  // Backend Registration
  // ==========================================================================

  /// Register a backend
  void register_backend(std::shared_ptr<Backend> backend) {
    std::lock_guard<std::mutex> lock(mutex_);
    backends_[backend->name()] = std::move(backend);
  }

  /// Register an operation handler for a specific backend
  void register_handler(const std::string &op_type,
                        const std::string &backend_name, OpHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    handlers_[op_type][backend_name] = std::move(handler);
  }

  // ==========================================================================
  // Backend Selection
  // ==========================================================================

  /// Get backend by name
  [[nodiscard]] Backend *get_backend(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(name);
    return it != backends_.end() ? it->second.get() : nullptr;
  }

  /// Get default available backend (CUDA > CPU)
  [[nodiscard]] Backend *get_default_backend() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Priority order: cuda, rocm, oneapi, cpu
    static const std::vector<std::string> priority = {"cuda", "rocm", "oneapi",
                                                      "cpu"};

    for (const auto &name : priority) {
      auto it = backends_.find(name);
      if (it != backends_.end() && it->second->is_available()) {
        return it->second.get();
      }
    }

    return nullptr;
  }

  /// Select best backend for an operation
  [[nodiscard]] Backend *
  select_backend(const Node &node,
                 const BackendCriteria &criteria = BackendCriteria()) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const std::string &op = node.op_type();

    // If specific backend required
    if (criteria.preference == BackendPref::Required) {
      auto it = backends_.find(criteria.backend_name);
      if (it != backends_.end() && it->second->is_available()) {
        return it->second.get();
      }
      return nullptr;
    }

    // Try preferred backend first
    if (!criteria.backend_name.empty()) {
      auto it = backends_.find(criteria.backend_name);
      if (it != backends_.end() && it->second->is_available() &&
          it->second->supports_op(op)) {
        return it->second.get();
      }
    }

    // Find best available backend that supports this operation
    for (const auto &[name, backend] : backends_) {
      if (backend->is_available() && backend->supports_op(op)) {
        return backend.get();
      }
    }

    return nullptr;
  }

  // ==========================================================================
  // Operation Dispatch
  // ==========================================================================

  /// Dispatch a node to appropriate backend/kernel
  [[nodiscard]] Status
  dispatch(const Node &node, std::unordered_map<std::string, void *> &tensors,
           const BackendCriteria &criteria = BackendCriteria()) {
    const std::string &op = node.op_type();

    // Check for cuDNN/cuBLAS optimized paths first
    if (should_use_cudnn(op, criteria)) {
      auto status = dispatch_cudnn(node, tensors);
      if (status.ok()) {
        return status;
      }
      // Fall through to regular dispatch on failure
    }

    if (should_use_cublas(op, criteria)) {
      auto status = dispatch_cublas(node, tensors);
      if (status.ok()) {
        return status;
      }
    }

    // Find handler for this operation
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto op_it = handlers_.find(op);
      if (op_it != handlers_.end()) {
        // Try to find handler for preferred backend
        std::string target_backend =
            criteria.backend_name.empty()
                ? (get_default_backend() ? get_default_backend()->name()
                                         : "cpu")
                : criteria.backend_name;

        auto handler_it = op_it->second.find(target_backend);
        if (handler_it != op_it->second.end()) {
          return handler_it->second(node, tensors);
        }

        // Try any available handler
        for (const auto &[backend_name, handler] : op_it->second) {
          auto backend_it = backends_.find(backend_name);
          if (backend_it != backends_.end() &&
              backend_it->second->is_available()) {
            return handler(node, tensors);
          }
        }
      }
    }

    // Use KernelRegistry as fallback
    auto *kernel = KernelRegistry::instance().dispatch(
        OpSignature{op, {}, {}},
        criteria.backend_name.empty() ? "cpu" : criteria.backend_name);

    if (kernel) {
      // Kernel execution would go here
      return Status::Ok();
    }

    return Status::Error(StatusCode::UnsupportedOp,
                         "No handler found for operation: " + op);
  }

  /// Execute entire graph with automatic dispatch
  [[nodiscard]] Status
  execute_graph(const GraphIR &graph,
                const std::unordered_map<std::string, const void *> &inputs,
                std::unordered_map<std::string, void *> &outputs,
                const BackendCriteria &criteria = BackendCriteria()) {

    Backend *backend =
        select_backend(Node(), // Empty node for general selection
                       criteria);

    if (!backend) {
      return Status::Error(StatusCode::NotFound, "No suitable backend found");
    }

    return backend->execute(graph, inputs, outputs);
  }

  // ==========================================================================
  // Utility
  // ==========================================================================

  /// List available backends
  [[nodiscard]] std::vector<std::string> list_backends() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(backends_.size());
    for (const auto &[name, _] : backends_) {
      names.push_back(name);
    }
    return names;
  }

  /// Check if a specific backend is available
  [[nodiscard]] bool is_backend_available(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(name);
    return it != backends_.end() && it->second->is_available();
  }

  /// Get dispatch statistics
  struct DispatchStats {
    size_t total_dispatches = 0;
    size_t cudnn_dispatches = 0;
    size_t cublas_dispatches = 0;
    size_t fallback_dispatches = 0;
  };

  [[nodiscard]] DispatchStats get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  void reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = DispatchStats{};
  }

private:
  Dispatcher() = default;

  // ==========================================================================
  // cuDNN/cuBLAS Dispatch Helpers
  // ==========================================================================

  [[nodiscard]] bool should_use_cudnn(const std::string &op,
                                      const BackendCriteria &criteria) const {
    if (!criteria.use_cudnn)
      return false;

#ifdef ZENITH_HAS_CUDNN
    if (!cudnn::is_cudnn_available())
      return false;

    // Operations supported by cuDNN
    static const std::vector<std::string> cudnn_ops = {"Conv",
                                                       "Conv2D",
                                                       "Relu",
                                                       "Sigmoid",
                                                       "Tanh",
                                                       "MaxPool",
                                                       "AvgPool",
                                                       "Softmax",
                                                       "BatchNormalization",
                                                       "GlobalAveragePool"};

    for (const auto &cudnn_op : cudnn_ops) {
      if (op == cudnn_op)
        return true;
    }
#endif
    return false;
  }

  [[nodiscard]] bool should_use_cublas(const std::string &op,
                                       const BackendCriteria &criteria) const {
    if (!criteria.use_cublas)
      return false;

#ifdef ZENITH_HAS_CUDA
    if (!cublas::is_cublas_available())
      return false;

    // Operations supported by cuBLAS
    static const std::vector<std::string> cublas_ops = {"MatMul", "Gemm"};

    for (const auto &cublas_op : cublas_ops) {
      if (op == cublas_op)
        return true;
    }
#endif
    return false;
  }

  [[nodiscard]] Status
  dispatch_cudnn(const Node &node,
                 std::unordered_map<std::string, void *> &tensors) {
#ifdef ZENITH_HAS_CUDNN
    const std::string &op = node.op_type();

    // ReLU dispatch
    if (op == "Relu") {
      if (node.inputs().empty() || node.outputs().empty()) {
        return Status::Error(StatusCode::InvalidInput,
                             "ReLU requires input and output tensors");
      }

      const auto &input_desc = node.inputs()[0];
      void *input = tensors[input_desc.name()];
      void *output = tensors[node.outputs()[0].name()];

      if (!input || !output) {
        return Status::Error(StatusCode::InvalidInput, "Missing tensor data");
      }

      auto &shape = input_desc.shape();
      int N = shape.rank() >= 1 ? shape[0] : 1;
      int C = shape.rank() >= 2 ? shape[1] : 1;
      int H = shape.rank() >= 3 ? shape[2] : 1;
      int W = shape.rank() >= 4 ? shape[3] : 1;

      auto status =
          cudnn::relu_forward(static_cast<float *>(input),
                              static_cast<float *>(output), N, C, H, W);

      if (status.ok()) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.cudnn_dispatches++;
        stats_.total_dispatches++;
      }
      return status;
    }

    // Add more cuDNN dispatches here...

    return Status::Error(StatusCode::UnsupportedOp,
                         "cuDNN dispatch not implemented for: " + op);
#else
    return Status::Error(StatusCode::NotImplemented, "cuDNN not available");
#endif
  }

  [[nodiscard]] Status
  dispatch_cublas(const Node &node,
                  std::unordered_map<std::string, void *> &tensors) {
#ifdef ZENITH_HAS_CUDA
    const std::string &op = node.op_type();

    // MatMul / Gemm dispatch
    if (op == "MatMul" || op == "Gemm") {
      if (node.inputs().size() < 2 || node.outputs().empty()) {
        return Status::Error(StatusCode::InvalidInput,
                             "MatMul requires 2 inputs and 1 output");
      }

      const auto &a_desc = node.inputs()[0];
      const auto &b_desc = node.inputs()[1];
      const auto &c_desc = node.outputs()[0];

      void *A = tensors[a_desc.name()];
      void *B = tensors[b_desc.name()];
      void *C = tensors[c_desc.name()];

      if (!A || !B || !C) {
        return Status::Error(StatusCode::InvalidInput,
                             "Missing tensor data for MatMul");
      }

      // Get dimensions from shapes
      auto &a_shape = a_desc.shape();
      auto &b_shape = b_desc.shape();

      int M = a_shape.rank() >= 1 ? a_shape[0] : 1;
      int K = a_shape.rank() >= 2 ? a_shape[1] : 1;
      int N = b_shape.rank() >= 2 ? b_shape[1] : 1;

      auto status =
          cublas::gemm_f32(static_cast<float *>(A), static_cast<float *>(B),
                           static_cast<float *>(C), M, N, K);

      if (status.ok()) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.cublas_dispatches++;
        stats_.total_dispatches++;
      }
      return status;
    }

    return Status::Error(StatusCode::UnsupportedOp,
                         "cuBLAS dispatch not implemented for: " + op);
#else
    return Status::Error(StatusCode::NotImplemented, "cuBLAS not available");
#endif
  }

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
  std::unordered_map<std::string, std::unordered_map<std::string, OpHandler>>
      handlers_;
  DispatchStats stats_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get default dispatcher
inline Dispatcher &dispatcher() { return Dispatcher::instance(); }

/// Quick dispatch helper
inline Status dispatch_op(const Node &node,
                          std::unordered_map<std::string, void *> &tensors) {
  return Dispatcher::instance().dispatch(node, tensors);
}

} // namespace zenith

#endif // ZENITH_DISPATCHER_HPP
