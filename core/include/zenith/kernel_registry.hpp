// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_KERNEL_REGISTRY_HPP
#define ZENITH_KERNEL_REGISTRY_HPP

#include "kernel.hpp"
#include "op_signature.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace zenith {

/// Global registry for kernel implementations.
/// Based on CetakBiru 4.4 - KernelRegistry class diagram.
/// Follows TensorRT IPluginRegistry pattern.
class KernelRegistry {
public:
  /// Get the singleton instance
  static KernelRegistry &instance() {
    static KernelRegistry registry;
    return registry;
  }

  // Non-copyable, non-movable
  KernelRegistry(const KernelRegistry &) = delete;
  KernelRegistry &operator=(const KernelRegistry &) = delete;
  KernelRegistry(KernelRegistry &&) = delete;
  KernelRegistry &operator=(KernelRegistry &&) = delete;

  /// Register a kernel creator
  /// @param creator The kernel creator to register
  void register_creator(std::shared_ptr<KernelCreator> creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string &op = creator->op_type();
    creators_[op] = std::move(creator);
  }

  /// Register a kernel instance directly
  /// @param signature The operation signature
  /// @param kernel The kernel implementation
  void register_kernel(const OpSignature &signature,
                       std::shared_ptr<Kernel> kernel) {
    std::lock_guard<std::mutex> lock(mutex_);
    kernels_[signature] = std::move(kernel);
  }

  /// Get a kernel creator by operation type
  /// @param op_type The operation type
  /// @return Pointer to creator, or nullptr if not found
  [[nodiscard]] KernelCreator *get_creator(const std::string &op_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = creators_.find(op_type);
    if (it != creators_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  /// Dispatch: find the best kernel for given signature and target
  /// @param signature The operation signature
  /// @param target The target backend name
  /// @return Pointer to kernel, or nullptr if not found
  [[nodiscard]] Kernel *dispatch(const OpSignature &signature,
                                 const std::string &target) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // First, try exact match
    auto it = kernels_.find(signature);
    if (it != kernels_.end()) {
      return it->second.get();
    }

    // Second, try finding a kernel that supports this signature
    for (const auto &[sig, kernel] : kernels_) {
      if (kernel->supports(signature)) {
        return kernel.get();
      }
    }

    // Third, try creating from creator
    auto creator_it = creators_.find(signature.op_type);
    if (creator_it != creators_.end()) {
      // Note: In a full implementation, we'd cache the created kernel
      return nullptr; // Caller should use create_kernel instead
    }

    return nullptr;
  }

  /// Create a kernel using registered creator
  /// @param op_type The operation type
  /// @param attrs Attributes for kernel creation
  /// @return Unique pointer to created kernel, or nullptr
  [[nodiscard]] std::unique_ptr<Kernel>
  create_kernel(const std::string &op_type, const AttributeMap &attrs) const {
    auto *creator = get_creator(op_type);
    if (creator) {
      return creator->create(attrs);
    }
    return nullptr;
  }

  /// List all registered operation types
  [[nodiscard]] std::vector<std::string> list_ops() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ops;
    ops.reserve(creators_.size());
    for (const auto &[op, _] : creators_) {
      ops.push_back(op);
    }
    return ops;
  }

  /// Check if an operation type is registered
  [[nodiscard]] bool has_op(const std::string &op_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return creators_.find(op_type) != creators_.end() ||
           std::any_of(kernels_.begin(), kernels_.end(), [&](const auto &pair) {
             return pair.first.op_type == op_type;
           });
  }

  /// Get count of registered kernels
  [[nodiscard]] size_t kernel_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.size();
  }

  /// Get count of registered creators
  [[nodiscard]] size_t creator_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return creators_.size();
  }

  /// Clear all registrations (mainly for testing)
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    kernels_.clear();
    creators_.clear();
  }

private:
  KernelRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<OpSignature, std::shared_ptr<Kernel>> kernels_;
  std::unordered_map<std::string, std::shared_ptr<KernelCreator>> creators_;
};

/// Helper macro for kernel registration
#define ZENITH_REGISTER_KERNEL(op_type, kernel_class)                          \
  static bool _zenith_kernel_##kernel_class##_registered = []() {              \
    zenith::KernelRegistry::instance().register_creator(                       \
        std::make_shared<kernel_class##Creator>());                            \
    return true;                                                               \
  }()

} // namespace zenith

#endif // ZENITH_KERNEL_REGISTRY_HPP
