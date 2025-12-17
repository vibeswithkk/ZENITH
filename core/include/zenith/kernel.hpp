// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_KERNEL_HPP
#define ZENITH_KERNEL_HPP

#include "op_signature.hpp"
#include "types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace zenith {

/// Forward declarations
class Backend;

/// Kernel execution arguments
struct KernelArgs {
  std::vector<const void *> inputs;
  std::vector<void *> outputs;
  std::vector<Shape> input_shapes;
  std::vector<Shape> output_shapes;
  void *workspace = nullptr;
  size_t workspace_size = 0;
};

/// Abstract base class for all computational kernels.
/// Follows the TensorRT IPlugin pattern from CetakBiru 4.4.
class Kernel {
public:
  virtual ~Kernel() = default;

  /// Get the kernel name
  [[nodiscard]] virtual std::string name() const = 0;

  /// Get the operation type this kernel implements
  [[nodiscard]] virtual std::string op_type() const = 0;

  /// Get supported data types
  [[nodiscard]] virtual std::vector<DataType> supported_dtypes() const {
    return {DataType::Float32};
  }

  /// Check if kernel supports given signature
  [[nodiscard]] virtual bool supports(const OpSignature &sig) const {
    if (sig.op_type != op_type())
      return false;
    auto dtypes = supported_dtypes();
    for (auto dt : dtypes) {
      if (dt == sig.dtype)
        return true;
    }
    return false;
  }

  /// Compute required workspace size
  [[nodiscard]] virtual size_t
  get_workspace_size(const std::vector<Shape> &input_shapes) const {
    (void)input_shapes;
    return 0;
  }

  /// Compute output shapes given input shapes
  [[nodiscard]] virtual std::vector<Shape>
  infer_output_shapes(const std::vector<Shape> &input_shapes,
                      const AttributeMap &attrs) const = 0;

  /// Execute the kernel
  virtual Status execute(const KernelArgs &args, Backend *backend) = 0;

  /// Clone the kernel
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone() const = 0;
};

/// Base class for kernel factories (creators)
class KernelCreator {
public:
  virtual ~KernelCreator() = default;

  /// Get the operation type this creator handles
  [[nodiscard]] virtual std::string op_type() const = 0;

  /// Get creator version
  [[nodiscard]] virtual std::string version() const { return "1.0"; }

  /// Create a kernel instance
  [[nodiscard]] virtual std::unique_ptr<Kernel>
  create(const AttributeMap &attrs) const = 0;
};

} // namespace zenith

#endif // ZENITH_KERNEL_HPP
