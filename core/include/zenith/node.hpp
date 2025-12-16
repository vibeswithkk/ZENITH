// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_NODE_HPP
#define ZENITH_NODE_HPP

#include "tensor.hpp"
#include "types.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace zenith {

// ============================================================================
// Forward Declarations
// ============================================================================

class GraphIR;

// ============================================================================
// Node - Represents a single operation in the computation graph
// ============================================================================

/// Represents a single operation (node) in the computation graph.
/// Based on the class diagram in section 4.4 of the blueprint.
class Node {
public:
  /// Unique identifier counter
  static inline size_t next_id_ = 0;

  Node() : id_(next_id_++) {}

  Node(std::string op_type, std::string name,
       std::vector<TensorDescriptor> inputs,
       std::vector<TensorDescriptor> outputs, AttributeMap attrs = {})
      : id_(next_id_++), op_type_(std::move(op_type)), name_(std::move(name)),
        inputs_(std::move(inputs)), outputs_(std::move(outputs)),
        attrs_(std::move(attrs)) {}

  // ========================================================================
  // Accessors
  // ========================================================================

  [[nodiscard]] size_t id() const { return id_; }
  [[nodiscard]] const std::string &op_type() const { return op_type_; }
  [[nodiscard]] const std::string &name() const { return name_; }
  [[nodiscard]] const std::vector<TensorDescriptor> &inputs() const {
    return inputs_;
  }
  [[nodiscard]] const std::vector<TensorDescriptor> &outputs() const {
    return outputs_;
  }
  [[nodiscard]] const AttributeMap &attrs() const { return attrs_; }

  // ========================================================================
  // Mutators
  // ========================================================================

  void set_op_type(std::string op_type) { op_type_ = std::move(op_type); }
  void set_name(std::string name) { name_ = std::move(name); }

  void add_input(TensorDescriptor input) {
    inputs_.push_back(std::move(input));
  }

  void add_output(TensorDescriptor output) {
    outputs_.push_back(std::move(output));
  }

  void set_attr(const std::string &key, AttributeValue value) {
    attrs_[key] = std::move(value);
  }

  // ========================================================================
  // Attribute Access Helpers
  // ========================================================================

  /// Get attribute value with type checking
  template <typename T>
  [[nodiscard]] std::optional<T> get_attr(const std::string &key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end())
      return std::nullopt;
    if (auto *val = std::get_if<T>(&it->second)) {
      return *val;
    }
    return std::nullopt;
  }

  /// Check if attribute exists
  [[nodiscard]] bool has_attr(const std::string &key) const {
    return attrs_.find(key) != attrs_.end();
  }

  /// Get integer attribute
  [[nodiscard]] std::optional<int64_t>
  get_attr_int(const std::string &key) const {
    return get_attr<int64_t>(key);
  }

  /// Get float attribute
  [[nodiscard]] std::optional<double>
  get_attr_float(const std::string &key) const {
    return get_attr<double>(key);
  }

  /// Get string attribute
  [[nodiscard]] std::optional<std::string>
  get_attr_string(const std::string &key) const {
    return get_attr<std::string>(key);
  }

  /// Get vector of integers attribute
  [[nodiscard]] const std::vector<int64_t> *
  get_attr_ints(const std::string &key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end())
      return nullptr;
    if (auto *val = std::get_if<std::vector<int64_t>>(&it->second)) {
      return val;
    }
    return nullptr;
  }

  /// Get vector of floats attribute
  [[nodiscard]] const std::vector<double> *
  get_attr_floats(const std::string &key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end())
      return nullptr;
    if (auto *val = std::get_if<std::vector<double>>(&it->second)) {
      return val;
    }
    return nullptr;
  }

  // ========================================================================
  // Utility Methods
  // ========================================================================

  /// Get number of inputs
  [[nodiscard]] size_t num_inputs() const { return inputs_.size(); }

  /// Get number of outputs
  [[nodiscard]] size_t num_outputs() const { return outputs_.size(); }

  /// Check if this is a specific operation type
  [[nodiscard]] bool is_op(const std::string &op) const {
    return op_type_ == op;
  }

  /// Clone this node
  [[nodiscard]] std::unique_ptr<Node> clone() const {
    return std::make_unique<Node>(op_type_, name_, inputs_, outputs_, attrs_);
  }

private:
  size_t id_;
  std::string op_type_; // e.g., "Conv", "MatMul", "Relu"
  std::string name_;    // Unique name in the graph
  std::vector<TensorDescriptor> inputs_;
  std::vector<TensorDescriptor> outputs_;
  AttributeMap attrs_;
};

// ============================================================================
// Common Operation Types (as constants)
// ============================================================================

namespace ops {
// Activation operations
inline constexpr const char *RELU = "Relu";
inline constexpr const char *GELU = "Gelu";
inline constexpr const char *SIGMOID = "Sigmoid";
inline constexpr const char *TANH = "Tanh";
inline constexpr const char *SOFTMAX = "Softmax";

// Linear operations
inline constexpr const char *MATMUL = "MatMul";
inline constexpr const char *GEMM = "Gemm";
inline constexpr const char *LINEAR = "Linear";

// Convolution operations
inline constexpr const char *CONV = "Conv";
inline constexpr const char *CONV_TRANSPOSE = "ConvTranspose";

// Normalization operations
inline constexpr const char *BATCH_NORM = "BatchNormalization";
inline constexpr const char *LAYER_NORM = "LayerNormalization";
inline constexpr const char *INSTANCE_NORM = "InstanceNormalization";

// Pooling operations
inline constexpr const char *MAX_POOL = "MaxPool";
inline constexpr const char *AVG_POOL = "AveragePool";
inline constexpr const char *GLOBAL_AVG_POOL = "GlobalAveragePool";

// Element-wise operations
inline constexpr const char *ADD = "Add";
inline constexpr const char *SUB = "Sub";
inline constexpr const char *MUL = "Mul";
inline constexpr const char *DIV = "Div";

// Shape operations
inline constexpr const char *RESHAPE = "Reshape";
inline constexpr const char *TRANSPOSE = "Transpose";
inline constexpr const char *FLATTEN = "Flatten";
inline constexpr const char *CONCAT = "Concat";

// Special
inline constexpr const char *IDENTITY = "Identity";
inline constexpr const char *CONSTANT = "Constant";
} // namespace ops

} // namespace zenith

#endif // ZENITH_NODE_HPP
