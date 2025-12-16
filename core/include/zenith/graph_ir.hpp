// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_GRAPH_IR_HPP
#define ZENITH_GRAPH_IR_HPP

#include "node.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace zenith {

// ============================================================================
// GraphIR - The unified intermediate representation for computation graphs
// ============================================================================

/// The unified intermediate representation for computation graphs.
/// This is the core data structure that all framework adapters convert to,
/// and all optimization passes operate on.
///
/// Based on the class diagram in section 4.4 of the blueprint.
class GraphIR {
public:
  GraphIR() = default;
  explicit GraphIR(std::string name) : name_(std::move(name)) {}

  // ========================================================================
  // Graph Information
  // ========================================================================

  [[nodiscard]] const std::string &name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

  // ========================================================================
  // Node Management
  // ========================================================================

  /// Add a node to the graph
  Node *add_node(std::unique_ptr<Node> node) {
    auto *ptr = node.get();
    nodes_.push_back(std::move(node));
    name_to_node_[ptr->name()] = ptr;
    return ptr;
  }

  /// Add a node by constructing it in place
  template <typename... Args> Node *add_node(Args &&...args) {
    auto node = std::make_unique<Node>(std::forward<Args>(args)...);
    return add_node(std::move(node));
  }

  /// Get node by name
  [[nodiscard]] Node *get_node(const std::string &name) const {
    auto it = name_to_node_.find(name);
    if (it != name_to_node_.end()) {
      return it->second;
    }
    return nullptr;
  }

  /// Get all nodes
  [[nodiscard]] const std::vector<std::unique_ptr<Node>> &nodes() const {
    return nodes_;
  }

  /// Get number of nodes
  [[nodiscard]] size_t num_nodes() const { return nodes_.size(); }

  /// Remove a node by name
  bool remove_node(const std::string &name) {
    auto it = name_to_node_.find(name);
    if (it == name_to_node_.end())
      return false;

    auto node_it =
        std::find_if(nodes_.begin(), nodes_.end(),
                     [&name](const auto &n) { return n->name() == name; });

    if (node_it != nodes_.end()) {
      nodes_.erase(node_it);
      name_to_node_.erase(it);
      return true;
    }
    return false;
  }

  // ========================================================================
  // Graph Inputs/Outputs
  // ========================================================================

  /// Set graph input tensors
  void set_inputs(std::vector<TensorDescriptor> inputs) {
    inputs_ = std::move(inputs);
  }

  /// Set graph output tensors
  void set_outputs(std::vector<TensorDescriptor> outputs) {
    outputs_ = std::move(outputs);
  }

  /// Add a graph input
  void add_input(TensorDescriptor input) {
    inputs_.push_back(std::move(input));
  }

  /// Add a graph output
  void add_output(TensorDescriptor output) {
    outputs_.push_back(std::move(output));
  }

  [[nodiscard]] const std::vector<TensorDescriptor> &inputs() const {
    return inputs_;
  }
  [[nodiscard]] const std::vector<TensorDescriptor> &outputs() const {
    return outputs_;
  }

  // ========================================================================
  // Constants/Weights
  // ========================================================================

  /// Add constant tensor data (weights, biases, etc.)
  void add_constant(std::string name, TensorData data) {
    constants_[std::move(name)] = std::move(data);
  }

  /// Get constant data
  [[nodiscard]] const TensorData *get_constant(const std::string &name) const {
    auto it = constants_.find(name);
    if (it != constants_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  [[nodiscard]] const std::unordered_map<std::string, TensorData> &
  constants() const {
    return constants_;
  }

  // ========================================================================
  // Graph Traversal
  // ========================================================================

  /// Find all nodes of a specific operation type
  [[nodiscard]] std::vector<Node *>
  find_nodes_by_op(const std::string &op_type) const {
    std::vector<Node *> result;
    for (const auto &node : nodes_) {
      if (node->op_type() == op_type) {
        result.push_back(node.get());
      }
    }
    return result;
  }

  /// Get nodes in topological order (for execution)
  [[nodiscard]] std::vector<Node *> topological_order() const {
    // Simplified implementation - returns nodes in insertion order
    // Full topological sort will be implemented in optimization phase
    std::vector<Node *> result;
    result.reserve(nodes_.size());
    for (const auto &node : nodes_) {
      result.push_back(node.get());
    }
    return result;
  }

  // ========================================================================
  // Validation
  // ========================================================================

  /// Validate the graph structure
  [[nodiscard]] Status validate() const {
    // Check for empty graph
    if (nodes_.empty()) {
      return Status::Error(StatusCode::InvalidGraph, "Graph has no nodes");
    }

    // Check for inputs
    if (inputs_.empty()) {
      return Status::Error(StatusCode::InvalidGraph, "Graph has no inputs");
    }

    // Check for outputs
    if (outputs_.empty()) {
      return Status::Error(StatusCode::InvalidGraph, "Graph has no outputs");
    }

    // Check for duplicate node names
    std::unordered_map<std::string, bool> seen;
    for (const auto &node : nodes_) {
      if (seen.count(node->name())) {
        return Status::Error(StatusCode::InvalidGraph,
                             "Duplicate node name: " + node->name());
      }
      seen[node->name()] = true;
    }

    return Status::Ok();
  }

  // ========================================================================
  // Cloning
  // ========================================================================

  /// Deep clone the graph
  [[nodiscard]] std::unique_ptr<GraphIR> clone() const {
    auto new_graph = std::make_unique<GraphIR>(name_);
    new_graph->inputs_ = inputs_;
    new_graph->outputs_ = outputs_;
    new_graph->constants_ = constants_;

    for (const auto &node : nodes_) {
      new_graph->add_node(node->clone());
    }

    return new_graph;
  }

  // ========================================================================
  // Statistics
  // ========================================================================

  /// Count nodes by operation type
  [[nodiscard]] std::unordered_map<std::string, size_t> count_ops() const {
    std::unordered_map<std::string, size_t> counts;
    for (const auto &node : nodes_) {
      counts[node->op_type()]++;
    }
    return counts;
  }

  /// Print graph summary
  [[nodiscard]] std::string summary() const {
    std::string result = "GraphIR: " + name_ + "\n";
    result += "  Inputs: " + std::to_string(inputs_.size()) + "\n";
    result += "  Outputs: " + std::to_string(outputs_.size()) + "\n";
    result += "  Nodes: " + std::to_string(nodes_.size()) + "\n";
    result += "  Constants: " + std::to_string(constants_.size()) + "\n";

    auto op_counts = count_ops();
    result += "  Operations:\n";
    for (const auto &[op, count] : op_counts) {
      result += "    " + op + ": " + std::to_string(count) + "\n";
    }

    return result;
  }

private:
  std::string name_;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<std::string, Node *> name_to_node_;
  std::vector<TensorDescriptor> inputs_;
  std::vector<TensorDescriptor> outputs_;
  std::unordered_map<std::string, TensorData> constants_;
};

} // namespace zenith

#endif // ZENITH_GRAPH_IR_HPP
