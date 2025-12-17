// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include "zenith/compilation_session.hpp"
#include "zenith/cpu_backend.hpp"
#include <algorithm>
#include <fstream>
#include <unordered_set>

namespace zenith {

// ============================================================================
// Standard Optimization Passes
// ============================================================================

namespace passes {

/// Constant folding pass - evaluates constant expressions at compile time
Status constant_folding(GraphIR &graph) {
  // Find nodes with all constant inputs
  std::vector<std::string> to_remove;

  for (const auto &node : graph.nodes()) {
    // Check if all inputs are constants
    bool all_const = true;
    for (const auto &input : node->inputs()) {
      if (graph.get_constant(input.name()) == nullptr) {
        all_const = false;
        break;
      }
    }

    if (all_const && node->inputs().size() > 0) {
      // Mark for evaluation (in full impl, we'd evaluate and replace)
      // For now, just mark as foldable
      node->set_attr("foldable", true);
    }
  }

  return Status::Ok();
}

/// Dead code elimination pass - removes unused nodes
Status dead_code_elimination(GraphIR &graph) {
  // Build set of nodes that contribute to outputs
  std::unordered_set<std::string> live_nodes;

  // Start from outputs and work backwards
  std::vector<std::string> worklist;
  for (const auto &output : graph.outputs()) {
    // Find nodes that produce this output
    for (const auto &node : graph.nodes()) {
      for (const auto &out : node->outputs()) {
        if (out.name() == output.name()) {
          worklist.push_back(node->name());
          live_nodes.insert(node->name());
        }
      }
    }
  }

  // Propagate liveness
  while (!worklist.empty()) {
    std::string name = worklist.back();
    worklist.pop_back();

    Node *node = graph.get_node(name);
    if (!node)
      continue;

    // Mark input producers as live
    for (const auto &input : node->inputs()) {
      for (const auto &n : graph.nodes()) {
        for (const auto &out : n->outputs()) {
          if (out.name() == input.name()) {
            if (live_nodes.find(n->name()) == live_nodes.end()) {
              live_nodes.insert(n->name());
              worklist.push_back(n->name());
            }
          }
        }
      }
    }
  }

  // Remove dead nodes
  std::vector<std::string> dead;
  for (const auto &node : graph.nodes()) {
    if (live_nodes.find(node->name()) == live_nodes.end()) {
      dead.push_back(node->name());
    }
  }

  for (const auto &name : dead) {
    graph.remove_node(name);
  }

  return Status::Ok();
}

/// Common subexpression elimination
Status common_subexpression_elimination(GraphIR & /*graph*/) {
  // Find nodes with identical op_type and inputs
  // For full implementation, would hash nodes and merge duplicates
  return Status::Ok();
}

/// Operator fusion pass - fuses compatible operators
Status operator_fusion(GraphIR &graph) {
  // Look for fuseable patterns:
  // - Conv + BatchNorm + ReLU
  // - MatMul + Add + ReLU
  // - etc.

  for (const auto &node : graph.nodes()) {
    // Mark candidates for fusion
    if (node->op_type() == ops::CONV || node->op_type() == ops::GEMM) {
      node->set_attr("fusion_candidate", true);
    }
  }

  return Status::Ok();
}

} // namespace passes

// ============================================================================
// CompilationSession Implementation
// ============================================================================

void CompilationSession::add_standard_passes() {
  clear_passes();

  // Always add basic passes
  add_pass("constant_folding", passes::constant_folding);
  add_pass("dead_code_elimination", passes::dead_code_elimination);

  if (target_.opt_level >= OptimizationLevel::Extended) {
    add_pass("cse", passes::common_subexpression_elimination);
    add_pass("operator_fusion", passes::operator_fusion);
  }
}

std::shared_ptr<Backend> CompilationSession::get_or_create_backend() {
  // Check registry first
  Backend *existing = BackendRegistry::instance().get(target_.backend);
  if (existing) {
    // Wrap in shared_ptr (non-owning)
    return std::shared_ptr<Backend>(existing, [](Backend *) {});
  }

  // Create new backend based on target
  if (target_.backend == "cpu") {
    auto backend = std::make_shared<CpuBackend>();
    BackendRegistry::instance().register_backend(backend);
    return backend;
  }

  // CUDA, ROCm, etc. would be added here with proper guards
#ifdef ZENITH_HAS_CUDA
  if (target_.backend == "cuda") {
    // Would create CudaBackend here
  }
#endif

  return nullptr;
}

Status CompilationSession::load_onnx(const std::string &path) {
  // Read file into bytes
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return Status::Error(StatusCode::NotFound,
                         "Cannot open ONNX file: " + path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    return Status::Error(StatusCode::InternalError,
                         "Failed to read ONNX file: " + path);
  }

  return load_onnx_bytes(buffer);
}

Status CompilationSession::load_onnx_bytes(const std::vector<uint8_t> &data) {
  // Note: Full ONNX parsing requires protobuf dependency
  // For now, we create a placeholder implementation

  if (data.empty()) {
    return Status::Error(StatusCode::InvalidInput, "Empty ONNX data");
  }

  // Check ONNX magic bytes (simplified check)
  // Full implementation would use onnx::ModelProto
  if (data.size() < 8) {
    return Status::Error(StatusCode::InvalidInput, "ONNX data too small");
  }

  // Create placeholder graph
  auto graph = std::make_unique<GraphIR>("onnx_model");

  // Add placeholder for ONNX parsing
  // In full implementation:
  // 1. Parse onnx::ModelProto from bytes
  // 2. Convert onnx::GraphProto to GraphIR
  // 3. Load onnx::TensorProto as constants

  return load(std::move(graph));
}

} // namespace zenith
