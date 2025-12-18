// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Dead Code Elimination Pass
// Menghapus node yang tidak berkontribusi ke output graph.
// Referensi: TensorFlow Grappler Pruning Optimizer

#ifndef ZENITH_OPTIMIZATION_DEAD_CODE_ELIMINATION_HPP
#define ZENITH_OPTIMIZATION_DEAD_CODE_ELIMINATION_HPP

#include "../graph_ir.hpp"
#include "../graph_optimizer.hpp"
#include <queue>
#include <unordered_set>

namespace zenith {
namespace optimizer {

/// Dead Code Elimination Pass
/// Menghapus node yang tidak reachable dari output graph.
/// Algoritma: Backward reachability dari outputs ke inputs.
class DeadCodeEliminationPass : public OptimizationPass {
public:
  [[nodiscard]] std::string name() const override {
    return "DeadCodeElimination";
  }

  [[nodiscard]] std::string description() const override {
    return "Menghapus node yang tidak berkontribusi ke output graph";
  }

  [[nodiscard]] int optimization_level() const override {
    return 1; // O1 - optimisasi dasar
  }

  Status run(GraphIR *graph) override {
    if (!graph) {
      return Status::Error(StatusCode::InvalidArgument, "Null graph");
    }

    // Kumpulkan semua node yang reachable dari outputs
    std::unordered_set<std::string> reachable;
    std::queue<std::string> worklist;

    // Mulai dari output nodes
    for (const auto &output : graph->outputs()) {
      worklist.push(output.name());
    }

    // BFS backward dari outputs
    while (!worklist.empty()) {
      std::string current = worklist.front();
      worklist.pop();

      if (reachable.count(current)) {
        continue; // Sudah diproses
      }
      reachable.insert(current);

      // Temukan node yang memproduksi tensor ini
      const Node *producer = find_producer_node(*graph, current);
      if (producer) {
        // Tambahkan semua input dari producer ke worklist
        for (const auto &input : producer->inputs()) {
          if (!reachable.count(input.name())) {
            worklist.push(input.name());
          }
        }
        // Mark producer node name sebagai reachable
        reachable.insert(producer->name());
      }
    }

    // Kumpulkan node yang tidak reachable (dead nodes)
    std::vector<std::string> dead_nodes;
    for (const auto &node : graph->nodes()) {
      if (!reachable.count(node->name())) {
        dead_nodes.push_back(node->name());
      }
    }

    // Hapus dead nodes
    for (const auto &dead_name : dead_nodes) {
      graph->remove_node(dead_name);
    }

    removed_count_ = dead_nodes.size();

    return Status::Ok();
  }

  /// Jumlah node yang dihapus di run terakhir
  [[nodiscard]] size_t removed_count() const { return removed_count_; }

private:
  size_t removed_count_ = 0;

  /// Temukan node yang memproduksi tensor dengan nama tertentu
  const Node *find_producer_node(const GraphIR &graph,
                                 const std::string &tensor_name) {
    for (const auto &node : graph.nodes()) {
      for (const auto &output : node->outputs()) {
        if (output.name() == tensor_name) {
          return node.get();
        }
      }
    }
    return nullptr;
  }
};

} // namespace optimizer
} // namespace zenith

#endif // ZENITH_OPTIMIZATION_DEAD_CODE_ELIMINATION_HPP
