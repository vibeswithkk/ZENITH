// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include "zenith/graph_ir.hpp"
#include "zenith/onnx_loader.hpp"
#include "zenith/onnx_saver.hpp"

namespace zenith {

// ============================================================================
// ONNX Import/Export Utility Functions
// ============================================================================

/// Load a GraphIR from an ONNX file
Status load_onnx(const std::string &path, GraphIR *graph) {
  onnx::OnnxLoader loader;
  return loader.load(path, graph);
}

/// Load a GraphIR from ONNX bytes in memory
Status load_onnx_from_bytes(const uint8_t *data, size_t size, GraphIR *graph) {
  onnx::OnnxLoader loader;
  return loader.load_from_bytes(data, size, graph);
}

/// Save a GraphIR to an ONNX file
Status save_onnx(const GraphIR &graph, const std::string &path) {
  onnx::OnnxSaver saver;
  return saver.save(graph, path);
}

/// Save a GraphIR to ONNX bytes in memory
Status save_onnx_to_bytes(const GraphIR &graph, std::vector<uint8_t> *output) {
  onnx::OnnxSaver saver;
  return saver.save_to_bytes(graph, output);
}

} // namespace zenith
