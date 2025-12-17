// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_HPP
#define ZENITH_HPP

/// Zenith: Cross-Platform ML Optimization Framework
///
/// This is the main header file that includes all core components.
/// Include this header to use Zenith in your project.
///
/// Example:
///   #include <zenith/zenith.hpp>
///   zenith::CompilationSession session;
///   session.load_onnx("model.onnx");
///   session.compile();

#include <string>

// Core types and data structures
#include "zenith/graph_ir.hpp"
#include "zenith/node.hpp"
#include "zenith/tensor.hpp"
#include "zenith/types.hpp"

// Compilation and execution
#include "zenith/compilation_session.hpp"
#include "zenith/compiled_artifact.hpp"
#include "zenith/kernel.hpp"
#include "zenith/kernel_registry.hpp"
#include "zenith/op_signature.hpp"
#include "zenith/target_descriptor.hpp"

// Backend abstraction
#include "zenith/backend.hpp"
#include "zenith/dispatcher.hpp"

// CUDA support (conditional)
#ifdef ZENITH_HAS_CUDA
#include "zenith/cublas_ops.hpp"
#include "zenith/cuda_backend.hpp"
#ifdef ZENITH_HAS_CUDNN
#include "zenith/cudnn_ops.hpp"
#endif
#endif

namespace zenith {

/// Library version information
constexpr const char *VERSION = "0.1.0";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

/// Get version string
inline std::string get_version() { return VERSION; }

} // namespace zenith

#endif // ZENITH_HPP
