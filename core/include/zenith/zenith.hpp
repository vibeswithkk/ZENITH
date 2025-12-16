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
///   zenith::GraphIR graph("my_model");
///   graph.add_node(...);

#include <string>

#include "zenith/graph_ir.hpp"
#include "zenith/node.hpp"
#include "zenith/tensor.hpp"
#include "zenith/types.hpp"

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
