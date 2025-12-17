// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_TARGET_DESCRIPTOR_HPP
#define ZENITH_TARGET_DESCRIPTOR_HPP

#include "types.hpp"
#include <string>

namespace zenith {

/// Optimization level for compilation
enum class OptimizationLevel : uint8_t {
  None = 0,     // No optimization
  Basic = 1,    // Constant folding, dead code elimination
  Extended = 2, // + Operator fusion, layout optimization
  Full = 3,     // + Auto-tuning, aggressive fusion
};

/// Target hardware descriptor for compilation.
/// Based on CetakBiru 4.4 - CompilationSession requirements.
struct TargetDescriptor {
  std::string backend = "cpu"; // "cpu", "cuda", "rocm", "oneapi"
  int device_id = 0;           // Device index
  DataType preferred_dtype = DataType::Float32;
  OptimizationLevel opt_level = OptimizationLevel::Extended;
  bool enable_fp16 = false;      // Enable FP16 compute
  bool enable_int8 = false;      // Enable INT8 quantization
  size_t max_workspace_size = 0; // Max workspace memory (0 = auto)

  TargetDescriptor() = default;

  explicit TargetDescriptor(std::string backend_name)
      : backend(std::move(backend_name)) {}

  TargetDescriptor(std::string backend_name, int dev_id)
      : backend(std::move(backend_name)), device_id(dev_id) {}

  /// Check if target is GPU-based
  [[nodiscard]] bool is_gpu() const {
    return backend == "cuda" || backend == "rocm" || backend == "oneapi";
  }

  /// Check if target is CPU-based
  [[nodiscard]] bool is_cpu() const {
    return backend == "cpu" || backend == "cpu_avx2" || backend == "cpu_avx512";
  }

  /// String representation
  [[nodiscard]] std::string to_string() const {
    std::string result = backend + ":" + std::to_string(device_id);
    result += " opt=" + std::to_string(static_cast<int>(opt_level));
    if (enable_fp16)
      result += " fp16";
    if (enable_int8)
      result += " int8";
    return result;
  }
};

} // namespace zenith

#endif // ZENITH_TARGET_DESCRIPTOR_HPP
