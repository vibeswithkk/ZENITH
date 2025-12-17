// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_OP_SIGNATURE_HPP
#define ZENITH_OP_SIGNATURE_HPP

#include "types.hpp"
#include <functional>
#include <string>
#include <vector>

namespace zenith {

/// Operation signature for kernel dispatch.
/// Uniquely identifies an operation variant for kernel selection.
/// Based on CetakBiru 4.4 - KernelRegistry requirements.
struct OpSignature {
  std::string op_type;
  DataType dtype = DataType::Float32;
  std::vector<int64_t> shape_pattern;

  OpSignature() = default;

  OpSignature(std::string op, DataType dt)
      : op_type(std::move(op)), dtype(dt) {}

  OpSignature(std::string op, DataType dt, std::vector<int64_t> pattern)
      : op_type(std::move(op)), dtype(dt), shape_pattern(std::move(pattern)) {}

  /// Equality comparison
  bool operator==(const OpSignature &other) const {
    return op_type == other.op_type && dtype == other.dtype &&
           shape_pattern == other.shape_pattern;
  }

  bool operator!=(const OpSignature &other) const { return !(*this == other); }

  /// Hash function for use in unordered containers
  [[nodiscard]] size_t hash() const {
    size_t h = std::hash<std::string>{}(op_type);
    h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(dtype)) << 1;
    for (auto dim : shape_pattern) {
      h ^= std::hash<int64_t>{}(dim) << 2;
    }
    return h;
  }

  /// String representation for debugging
  [[nodiscard]] std::string to_string() const {
    std::string result = op_type + "_" + dtype_to_string(dtype);
    if (!shape_pattern.empty()) {
      result += "[";
      for (size_t i = 0; i < shape_pattern.size(); ++i) {
        if (i > 0)
          result += ",";
        result += std::to_string(shape_pattern[i]);
      }
      result += "]";
    }
    return result;
  }
};

} // namespace zenith

/// Hash specialization for std::unordered_map support
namespace std {
template <> struct hash<zenith::OpSignature> {
  size_t operator()(const zenith::OpSignature &sig) const { return sig.hash(); }
};
} // namespace std

#endif // ZENITH_OP_SIGNATURE_HPP
