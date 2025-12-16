// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_TYPES_HPP
#define ZENITH_TYPES_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace zenith {

// ============================================================================
// Basic Types
// ============================================================================

/// Supported data types for tensors
enum class DataType : uint8_t {
  Float32 = 0,
  Float16 = 1,
  BFloat16 = 2,
  Float64 = 3,
  Int8 = 4,
  Int16 = 5,
  Int32 = 6,
  Int64 = 7,
  UInt8 = 8,
  Bool = 9,
};

/// Get the size in bytes for a data type
constexpr size_t dtype_size(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return 4;
  case DataType::Float16:
    return 2;
  case DataType::BFloat16:
    return 2;
  case DataType::Float64:
    return 8;
  case DataType::Int8:
    return 1;
  case DataType::Int16:
    return 2;
  case DataType::Int32:
    return 4;
  case DataType::Int64:
    return 8;
  case DataType::UInt8:
    return 1;
  case DataType::Bool:
    return 1;
  }
  return 0;
}

/// Get string representation of data type
inline std::string dtype_to_string(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return "float32";
  case DataType::Float16:
    return "float16";
  case DataType::BFloat16:
    return "bfloat16";
  case DataType::Float64:
    return "float64";
  case DataType::Int8:
    return "int8";
  case DataType::Int16:
    return "int16";
  case DataType::Int32:
    return "int32";
  case DataType::Int64:
    return "int64";
  case DataType::UInt8:
    return "uint8";
  case DataType::Bool:
    return "bool";
  }
  return "unknown";
}

/// Memory layout for tensors
enum class Layout : uint8_t {
  NCHW = 0, // Batch, Channels, Height, Width (PyTorch default)
  NHWC = 1, // Batch, Height, Width, Channels (TensorFlow default)
  NC = 2,   // Batch, Channels (1D)
};

/// Attribute value type (for operator attributes)
using AttributeValue =
    std::variant<int64_t, double, std::string, std::vector<int64_t>,
                 std::vector<double>, std::vector<std::string>, bool>;

/// Attribute map type
using AttributeMap = std::unordered_map<std::string, AttributeValue>;

// ============================================================================
// Shape and Dimension
// ============================================================================

/// Represents tensor dimensions
class Shape {
public:
  Shape() = default;
  Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}
  explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

  /// Get number of dimensions
  [[nodiscard]] size_t rank() const { return dims_.size(); }

  /// Get total number of elements
  [[nodiscard]] int64_t numel() const {
    if (dims_.empty())
      return 0;
    int64_t total = 1;
    for (auto d : dims_) {
      if (d < 0)
        return -1; // Dynamic dimension
      total *= d;
    }
    return total;
  }

  /// Get dimension at index
  [[nodiscard]] int64_t operator[](size_t idx) const { return dims_[idx]; }
  int64_t &operator[](size_t idx) { return dims_[idx]; }

  /// Get underlying vector
  [[nodiscard]] const std::vector<int64_t> &dims() const { return dims_; }

  /// Check if shape has dynamic dimensions
  [[nodiscard]] bool is_dynamic() const {
    for (auto d : dims_) {
      if (d < 0)
        return true;
    }
    return false;
  }

  /// Compare equality
  bool operator==(const Shape &other) const { return dims_ == other.dims_; }
  bool operator!=(const Shape &other) const { return dims_ != other.dims_; }

private:
  std::vector<int64_t> dims_;
};

// ============================================================================
// Status and Error Handling
// ============================================================================

/// Result status for operations
enum class StatusCode {
  Ok = 0,
  InvalidArgument,
  InvalidInput,
  InvalidOutput,
  NotFound,
  AlreadyExists,
  OutOfMemory,
  NotImplemented,
  UnsupportedOp,
  InternalError,
  InvalidGraph,
  OptimizationFailed,
};

/// Status class for operation results
class Status {
public:
  Status() : code_(StatusCode::Ok) {}
  Status(StatusCode code, std::string message = "")
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] bool ok() const { return code_ == StatusCode::Ok; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] const std::string &message() const { return message_; }

  static Status Ok() { return Status(); }
  static Status Error(StatusCode code, const std::string &msg) {
    return Status(code, msg);
  }

private:
  StatusCode code_;
  std::string message_;
};

} // namespace zenith

#endif // ZENITH_TYPES_HPP
