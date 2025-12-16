// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_TENSOR_HPP
#define ZENITH_TENSOR_HPP

#include "types.hpp"
#include <memory>
#include <string>

namespace zenith {

// ============================================================================
// TensorDescriptor
// ============================================================================

/// Describes a tensor's metadata without holding actual data.
/// This is the primary way tensors are represented in the Graph IR.
class TensorDescriptor {
public:
  TensorDescriptor() = default;

  TensorDescriptor(std::string name, Shape shape, DataType dtype,
                   Layout layout = Layout::NCHW)
      : name_(std::move(name)), shape_(std::move(shape)), dtype_(dtype),
        layout_(layout) {}

  // Accessors
  [[nodiscard]] const std::string &name() const { return name_; }
  [[nodiscard]] const Shape &shape() const { return shape_; }
  [[nodiscard]] DataType dtype() const { return dtype_; }
  [[nodiscard]] Layout layout() const { return layout_; }

  // Mutators
  void set_name(std::string name) { name_ = std::move(name); }
  void set_shape(Shape shape) { shape_ = std::move(shape); }
  void set_dtype(DataType dtype) { dtype_ = dtype; }
  void set_layout(Layout layout) { layout_ = layout; }

  /// Calculate the size in bytes
  [[nodiscard]] size_t size_bytes() const {
    auto elements = shape_.numel();
    if (elements < 0)
      return 0; // Dynamic shape
    return static_cast<size_t>(elements) * dtype_size(dtype_);
  }

  /// Check if tensor has a valid shape
  [[nodiscard]] bool is_valid() const {
    return !name_.empty() && shape_.rank() > 0;
  }

private:
  std::string name_;
  Shape shape_;
  DataType dtype_ = DataType::Float32;
  Layout layout_ = Layout::NCHW;
};

// ============================================================================
// TensorData (for holding actual tensor data)
// ============================================================================

/// Holds actual tensor data. Used for constant tensors and weights.
class TensorData {
public:
  TensorData() = default;

  TensorData(TensorDescriptor desc, std::vector<uint8_t> data)
      : descriptor_(std::move(desc)), data_(std::move(data)) {}

  [[nodiscard]] const TensorDescriptor &descriptor() const {
    return descriptor_;
  }
  [[nodiscard]] const std::vector<uint8_t> &data() const { return data_; }
  [[nodiscard]] std::vector<uint8_t> &data() { return data_; }

  /// Get size of data in bytes
  [[nodiscard]] size_t size() const { return data_.size(); }

  /// Access data as typed pointer
  template <typename T> [[nodiscard]] const T *data_as() const {
    return reinterpret_cast<const T *>(data_.data());
  }

  template <typename T> [[nodiscard]] T *data_as() {
    return reinterpret_cast<T *>(data_.data());
  }

private:
  TensorDescriptor descriptor_;
  std::vector<uint8_t> data_;
};

} // namespace zenith

#endif // ZENITH_TENSOR_HPP
