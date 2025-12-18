// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// ONNX Saver - Export Zenith GraphIR to ONNX format
// Produces binary protobuf format compatible with ONNX runtime

#ifndef ZENITH_ONNX_SAVER_HPP
#define ZENITH_ONNX_SAVER_HPP

#include "graph_ir.hpp"
#include "onnx_loader.hpp" // For type mappings
#include "types.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace zenith {
namespace onnx {

// ============================================================================
// Simple Binary Writer for Protobuf
// ============================================================================

/// Minimal protobuf wire format writer (no dependency on protobuf library)
class ProtobufWriter {
public:
  ProtobufWriter() = default;

  /// Get the written bytes
  const std::vector<uint8_t> &data() const { return buffer_; }
  size_t size() const { return buffer_.size(); }

  /// Write varint
  void write_varint(uint64_t value) {
    while (value >= 0x80) {
      buffer_.push_back(static_cast<uint8_t>(value | 0x80));
      value >>= 7;
    }
    buffer_.push_back(static_cast<uint8_t>(value));
  }

  /// Write signed varint (zigzag encoding)
  void write_svarint(int64_t value) {
    uint64_t encoded = (static_cast<uint64_t>(value) << 1) ^
                       static_cast<uint64_t>(value >> 63);
    write_varint(encoded);
  }

  /// Write fixed 32-bit value
  void write_fixed32(uint32_t value) {
    buffer_.push_back(static_cast<uint8_t>(value));
    buffer_.push_back(static_cast<uint8_t>(value >> 8));
    buffer_.push_back(static_cast<uint8_t>(value >> 16));
    buffer_.push_back(static_cast<uint8_t>(value >> 24));
  }

  /// Write fixed 64-bit value
  void write_fixed64(uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      buffer_.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
  }

  /// Write bytes with length prefix
  void write_bytes(const uint8_t *data, size_t size) {
    write_varint(size);
    buffer_.insert(buffer_.end(), data, data + size);
  }

  /// Write string
  void write_string(const std::string &str) {
    write_bytes(reinterpret_cast<const uint8_t *>(str.data()), str.size());
  }

  /// Write field tag
  void write_tag(int field_num, int wire_type) {
    write_varint(static_cast<uint64_t>((field_num << 3) | wire_type));
  }

  /// Append another writer's data as embedded message
  void write_embedded(int field_num, const ProtobufWriter &embedded) {
    write_tag(field_num, 2); // wire type 2 = length-delimited
    write_bytes(embedded.data().data(), embedded.size());
  }

  /// Write varint field
  void write_varint_field(int field_num, uint64_t value) {
    write_tag(field_num, 0); // wire type 0 = varint
    write_varint(value);
  }

  /// Write string field
  void write_string_field(int field_num, const std::string &value) {
    write_tag(field_num, 2); // wire type 2 = length-delimited
    write_string(value);
  }

  /// Write float field
  void write_float_field(int field_num, float value) {
    write_tag(field_num, 5); // wire type 5 = 32-bit
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_fixed32(bits);
  }

private:
  std::vector<uint8_t> buffer_;
};

// ============================================================================
// ONNX Saver Class
// ============================================================================

/// Exports Zenith GraphIR to ONNX format
class OnnxSaver {
public:
  /// ONNX IR version (8 = ONNX 1.13)
  static constexpr int64_t IR_VERSION = 8;

  /// ONNX opset version (17 = PyTorch 2.0 compatible)
  static constexpr int64_t OPSET_VERSION = 17;

  /// Save GraphIR to ONNX file
  Status save(const GraphIR &graph, const std::string &path) {
    std::vector<uint8_t> output;
    auto status = save_to_bytes(graph, &output);
    if (!status.ok())
      return status;

    std::ofstream file(path, std::ios::binary);
    if (!file) {
      return Status::Error(StatusCode::InternalError,
                           "Failed to create file: " + path);
    }

    file.write(reinterpret_cast<const char *>(output.data()),
               static_cast<std::streamsize>(output.size()));
    return Status::Ok();
  }

  /// Save GraphIR to memory buffer
  Status save_to_bytes(const GraphIR &graph, std::vector<uint8_t> *output) {
    if (!output) {
      return Status::Error(StatusCode::InvalidArgument, "Null output pointer");
    }

    try {
      // Validate graph first
      auto validation = graph.validate();
      if (!validation.ok()) {
        return validation;
      }

      ProtobufWriter model_writer;
      write_model_proto(graph, model_writer);
      *output = model_writer.data();
      return Status::Ok();

    } catch (const std::exception &e) {
      return Status::Error(StatusCode::InternalError,
                           std::string("Serialization error: ") + e.what());
    }
  }

private:
  /// Write ONNX ModelProto
  void write_model_proto(const GraphIR &graph, ProtobufWriter &writer) {
    // Field 1: ir_version (int64)
    writer.write_varint_field(1, IR_VERSION);

    // Field 2: opset_import (repeated OperatorSetIdProto)
    ProtobufWriter opset_writer;
    opset_writer.write_string_field(1, ""); // domain (empty = default)
    opset_writer.write_varint_field(2, OPSET_VERSION); // version
    writer.write_embedded(2, opset_writer);

    // Field 3: producer_name
    writer.write_string_field(3, "Zenith");

    // Field 4: producer_version
    writer.write_string_field(4, "1.0.0");

    // Field 7: graph (GraphProto)
    ProtobufWriter graph_writer;
    write_graph_proto(graph, graph_writer);
    writer.write_embedded(7, graph_writer);
  }

  /// Write ONNX GraphProto
  void write_graph_proto(const GraphIR &graph, ProtobufWriter &writer) {
    // Field 2: name
    writer.write_string_field(2, graph.name());

    // Field 5: initializer (repeated TensorProto) - weights
    for (const auto &[name, tensor_data] : graph.constants()) {
      ProtobufWriter init_writer;
      write_tensor_proto(name, tensor_data, init_writer);
      writer.write_embedded(5, init_writer);
    }

    // Field 11: input (repeated ValueInfoProto)
    for (const auto &input : graph.inputs()) {
      ProtobufWriter input_writer;
      write_value_info(input, input_writer);
      writer.write_embedded(11, input_writer);
    }

    // Field 12: output (repeated ValueInfoProto)
    for (const auto &output : graph.outputs()) {
      ProtobufWriter output_writer;
      write_value_info(output, output_writer);
      writer.write_embedded(12, output_writer);
    }

    // Field 1: node (repeated NodeProto)
    for (const auto &node : graph.nodes()) {
      ProtobufWriter node_writer;
      write_node_proto(*node, node_writer);
      writer.write_embedded(1, node_writer);
    }
  }

  /// Write ONNX NodeProto
  void write_node_proto(const Node &node, ProtobufWriter &writer) {
    // Field 1: input (repeated string)
    for (const auto &input : node.inputs()) {
      writer.write_string_field(1, input.name());
    }

    // Field 2: output (repeated string)
    for (const auto &output : node.outputs()) {
      writer.write_string_field(2, output.name());
    }

    // Field 3: name
    writer.write_string_field(3, node.name());

    // Field 4: op_type
    writer.write_string_field(4, node.op_type());

    // Field 5: attribute (repeated AttributeProto)
    for (const auto &[attr_name, attr_value] : node.attrs()) {
      ProtobufWriter attr_writer;
      write_attribute(attr_name, attr_value, attr_writer);
      writer.write_embedded(5, attr_writer);
    }
  }

  /// Write ONNX AttributeProto
  void write_attribute(const std::string &name, const AttributeValue &value,
                       ProtobufWriter &writer) {
    // Field 1: name
    writer.write_string_field(1, name);

    // Write value based on type
    std::visit(
        [&writer](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int64_t>) {
            // Field 3: i (int64)
            writer.write_varint_field(3, static_cast<uint64_t>(arg));
            // Field 20: type = INT (2)
            writer.write_varint_field(20, 2);
          } else if constexpr (std::is_same_v<T, double>) {
            // Field 2: f (float)
            writer.write_float_field(2, static_cast<float>(arg));
            // Field 20: type = FLOAT (1)
            writer.write_varint_field(20, 1);
          } else if constexpr (std::is_same_v<T, std::string>) {
            // Field 4: s (bytes)
            writer.write_string_field(4, arg);
            // Field 20: type = STRING (3)
            writer.write_varint_field(20, 3);
          } else if constexpr (std::is_same_v<T, bool>) {
            // Field 3: i (int64)
            writer.write_varint_field(3, arg ? 1 : 0);
            // Field 20: type = INT (2)
            writer.write_varint_field(20, 2);
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            // Field 7: ints (repeated int64)
            for (auto val : arg) {
              writer.write_varint_field(7, static_cast<uint64_t>(val));
            }
            // Field 20: type = INTS (7)
            writer.write_varint_field(20, 7);
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            // Field 6: floats (repeated float)
            for (auto val : arg) {
              writer.write_float_field(6, static_cast<float>(val));
            }
            // Field 20: type = FLOATS (6)
            writer.write_varint_field(20, 6);
          } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            // Field 8: strings (repeated bytes)
            for (const auto &s : arg) {
              writer.write_string_field(8, s);
            }
            // Field 20: type = STRINGS (8)
            writer.write_varint_field(20, 8);
          }
        },
        value);
  }

  /// Write ONNX ValueInfoProto
  void write_value_info(const TensorDescriptor &desc, ProtobufWriter &writer) {
    // Field 1: name
    writer.write_string_field(1, desc.name());

    // Field 2: type (TypeProto)
    ProtobufWriter type_writer;
    write_type_proto(desc, type_writer);
    writer.write_embedded(2, type_writer);
  }

  /// Write ONNX TypeProto
  void write_type_proto(const TensorDescriptor &desc, ProtobufWriter &writer) {
    // Field 1: tensor_type (Tensor)
    ProtobufWriter tensor_type_writer;
    write_tensor_type(desc, tensor_type_writer);
    writer.write_embedded(1, tensor_type_writer);
  }

  /// Write ONNX TensorTypeProto
  void write_tensor_type(const TensorDescriptor &desc, ProtobufWriter &writer) {
    // Field 1: elem_type
    OnnxDataType onnx_dtype = zenith_to_onnx_dtype(desc.dtype());
    writer.write_varint_field(1, static_cast<uint64_t>(onnx_dtype));

    // Field 2: shape (TensorShapeProto)
    if (desc.shape().rank() > 0) {
      ProtobufWriter shape_writer;
      write_tensor_shape(desc.shape(), shape_writer);
      writer.write_embedded(2, shape_writer);
    }
  }

  /// Write ONNX TensorShapeProto
  void write_tensor_shape(const Shape &shape, ProtobufWriter &writer) {
    for (size_t i = 0; i < shape.rank(); ++i) {
      // Field 1: dim (repeated Dimension)
      ProtobufWriter dim_writer;
      int64_t dim_val = shape[i];
      if (dim_val >= 0) {
        // Field 1: dim_value
        dim_writer.write_varint_field(1, static_cast<uint64_t>(dim_val));
      } else {
        // Field 2: dim_param (dynamic dimension)
        dim_writer.write_string_field(2, "dynamic_" + std::to_string(i));
      }
      writer.write_embedded(1, dim_writer);
    }
  }

  /// Write ONNX TensorProto (for initializers/weights)
  void write_tensor_proto(const std::string &name, const TensorData &tensor,
                          ProtobufWriter &writer) {
    const auto &desc = tensor.descriptor();

    // Field 1: dims (repeated int64)
    for (size_t i = 0; i < desc.shape().rank(); ++i) {
      writer.write_varint_field(1, static_cast<uint64_t>(desc.shape()[i]));
    }

    // Field 2: data_type
    OnnxDataType onnx_dtype = zenith_to_onnx_dtype(desc.dtype());
    writer.write_varint_field(2, static_cast<uint64_t>(onnx_dtype));

    // Field 8: name
    writer.write_string_field(8, name);

    // Field 9: raw_data
    if (!tensor.data().empty()) {
      writer.write_tag(9, 2); // wire type 2
      writer.write_bytes(tensor.data().data(), tensor.data().size());
    }
  }
};

} // namespace onnx
} // namespace zenith

#endif // ZENITH_ONNX_SAVER_HPP
