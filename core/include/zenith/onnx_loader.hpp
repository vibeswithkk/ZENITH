// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// ONNX Loader - Import ONNX models to Zenith GraphIR
// Supports both binary protobuf and JSON formats

#ifndef ZENITH_ONNX_LOADER_HPP
#define ZENITH_ONNX_LOADER_HPP

#include "graph_ir.hpp"
#include "types.hpp"
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace zenith {
namespace onnx {

// ============================================================================
// ONNX Data Type Mapping
// ============================================================================

/// ONNX TensorProto DataType values (from onnx.proto)
enum class OnnxDataType : int32_t {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
};

/// Convert ONNX data type to Zenith data type
inline DataType onnx_to_zenith_dtype(OnnxDataType onnx_dtype) {
  switch (onnx_dtype) {
  case OnnxDataType::FLOAT:
    return DataType::Float32;
  case OnnxDataType::FLOAT16:
    return DataType::Float16;
  case OnnxDataType::BFLOAT16:
    return DataType::BFloat16;
  case OnnxDataType::DOUBLE:
    return DataType::Float64;
  case OnnxDataType::INT8:
    return DataType::Int8;
  case OnnxDataType::INT16:
    return DataType::Int16;
  case OnnxDataType::INT32:
    return DataType::Int32;
  case OnnxDataType::INT64:
    return DataType::Int64;
  case OnnxDataType::UINT8:
    return DataType::UInt8;
  case OnnxDataType::BOOL:
    return DataType::Bool;
  default:
    return DataType::Float32; // Default fallback
  }
}

/// Convert Zenith data type to ONNX data type
inline OnnxDataType zenith_to_onnx_dtype(DataType zenith_dtype) {
  switch (zenith_dtype) {
  case DataType::Float32:
    return OnnxDataType::FLOAT;
  case DataType::Float16:
    return OnnxDataType::FLOAT16;
  case DataType::BFloat16:
    return OnnxDataType::BFLOAT16;
  case DataType::Float64:
    return OnnxDataType::DOUBLE;
  case DataType::Int8:
    return OnnxDataType::INT8;
  case DataType::Int16:
    return OnnxDataType::INT16;
  case DataType::Int32:
    return OnnxDataType::INT32;
  case DataType::Int64:
    return OnnxDataType::INT64;
  case DataType::UInt8:
    return OnnxDataType::UINT8;
  case DataType::Bool:
    return OnnxDataType::BOOL;
  }
  return OnnxDataType::FLOAT;
}

// ============================================================================
// Simple Binary Reader for Protobuf
// ============================================================================

/// Minimal protobuf wire type parser (no dependency on protobuf library)
class ProtobufReader {
public:
  explicit ProtobufReader(const uint8_t *data, size_t size)
      : data_(data), size_(size), pos_(0) {}

  bool has_more() const { return pos_ < size_; }
  size_t position() const { return pos_; }

  /// Read varint (variable-length integer)
  uint64_t read_varint() {
    uint64_t result = 0;
    int shift = 0;
    while (pos_ < size_) {
      uint8_t byte = data_[pos_++];
      result |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0)
        break;
      shift += 7;
    }
    return result;
  }

  /// Read fixed 32-bit value
  uint32_t read_fixed32() {
    if (pos_ + 4 > size_)
      return 0;
    uint32_t result = 0;
    result |= static_cast<uint32_t>(data_[pos_++]);
    result |= static_cast<uint32_t>(data_[pos_++]) << 8;
    result |= static_cast<uint32_t>(data_[pos_++]) << 16;
    result |= static_cast<uint32_t>(data_[pos_++]) << 24;
    return result;
  }

  /// Read fixed 64-bit value
  uint64_t read_fixed64() {
    if (pos_ + 8 > size_)
      return 0;
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i) {
      result |= static_cast<uint64_t>(data_[pos_++]) << (i * 8);
    }
    return result;
  }

  /// Read length-delimited bytes
  std::vector<uint8_t> read_bytes() {
    uint64_t len = read_varint();
    if (pos_ + len > size_)
      return {};
    std::vector<uint8_t> result(data_ + pos_, data_ + pos_ + len);
    pos_ += len;
    return result;
  }

  /// Read string
  std::string read_string() {
    auto bytes = read_bytes();
    return std::string(bytes.begin(), bytes.end());
  }

  /// Skip a field based on wire type
  void skip_field(int wire_type) {
    switch (wire_type) {
    case 0: // Varint
      read_varint();
      break;
    case 1: // 64-bit
      pos_ += 8;
      break;
    case 2: // Length-delimited
      read_bytes();
      break;
    case 5: // 32-bit
      pos_ += 4;
      break;
    default:
      break;
    }
  }

  /// Get sub-reader for embedded message
  ProtobufReader sub_reader(size_t len) {
    ProtobufReader sub(data_ + pos_, std::min(len, size_ - pos_));
    pos_ += len;
    return sub;
  }

private:
  const uint8_t *data_;
  size_t size_;
  size_t pos_;
};

// ============================================================================
// ONNX Loader Class
// ============================================================================

/// Loads ONNX models into Zenith GraphIR
class OnnxLoader {
public:
  /// Load ONNX model from file path
  Status load(const std::string &path, GraphIR *graph) {
    if (!graph)
      return Status::Error(StatusCode::InvalidArgument, "Null graph pointer");

    // Read file contents
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
      return Status::Error(StatusCode::NotFound,
                           "Failed to open file: " + path);
    }

    auto file_size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(file_size);
    file.read(reinterpret_cast<char *>(data.data()), file_size);

    return load_from_bytes(data.data(), data.size(), graph);
  }

  /// Load ONNX model from memory
  Status load_from_bytes(const uint8_t *data, size_t size, GraphIR *graph) {
    if (!graph)
      return Status::Error(StatusCode::InvalidArgument, "Null graph pointer");
    if (!data || size == 0)
      return Status::Error(StatusCode::InvalidArgument, "Empty data");

    try {
      return parse_model_proto(data, size, graph);
    } catch (const std::exception &e) {
      return Status::Error(StatusCode::InternalError,
                           std::string("Parse error: ") + e.what());
    }
  }

private:
  /// Parse ONNX ModelProto
  Status parse_model_proto(const uint8_t *data, size_t size, GraphIR *graph) {
    ProtobufReader reader(data, size);
    std::string model_name = "onnx_model";

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 7: { // graph (GraphProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          auto status = parse_graph_proto(sub, graph);
          if (!status.ok())
            return status;
        }
        break;
      }
      case 1:  // ir_version
      case 2:  // opset_import
      case 3:  // producer_name
      case 4:  // producer_version
      case 5:  // domain
      case 6:  // model_version
      case 8:  // metadata_props
      case 14: // training_info
      case 20: // functions
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    graph->set_name(model_name);
    return Status::Ok();
  }

  /// Parse ONNX GraphProto
  Status parse_graph_proto(ProtobufReader &reader, GraphIR *graph) {
    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: { // node (repeated NodeProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          auto status = parse_node_proto(sub, graph);
          if (!status.ok())
            return status;
        }
        break;
      }
      case 2: { // name
        if (wire_type == 2) {
          graph->set_name(reader.read_string());
        }
        break;
      }
      case 5: { // initializer (repeated TensorProto) - weights
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          parse_initializer(sub, graph);
        }
        break;
      }
      case 11: { // input (repeated ValueInfoProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          auto desc = parse_value_info(sub);
          if (desc.is_valid()) {
            graph->add_input(std::move(desc));
          }
        }
        break;
      }
      case 12: { // output (repeated ValueInfoProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          auto desc = parse_value_info(sub);
          if (desc.is_valid()) {
            graph->add_output(std::move(desc));
          }
        }
        break;
      }
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    return Status::Ok();
  }

  /// Parse ONNX NodeProto
  Status parse_node_proto(ProtobufReader &reader, GraphIR *graph) {
    std::string op_type;
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    AttributeMap attrs;

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // input (repeated string)
        if (wire_type == 2) {
          input_names.push_back(reader.read_string());
        }
        break;
      case 2: // output (repeated string)
        if (wire_type == 2) {
          output_names.push_back(reader.read_string());
        }
        break;
      case 3: // name
        if (wire_type == 2) {
          name = reader.read_string();
        }
        break;
      case 4: // op_type
        if (wire_type == 2) {
          op_type = reader.read_string();
        }
        break;
      case 5: // attribute (repeated AttributeProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          parse_attribute(sub, attrs);
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    // Create TensorDescriptors for inputs/outputs
    std::vector<TensorDescriptor> inputs;
    for (const auto &in_name : input_names) {
      inputs.emplace_back(in_name, Shape{}, DataType::Float32);
    }

    std::vector<TensorDescriptor> outputs;
    for (const auto &out_name : output_names) {
      outputs.emplace_back(out_name, Shape{}, DataType::Float32);
    }

    // Generate unique name if not provided
    if (name.empty()) {
      name = op_type + "_" + std::to_string(graph->num_nodes());
    }

    graph->add_node(op_type, name, std::move(inputs), std::move(outputs),
                    std::move(attrs));

    return Status::Ok();
  }

  /// Parse ONNX AttributeProto
  void parse_attribute(ProtobufReader &reader, AttributeMap &attrs) {
    std::string attr_name;
    int attr_type = 0;
    AttributeValue value;

    // Temporary storage
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string string_val;
    std::vector<int64_t> ints_val;
    std::vector<double> floats_val;

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // name
        if (wire_type == 2) {
          attr_name = reader.read_string();
        }
        break;
      case 2: // f (float)
        if (wire_type == 5) {
          uint32_t bits = reader.read_fixed32();
          float f;
          std::memcpy(&f, &bits, sizeof(f));
          float_val = static_cast<double>(f);
          attr_type = 1;
        }
        break;
      case 3: // i (int64)
        if (wire_type == 0) {
          int_val = static_cast<int64_t>(reader.read_varint());
          attr_type = 2;
        }
        break;
      case 4: // s (bytes/string)
        if (wire_type == 2) {
          string_val = reader.read_string();
          attr_type = 3;
        }
        break;
      case 6: // floats (repeated float)
        if (wire_type == 2) {
          auto bytes = reader.read_bytes();
          size_t count = bytes.size() / 4;
          for (size_t i = 0; i < count; ++i) {
            float f;
            std::memcpy(&f, bytes.data() + i * 4, sizeof(f));
            floats_val.push_back(static_cast<double>(f));
          }
          attr_type = 6;
        }
        break;
      case 7: // ints (repeated int64)
        if (wire_type == 2) {
          auto bytes = reader.read_bytes();
          ProtobufReader ints_reader(bytes.data(), bytes.size());
          while (ints_reader.has_more()) {
            ints_val.push_back(static_cast<int64_t>(ints_reader.read_varint()));
          }
          attr_type = 7;
        }
        break;
      case 20: // type
        if (wire_type == 0) {
          attr_type = static_cast<int>(reader.read_varint());
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    // Store the attribute with appropriate type
    if (!attr_name.empty()) {
      switch (attr_type) {
      case 1: // FLOAT
        attrs[attr_name] = float_val;
        break;
      case 2: // INT
        attrs[attr_name] = int_val;
        break;
      case 3: // STRING
        attrs[attr_name] = string_val;
        break;
      case 6: // FLOATS
        attrs[attr_name] = floats_val;
        break;
      case 7: // INTS
        attrs[attr_name] = ints_val;
        break;
      default:
        break;
      }
    }
  }

  /// Parse ONNX ValueInfoProto
  TensorDescriptor parse_value_info(ProtobufReader &reader) {
    std::string name;
    Shape shape;
    DataType dtype = DataType::Float32;

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // name
        if (wire_type == 2) {
          name = reader.read_string();
        }
        break;
      case 2: // type (TypeProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          parse_type_proto(sub, shape, dtype);
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    return TensorDescriptor(name, shape, dtype);
  }

  /// Parse ONNX TypeProto
  void parse_type_proto(ProtobufReader &reader, Shape &shape, DataType &dtype) {
    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // tensor_type (Tensor)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          parse_tensor_type(sub, shape, dtype);
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }
  }

  /// Parse ONNX TensorTypeProto
  void parse_tensor_type(ProtobufReader &reader, Shape &shape,
                         DataType &dtype) {
    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // elem_type
        if (wire_type == 0) {
          int onnx_dtype = static_cast<int>(reader.read_varint());
          dtype = onnx_to_zenith_dtype(static_cast<OnnxDataType>(onnx_dtype));
        }
        break;
      case 2: // shape (TensorShapeProto)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          shape = parse_tensor_shape(sub);
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }
  }

  /// Parse ONNX TensorShapeProto
  Shape parse_tensor_shape(ProtobufReader &reader) {
    std::vector<int64_t> dims;

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // dim (repeated Dimension)
        if (wire_type == 2) {
          uint64_t len = reader.read_varint();
          auto sub = reader.sub_reader(static_cast<size_t>(len));
          int64_t dim_val = parse_dimension(sub);
          dims.push_back(dim_val);
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    return Shape(dims);
  }

  /// Parse ONNX Dimension
  int64_t parse_dimension(ProtobufReader &reader) {
    int64_t value = -1; // -1 for dynamic

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // dim_value
        if (wire_type == 0) {
          value = static_cast<int64_t>(reader.read_varint());
        }
        break;
      case 2: // dim_param (string, dynamic)
        if (wire_type == 2) {
          reader.read_string(); // Ignore, mark as dynamic
          value = -1;
        }
        break;
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    return value;
  }

  /// Parse ONNX TensorProto (initializer/weights)
  void parse_initializer(ProtobufReader &reader, GraphIR *graph) {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = 1; // FLOAT
    std::vector<uint8_t> raw_data;

    while (reader.has_more()) {
      uint64_t tag = reader.read_varint();
      int field_num = static_cast<int>(tag >> 3);
      int wire_type = static_cast<int>(tag & 0x7);

      switch (field_num) {
      case 1: // dims (repeated int64)
        if (wire_type == 0) {
          dims.push_back(static_cast<int64_t>(reader.read_varint()));
        } else if (wire_type == 2) {
          auto bytes = reader.read_bytes();
          ProtobufReader dim_reader(bytes.data(), bytes.size());
          while (dim_reader.has_more()) {
            dims.push_back(static_cast<int64_t>(dim_reader.read_varint()));
          }
        }
        break;
      case 2: // data_type
        if (wire_type == 0) {
          data_type = static_cast<int32_t>(reader.read_varint());
        }
        break;
      case 8: // name
        if (wire_type == 2) {
          name = reader.read_string();
        }
        break;
      case 9: // raw_data
        if (wire_type == 2) {
          raw_data = reader.read_bytes();
        }
        break;
      case 4:  // float_data (repeated float)
      case 5:  // int32_data (repeated int32)
      case 6:  // string_data
      case 7:  // int64_data (repeated int64)
      case 10: // external_data
      case 11: // data_location
      case 12: // double_data
      case 13: // uint64_data
      default:
        reader.skip_field(wire_type);
        break;
      }
    }

    // Create TensorData from parsed info
    if (!name.empty() && !raw_data.empty()) {
      DataType dtype =
          onnx_to_zenith_dtype(static_cast<OnnxDataType>(data_type));
      TensorDescriptor desc(name, Shape(dims), dtype);
      TensorData tensor_data(std::move(desc), std::move(raw_data));
      graph->add_constant(name, std::move(tensor_data));
    }
  }
};

} // namespace onnx
} // namespace zenith

#endif // ZENITH_ONNX_LOADER_HPP
