// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#ifndef ZENITH_CPU_BACKEND_HPP
#define ZENITH_CPU_BACKEND_HPP

#include "backend.hpp"
#include "kernels.hpp"
#include <cstring>
#include <functional>
#include <unordered_map>

namespace zenith {

// ============================================================================
// CpuBackend - CPU execution backend with SIMD optimizations
// ============================================================================

/// CPU backend implementation using SIMD intrinsics (AVX2, AVX-512, NEON)
/// Based on CetakBiru Section 5.1 - Phase 1 requirements
class CpuBackend : public Backend {
public:
  CpuBackend() { initialize_op_registry(); }

  // ========================================================================
  // Backend Interface Implementation
  // ========================================================================

  [[nodiscard]] std::string name() const override { return "cpu"; }

  [[nodiscard]] std::string description() const override {
    std::string desc = "CPU Backend";
    auto features = kernels::get_cpu_features();

    if (features.has_avx512f) {
      desc += " (AVX-512)";
    } else if (features.has_avx2) {
      desc += " (AVX2";
      if (features.has_fma) {
        desc += "+FMA";
      }
      desc += ")";
    } else if (features.has_sse4_2) {
      desc += " (SSE4.2)";
    } else if (features.has_neon) {
      desc += " (NEON)";
    }

    return desc;
  }

  [[nodiscard]] bool is_available() const override {
    return true; // CPU is always available
  }

  [[nodiscard]] std::string version() const override {
    auto features = kernels::get_cpu_features();
    if (features.has_avx512f)
      return "avx512";
    if (features.has_avx2)
      return "avx2";
    if (features.has_sse4_2)
      return "sse4.2";
    if (features.has_neon)
      return "neon";
    return "scalar";
  }

  [[nodiscard]] Status
  execute(const GraphIR &graph,
          const std::unordered_map<std::string, const void *> &inputs,
          std::unordered_map<std::string, void *> &outputs) override {

    // Get topological order for execution
    auto nodes = graph.topological_order();

    // Tensor storage during execution
    std::unordered_map<std::string, void *> tensor_data;

    // Copy input data to internal storage
    for (const auto &input : graph.inputs()) {
      auto it = inputs.find(input.name());
      if (it == inputs.end()) {
        return Status::Error(StatusCode::InvalidInput,
                             "Missing input: " + input.name());
      }
      // Allocate and copy input
      size_t size = input.size_bytes();
      void *data = allocate(size);
      std::memcpy(data, it->second, size);
      tensor_data[input.name()] = data;
    }

    // Load constants
    for (const auto &[name, const_data] : graph.constants()) {
      void *data = allocate(const_data.size());
      std::memcpy(data, const_data.data().data(), const_data.size());
      tensor_data[name] = data;
    }

    // Execute each node in order
    for (const auto *node : nodes) {
      auto status = execute_node(*node, tensor_data);
      if (!status.ok()) {
        cleanup(tensor_data);
        return status;
      }
    }

    // Copy output data
    for (const auto &output : graph.outputs()) {
      auto it = outputs.find(output.name());
      if (it == outputs.end()) {
        cleanup(tensor_data);
        return Status::Error(StatusCode::InvalidOutput,
                             "Missing output buffer: " + output.name());
      }
      auto data_it = tensor_data.find(output.name());
      if (data_it != tensor_data.end()) {
        std::memcpy(it->second, data_it->second, output.size_bytes());
      }
    }

    cleanup(tensor_data);
    return Status::Ok();
  }

  [[nodiscard]] void *allocate(size_t size_bytes) override {
    return kernels::aligned_alloc(size_bytes, 32); // 32-byte alignment for AVX2
  }

  void deallocate(void *ptr) override { kernels::aligned_free(ptr); }

private:
  // Operation handler type
  using OpHandler = std::function<Status(
      const Node &, std::unordered_map<std::string, void *> &)>;

  std::unordered_map<std::string, OpHandler> op_registry_;

  void cleanup(std::unordered_map<std::string, void *> &tensor_data) {
    for (auto &[_, ptr] : tensor_data) {
      deallocate(ptr);
    }
    tensor_data.clear();
  }

  void initialize_op_registry() {
    // Register operation handlers
    op_registry_["Relu"] = [this](const Node &node, auto &tensors) {
      return execute_relu(node, tensors);
    };
    op_registry_["MatMul"] = [this](const Node &node, auto &tensors) {
      return execute_matmul(node, tensors);
    };
    op_registry_["Gemm"] = [this](const Node &node, auto &tensors) {
      return execute_matmul(node, tensors); // GEMM uses same path
    };
    op_registry_["Add"] = [this](const Node &node, auto &tensors) {
      return execute_add(node, tensors);
    };
    op_registry_["Conv"] = [this](const Node &node, auto &tensors) {
      return execute_conv2d(node, tensors);
    };
    op_registry_["Conv2D"] = [this](const Node &node, auto &tensors) {
      return execute_conv2d(node, tensors);
    };
    op_registry_["Sigmoid"] = [this](const Node &node, auto &tensors) {
      return execute_sigmoid(node, tensors);
    };
    op_registry_["Tanh"] = [this](const Node &node, auto &tensors) {
      return execute_tanh(node, tensors);
    };
    op_registry_["Softmax"] = [this](const Node &node, auto &tensors) {
      return execute_softmax(node, tensors);
    };
    op_registry_["MaxPool"] = [this](const Node &node, auto &tensors) {
      return execute_maxpool(node, tensors);
    };
    op_registry_["Identity"] = [this](const Node &node, auto &tensors) {
      return execute_identity(node, tensors);
    };
  }

  Status execute_node(const Node &node,
                      std::unordered_map<std::string, void *> &tensors) {
    auto it = op_registry_.find(node.op_type());
    if (it != op_registry_.end()) {
      return it->second(node, tensors);
    }
    return Status::Error(StatusCode::UnsupportedOp,
                         "Unsupported operation: " + node.op_type());
  }

  // ========================================================================
  // Operation Implementations
  // ========================================================================

  Status execute_relu(const Node &node,
                      std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().empty() || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "ReLU requires I/O");
    }

    const auto &input = node.inputs()[0];
    const auto &output = node.outputs()[0];

    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput,
                           "Input not found: " + input.name());
    }

    size_t size = input.shape().numel();

    // Allocate output (or reuse if in-place)
    void *out_data = allocate(output.size_bytes());
    std::memcpy(out_data, it->second, output.size_bytes());

    kernels::relu_f32(static_cast<float *>(out_data), size);

    tensors[output.name()] = out_data;
    return Status::Ok();
  }

  Status execute_matmul(const Node &node,
                        std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().size() < 2 || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput,
                           "MatMul requires 2 inputs");
    }

    const auto &A = node.inputs()[0];
    const auto &B = node.inputs()[1];
    const auto &C = node.outputs()[0];

    auto a_it = tensors.find(A.name());
    auto b_it = tensors.find(B.name());

    if (a_it == tensors.end() || b_it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "MatMul inputs not found");
    }

    // Get dimensions (assume 2D matrices for now)
    const auto &a_shape = A.shape().dims();
    const auto &b_shape = B.shape().dims();

    if (a_shape.size() < 2 || b_shape.size() < 2) {
      return Status::Error(StatusCode::InvalidInput,
                           "MatMul requires 2D+ tensors");
    }

    int M = a_shape[a_shape.size() - 2];
    int K = a_shape[a_shape.size() - 1];
    int N = b_shape[b_shape.size() - 1];

    void *out_data = allocate(C.size_bytes());

    kernels::matmul_f32(static_cast<const float *>(a_it->second),
                        static_cast<const float *>(b_it->second),
                        static_cast<float *>(out_data), M, N, K);

    tensors[C.name()] = out_data;
    return Status::Ok();
  }

  Status execute_add(const Node &node,
                     std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().size() < 2 || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "Add requires 2 inputs");
    }

    const auto &A = node.inputs()[0];
    const auto &B = node.inputs()[1];
    const auto &C = node.outputs()[0];

    auto a_it = tensors.find(A.name());
    auto b_it = tensors.find(B.name());

    if (a_it == tensors.end() || b_it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Add inputs not found");
    }

    size_t size = A.shape().numel();
    void *out_data = allocate(C.size_bytes());

    kernels::add_f32(static_cast<const float *>(a_it->second),
                     static_cast<const float *>(b_it->second),
                     static_cast<float *>(out_data), size);

    tensors[C.name()] = out_data;
    return Status::Ok();
  }

  Status execute_conv2d(const Node &node,
                        std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().size() < 2 || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "Conv requires 2 inputs");
    }

    const auto &input = node.inputs()[0];
    const auto &weight = node.inputs()[1];
    const auto &output = node.outputs()[0];

    auto in_it = tensors.find(input.name());
    auto w_it = tensors.find(weight.name());

    if (in_it == tensors.end() || w_it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Conv inputs not found");
    }

    // Get dimensions [N, C, H, W]
    const auto &in_shape = input.shape().dims();
    const auto &w_shape = weight.shape().dims();

    if (in_shape.size() != 4 || w_shape.size() != 4) {
      return Status::Error(StatusCode::InvalidInput,
                           "Conv requires 4D tensors");
    }

    // Get attributes with defaults
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;

    if (auto *strides = node.get_attr_ints("strides")) {
      if (strides->size() >= 2) {
        stride_h = (*strides)[0];
        stride_w = (*strides)[1];
      }
    }
    if (auto *pads = node.get_attr_ints("pads")) {
      if (pads->size() >= 2) {
        pad_h = (*pads)[0];
        pad_w = (*pads)[1];
      }
    }

    void *out_data = allocate(output.size_bytes());

    // Check for bias
    const float *bias = nullptr;
    if (node.inputs().size() > 2) {
      auto b_it = tensors.find(node.inputs()[2].name());
      if (b_it != tensors.end()) {
        bias = static_cast<const float *>(b_it->second);
      }
    }

    kernels::conv2d_f32(static_cast<const float *>(in_it->second),
                        static_cast<const float *>(w_it->second), bias,
                        static_cast<float *>(out_data), in_shape[0],
                        in_shape[1], in_shape[2], in_shape[3], w_shape[0],
                        w_shape[2], w_shape[3], stride_h, stride_w, pad_h,
                        pad_w);

    tensors[output.name()] = out_data;
    return Status::Ok();
  }

  Status execute_sigmoid(const Node &node,
                         std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().empty() || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "Sigmoid requires I/O");
    }

    const auto &input = node.inputs()[0];
    const auto &output = node.outputs()[0];

    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Input not found");
    }

    size_t size = input.shape().numel();
    void *out_data = allocate(output.size_bytes());
    std::memcpy(out_data, it->second, output.size_bytes());

    kernels::sigmoid_f32(static_cast<float *>(out_data), size);

    tensors[output.name()] = out_data;
    return Status::Ok();
  }

  Status execute_tanh(const Node &node,
                      std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().empty() || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "Tanh requires I/O");
    }

    const auto &input = node.inputs()[0];
    const auto &output = node.outputs()[0];

    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Input not found");
    }

    size_t size = input.shape().numel();
    void *out_data = allocate(output.size_bytes());
    std::memcpy(out_data, it->second, output.size_bytes());

    kernels::tanh_f32(static_cast<float *>(out_data), size);

    tensors[output.name()] = out_data;
    return Status::Ok();
  }

  Status execute_softmax(const Node &node,
                         std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().empty() || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "Softmax requires I/O");
    }

    const auto &input = node.inputs()[0];
    const auto &output = node.outputs()[0];

    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Input not found");
    }

    size_t size = input.shape().numel();
    size_t axis_size = input.shape().dims().back();

    void *out_data = allocate(output.size_bytes());

    kernels::softmax_f32(static_cast<const float *>(it->second),
                         static_cast<float *>(out_data), size, axis_size);

    tensors[output.name()] = out_data;
    return Status::Ok();
  }

  Status execute_maxpool(const Node &node,
                         std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().empty() || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "MaxPool requires I/O");
    }

    const auto &input = node.inputs()[0];
    const auto &output = node.outputs()[0];

    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Input not found");
    }

    const auto &in_shape = input.shape().dims();
    if (in_shape.size() != 4) {
      return Status::Error(StatusCode::InvalidInput, "MaxPool requires 4D");
    }

    int pool_h = 2, pool_w = 2;
    int stride_h = 2, stride_w = 2;

    if (auto *kernel = node.get_attr_ints("kernel_shape")) {
      if (kernel->size() >= 2) {
        pool_h = (*kernel)[0];
        pool_w = (*kernel)[1];
      }
    }
    if (auto *strides = node.get_attr_ints("strides")) {
      if (strides->size() >= 2) {
        stride_h = (*strides)[0];
        stride_w = (*strides)[1];
      }
    }

    void *out_data = allocate(output.size_bytes());

    kernels::maxpool2d_f32(static_cast<const float *>(it->second),
                           static_cast<float *>(out_data), in_shape[0],
                           in_shape[1], in_shape[2], in_shape[3], pool_h,
                           pool_w, stride_h, stride_w);

    tensors[output.name()] = out_data;
    return Status::Ok();
  }

  Status execute_identity(const Node &node,
                          std::unordered_map<std::string, void *> &tensors) {
    if (node.inputs().empty() || node.outputs().empty()) {
      return Status::Error(StatusCode::InvalidInput, "Identity requires I/O");
    }

    const auto &input = node.inputs()[0];
    const auto &output = node.outputs()[0];

    auto it = tensors.find(input.name());
    if (it == tensors.end()) {
      return Status::Error(StatusCode::InvalidInput, "Input not found");
    }

    void *out_data = allocate(output.size_bytes());
    std::memcpy(out_data, it->second, output.size_bytes());

    tensors[output.name()] = out_data;
    return Status::Ok();
  }
};

/// Factory function to create CPU backend
inline std::shared_ptr<CpuBackend> create_cpu_backend() {
  return std::make_shared<CpuBackend>();
}

} // namespace zenith

#endif // ZENITH_CPU_BACKEND_HPP
