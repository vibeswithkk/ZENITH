// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// CUDA Backend for Zenith Framework
// This file provides GPU execution using CUDA for NVIDIA GPUs.
// Designed for use in Google Colab and other CUDA-enabled environments.

#ifndef ZENITH_CUDA_BACKEND_HPP
#define ZENITH_CUDA_BACKEND_HPP

#include "backend.hpp"
#include "graph_ir.hpp"

#ifdef ZENITH_HAS_CUDA

#include <cuda_runtime.h>

namespace zenith {

// ============================================================================
// CUDA Error Handling Macros
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      return Status::Error(StatusCode::InternalError,                          \
                           std::string("CUDA error: ") +                       \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ============================================================================
// CudaBackend - GPU execution backend using CUDA
// ============================================================================

/// CUDA backend implementation for NVIDIA GPUs
/// Based on CetakBiru Section 5.1 - Phase 1 requirements
class CudaBackend : public Backend {
public:
  CudaBackend() : device_id_(0), initialized_(false) {}

  ~CudaBackend() {
    if (initialized_) {
      cudaDeviceReset();
    }
  }

  // ========================================================================
  // Backend Interface Implementation
  // ========================================================================

  [[nodiscard]] std::string name() const override { return "cuda"; }

  [[nodiscard]] std::string description() const override {
    if (!is_available()) {
      return "CUDA Backend (not available)";
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);

    return std::string("CUDA Backend (") + prop.name + ", SM " +
           std::to_string(prop.major) + "." + std::to_string(prop.minor) +
           ", " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB)";
  }

  [[nodiscard]] bool is_available() const override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
  }

  [[nodiscard]] std::string version() const override {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
  }

  Status initialize() {
    if (initialized_)
      return Status::Ok();

    if (!is_available()) {
      return Status::Error(StatusCode::NotFound, "No CUDA devices available");
    }

    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
      return Status::Error(StatusCode::InternalError,
                           std::string("Failed to set CUDA device: ") +
                               cudaGetErrorString(err));
    }

    initialized_ = true;
    return Status::Ok();
  }

  [[nodiscard]] Status
  execute(const GraphIR &graph,
          const std::unordered_map<std::string, const void *> &inputs,
          std::unordered_map<std::string, void *> &outputs) override {

    if (!initialized_) {
      auto status = initialize();
      if (!status.ok())
        return status;
    }

    // Get topological order for execution
    auto nodes = graph.topological_order();

    // Device tensor storage during execution
    std::unordered_map<std::string, void *> device_data;

    // Copy inputs to device
    for (const auto &input : graph.inputs()) {
      auto it = inputs.find(input.name());
      if (it == inputs.end()) {
        cleanup(device_data);
        return Status::Error(StatusCode::InvalidInput,
                             "Missing input: " + input.name());
      }

      size_t size = input.size_bytes();
      void *d_ptr = nullptr;

      cudaError_t err = cudaMalloc(&d_ptr, size);
      if (err != cudaSuccess) {
        cleanup(device_data);
        return Status::Error(StatusCode::OutOfMemory,
                             "Failed to allocate device memory");
      }

      err = cudaMemcpy(d_ptr, it->second, size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        cudaFree(d_ptr);
        cleanup(device_data);
        return Status::Error(StatusCode::InternalError,
                             "Failed to copy to device");
      }

      device_data[input.name()] = d_ptr;
    }

    // Load constants to device
    for (const auto &[name, const_data] : graph.constants()) {
      void *d_ptr = nullptr;
      size_t size = const_data.size();

      cudaError_t err = cudaMalloc(&d_ptr, size);
      if (err != cudaSuccess) {
        cleanup(device_data);
        return Status::Error(StatusCode::OutOfMemory,
                             "Failed to allocate device memory for constant");
      }

      err = cudaMemcpy(d_ptr, const_data.data().data(), size,
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        cudaFree(d_ptr);
        cleanup(device_data);
        return Status::Error(StatusCode::InternalError,
                             "Failed to copy constant to device");
      }

      device_data[name] = d_ptr;
    }

    // Execute each node
    for (const auto *node : nodes) {
      auto status = execute_node(*node, device_data);
      if (!status.ok()) {
        cleanup(device_data);
        return status;
      }
    }

    // Synchronize before copying back
    cudaDeviceSynchronize();

    // Copy outputs back to host
    for (const auto &output : graph.outputs()) {
      auto out_it = outputs.find(output.name());
      if (out_it == outputs.end()) {
        cleanup(device_data);
        return Status::Error(StatusCode::InvalidOutput,
                             "Missing output buffer: " + output.name());
      }

      auto data_it = device_data.find(output.name());
      if (data_it != device_data.end()) {
        cudaError_t err =
            cudaMemcpy(out_it->second, data_it->second, output.size_bytes(),
                       cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
          cleanup(device_data);
          return Status::Error(StatusCode::InternalError,
                               "Failed to copy output to host");
        }
      }
    }

    cleanup(device_data);
    return Status::Ok();
  }

  [[nodiscard]] void *allocate(size_t size_bytes) override {
    void *ptr = nullptr;
    cudaMalloc(&ptr, size_bytes);
    return ptr;
  }

  void deallocate(void *ptr) override {
    if (ptr) {
      cudaFree(ptr);
    }
  }

  Status copy_to_device(void *dst, const void *src,
                        size_t size_bytes) override {
    cudaError_t err = cudaMemcpy(dst, src, size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      return Status::Error(StatusCode::InternalError,
                           std::string("cudaMemcpy H2D failed: ") +
                               cudaGetErrorString(err));
    }
    return Status::Ok();
  }

  Status copy_to_host(void *dst, const void *src, size_t size_bytes) override {
    cudaError_t err = cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      return Status::Error(StatusCode::InternalError,
                           std::string("cudaMemcpy D2H failed: ") +
                               cudaGetErrorString(err));
    }
    return Status::Ok();
  }

private:
  int device_id_;
  bool initialized_;

  void cleanup(std::unordered_map<std::string, void *> &device_data) {
    for (auto &[_, ptr] : device_data) {
      if (ptr) {
        cudaFree(ptr);
      }
    }
    device_data.clear();
  }

  Status execute_node(const Node &node,
                      std::unordered_map<std::string, void *> &tensors) {
    const std::string &op = node.op_type();

    // Dispatch to kernel implementations
    // These will be implemented in cuda_kernels.cu
    if (op == "Relu") {
      return execute_relu(node, tensors);
    } else if (op == "MatMul" || op == "Gemm") {
      return execute_matmul(node, tensors);
    } else if (op == "Add") {
      return execute_add(node, tensors);
    } else if (op == "Conv" || op == "Conv2D") {
      return execute_conv2d(node, tensors);
    } else if (op == "Identity") {
      return execute_identity(node, tensors);
    }

    return Status::Error(StatusCode::UnsupportedOp,
                         "Unsupported CUDA operation: " + op);
  }

  // Forward declarations - implementations in cuda_kernels.cu
  Status execute_relu(const Node &node,
                      std::unordered_map<std::string, void *> &tensors);
  Status execute_matmul(const Node &node,
                        std::unordered_map<std::string, void *> &tensors);
  Status execute_add(const Node &node,
                     std::unordered_map<std::string, void *> &tensors);
  Status execute_conv2d(const Node &node,
                        std::unordered_map<std::string, void *> &tensors);
  Status execute_identity(const Node &node,
                          std::unordered_map<std::string, void *> &tensors);
};

/// Factory function
inline std::shared_ptr<CudaBackend> create_cuda_backend() {
  return std::make_shared<CudaBackend>();
}

} // namespace zenith

#else // !ZENITH_HAS_CUDA

namespace zenith {

/// Stub CudaBackend when CUDA is not available
class CudaBackend : public Backend {
public:
  [[nodiscard]] std::string name() const override { return "cuda"; }
  [[nodiscard]] std::string description() const override {
    return "CUDA Backend (not compiled with CUDA support)";
  }
  [[nodiscard]] bool is_available() const override { return false; }
  [[nodiscard]] std::string version() const override { return "N/A"; }

  [[nodiscard]] Status
  execute(const GraphIR &,
          const std::unordered_map<std::string, const void *> &,
          std::unordered_map<std::string, void *> &) override {
    return Status::Error(StatusCode::NotImplemented,
                         "CUDA support not compiled");
  }

  [[nodiscard]] void *allocate(size_t) override { return nullptr; }
  void deallocate(void *) override {}
};

inline std::shared_ptr<CudaBackend> create_cuda_backend() {
  return std::make_shared<CudaBackend>();
}

} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_CUDA_BACKEND_HPP
