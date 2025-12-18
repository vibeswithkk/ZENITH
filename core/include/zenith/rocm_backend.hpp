/**
 * @file rocm_backend.hpp
 * @brief ROCm/HIP Backend for AMD GPUs
 *
 * Portable implementation using HIP API for cross-platform GPU computing.
 * HIP provides CUDA-compatible API that works on both AMD and NVIDIA GPUs.
 *
 * This header provides:
 * - Conditional compilation with ZENITH_HAS_ROCM
 * - HIP-based memory management
 * - Kernel launch interface mirroring CUDA backend
 *
 * Build with: -DZENITH_HAS_ROCM=1 -I/opt/rocm/include
 * Link with: -L/opt/rocm/lib -lamdhip64
 *
 * Copyright 2025 Wahyu Ardiansyah
 * Licensed under the Apache License, Version 2.0
 */

#ifndef ZENITH_ROCM_BACKEND_HPP
#define ZENITH_ROCM_BACKEND_HPP

#include "backend.hpp"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef ZENITH_HAS_ROCM
#include <hip/hip_runtime.h>
#ifdef ZENITH_HAS_ROCBLAS
#include <rocblas/rocblas.h>
#endif
#ifdef ZENITH_HAS_MIOPEN
#include <miopen/miopen.h>
#endif
#endif

namespace zenith {

/**
 * @brief ROCm backend for AMD GPU execution.
 *
 * Implements the Backend interface using AMD HIP API.
 * When ZENITH_HAS_ROCM is not defined, all methods return
 * appropriate error states for graceful fallback.
 */
class ROCmBackend : public Backend {
public:
  explicit ROCmBackend(int device_id = 0)
      : device_id_(device_id), initialized_(false) {}

  ~ROCmBackend() override {
    if (initialized_) {
      cleanup();
    }
  }

  // Non-copyable, non-movable (owns HIP resources)
  ROCmBackend(const ROCmBackend &) = delete;
  ROCmBackend &operator=(const ROCmBackend &) = delete;
  ROCmBackend(ROCmBackend &&) = delete;
  ROCmBackend &operator=(ROCmBackend &&) = delete;

  // ========================================================================
  // Backend Identity
  // ========================================================================

  [[nodiscard]] std::string name() const override { return "rocm"; }

  [[nodiscard]] std::string description() const override {
    return "AMD ROCm/HIP Backend for GPU acceleration";
  }

  // ========================================================================
  // Availability and Capabilities
  // ========================================================================

  [[nodiscard]] bool is_available() const override {
#ifdef ZENITH_HAS_ROCM
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    return (err == hipSuccess && device_count > 0);
#else
    return false;
#endif
  }

  [[nodiscard]] std::string version() const override {
#ifdef ZENITH_HAS_ROCM
    int runtime_ver = 0;
    hipRuntimeGetVersion(&runtime_ver);
    return "HIP " + std::to_string(runtime_ver / 10000000) + "." +
           std::to_string((runtime_ver / 100000) % 100);
#else
    return "ROCm not available";
#endif
  }

  [[nodiscard]] bool supports_dtype(DataType dtype) const override {
    switch (dtype) {
    case DataType::Float32:
    case DataType::Float16:
    case DataType::Float64:
    case DataType::Int32:
    case DataType::Int64:
      return true;
    case DataType::BFloat16:
#ifdef ZENITH_HAS_ROCM
      return true; // AMD MI-series GPUs support BF16
#else
      return false;
#endif
    default:
      return false;
    }
  }

  // ========================================================================
  // Initialization
  // ========================================================================

  bool initialize() {
#ifdef ZENITH_HAS_ROCM
    hipError_t err = hipSetDevice(device_id_);
    if (err != hipSuccess) {
      last_error_ =
          std::string("Failed to set device: ") + hipGetErrorString(err);
      return false;
    }
    initialized_ = true;
    return true;
#else
    last_error_ = "ROCm support not compiled. Rebuild with -DZENITH_HAS_ROCM=1";
    return false;
#endif
  }

  void cleanup() {
#ifdef ZENITH_HAS_ROCM
    if (initialized_) {
      hipDeviceReset();
      initialized_ = false;
    }
#endif
  }

  // ========================================================================
  // Execution
  // ========================================================================

  [[nodiscard]] Status
  execute(const GraphIR &graph,
          const std::unordered_map<std::string, const void *> &inputs,
          std::unordered_map<std::string, void *> &outputs) override {
#ifdef ZENITH_HAS_ROCM
    if (!initialized_ && !initialize()) {
      return Status(StatusCode::InternalError,
                    "ROCm backend initialization failed: " + last_error_);
    }

    // TODO: Implement full graph execution
    // For now, return success for empty graphs
    if (graph.num_nodes() == 0) {
      return Status::Ok();
    }

    return Status(StatusCode::NotImplemented,
                  "ROCm graph execution not yet implemented");
#else
    (void)graph;
    (void)inputs;
    (void)outputs;
    return Status(StatusCode::NotImplemented, "ROCm not available");
#endif
  }

  // ========================================================================
  // Memory Management
  // ========================================================================

  [[nodiscard]] void *allocate(size_t size_bytes) override {
#ifdef ZENITH_HAS_ROCM
    void *ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size_bytes);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipMalloc failed: ") +
                               hipGetErrorString(err));
    }
    return ptr;
#else
    (void)size_bytes;
    throw std::runtime_error("ROCm not available");
#endif
  }

  void deallocate(void *ptr) override {
#ifdef ZENITH_HAS_ROCM
    if (ptr) {
      hipFree(ptr);
    }
#else
    (void)ptr;
#endif
  }

  Status copy_to_device(void *dst, const void *src,
                        size_t size_bytes) override {
#ifdef ZENITH_HAS_ROCM
    hipError_t err = hipMemcpy(dst, src, size_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      return Status(StatusCode::InternalError,
                    std::string("hipMemcpy H2D failed: ") +
                        hipGetErrorString(err));
    }
    return Status::Ok();
#else
    (void)dst;
    (void)src;
    (void)size_bytes;
    return Status(StatusCode::NotImplemented, "ROCm not available");
#endif
  }

  Status copy_to_host(void *dst, const void *src, size_t size_bytes) override {
#ifdef ZENITH_HAS_ROCM
    hipError_t err = hipMemcpy(dst, src, size_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      return Status(StatusCode::InternalError,
                    std::string("hipMemcpy D2H failed: ") +
                        hipGetErrorString(err));
    }
    return Status::Ok();
#else
    (void)dst;
    (void)src;
    (void)size_bytes;
    return Status(StatusCode::NotImplemented, "ROCm not available");
#endif
  }

  // ========================================================================
  // Synchronization
  // ========================================================================

  void synchronize() override {
#ifdef ZENITH_HAS_ROCM
    hipDeviceSynchronize();
#endif
  }

  // ========================================================================
  // Device Information (ROCm-specific methods)
  // ========================================================================

  [[nodiscard]] std::string get_device_name() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return std::string(props.name);
    }
#endif
    return "ROCm Device (not initialized)";
  }

  [[nodiscard]] size_t get_total_memory() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return props.totalGlobalMem;
    }
#endif
    return 0;
  }

  [[nodiscard]] size_t get_free_memory() const {
#ifdef ZENITH_HAS_ROCM
    size_t free_mem = 0, total_mem = 0;
    if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
      return free_mem;
    }
#endif
    return 0;
  }

  [[nodiscard]] std::string get_last_error() const { return last_error_; }

  /**
   * @brief Get compute capability (major, minor).
   */
  [[nodiscard]] std::pair<int, int> get_compute_capability() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return {props.major, props.minor};
    }
#endif
    return {0, 0};
  }

  /**
   * @brief Get number of compute units.
   */
  [[nodiscard]] int get_compute_units() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return props.multiProcessorCount;
    }
#endif
    return 0;
  }

  /**
   * @brief Get maximum threads per block.
   */
  [[nodiscard]] int get_max_threads_per_block() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return props.maxThreadsPerBlock;
    }
#endif
    return 256; // Safe default
  }

  /**
   * @brief Get warp size (wavefront size for AMD).
   * AMD GPUs use 64-thread wavefronts, unlike NVIDIA's 32-thread warps.
   */
  [[nodiscard]] int get_warp_size() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return props.warpSize; // 64 for AMD, 32 for NVIDIA via HIP
    }
#endif
    return 64; // AMD default
  }

private:
  int device_id_;
  bool initialized_;
  std::string last_error_;
};

// Factory function
inline std::unique_ptr<ROCmBackend> create_rocm_backend(int device_id = 0) {
  return std::make_unique<ROCmBackend>(device_id);
}

} // namespace zenith

#endif // ZENITH_ROCM_BACKEND_HPP
