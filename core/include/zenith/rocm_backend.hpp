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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ZENITH_HAS_ROCM
#include <hip/hip_runtime.h>
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

  std::string name() const override { return "rocm"; }

  bool is_available() const override {
#ifdef ZENITH_HAS_ROCM
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    return (err == hipSuccess && device_count > 0);
#else
    return false;
#endif
  }

  bool initialize() override {
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

  void cleanup() override {
#ifdef ZENITH_HAS_ROCM
    if (initialized_) {
      hipDeviceReset();
      initialized_ = false;
    }
#endif
  }

  void *allocate(size_t size) override {
#ifdef ZENITH_HAS_ROCM
    void *ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipMalloc failed: ") +
                               hipGetErrorString(err));
    }
    return ptr;
#else
    (void)size;
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

  void copy_to_device(void *dst, const void *src, size_t size) override {
#ifdef ZENITH_HAS_ROCM
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipMemcpy H2D failed: ") +
                               hipGetErrorString(err));
    }
#else
    (void)dst;
    (void)src;
    (void)size;
    throw std::runtime_error("ROCm not available");
#endif
  }

  void copy_to_host(void *dst, const void *src, size_t size) override {
#ifdef ZENITH_HAS_ROCM
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipMemcpy D2H failed: ") +
                               hipGetErrorString(err));
    }
#else
    (void)dst;
    (void)src;
    (void)size;
    throw std::runtime_error("ROCm not available");
#endif
  }

  void synchronize() override {
#ifdef ZENITH_HAS_ROCM
    hipDeviceSynchronize();
#endif
  }

  std::string get_device_name() const override {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return std::string(props.name);
    }
#endif
    return "ROCm Device (not initialized)";
  }

  size_t get_total_memory() const override {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return props.totalGlobalMem;
    }
#endif
    return 0;
  }

  size_t get_free_memory() const override {
#ifdef ZENITH_HAS_ROCM
    size_t free_mem = 0, total_mem = 0;
    if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
      return free_mem;
    }
#endif
    return 0;
  }

  std::string get_last_error() const override { return last_error_; }

  // ROCm-specific methods

  /**
   * @brief Get compute capability (major, minor).
   */
  std::pair<int, int> get_compute_capability() const {
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
  int get_compute_units() const {
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
  int get_max_threads_per_block() const {
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
   */
  int get_warp_size() const {
#ifdef ZENITH_HAS_ROCM
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id_) == hipSuccess) {
      return props.warpSize; // 64 for AMD, 32 for NVIDIA
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
inline std::unique_ptr<Backend> create_rocm_backend(int device_id = 0) {
  return std::make_unique<ROCmBackend>(device_id);
}

} // namespace zenith

#endif // ZENITH_ROCM_BACKEND_HPP
