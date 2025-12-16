/**
 * @file oneapi_backend.hpp
 * @brief Intel oneAPI/SYCL Backend
 *
 * Cross-platform implementation using SYCL for Intel, NVIDIA, and AMD GPUs.
 * Based on Intel DPC++ compiler and oneAPI ecosystem.
 *
 * This header provides:
 * - Conditional compilation with ZENITH_HAS_ONEAPI
 * - SYCL-based queue and memory management
 * - Unified Shared Memory (USM) support
 *
 * Build with: -DZENITH_HAS_ONEAPI=1 -fsycl
 * Link with: appropriate SYCL runtime
 *
 * Copyright 2025 Wahyu Ardiansyah
 * Licensed under the Apache License, Version 2.0
 */

#ifndef ZENITH_ONEAPI_BACKEND_HPP
#define ZENITH_ONEAPI_BACKEND_HPP

#include "backend.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ZENITH_HAS_ONEAPI
#include <sycl/sycl.hpp>
#endif

namespace zenith {

/**
 * @brief oneAPI/SYCL backend for cross-platform GPU execution.
 *
 * Implements the Backend interface using SYCL API.
 * Supports Intel, NVIDIA, and AMD GPUs through appropriate
 * SYCL implementations.
 */
class OneAPIBackend : public Backend {
public:
  enum class DeviceType { GPU, CPU, ACCELERATOR, ANY };

  explicit OneAPIBackend(DeviceType type = DeviceType::GPU)
      : device_type_(type), initialized_(false) {}

  ~OneAPIBackend() override { cleanup(); }

  std::string name() const override { return "oneapi"; }

  bool is_available() const override {
#ifdef ZENITH_HAS_ONEAPI
    try {
      auto devices = get_sycl_devices();
      return !devices.empty();
    } catch (...) {
      return false;
    }
#else
    return false;
#endif
  }

  bool initialize() override {
#ifdef ZENITH_HAS_ONEAPI
    try {
      // Create queue with appropriate device selector
      sycl::device selected_device;

      switch (device_type_) {
      case DeviceType::GPU:
        selected_device = sycl::device(sycl::gpu_selector_v);
        break;
      case DeviceType::CPU:
        selected_device = sycl::device(sycl::cpu_selector_v);
        break;
      case DeviceType::ACCELERATOR:
        selected_device = sycl::device(sycl::accelerator_selector_v);
        break;
      default:
        selected_device = sycl::device(sycl::default_selector_v);
      }

      queue_ = std::make_unique<sycl::queue>(
          selected_device,
          sycl::property_list{sycl::property::queue::in_order()});

      initialized_ = true;
      return true;

    } catch (const sycl::exception &e) {
      last_error_ = std::string("SYCL error: ") + e.what();
      return false;
    } catch (const std::exception &e) {
      last_error_ = std::string("Error: ") + e.what();
      return false;
    }
#else
    last_error_ =
        "oneAPI support not compiled. Rebuild with -DZENITH_HAS_ONEAPI=1";
    return false;
#endif
  }

  void cleanup() override {
#ifdef ZENITH_HAS_ONEAPI
    if (initialized_ && queue_) {
      queue_->wait();
      queue_.reset();
      initialized_ = false;
    }
#endif
  }

  void *allocate(size_t size) override {
#ifdef ZENITH_HAS_ONEAPI
    if (!queue_) {
      throw std::runtime_error("Queue not initialized");
    }
    // Use Unified Shared Memory (device allocation)
    void *ptr = sycl::malloc_device(size, *queue_);
    if (!ptr) {
      throw std::runtime_error("sycl::malloc_device failed");
    }
    return ptr;
#else
    (void)size;
    throw std::runtime_error("oneAPI not available");
#endif
  }

  void deallocate(void *ptr) override {
#ifdef ZENITH_HAS_ONEAPI
    if (ptr && queue_) {
      sycl::free(ptr, *queue_);
    }
#else
    (void)ptr;
#endif
  }

  void copy_to_device(void *dst, const void *src, size_t size) override {
#ifdef ZENITH_HAS_ONEAPI
    if (!queue_) {
      throw std::runtime_error("Queue not initialized");
    }
    queue_->memcpy(dst, src, size).wait();
#else
    (void)dst;
    (void)src;
    (void)size;
    throw std::runtime_error("oneAPI not available");
#endif
  }

  void copy_to_host(void *dst, const void *src, size_t size) override {
#ifdef ZENITH_HAS_ONEAPI
    if (!queue_) {
      throw std::runtime_error("Queue not initialized");
    }
    queue_->memcpy(dst, src, size).wait();
#else
    (void)dst;
    (void)src;
    (void)size;
    throw std::runtime_error("oneAPI not available");
#endif
  }

  void synchronize() override {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      queue_->wait();
    }
#endif
  }

  std::string get_device_name() const override {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      return queue_->get_device().get_info<sycl::info::device::name>();
    }
#endif
    return "oneAPI Device (not initialized)";
  }

  size_t get_total_memory() const override {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      return queue_->get_device()
          .get_info<sycl::info::device::global_mem_size>();
    }
#endif
    return 0;
  }

  size_t get_free_memory() const override {
    // SYCL doesn't provide direct free memory query
    // Return total as approximation
    return get_total_memory();
  }

  std::string get_last_error() const override { return last_error_; }

  // oneAPI-specific methods

  /**
   * @brief Get maximum work group size.
   */
  size_t get_max_work_group_size() const {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      return queue_->get_device()
          .get_info<sycl::info::device::max_work_group_size>();
    }
#endif
    return 256;
  }

  /**
   * @brief Get maximum compute units.
   */
  unsigned int get_max_compute_units() const {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      return queue_->get_device()
          .get_info<sycl::info::device::max_compute_units>();
    }
#endif
    return 1;
  }

  /**
   * @brief Check if device supports FP16.
   */
  bool supports_fp16() const {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      auto aspects =
          queue_->get_device().get_info<sycl::info::device::aspects>();
      for (auto asp : aspects) {
        if (asp == sycl::aspect::fp16) {
          return true;
        }
      }
    }
#endif
    return false;
  }

  /**
   * @brief Check if device supports FP64.
   */
  bool supports_fp64() const {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      auto aspects =
          queue_->get_device().get_info<sycl::info::device::aspects>();
      for (auto asp : aspects) {
        if (asp == sycl::aspect::fp64) {
          return true;
        }
      }
    }
#endif
    return false;
  }

  /**
   * @brief Get device vendor.
   */
  std::string get_vendor() const {
#ifdef ZENITH_HAS_ONEAPI
    if (queue_) {
      return queue_->get_device().get_info<sycl::info::device::vendor>();
    }
#endif
    return "Unknown";
  }

  /**
   * @brief Allocate shared memory (accessible from host and device).
   */
  void *allocate_shared(size_t size) {
#ifdef ZENITH_HAS_ONEAPI
    if (!queue_) {
      throw std::runtime_error("Queue not initialized");
    }
    return sycl::malloc_shared(size, *queue_);
#else
    (void)size;
    throw std::runtime_error("oneAPI not available");
#endif
  }

private:
#ifdef ZENITH_HAS_ONEAPI
  std::unique_ptr<sycl::queue> queue_;

  static std::vector<sycl::device> get_sycl_devices() {
    std::vector<sycl::device> result;
    auto platforms = sycl::platform::get_platforms();
    for (auto &p : platforms) {
      auto devices = p.get_devices();
      result.insert(result.end(), devices.begin(), devices.end());
    }
    return result;
  }
#endif

  DeviceType device_type_;
  bool initialized_;
  std::string last_error_;
};

// Factory function
inline std::unique_ptr<Backend> create_oneapi_backend(
    OneAPIBackend::DeviceType type = OneAPIBackend::DeviceType::GPU) {
  return std::make_unique<OneAPIBackend>(type);
}

} // namespace zenith

#endif // ZENITH_ONEAPI_BACKEND_HPP
