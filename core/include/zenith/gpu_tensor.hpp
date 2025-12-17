// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// GPU Tensor - Persistent GPU Memory Management
// Keeps data on GPU to avoid expensive H2D/D2H copies on every operation.

#ifndef ZENITH_GPU_TENSOR_HPP
#define ZENITH_GPU_TENSOR_HPP

#include "types.hpp"

#ifdef ZENITH_HAS_CUDA

#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace zenith {

// ============================================================================
// GPU Memory Pool
// ============================================================================

/// Simple memory pool that caches allocations by size
class GpuMemoryPool {
public:
  static GpuMemoryPool &instance() {
    static GpuMemoryPool pool;
    return pool;
  }

  /// Allocate or reuse memory from pool
  void *allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we have a cached allocation of this size
    auto it = free_blocks_.find(size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      void *ptr = it->second.back();
      it->second.pop_back();
      stats_.cache_hits++;
      return ptr;
    }

    // Allocate new memory
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      return nullptr;
    }

    stats_.allocations++;
    stats_.total_allocated += size;
    return ptr;
  }

  /// Return memory to pool (instead of freeing)
  void deallocate(void *ptr, size_t size) {
    if (ptr == nullptr)
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Add to free list for this size
    free_blocks_[size].push_back(ptr);
    stats_.cache_returns++;
  }

  /// Force free a specific pointer
  void free(void *ptr) {
    if (ptr) {
      cudaFree(ptr);
    }
  }

  /// Clear all cached memory
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &[size, ptrs] : free_blocks_) {
      for (void *ptr : ptrs) {
        cudaFree(ptr);
      }
    }
    free_blocks_.clear();
  }

  /// Get pool statistics
  struct Stats {
    size_t allocations = 0;
    size_t cache_hits = 0;
    size_t cache_returns = 0;
    size_t total_allocated = 0;
  };

  Stats get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  ~GpuMemoryPool() { clear(); }

private:
  GpuMemoryPool() = default;

  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::vector<void *>> free_blocks_;
  Stats stats_;
};

// ============================================================================
// GPU Tensor Class
// ============================================================================

/// Tensor that lives on GPU with automatic memory management
class GpuTensor {
public:
  /// Create empty tensor (no allocation)
  GpuTensor() : data_(nullptr), size_bytes_(0), owns_memory_(false) {}

  /// Create tensor with given shape and dtype
  GpuTensor(const Shape &shape, DataType dtype = DataType::Float32)
      : shape_(shape), dtype_(dtype), owns_memory_(true) {
    size_bytes_ = shape.numel() * dtype_size(dtype);
    data_ = GpuMemoryPool::instance().allocate(size_bytes_);
  }

  /// Move constructor
  GpuTensor(GpuTensor &&other) noexcept
      : data_(other.data_), shape_(std::move(other.shape_)),
        dtype_(other.dtype_), size_bytes_(other.size_bytes_),
        owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.owns_memory_ = false;
  }

  /// Move assignment
  GpuTensor &operator=(GpuTensor &&other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      shape_ = std::move(other.shape_);
      dtype_ = other.dtype_;
      size_bytes_ = other.size_bytes_;
      owns_memory_ = other.owns_memory_;
      other.data_ = nullptr;
      other.owns_memory_ = false;
    }
    return *this;
  }

  /// Destructor - returns memory to pool
  ~GpuTensor() { release(); }

  // No copy
  GpuTensor(const GpuTensor &) = delete;
  GpuTensor &operator=(const GpuTensor &) = delete;

  // ==========================================================================
  // Factory Methods
  // ==========================================================================

  /// Create empty tensor on GPU
  static GpuTensor empty(const Shape &shape,
                         DataType dtype = DataType::Float32) {
    return GpuTensor(shape, dtype);
  }

  /// Create tensor from host data (copies H2D)
  static GpuTensor from_host(const void *host_data, const Shape &shape,
                             DataType dtype = DataType::Float32) {
    GpuTensor tensor(shape, dtype);
    if (tensor.data_ && host_data) {
      cudaMemcpy(tensor.data_, host_data, tensor.size_bytes_,
                 cudaMemcpyHostToDevice);
    }
    return tensor;
  }

  // ==========================================================================
  // Data Access
  // ==========================================================================

  /// Get device pointer
  void *data() { return data_; }
  const void *data() const { return data_; }

  /// Get typed device pointer
  template <typename T> T *data_ptr() { return static_cast<T *>(data_); }

  template <typename T> const T *data_ptr() const {
    return static_cast<const T *>(data_);
  }

  /// Copy data to host
  void to_host(void *host_dst) const {
    if (data_ && host_dst) {
      cudaMemcpy(host_dst, data_, size_bytes_, cudaMemcpyDeviceToHost);
    }
  }

  // ==========================================================================
  // Properties
  // ==========================================================================

  const Shape &shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  size_t size_bytes() const { return size_bytes_; }
  size_t numel() const { return shape_.numel(); }
  bool is_valid() const { return data_ != nullptr; }

  /// Get dimension
  int64_t dim(size_t i) const { return shape_[i]; }
  size_t ndim() const { return shape_.rank(); }

private:
  void release() {
    if (owns_memory_ && data_) {
      GpuMemoryPool::instance().deallocate(data_, size_bytes_);
    }
    data_ = nullptr;
    owns_memory_ = false;
  }

  void *data_;
  Shape shape_;
  DataType dtype_;
  size_t size_bytes_;
  bool owns_memory_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Get memory pool statistics
inline GpuMemoryPool::Stats get_gpu_memory_stats() {
  return GpuMemoryPool::instance().get_stats();
}

/// Clear GPU memory pool
inline void clear_gpu_memory_pool() { GpuMemoryPool::instance().clear(); }

} // namespace zenith

#else // !ZENITH_HAS_CUDA

namespace zenith {

// Stub classes when CUDA not available
class GpuTensor {
public:
  bool is_valid() const { return false; }
};

} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_GPU_TENSOR_HPP
