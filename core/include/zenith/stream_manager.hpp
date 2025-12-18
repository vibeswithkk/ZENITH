// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// CUDA Stream Manager untuk Performance Optimization
// Berdasarkan CetakBiru.md: Hardware-aware scheduling, throughput optimization
// Referensi: NVIDIA CUDA Best Practices, cuDNN Streams, PyTorch CUDA Graphs

#ifndef ZENITH_STREAM_MANAGER_HPP
#define ZENITH_STREAM_MANAGER_HPP

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef ZENITH_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace zenith {
namespace cuda {

// ============================================================================
// Stream Priority
// ============================================================================

enum class StreamPriority {
  Low = 0,
  Normal = 1,
  High = 2,
};

// ============================================================================
// Stream Wrapper
// ============================================================================

#ifdef ZENITH_HAS_CUDA
/// RAII wrapper untuk CUDA stream
class Stream {
public:
  explicit Stream(StreamPriority priority = StreamPriority::Normal) {
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    int cuda_priority;
    switch (priority) {
    case StreamPriority::High:
      cuda_priority = greatest_priority;
      break;
    case StreamPriority::Low:
      cuda_priority = least_priority;
      break;
    default:
      cuda_priority = (least_priority + greatest_priority) / 2;
    }

    cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking,
                                 cuda_priority);
    priority_ = priority;
  }

  ~Stream() {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Non-copyable
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  // Movable
  Stream(Stream &&other) noexcept
      : stream_(other.stream_), priority_(other.priority_) {
    other.stream_ = nullptr;
  }

  Stream &operator=(Stream &&other) noexcept {
    if (this != &other) {
      if (stream_) {
        cudaStreamDestroy(stream_);
      }
      stream_ = other.stream_;
      priority_ = other.priority_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  /// Get native CUDA stream
  [[nodiscard]] cudaStream_t get() const { return stream_; }

  /// Synchronize stream
  void synchronize() { cudaStreamSynchronize(stream_); }

  /// Check if stream is idle
  [[nodiscard]] bool is_idle() const {
    cudaError_t status = cudaStreamQuery(stream_);
    return status == cudaSuccess;
  }

  /// Get priority
  [[nodiscard]] StreamPriority priority() const { return priority_; }

private:
  cudaStream_t stream_ = nullptr;
  StreamPriority priority_ = StreamPriority::Normal;
};

// ============================================================================
// Event Wrapper
// ============================================================================

/// RAII wrapper untuk CUDA event
class Event {
public:
  explicit Event(bool enable_timing = true) {
    unsigned int flags = cudaEventDefault;
    if (!enable_timing) {
      flags = cudaEventDisableTiming;
    }
    cudaEventCreateWithFlags(&event_, flags);
    timing_enabled_ = enable_timing;
  }

  ~Event() {
    if (event_) {
      cudaEventDestroy(event_);
    }
  }

  // Non-copyable
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  // Movable
  Event(Event &&other) noexcept
      : event_(other.event_), timing_enabled_(other.timing_enabled_) {
    other.event_ = nullptr;
  }

  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      if (event_) {
        cudaEventDestroy(event_);
      }
      event_ = other.event_;
      timing_enabled_ = other.timing_enabled_;
      other.event_ = nullptr;
    }
    return *this;
  }

  /// Get native CUDA event
  [[nodiscard]] cudaEvent_t get() const { return event_; }

  /// Record event on stream
  void record(cudaStream_t stream = 0) { cudaEventRecord(event_, stream); }

  /// Wait for event
  void synchronize() { cudaEventSynchronize(event_); }

  /// Check if event completed
  [[nodiscard]] bool is_complete() const {
    return cudaEventQuery(event_) == cudaSuccess;
  }

  /// Elapsed time between two events (ms)
  [[nodiscard]] float elapsed_ms(const Event &start) const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start.event_, event_);
    return ms;
  }

private:
  cudaEvent_t event_ = nullptr;
  bool timing_enabled_ = true;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// Stream Pool
// ============================================================================

#ifdef ZENITH_HAS_CUDA
/// Pool of reusable CUDA streams
class StreamPool {
public:
  explicit StreamPool(size_t initial_size = 4) {
    for (size_t i = 0; i < initial_size; ++i) {
      available_.push_back(std::make_unique<Stream>());
    }
  }

  /// Acquire a stream from pool
  std::unique_ptr<Stream> acquire() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (available_.empty()) {
      // Create new stream
      return std::make_unique<Stream>();
    }

    auto stream = std::move(available_.back());
    available_.pop_back();
    return stream;
  }

  /// Release stream back to pool
  void release(std::unique_ptr<Stream> stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push_back(std::move(stream));
  }

  /// Get pool size
  [[nodiscard]] size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_.size();
  }

private:
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Stream>> available_;
};
#endif

// ============================================================================
// Async Memory Operations
// ============================================================================

#ifdef ZENITH_HAS_CUDA
/// Async memory copy H2D
inline void async_copy_h2d(void *dst, const void *src, size_t size,
                           cudaStream_t stream = 0) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

/// Async memory copy D2H
inline void async_copy_d2h(void *dst, const void *src, size_t size,
                           cudaStream_t stream = 0) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

/// Async memory copy D2D
inline void async_copy_d2d(void *dst, const void *src, size_t size,
                           cudaStream_t stream = 0) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
}

/// Async memset
inline void async_memset(void *ptr, int value, size_t size,
                         cudaStream_t stream = 0) {
  cudaMemsetAsync(ptr, value, size, stream);
}

// ============================================================================
// Pinned Memory Allocator
// ============================================================================

/// Allocate pinned (page-locked) host memory
inline void *alloc_pinned(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate pinned memory");
  }
  return ptr;
}

/// Free pinned memory
inline void free_pinned(void *ptr) {
  if (ptr) {
    cudaFreeHost(ptr);
  }
}

/// RAII wrapper for pinned memory
class PinnedBuffer {
public:
  explicit PinnedBuffer(size_t size) : size_(size) {
    ptr_ = alloc_pinned(size);
  }

  ~PinnedBuffer() { free_pinned(ptr_); }

  // Non-copyable
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  // Movable
  PinnedBuffer(PinnedBuffer &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  PinnedBuffer &operator=(PinnedBuffer &&other) noexcept {
    if (this != &other) {
      free_pinned(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  [[nodiscard]] void *data() { return ptr_; }
  [[nodiscard]] const void *data() const { return ptr_; }
  [[nodiscard]] size_t size() const { return size_; }

  template <typename T> [[nodiscard]] T *as() { return static_cast<T *>(ptr_); }

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// Pipeline Manager
// ============================================================================

#ifdef ZENITH_HAS_CUDA
/// Manage multi-stream pipeline for overlapping compute and transfer
class PipelineManager {
public:
  explicit PipelineManager(int num_stages = 3) : num_stages_(num_stages) {
    for (int i = 0; i < num_stages; ++i) {
      compute_streams_.push_back(
          std::make_unique<Stream>(StreamPriority::High));
      transfer_streams_.push_back(
          std::make_unique<Stream>(StreamPriority::Normal));
      events_.push_back(std::make_unique<Event>(false));
    }
  }

  /// Get compute stream for stage
  [[nodiscard]] cudaStream_t compute_stream(int stage) const {
    return compute_streams_[stage % num_stages_]->get();
  }

  /// Get transfer stream for stage
  [[nodiscard]] cudaStream_t transfer_stream(int stage) const {
    return transfer_streams_[stage % num_stages_]->get();
  }

  /// Record event on compute stream
  void record_compute_done(int stage) {
    events_[stage % num_stages_]->record(compute_stream(stage));
  }

  /// Wait for compute on transfer stream
  void wait_compute_on_transfer(int stage) {
    cudaStreamWaitEvent(transfer_stream(stage),
                        events_[stage % num_stages_]->get(), 0);
  }

  /// Wait for transfer on compute stream
  void wait_transfer_on_compute(int stage) {
    // Create temp event for transfer
    Event transfer_event(false);
    transfer_event.record(transfer_stream(stage));
    cudaStreamWaitEvent(compute_stream(stage), transfer_event.get(), 0);
  }

  /// Synchronize all streams
  void synchronize_all() {
    for (auto &stream : compute_streams_) {
      stream->synchronize();
    }
    for (auto &stream : transfer_streams_) {
      stream->synchronize();
    }
  }

  /// Get number of pipeline stages
  [[nodiscard]] int num_stages() const { return num_stages_; }

private:
  int num_stages_;
  std::vector<std::unique_ptr<Stream>> compute_streams_;
  std::vector<std::unique_ptr<Stream>> transfer_streams_;
  std::vector<std::unique_ptr<Event>> events_;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// Kernel Launcher with Stream Support
// ============================================================================

#ifdef ZENITH_HAS_CUDA
/// Configuration for kernel launch
struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared_mem = 0;
  cudaStream_t stream = 0;

  LaunchConfig(dim3 g, dim3 b, size_t sm = 0, cudaStream_t s = 0)
      : grid(g), block(b), shared_mem(sm), stream(s) {}

  /// Create 1D launch config
  static LaunchConfig linear(size_t total_threads, size_t block_size = 256,
                             cudaStream_t stream = 0) {
    size_t num_blocks = (total_threads + block_size - 1) / block_size;
    return LaunchConfig(dim3(num_blocks), dim3(block_size), 0, stream);
  }

  /// Create 2D launch config
  static LaunchConfig grid2d(size_t width, size_t height, size_t tile_x = 16,
                             size_t tile_y = 16, cudaStream_t stream = 0) {
    dim3 grid((width + tile_x - 1) / tile_x, (height + tile_y - 1) / tile_y);
    dim3 block(tile_x, tile_y);
    return LaunchConfig(grid, block, 0, stream);
  }
};

/// Compute optimal block size for occupancy
inline int get_optimal_block_size(const void *kernel, size_t dynamic_smem = 0) {
  int min_grid_size, block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel,
                                     dynamic_smem);
  return block_size;
}

/// Get max active blocks per SM
inline int get_max_active_blocks(const void *kernel, int block_size,
                                 size_t dynamic_smem = 0) {
  int num_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel, block_size,
                                                dynamic_smem);
  return num_blocks;
}
#endif // ZENITH_HAS_CUDA

// ============================================================================
// Performance Hints
// ============================================================================

/// Performance optimization hints
struct PerformanceHints {
  /// Enable kernel fusion
  bool enable_fusion = true;

  /// Enable async memory operations
  bool enable_async = true;

  /// Use pinned memory for transfers
  bool use_pinned_memory = true;

  /// Number of pipeline stages
  int pipeline_stages = 3;

  /// Preferred block size (0 = auto)
  int preferred_block_size = 0;

  /// Enable TF32 for FP32 GEMM
  bool enable_tf32 = true;

  /// Enable Tensor Cores
  bool enable_tensor_cores = true;

  /// Cache size limit (bytes)
  size_t cache_size_limit = 1024 * 1024 * 512; // 512 MB

  /// Prefetch batch size
  int prefetch_batch_size = 4;
};

// ============================================================================
// Global Default Hints
// ============================================================================

inline PerformanceHints &default_hints() {
  static PerformanceHints hints;
  return hints;
}

inline void set_default_hints(const PerformanceHints &hints) {
  default_hints() = hints;
}

} // namespace cuda
} // namespace zenith

#endif // ZENITH_STREAM_MANAGER_HPP
