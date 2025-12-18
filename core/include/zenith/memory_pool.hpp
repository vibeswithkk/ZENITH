// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Memory Pool Allocator untuk GPU Memory Management
// Berdasarkan CetakBiru.md: Observabilitas dan pengukuran penggunaan memori
// Referensi: PyTorch Caching Allocator, NVIDIA Memory Pool, cuMemPool

#ifndef ZENITH_MEMORY_POOL_HPP
#define ZENITH_MEMORY_POOL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef ZENITH_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace zenith {
namespace memory {

// ============================================================================
// Memory Statistics
// ============================================================================

/// Statistik penggunaan memori
struct MemoryStats {
  /// Total bytes yang dialokasikan saat ini
  size_t allocated_bytes = 0;

  /// Total bytes yang di-cache (tersedia untuk reuse)
  size_t cached_bytes = 0;

  /// Peak allocated bytes sepanjang waktu
  size_t peak_allocated_bytes = 0;

  /// Jumlah alokasi yang berhasil
  size_t num_allocations = 0;

  /// Jumlah dealokasi
  size_t num_deallocations = 0;

  /// Cache hits (reuse dari pool)
  size_t cache_hits = 0;

  /// Cache misses (perlu alokasi baru)
  size_t cache_misses = 0;

  /// Total bytes yang pernah dialokasikan (lifetime)
  size_t total_allocated_lifetime = 0;

  /// Jumlah block di cache
  size_t cached_blocks = 0;

  /// Percentage of cache hits
  [[nodiscard]] double cache_hit_rate() const {
    size_t total = cache_hits + cache_misses;
    return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
  }

  /// Get summary string
  [[nodiscard]] std::string summary() const {
    std::ostringstream ss;
    ss << "MemoryStats:\n";
    ss << "  Allocated: " << (allocated_bytes / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Cached: " << (cached_bytes / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Peak: " << (peak_allocated_bytes / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Cache hit rate: " << (cache_hit_rate() * 100.0) << "%\n";
    ss << "  Allocations: " << num_allocations << "\n";
    ss << "  Deallocations: " << num_deallocations << "\n";
    ss << "  Cached blocks: " << cached_blocks << "\n";
    return ss.str();
  }
};

// ============================================================================
// Memory Block
// ============================================================================

/// Block memori yang dialokasikan
struct MemoryBlock {
  void *ptr = nullptr;
  size_t size = 0;
  bool in_use = false;
  size_t allocation_id = 0;

  MemoryBlock() = default;
  MemoryBlock(void *p, size_t s, size_t id)
      : ptr(p), size(s), in_use(true), allocation_id(id) {}
};

// ============================================================================
// Memory Pool Configuration
// ============================================================================

/// Konfigurasi untuk memory pool
struct PoolConfig {
  /// Minimum block size (default 512 bytes)
  size_t min_block_size = 512;

  /// Maximum cached memory (default 1GB)
  size_t max_cached_bytes = 1024ULL * 1024 * 1024;

  /// Block size granularity (default 512 bytes)
  /// Semua alokasi akan dibulatkan ke kelipatan ini
  size_t block_granularity = 512;

  /// Maximum number of cached blocks per size
  size_t max_blocks_per_size = 32;

  /// Enable defragmentation
  bool enable_defrag = true;

  /// Growth factor untuk large allocations
  double growth_factor = 1.2;
};

// ============================================================================
// Base Allocator Interface
// ============================================================================

/// Interface untuk allocator
class IAllocator {
public:
  virtual ~IAllocator() = default;

  /// Alokasi memory
  virtual void *allocate(size_t size) = 0;

  /// Dealokasi memory
  virtual void deallocate(void *ptr) = 0;

  /// Get statistics
  virtual MemoryStats get_stats() const = 0;

  /// Clear cache
  virtual void clear_cache() = 0;

  /// Reset statistics
  virtual void reset_stats() = 0;
};

// ============================================================================
// CPU Memory Pool
// ============================================================================

/// Memory pool untuk CPU memory
class CpuMemoryPool : public IAllocator {
public:
  explicit CpuMemoryPool(PoolConfig config = {}) : config_(std::move(config)) {}

  ~CpuMemoryPool() override { clear_all(); }

  void *allocate(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (size == 0)
      return nullptr;

    // Round up size
    size_t aligned_size = align_size(size);

    // Try to find in cache
    auto it = free_blocks_.find(aligned_size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      // Cache hit!
      MemoryBlock block = it->second.front();
      it->second.pop_front();
      block.in_use = true;
      block.allocation_id = next_alloc_id_++;

      stats_.cache_hits++;
      stats_.allocated_bytes += block.size;
      stats_.cached_bytes -= block.size;
      stats_.cached_blocks--;
      stats_.peak_allocated_bytes =
          std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);

      allocated_blocks_[block.ptr] = block;
      return block.ptr;
    }

    // Cache miss - allocate new
    void *ptr = std::malloc(aligned_size);
    if (!ptr) {
      throw std::bad_alloc();
    }

    MemoryBlock block(ptr, aligned_size, next_alloc_id_++);
    allocated_blocks_[ptr] = block;

    stats_.cache_misses++;
    stats_.num_allocations++;
    stats_.allocated_bytes += aligned_size;
    stats_.total_allocated_lifetime += aligned_size;
    stats_.peak_allocated_bytes =
        std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);

    return ptr;
  }

  void deallocate(void *ptr) override {
    if (!ptr)
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
      // Not from this pool, ignore
      return;
    }

    MemoryBlock block = it->second;
    allocated_blocks_.erase(it);

    stats_.allocated_bytes -= block.size;
    stats_.num_deallocations++;

    // Check if we should cache this block
    if (stats_.cached_bytes + block.size <= config_.max_cached_bytes) {
      block.in_use = false;
      free_blocks_[block.size].push_back(block);
      stats_.cached_bytes += block.size;
      stats_.cached_blocks++;
    } else {
      // Cache full, actually free
      std::free(ptr);
    }
  }

  MemoryStats get_stats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  void clear_cache() override {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &[size, blocks] : free_blocks_) {
      for (auto &block : blocks) {
        std::free(block.ptr);
      }
    }
    free_blocks_.clear();
    stats_.cached_bytes = 0;
    stats_.cached_blocks = 0;
  }

  void reset_stats() override {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = MemoryStats{};
  }

private:
  void clear_all() {
    clear_cache();

    // Free all allocated blocks (warning: application may still use them!)
    for (auto &[ptr, block] : allocated_blocks_) {
      std::free(ptr);
    }
    allocated_blocks_.clear();
  }

  size_t align_size(size_t size) const {
    size = std::max(size, config_.min_block_size);
    return ((size + config_.block_granularity - 1) /
            config_.block_granularity) *
           config_.block_granularity;
  }

  PoolConfig config_;
  mutable std::mutex mutex_;
  MemoryStats stats_;
  size_t next_alloc_id_ = 0;

  // Size -> list of free blocks
  std::map<size_t, std::deque<MemoryBlock>> free_blocks_;

  // Pointer -> allocated block
  std::map<void *, MemoryBlock> allocated_blocks_;
};

// ============================================================================
// GPU Memory Pool (CUDA)
// ============================================================================

#ifdef ZENITH_HAS_CUDA
/// Memory pool untuk CUDA GPU memory
class GpuMemoryPool : public IAllocator {
public:
  explicit GpuMemoryPool(PoolConfig config = {}, int device_id = 0)
      : config_(std::move(config)), device_id_(device_id) {}

  ~GpuMemoryPool() override { clear_all(); }

  void *allocate(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (size == 0)
      return nullptr;

    size_t aligned_size = align_size(size);

    // Try cache first
    auto it = free_blocks_.find(aligned_size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      MemoryBlock block = it->second.front();
      it->second.pop_front();
      block.in_use = true;
      block.allocation_id = next_alloc_id_++;

      stats_.cache_hits++;
      stats_.allocated_bytes += block.size;
      stats_.cached_bytes -= block.size;
      stats_.cached_blocks--;
      stats_.peak_allocated_bytes =
          std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);

      allocated_blocks_[block.ptr] = block;
      return block.ptr;
    }

    // Cache miss - allocate new
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, aligned_size);

    if (err != cudaSuccess) {
      // Try clearing cache and retry
      clear_cache_unsafe();
      err = cudaMalloc(&ptr, aligned_size);

      if (err != cudaSuccess) {
        throw std::bad_alloc();
      }
    }

    MemoryBlock block(ptr, aligned_size, next_alloc_id_++);
    allocated_blocks_[ptr] = block;

    stats_.cache_misses++;
    stats_.num_allocations++;
    stats_.allocated_bytes += aligned_size;
    stats_.total_allocated_lifetime += aligned_size;
    stats_.peak_allocated_bytes =
        std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);

    return ptr;
  }

  void deallocate(void *ptr) override {
    if (!ptr)
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
      return;
    }

    MemoryBlock block = it->second;
    allocated_blocks_.erase(it);

    stats_.allocated_bytes -= block.size;
    stats_.num_deallocations++;

    // Cache or free
    if (stats_.cached_bytes + block.size <= config_.max_cached_bytes) {
      block.in_use = false;
      free_blocks_[block.size].push_back(block);
      stats_.cached_bytes += block.size;
      stats_.cached_blocks++;
    } else {
      cudaFree(ptr);
    }
  }

  MemoryStats get_stats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  void clear_cache() override {
    std::lock_guard<std::mutex> lock(mutex_);
    clear_cache_unsafe();
  }

  void reset_stats() override {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = MemoryStats{};
  }

  /// Get CUDA device info
  std::string device_info() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);

    std::ostringstream ss;
    ss << "GPU " << device_id_ << ": " << prop.name << "\n";
    ss << "  Total memory: "
       << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    return ss.str();
  }

private:
  void clear_cache_unsafe() {
    for (auto &[size, blocks] : free_blocks_) {
      for (auto &block : blocks) {
        cudaFree(block.ptr);
      }
    }
    free_blocks_.clear();
    stats_.cached_bytes = 0;
    stats_.cached_blocks = 0;
  }

  void clear_all() {
    clear_cache();

    for (auto &[ptr, block] : allocated_blocks_) {
      cudaFree(ptr);
    }
    allocated_blocks_.clear();
  }

  size_t align_size(size_t size) const {
    size = std::max(size, config_.min_block_size);
    // For GPU, align to 256 bytes (cache line)
    constexpr size_t gpu_alignment = 256;
    return ((size + gpu_alignment - 1) / gpu_alignment) * gpu_alignment;
  }

  PoolConfig config_;
  int device_id_;
  mutable std::mutex mutex_;
  MemoryStats stats_;
  size_t next_alloc_id_ = 0;

  std::map<size_t, std::deque<MemoryBlock>> free_blocks_;
  std::map<void *, MemoryBlock> allocated_blocks_;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// Global Memory Pool Manager
// ============================================================================

/// Singleton manager untuk memory pools
class MemoryPoolManager {
public:
  static MemoryPoolManager &instance() {
    static MemoryPoolManager mgr;
    return mgr;
  }

  /// Get CPU pool
  CpuMemoryPool &cpu_pool() {
    if (!cpu_pool_) {
      cpu_pool_ = std::make_unique<CpuMemoryPool>();
    }
    return *cpu_pool_;
  }

#ifdef ZENITH_HAS_CUDA
  /// Get GPU pool for specific device
  GpuMemoryPool &gpu_pool(int device_id = 0) {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    auto it = gpu_pools_.find(device_id);
    if (it == gpu_pools_.end()) {
      gpu_pools_[device_id] =
          std::make_unique<GpuMemoryPool>(PoolConfig{}, device_id);
    }
    return *gpu_pools_[device_id];
  }
#endif

  /// Get all stats
  std::string get_summary() const {
    std::ostringstream ss;
    ss << "=== Memory Pool Summary ===\n\n";

    if (cpu_pool_) {
      ss << "CPU Pool:\n";
      ss << cpu_pool_->get_stats().summary() << "\n";
    }

#ifdef ZENITH_HAS_CUDA
    for (const auto &[device_id, pool] : gpu_pools_) {
      ss << "GPU " << device_id << " Pool:\n";
      ss << pool->get_stats().summary() << "\n";
    }
#endif

    return ss.str();
  }

  /// Clear all caches
  void clear_all_caches() {
    if (cpu_pool_) {
      cpu_pool_->clear_cache();
    }

#ifdef ZENITH_HAS_CUDA
    for (auto &[device_id, pool] : gpu_pools_) {
      pool->clear_cache();
    }
#endif
  }

private:
  MemoryPoolManager() = default;
  ~MemoryPoolManager() = default;

  MemoryPoolManager(const MemoryPoolManager &) = delete;
  MemoryPoolManager &operator=(const MemoryPoolManager &) = delete;

  std::unique_ptr<CpuMemoryPool> cpu_pool_;

#ifdef ZENITH_HAS_CUDA
  std::mutex gpu_mutex_;
  std::map<int, std::unique_ptr<GpuMemoryPool>> gpu_pools_;
#endif
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Allocate from CPU pool
inline void *cpu_allocate(size_t size) {
  return MemoryPoolManager::instance().cpu_pool().allocate(size);
}

/// Deallocate to CPU pool
inline void cpu_deallocate(void *ptr) {
  MemoryPoolManager::instance().cpu_pool().deallocate(ptr);
}

#ifdef ZENITH_HAS_CUDA
/// Allocate from GPU pool
inline void *gpu_allocate(size_t size, int device_id = 0) {
  return MemoryPoolManager::instance().gpu_pool(device_id).allocate(size);
}

/// Deallocate to GPU pool
inline void gpu_deallocate(void *ptr, int device_id = 0) {
  MemoryPoolManager::instance().gpu_pool(device_id).deallocate(ptr);
}
#endif

/// Get memory summary
inline std::string memory_summary() {
  return MemoryPoolManager::instance().get_summary();
}

/// Clear all caches
inline void clear_memory_caches() {
  MemoryPoolManager::instance().clear_all_caches();
}

} // namespace memory
} // namespace zenith

#endif // ZENITH_MEMORY_POOL_HPP
