// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Advanced Performance Optimizations
// Berdasarkan CetakBiru.md: Throughput optimization, cache locality
// Referensi: NVIDIA Tensor Cores, cuDNN algorithms, Flash Attention

#ifndef ZENITH_PERF_OPTIMIZATIONS_HPP
#define ZENITH_PERF_OPTIMIZATIONS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifdef ZENITH_HAS_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace zenith {
namespace perf {

// ============================================================================
// Memory Access Pattern Analyzer
// ============================================================================

/// Analyze memory access patterns for optimization
struct AccessPattern {
  enum class Type {
    Sequential, // Good for coalescing
    Strided,    // May have bank conflicts
    Random,     // Poor locality
    Tiled,      // Good for shared mem
  };

  Type type = Type::Sequential;
  size_t stride = 1;
  size_t tile_size = 32;
  bool is_coalesced = true;

  /// Recommend optimization based on pattern
  [[nodiscard]] std::string recommendation() const {
    switch (type) {
    case Type::Sequential:
      return "Coalesced access - optimal";
    case Type::Strided:
      return "Use shared memory tiling";
    case Type::Random:
      return "Consider texture memory or L2 prefetch";
    case Type::Tiled:
      return "Good for shared memory";
    }
    return "Unknown pattern";
  }
};

// ============================================================================
// Occupancy Calculator
// ============================================================================

/// Calculate theoretical occupancy
struct OccupancyInfo {
  int threads_per_block = 256;
  int blocks_per_sm = 0;
  int warps_per_sm = 0;
  int max_warps_per_sm = 64; // Varies by GPU
  float occupancy = 0.0f;

  int registers_per_thread = 32;
  int shared_mem_per_block = 0;

  /// Calculate occupancy percentage
  void calculate() {
    warps_per_sm = blocks_per_sm * (threads_per_block / 32);
    occupancy = static_cast<float>(warps_per_sm) / max_warps_per_sm;
  }

  [[nodiscard]] std::string summary() const {
    std::string result;
    result +=
        "Occupancy: " + std::to_string(static_cast<int>(occupancy * 100)) +
        "%\n";
    result += "Blocks/SM: " + std::to_string(blocks_per_sm) + "\n";
    result += "Warps/SM: " + std::to_string(warps_per_sm) + "/" +
              std::to_string(max_warps_per_sm) + "\n";
    return result;
  }
};

// ============================================================================
// Loop Tiling Parameters
// ============================================================================

/// Optimal tiling for matrix operations
struct TilingConfig {
  int tile_m = 128; // Output rows
  int tile_n = 128; // Output cols
  int tile_k = 8;   // Reduction dimension

  int warp_m = 32; // Warp tile M
  int warp_n = 32; // Warp tile N
  int warp_k = 8;  // Warp tile K

  /// GEMM tiling for different sizes
  static TilingConfig for_gemm(int M, int N, int K) {
    TilingConfig config;

    // Heuristics based on matrix size
    if (M >= 4096 && N >= 4096) {
      // Large matrices
      config.tile_m = 256;
      config.tile_n = 128;
      config.tile_k = 32;
    } else if (M >= 1024 && N >= 1024) {
      // Medium matrices
      config.tile_m = 128;
      config.tile_n = 128;
      config.tile_k = 16;
    } else {
      // Small matrices
      config.tile_m = 64;
      config.tile_n = 64;
      config.tile_k = 8;
    }

    return config;
  }

  /// Attention tiling
  static TilingConfig for_attention(int seq_len, int head_dim) {
    TilingConfig config;

    // Flash Attention style tiling
    if (seq_len <= 512) {
      config.tile_m = 64;
      config.tile_n = 64;
    } else if (seq_len <= 2048) {
      config.tile_m = 128;
      config.tile_n = 64;
    } else {
      // Very long sequences
      config.tile_m = 256;
      config.tile_n = 64;
    }

    config.tile_k = std::min(head_dim, 64);
    return config;
  }
};

// ============================================================================
// Vectorized Load/Store
// ============================================================================

/// Vectorization info for memory operations
struct VectorizationInfo {
  int vector_width = 4; // float4, int4
  bool can_vectorize = true;
  size_t alignment = 16; // bytes

  /// Check if address is aligned for vectorization
  static bool is_aligned(const void *ptr, size_t alignment = 16) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
  }

  /// Get recommended vector width for data type
  static int recommended_width(size_t element_size) {
    // Target 16 bytes for optimal memory transactions
    return std::min(16 / static_cast<int>(element_size), 4);
  }
};

// ============================================================================
// Warp Shuffle Patterns
// ============================================================================

/// Common warp shuffle patterns
struct WarpShufflePattern {
  enum class Type {
    Broadcast, // Broadcast from lane 0
    Reduction, // Tree reduction
    Scan,      // Prefix sum
    Transpose, // 4x4 transpose
  };

  Type type = Type::Reduction;
  int source_lane = 0;
  int width = 32;

  /// Get number of shuffle operations for reduction
  static int reduction_steps(int width = 32) {
    return static_cast<int>(std::log2(width));
  }
};

// ============================================================================
// Shared Memory Bank Conflict Analyzer
// ============================================================================

/// Analyze and avoid bank conflicts
struct BankConflictInfo {
  int num_banks = 32; // Modern GPUs
  int elements_per_bank = 1;
  bool has_conflict = false;
  int conflict_degree = 1;

  /// Check if stride causes bank conflicts
  static bool check_stride_conflict(size_t stride, size_t element_size,
                                    int num_banks = 32) {
    size_t bank_stride = stride * element_size;
    // Conflict if stride is multiple of num_banks
    return (bank_stride % (num_banks * 4)) == 0;
  }

  /// Recommend padding to avoid conflicts
  static size_t recommended_padding(size_t width, size_t element_size,
                                    int num_banks = 32) {
    size_t bank_width = num_banks * 4 / element_size;
    if (width % bank_width == 0) {
      return 1; // Add 1 element padding
    }
    return 0;
  }
};

// ============================================================================
// L2 Cache Optimization
// ============================================================================

/// L2 cache hints and prefetching
struct CacheOptimization {
  enum class AccessHint {
    Streaming,  // Use for one-time access
    Persistent, // Keep in cache
    Normal,     // Default
  };

  size_t l2_size = 0; // Will be queried from device
  size_t working_set_size = 0;
  bool fits_in_l2 = false;

#ifdef ZENITH_HAS_CUDA
  /// Query L2 cache size
  static size_t query_l2_size(int device = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.l2CacheSize;
  }

  /// Set L2 cache persistence
  static void set_cache_persistence(void *ptr, size_t size, bool persistent) {
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr = ptr;
    attr.accessPolicyWindow.num_bytes = size;
    attr.accessPolicyWindow.hitRatio = persistent ? 1.0f : 0.0f;
    attr.accessPolicyWindow.hitProp =
        persistent ? cudaAccessPropertyPersisting : cudaAccessPropertyStreaming;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    // Would be applied via cudaStreamSetAttribute
  }
#endif
};

// ============================================================================
// Arithmetic Intensity Calculator
// ============================================================================

/// Calculate arithmetic intensity (FLOPs / bytes)
struct ArithmeticIntensity {
  double flops = 0;
  double bytes_transferred = 0;
  double intensity = 0;

  // Roofline model thresholds (example for A100)
  static constexpr double memory_bandwidth_gbs = 2000.0; // GB/s
  static constexpr double peak_flops_tflops = 19.5;      // TF32 TFLOPS

  /// Calculate intensity
  void calculate() {
    if (bytes_transferred > 0) {
      intensity = flops / bytes_transferred;
    }
  }

  /// Check if compute or memory bound
  [[nodiscard]] bool is_compute_bound() const {
    double ridge_point =
        peak_flops_tflops * 1e12 / (memory_bandwidth_gbs * 1e9);
    return intensity >= ridge_point;
  }

  /// Recommendation based on intensity
  [[nodiscard]] std::string recommendation() const {
    if (is_compute_bound()) {
      return "Compute bound: focus on algorithmic optimization";
    }
    return "Memory bound: focus on cache/memory optimization";
  }

  /// Calculate for GEMM (C = A * B)
  static ArithmeticIntensity for_gemm(int M, int N, int K,
                                      size_t elem_size = 4) {
    ArithmeticIntensity ai;
    ai.flops = 2.0 * M * N * K; // 2 ops per multiply-add
    ai.bytes_transferred = elem_size * (M * K + K * N + M * N);
    ai.calculate();
    return ai;
  }

  /// Calculate for attention
  static ArithmeticIntensity for_attention(int batch, int heads, int seq,
                                           int dim, size_t elem_size = 4) {
    ArithmeticIntensity ai;
    // Q*K^T + softmax + V
    ai.flops = 4.0 * batch * heads * seq * seq * dim;
    ai.bytes_transferred = elem_size * batch * heads * 3 * seq * dim;
    ai.calculate();
    return ai;
  }
};

// ============================================================================
// Operation Fusion Analyzer
// ============================================================================

/// Analyze fusion opportunities
struct FusionOpportunity {
  enum class Type {
    PointwiseFusion, // e.g., add + relu
    BiasActivation,  // gemm + bias + activation
    ResidualFusion,  // add + normalization
    AttentionFusion, // qk + softmax + av
  };

  Type type;
  std::vector<std::string> ops_to_fuse;
  double estimated_speedup = 1.0;

  /// Check if fusion is beneficial
  [[nodiscard]] bool is_beneficial() const {
    return estimated_speedup > 1.05; // At least 5% speedup
  }
};

/// Analyze graph for fusion opportunities
class FusionAnalyzer {
public:
  /// Common fusion patterns
  static std::vector<FusionOpportunity>
  analyze_patterns(const std::vector<std::string> &ops) {
    std::vector<FusionOpportunity> opportunities;

    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      // MatMul/Conv + Bias + Activation
      if ((ops[i] == "MatMul" || ops[i] == "Conv") && i + 2 < ops.size() &&
          ops[i + 1] == "Add" &&
          (ops[i + 2] == "ReLU" || ops[i + 2] == "GELU")) {
        FusionOpportunity opp;
        opp.type = FusionOpportunity::Type::BiasActivation;
        opp.ops_to_fuse = {ops[i], ops[i + 1], ops[i + 2]};
        opp.estimated_speedup = 1.3; // ~30% speedup
        opportunities.push_back(opp);
        i += 2;
      }
      // Add + LayerNorm
      else if (ops[i] == "Add" && i + 1 < ops.size() &&
               ops[i + 1] == "LayerNorm") {
        FusionOpportunity opp;
        opp.type = FusionOpportunity::Type::ResidualFusion;
        opp.ops_to_fuse = {ops[i], ops[i + 1]};
        opp.estimated_speedup = 1.2;
        opportunities.push_back(opp);
        i += 1;
      }
      // Pointwise operations
      else if (is_pointwise(ops[i]) && i + 1 < ops.size() &&
               is_pointwise(ops[i + 1])) {
        FusionOpportunity opp;
        opp.type = FusionOpportunity::Type::PointwiseFusion;
        opp.ops_to_fuse = {ops[i], ops[i + 1]};
        opp.estimated_speedup = 1.15;
        opportunities.push_back(opp);
        i += 1;
      }
    }

    return opportunities;
  }

private:
  static bool is_pointwise(const std::string &op) {
    return op == "Add" || op == "Mul" || op == "ReLU" || op == "Sigmoid" ||
           op == "Tanh" || op == "GELU";
  }
};

// ============================================================================
// Performance Profiling Hints
// ============================================================================

/// Hints for profiling and optimization
struct ProfilingHints {
  /// Estimate kernel execution time
  static double estimate_kernel_time_ms(double flops, double tflops_capability,
                                        double efficiency = 0.7) {
    return flops / (tflops_capability * 1e9 * efficiency);
  }

  /// Estimate memory transfer time
  static double estimate_transfer_time_ms(size_t bytes, double bandwidth_gbs) {
    return static_cast<double>(bytes) / (bandwidth_gbs * 1e6);
  }

  /// Memory vs compute balance
  static std::string analyze_balance(double compute_time, double memory_time) {
    if (memory_time > compute_time * 1.2) {
      return "Memory bound - optimize data movement";
    } else if (compute_time > memory_time * 1.2) {
      return "Compute bound - optimize kernel efficiency";
    }
    return "Balanced workload";
  }
};

} // namespace perf
} // namespace zenith

#endif // ZENITH_PERF_OPTIMIZATIONS_HPP
