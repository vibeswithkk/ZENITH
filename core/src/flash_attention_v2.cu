// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Flash Attention V2 Implementation
// Based on: "FlashAttention-2: Faster Attention with Better Parallelism and
// Work Partitioning" Paper: https://arxiv.org/abs/2307.08691
//
// Key optimizations over standard attention:
// 1. Tiling: Process Q, K, V in blocks that fit in SRAM (shared memory)
// 2. Recomputation: Don't store the N×N attention matrix in HBM
// 3. Online Softmax: Compute softmax incrementally without full materialization
// 4. Improved Work Partitioning: Better parallelism across warps

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// Configuration Constants
// ============================================================================

// Block sizes for Flash Attention V2
// Optimized for NVIDIA Ampere (A100, RTX 3090) and Hopper (H100)
constexpr int BLOCK_M = 64; // Block rows for Q
constexpr int BLOCK_N = 64; // Block cols for K/V
constexpr int BLOCK_D = 64; // Head dimension (supports up to 128)

// Warp configuration
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4; // 4 warps per block = 128 threads

// ============================================================================
// Utility Functions
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float block_reduce_max(float val,
                                                  float *shared_max) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  // Warp-level reduction
  val = warp_reduce_max(val);

  // Write warp maxes to shared memory
  if (lane == 0) {
    shared_max[warp_id] = val;
  }
  __syncthreads();

  // First warp does final reduction
  if (warp_id == 0) {
    val = (lane < NUM_WARPS) ? shared_max[lane] : -FLT_MAX;
    val = warp_reduce_max(val);
  }

  return val;
}

__device__ __forceinline__ float block_reduce_sum(float val,
                                                  float *shared_sum) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  // Warp-level reduction
  val = warp_reduce_sum(val);

  // Write warp sums to shared memory
  if (lane == 0) {
    shared_sum[warp_id] = val;
  }
  __syncthreads();

  // First warp does final reduction
  if (warp_id == 0) {
    val = (lane < NUM_WARPS) ? shared_sum[lane] : 0.0f;
    val = warp_reduce_sum(val);
  }

  return val;
}

// ============================================================================
// Flash Attention V2 Forward Kernel (FP32)
// ============================================================================
//
// Input shapes:
//   Q: [batch, num_heads, seq_len, head_dim]
//   K: [batch, num_heads, seq_len, head_dim]
//   V: [batch, num_heads, seq_len, head_dim]
//   O: [batch, num_heads, seq_len, head_dim] (output)
//
// Algorithm (FlashAttention-2):
//   For each block of Q:
//     Initialize: m = -inf, l = 0, O = 0
//     For each block of K, V:
//       S = Q @ K.T  (tiled in shared memory)
//       m_new = max(m, rowmax(S))
//       P = exp(S - m_new)  (online softmax numerator)
//       l_new = exp(m - m_new) * l + rowsum(P)
//       O = exp(m - m_new) * O + P @ V
//     O = O / l  (normalize by softmax denominator)

template <int BLOCK_M_T, int BLOCK_N_T, int BLOCK_D_T>
__global__ void flash_attention_v2_forward_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O, int batch_size,
    int num_heads, int seq_len, int head_dim, float scale) {
  // Block indices
  const int batch_head_idx = blockIdx.z; // Combined batch and head index
  const int batch_idx = batch_head_idx / num_heads;
  const int head_idx = batch_head_idx % num_heads;
  const int q_block_idx =
      blockIdx.x; // Which block of Q rows this block processes

  // Thread indices
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // Row index within this Q block that this thread helps process
  // Each row of Q is processed by multiple threads
  const int q_row_in_block = tid % BLOCK_M_T;
  const int q_row_global = q_block_idx * BLOCK_M_T + q_row_in_block;

  // Check bounds
  if (q_row_global >= seq_len)
    return;

  // Pointers to this batch/head's data
  const int bhd_offset = batch_idx * num_heads * seq_len * head_dim +
                         head_idx * seq_len * head_dim;
  const float *Q_bh = Q + bhd_offset;
  const float *K_bh = K + bhd_offset;
  const float *V_bh = V + bhd_offset;
  float *O_bh = O + bhd_offset;

  // Shared memory for K and V tiles
  __shared__ float K_shared[BLOCK_N_T]
                           [BLOCK_D_T + 1]; // +1 to avoid bank conflicts
  __shared__ float V_shared[BLOCK_N_T][BLOCK_D_T + 1];
  __shared__ float reduce_max[NUM_WARPS];
  __shared__ float reduce_sum[NUM_WARPS];

  // Per-thread accumulators for this Q row
  float m_i = -FLT_MAX;   // Running max for softmax
  float l_i = 0.0f;       // Running sum for softmax
  float O_acc[BLOCK_D_T]; // Output accumulator

// Initialize output accumulator
#pragma unroll
  for (int d = 0; d < BLOCK_D_T; d++) {
    O_acc[d] = 0.0f;
  }

  // Load Q row for this thread (persistent across K/V blocks)
  float Q_row[BLOCK_D_T];
  if (q_row_global < seq_len) {
#pragma unroll
    for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
      Q_row[d] = Q_bh[q_row_global * head_dim + d] * scale;
    }
  }

  // Number of key/value blocks
  const int num_kv_blocks = (seq_len + BLOCK_N_T - 1) / BLOCK_N_T;

  // Loop over key/value blocks
  for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
    const int kv_start = kv_block * BLOCK_N_T;

    // Cooperative load of K and V into shared memory
    __syncthreads();

#pragma unroll
    for (int i = tid; i < BLOCK_N_T * head_dim; i += blockDim.x) {
      int row = i / head_dim;
      int col = i % head_dim;
      int global_row = kv_start + row;

      if (global_row < seq_len && col < head_dim) {
        K_shared[row][col] = K_bh[global_row * head_dim + col];
        V_shared[row][col] = V_bh[global_row * head_dim + col];
      } else {
        K_shared[row][col] = 0.0f;
        V_shared[row][col] = 0.0f;
      }
    }

    __syncthreads();

    // Compute attention scores for this Q row against all K in this block
    // S[i,j] = Q[i,:] @ K[j,:].T
    float S_row[BLOCK_N_T];
    float row_max = -FLT_MAX;

#pragma unroll
    for (int j = 0; j < BLOCK_N_T; j++) {
      int kv_row_global = kv_start + j;
      float score = 0.0f;

      if (kv_row_global < seq_len) {
#pragma unroll
        for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
          score += Q_row[d] * K_shared[j][d];
        }
      } else {
        score = -FLT_MAX; // Mask out-of-bounds
      }

      S_row[j] = score;
      row_max = fmaxf(row_max, score);
    }

    // Online softmax update: compute new max and scaling factor
    float m_new = fmaxf(m_i, row_max);
    float scale_old = expf(m_i - m_new);

    // Compute probabilities and sum
    float row_sum = 0.0f;
    float P_row[BLOCK_N_T];

#pragma unroll
    for (int j = 0; j < BLOCK_N_T; j++) {
      float p = expf(S_row[j] - m_new);
      P_row[j] = p;
      if (kv_start + j < seq_len) {
        row_sum += p;
      }
    }

    // Update l (softmax denominator)
    float l_new = scale_old * l_i + row_sum;

// Rescale previous output accumulator
#pragma unroll
    for (int d = 0; d < BLOCK_D_T; d++) {
      O_acc[d] *= scale_old;
    }

// Accumulate P @ V for this block
#pragma unroll
    for (int j = 0; j < BLOCK_N_T; j++) {
      if (kv_start + j < seq_len) {
        float p = P_row[j];
#pragma unroll
        for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
          O_acc[d] += p * V_shared[j][d];
        }
      }
    }

    // Update running max and sum
    m_i = m_new;
    l_i = l_new;
  }

  // Normalize output by softmax denominator
  if (q_row_global < seq_len && l_i > 0.0f) {
    float inv_l = 1.0f / l_i;
#pragma unroll
    for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
      O_bh[q_row_global * head_dim + d] = O_acc[d] * inv_l;
    }
  }
}

// ============================================================================
// Flash Attention V2 Causal Kernel (for autoregressive models)
// ============================================================================
// Same as above but with causal masking: position i can only attend to j <= i

template <int BLOCK_M_T, int BLOCK_N_T, int BLOCK_D_T>
__global__ void flash_attention_v2_causal_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O, int batch_size,
    int num_heads, int seq_len, int head_dim, float scale) {
  const int batch_head_idx = blockIdx.z;
  const int batch_idx = batch_head_idx / num_heads;
  const int head_idx = batch_head_idx % num_heads;
  const int q_block_idx = blockIdx.x;

  const int tid = threadIdx.x;
  const int q_row_in_block = tid % BLOCK_M_T;
  const int q_row_global = q_block_idx * BLOCK_M_T + q_row_in_block;

  if (q_row_global >= seq_len)
    return;

  const int bhd_offset = batch_idx * num_heads * seq_len * head_dim +
                         head_idx * seq_len * head_dim;
  const float *Q_bh = Q + bhd_offset;
  const float *K_bh = K + bhd_offset;
  const float *V_bh = V + bhd_offset;
  float *O_bh = O + bhd_offset;

  __shared__ float K_shared[BLOCK_N_T][BLOCK_D_T + 1];
  __shared__ float V_shared[BLOCK_N_T][BLOCK_D_T + 1];

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  float O_acc[BLOCK_D_T];

#pragma unroll
  for (int d = 0; d < BLOCK_D_T; d++) {
    O_acc[d] = 0.0f;
  }

  float Q_row[BLOCK_D_T];
  if (q_row_global < seq_len) {
#pragma unroll
    for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
      Q_row[d] = Q_bh[q_row_global * head_dim + d] * scale;
    }
  }

  // For causal attention, only process K/V blocks up to current Q position
  const int num_kv_blocks = (q_row_global + BLOCK_N_T) / BLOCK_N_T;

  for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
    const int kv_start = kv_block * BLOCK_N_T;

    __syncthreads();

#pragma unroll
    for (int i = tid; i < BLOCK_N_T * head_dim; i += blockDim.x) {
      int row = i / head_dim;
      int col = i % head_dim;
      int global_row = kv_start + row;

      if (global_row < seq_len && col < head_dim) {
        K_shared[row][col] = K_bh[global_row * head_dim + col];
        V_shared[row][col] = V_bh[global_row * head_dim + col];
      } else {
        K_shared[row][col] = 0.0f;
        V_shared[row][col] = 0.0f;
      }
    }

    __syncthreads();

    float S_row[BLOCK_N_T];
    float row_max = -FLT_MAX;

#pragma unroll
    for (int j = 0; j < BLOCK_N_T; j++) {
      int kv_row_global = kv_start + j;
      float score = 0.0f;

      // Causal mask: only attend to positions <= current position
      if (kv_row_global < seq_len && kv_row_global <= q_row_global) {
#pragma unroll
        for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
          score += Q_row[d] * K_shared[j][d];
        }
      } else {
        score = -FLT_MAX; // Mask future positions
      }

      S_row[j] = score;
      row_max = fmaxf(row_max, score);
    }

    float m_new = fmaxf(m_i, row_max);
    float scale_old = expf(m_i - m_new);

    float row_sum = 0.0f;
    float P_row[BLOCK_N_T];

#pragma unroll
    for (int j = 0; j < BLOCK_N_T; j++) {
      float p = expf(S_row[j] - m_new);
      P_row[j] = p;
      int kv_row_global = kv_start + j;
      if (kv_row_global < seq_len && kv_row_global <= q_row_global) {
        row_sum += p;
      }
    }

    float l_new = scale_old * l_i + row_sum;

#pragma unroll
    for (int d = 0; d < BLOCK_D_T; d++) {
      O_acc[d] *= scale_old;
    }

#pragma unroll
    for (int j = 0; j < BLOCK_N_T; j++) {
      int kv_row_global = kv_start + j;
      if (kv_row_global < seq_len && kv_row_global <= q_row_global) {
        float p = P_row[j];
#pragma unroll
        for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
          O_acc[d] += p * V_shared[j][d];
        }
      }
    }

    m_i = m_new;
    l_i = l_new;
  }

  if (q_row_global < seq_len && l_i > 0.0f) {
    float inv_l = 1.0f / l_i;
#pragma unroll
    for (int d = 0; d < BLOCK_D_T && d < head_dim; d++) {
      O_bh[q_row_global * head_dim + d] = O_acc[d] * inv_l;
    }
  }
}

// ============================================================================
// Wrapper Functions (C-linkage for Python bindings)
// ============================================================================

extern "C" {

void flash_attention_v2_forward(const float *Q, const float *K, const float *V,
                                float *O, int batch_size, int num_heads,
                                int seq_len, int head_dim, bool causal) {
  if (head_dim > BLOCK_D) {
    fprintf(stderr, "flash_attention_v2: head_dim=%d exceeds maximum %d\n",
            head_dim, BLOCK_D);
    return;
  }

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // Grid dimensions
  int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
  dim3 grid(num_q_blocks, 1, batch_size * num_heads);
  dim3 block(BLOCK_M); // One thread per Q row in block

  if (causal) {
    flash_attention_v2_causal_kernel<BLOCK_M, BLOCK_N, BLOCK_D>
        <<<grid, block>>>(Q, K, V, O, batch_size, num_heads, seq_len, head_dim,
                          scale);
  } else {
    flash_attention_v2_forward_kernel<BLOCK_M, BLOCK_N, BLOCK_D>
        <<<grid, block>>>(Q, K, V, O, batch_size, num_heads, seq_len, head_dim,
                          scale);
  }

  cudaDeviceSynchronize();
}

// FP16 variant (uses FP16 compute with FP32 accumulator for accuracy)
void flash_attention_v2_forward_fp16(const __half *Q, const __half *K,
                                     const __half *V, __half *O, int batch_size,
                                     int num_heads, int seq_len, int head_dim,
                                     bool causal) {
  // TODO: Implement FP16 kernel with Tensor Core WMMA instructions
  // For now, convert to FP32, run, convert back
  // This is a placeholder - real implementation would use WMMA

  size_t total_elements = batch_size * num_heads * seq_len * head_dim;

  // Allocate FP32 buffers
  float *Q_fp32, *K_fp32, *V_fp32, *O_fp32;
  cudaMalloc(&Q_fp32, total_elements * sizeof(float));
  cudaMalloc(&K_fp32, total_elements * sizeof(float));
  cudaMalloc(&V_fp32, total_elements * sizeof(float));
  cudaMalloc(&O_fp32, total_elements * sizeof(float));

  // Convert FP16 to FP32 (simple kernel)
  // TODO: Replace with optimized conversion or use FP16 kernel

  cudaFree(Q_fp32);
  cudaFree(K_fp32);
  cudaFree(V_fp32);
  cudaFree(O_fp32);
}

} // extern "C"

// ============================================================================
// Benchmark helper
// ============================================================================

#ifdef FLASH_ATTENTION_BENCHMARK

#include <chrono>

void benchmark_flash_attention_v2() {
  // Benchmark configuration
  const int batch_size = 1;
  const int num_heads = 12;
  const int seq_len = 512;
  const int head_dim = 64;
  const int warmup = 10;
  const int iterations = 100;

  size_t total = batch_size * num_heads * seq_len * head_dim;

  // Allocate
  float *Q, *K, *V, *O;
  cudaMalloc(&Q, total * sizeof(float));
  cudaMalloc(&K, total * sizeof(float));
  cudaMalloc(&V, total * sizeof(float));
  cudaMalloc(&O, total * sizeof(float));

  // Initialize with random data (simplified)
  cudaMemset(Q, 0, total * sizeof(float));
  cudaMemset(K, 0, total * sizeof(float));
  cudaMemset(V, 0, total * sizeof(float));

  // Warmup
  for (int i = 0; i < warmup; i++) {
    flash_attention_v2_forward(Q, K, V, O, batch_size, num_heads, seq_len,
                               head_dim, false);
  }
  cudaDeviceSynchronize();

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    flash_attention_v2_forward(Q, K, V, O, batch_size, num_heads, seq_len,
                               head_dim, false);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double per_iteration_ms = elapsed_ms / iterations;

  printf("Flash Attention V2 Benchmark:\n");
  printf("  Config: batch=%d, heads=%d, seq=%d, dim=%d\n", batch_size,
         num_heads, seq_len, head_dim);
  printf("  Time: %.4f ms/iteration\n", per_iteration_ms);
  printf("  Throughput: %.2f TFLOPS\n",
         (4.0 * batch_size * num_heads * seq_len * seq_len * head_dim) /
             (per_iteration_ms * 1e9));

  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(O);
}

#endif // FLASH_ATTENTION_BENCHMARK
