// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Fused CUDA Kernels untuk Kernel Fusion Optimization
// Berdasarkan CetakBiru.md: Mathematical Kernel Library (Zenith-MKL)
// Referensi: NVIDIA CUTLASS, cuDNN Graph API, APEX FusedLayerNorm

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants and Utilities
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr float GELU_COEF = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ __forceinline__ int div_ceil_device(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// Fused Bias + Activation Kernels
// Menggabungkan bias addition dengan activation untuk mengurangi memory traffic
// ============================================================================

/// Fused Bias + ReLU: Y = max(0, X + bias)
__global__ void fused_bias_relu_kernel(const float *__restrict__ input,
                                       const float *__restrict__ bias,
                                       float *__restrict__ output, int batch,
                                       int features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * features;

  if (idx < total) {
    int feature_idx = idx % features;
    float val = input[idx] + bias[feature_idx];
    output[idx] = fmaxf(0.0f, val);
  }
}

/// Fused Bias + GELU: Y = GELU(X + bias)
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
__global__ void fused_bias_gelu_kernel(const float *__restrict__ input,
                                       const float *__restrict__ bias,
                                       float *__restrict__ output, int batch,
                                       int features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * features;

  if (idx < total) {
    int feature_idx = idx % features;
    float x = input[idx] + bias[feature_idx];

    float x_cubed = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

/// Fused Bias + Sigmoid: Y = 1 / (1 + exp(-(X + bias)))
__global__ void fused_bias_sigmoid_kernel(const float *__restrict__ input,
                                          const float *__restrict__ bias,
                                          float *__restrict__ output, int batch,
                                          int features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * features;

  if (idx < total) {
    int feature_idx = idx % features;
    float val = input[idx] + bias[feature_idx];
    output[idx] = 1.0f / (1.0f + expf(-val));
  }
}

/// Fused Bias + Tanh: Y = tanh(X + bias)
__global__ void fused_bias_tanh_kernel(const float *__restrict__ input,
                                       const float *__restrict__ bias,
                                       float *__restrict__ output, int batch,
                                       int features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * features;

  if (idx < total) {
    int feature_idx = idx % features;
    float val = input[idx] + bias[feature_idx];
    output[idx] = tanhf(val);
  }
}

/// Fused Bias + SiLU (Swish): Y = (X + bias) * sigmoid(X + bias)
__global__ void fused_bias_silu_kernel(const float *__restrict__ input,
                                       const float *__restrict__ bias,
                                       float *__restrict__ output, int batch,
                                       int features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * features;

  if (idx < total) {
    int feature_idx = idx % features;
    float val = input[idx] + bias[feature_idx];
    float sig = 1.0f / (1.0f + expf(-val));
    output[idx] = val * sig;
  }
}

// ============================================================================
// Fused Residual + Activation Kernels
// Y = Activation(X + residual)
// ============================================================================

/// Fused Add + ReLU: Y = max(0, X + residual)
__global__ void fused_add_relu_kernel(const float *__restrict__ x,
                                      const float *__restrict__ residual,
                                      float *__restrict__ output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, x[idx] + residual[idx]);
  }
}

/// Fused Add + GELU: Y = GELU(X + residual)
__global__ void fused_add_gelu_kernel(const float *__restrict__ x,
                                      const float *__restrict__ residual,
                                      float *__restrict__ output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = x[idx] + residual[idx];
    float x_cubed = val * val * val;
    float inner = SQRT_2_OVER_PI * (val + GELU_COEF * x_cubed);
    output[idx] = 0.5f * val * (1.0f + tanhf(inner));
  }
}

// ============================================================================
// Fused Residual + LayerNorm Kernels
// Berdasarkan NVIDIA APEX FusedLayerNorm
// ============================================================================

/// Fused Add + LayerNorm: Y = LayerNorm(X + residual)
/// Menggunakan shared memory untuk mean dan variance computation
__global__ void fused_add_layernorm_kernel(const float *__restrict__ x,
                                           const float *__restrict__ residual,
                                           const float *__restrict__ gamma,
                                           const float *__restrict__ beta,
                                           float *__restrict__ output,
                                           int batch, int hidden, float eps) {
  // Shared memory untuk warp-level reduction
  __shared__ float s_mean;
  __shared__ float s_var;

  int batch_idx = blockIdx.x;
  if (batch_idx >= batch)
    return;

  const float *x_row = x + batch_idx * hidden;
  const float *res_row = residual + batch_idx * hidden;
  float *out_row = output + batch_idx * hidden;

  // Step 1: Compute sum and sum of squares
  float local_sum = 0.0f;
  float local_sum_sq = 0.0f;

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float val = x_row[i] + res_row[i];
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }

  // First thread of each warp contributes to shared memory
  __shared__ float warp_sums[32];
  __shared__ float warp_sum_sqs[32];

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  if (lane_id == 0) {
    warp_sums[warp_id] = local_sum;
    warp_sum_sqs[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    local_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    local_sum_sq = (lane_id < num_warps) ? warp_sum_sqs[lane_id] : 0.0f;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (lane_id == 0) {
      float mean = local_sum / static_cast<float>(hidden);
      float var = (local_sum_sq / static_cast<float>(hidden)) - (mean * mean);
      s_mean = mean;
      s_var = var;
    }
  }
  __syncthreads();

  // Step 2: Normalize and apply gamma/beta
  float mean = s_mean;
  float inv_std = rsqrtf(s_var + eps);

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float val = x_row[i] + res_row[i];
    float normalized = (val - mean) * inv_std;
    out_row[i] = normalized * gamma[i] + beta[i];
  }
}

/// Fused LayerNorm + Add: Y = LayerNorm(X) + residual
__global__ void fused_layernorm_add_kernel(const float *__restrict__ x,
                                           const float *__restrict__ residual,
                                           const float *__restrict__ gamma,
                                           const float *__restrict__ beta,
                                           float *__restrict__ output,
                                           int batch, int hidden, float eps) {
  __shared__ float s_mean;
  __shared__ float s_var;

  int batch_idx = blockIdx.x;
  if (batch_idx >= batch)
    return;

  const float *x_row = x + batch_idx * hidden;
  const float *res_row = residual + batch_idx * hidden;
  float *out_row = output + batch_idx * hidden;

  // Compute mean and variance of x only
  float local_sum = 0.0f;
  float local_sum_sq = 0.0f;

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float val = x_row[i];
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Warp reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }

  __shared__ float warp_sums[32];
  __shared__ float warp_sum_sqs[32];

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  if (lane_id == 0) {
    warp_sums[warp_id] = local_sum;
    warp_sum_sqs[warp_id] = local_sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    local_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    local_sum_sq = (lane_id < num_warps) ? warp_sum_sqs[lane_id] : 0.0f;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (lane_id == 0) {
      float mean = local_sum / static_cast<float>(hidden);
      float var = (local_sum_sq / static_cast<float>(hidden)) - (mean * mean);
      s_mean = mean;
      s_var = var;
    }
  }
  __syncthreads();

  float mean = s_mean;
  float inv_std = rsqrtf(s_var + eps);

  // Apply LayerNorm then add residual
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float normalized = (x_row[i] - mean) * inv_std;
    out_row[i] = normalized * gamma[i] + beta[i] + res_row[i];
  }
}

// ============================================================================
// Fused Linear + Bias + Activation (for GEMM epilogue fusion)
// Catatan: GEMM sendiri menggunakan cuBLAS, ini hanya untuk epilogue
// ============================================================================

/// Fused Bias + ReLU epilogue untuk setelah GEMM
__global__ void fused_gemm_epilogue_relu_kernel(float *__restrict__ C,
                                                const float *__restrict__ bias,
                                                int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;

  if (idx < total) {
    int col = idx % N;
    C[idx] = fmaxf(0.0f, C[idx] + bias[col]);
  }
}

/// Fused Bias + GELU epilogue untuk setelah GEMM
__global__ void fused_gemm_epilogue_gelu_kernel(float *__restrict__ C,
                                                const float *__restrict__ bias,
                                                int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;

  if (idx < total) {
    int col = idx % N;
    float x = C[idx] + bias[col];
    float x_cubed = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
    C[idx] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

// ============================================================================
// Wrapper Functions (C-linkage for easy binding)
// ============================================================================

extern "C" {

// Fused Bias + Activation
void fused_bias_relu(const float *input, const float *bias, float *output,
                     int batch, int features) {
  int total = batch * features;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_bias_relu_kernel<<<blocks, BLOCK_SIZE>>>(input, bias, output, batch,
                                                 features);
}

void fused_bias_gelu(const float *input, const float *bias, float *output,
                     int batch, int features) {
  int total = batch * features;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_bias_gelu_kernel<<<blocks, BLOCK_SIZE>>>(input, bias, output, batch,
                                                 features);
}

void fused_bias_sigmoid(const float *input, const float *bias, float *output,
                        int batch, int features) {
  int total = batch * features;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_bias_sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(input, bias, output, batch,
                                                    features);
}

void fused_bias_tanh(const float *input, const float *bias, float *output,
                     int batch, int features) {
  int total = batch * features;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_bias_tanh_kernel<<<blocks, BLOCK_SIZE>>>(input, bias, output, batch,
                                                 features);
}

void fused_bias_silu(const float *input, const float *bias, float *output,
                     int batch, int features) {
  int total = batch * features;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_bias_silu_kernel<<<blocks, BLOCK_SIZE>>>(input, bias, output, batch,
                                                 features);
}

// Fused Add + Activation
void fused_add_relu(const float *x, const float *residual, float *output,
                    size_t size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_add_relu_kernel<<<blocks, BLOCK_SIZE>>>(x, residual, output, size);
}

void fused_add_gelu(const float *x, const float *residual, float *output,
                    size_t size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_add_gelu_kernel<<<blocks, BLOCK_SIZE>>>(x, residual, output, size);
}

// Fused LayerNorm + Residual
void fused_add_layernorm(const float *x, const float *residual,
                         const float *gamma, const float *beta, float *output,
                         int batch, int hidden, float eps) {
  int block_size = (hidden < BLOCK_SIZE) ? hidden : BLOCK_SIZE;
  fused_add_layernorm_kernel<<<batch, block_size>>>(x, residual, gamma, beta,
                                                    output, batch, hidden, eps);
}

void fused_layernorm_add(const float *x, const float *residual,
                         const float *gamma, const float *beta, float *output,
                         int batch, int hidden, float eps) {
  int block_size = (hidden < BLOCK_SIZE) ? hidden : BLOCK_SIZE;
  fused_layernorm_add_kernel<<<batch, block_size>>>(x, residual, gamma, beta,
                                                    output, batch, hidden, eps);
}

// GEMM Epilogue Fusion
void fused_gemm_epilogue_relu(float *C, const float *bias, int M, int N) {
  int total = M * N;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_gemm_epilogue_relu_kernel<<<blocks, BLOCK_SIZE>>>(C, bias, M, N);
}

void fused_gemm_epilogue_gelu(float *C, const float *bias, int M, int N) {
  int total = M * N;
  int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fused_gemm_epilogue_gelu_kernel<<<blocks, BLOCK_SIZE>>>(C, bias, M, N);
}

} // extern "C"
