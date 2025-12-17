// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// CUDA Kernel Implementations for Zenith Framework
// This file contains CUDA __global__ kernels for GPU execution.
// Designed for use in Google Colab and other CUDA-enabled environments.

#ifndef ZENITH_CUDA_KERNELS_CU
#define ZENITH_CUDA_KERNELS_CU

#ifdef ZENITH_HAS_CUDA

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace zenith {
namespace cuda_kernels {

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;

// ============================================================================
// Utility Functions
// ============================================================================

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// Element-wise Kernels
// ============================================================================

/// ReLU activation kernel: max(0, x)
__global__ void relu_kernel(float *data, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = fmaxf(0.0f, data[idx]);
  }
}

/// Sigmoid activation kernel: 1 / (1 + exp(-x))
__global__ void sigmoid_kernel(float *data, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = 1.0f / (1.0f + expf(-data[idx]));
  }
}

/// Tanh activation kernel
__global__ void tanh_kernel(float *data, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = tanhf(data[idx]);
  }
}

/// Leaky ReLU kernel
__global__ void leaky_relu_kernel(float *data, size_t size, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = data[idx];
    data[idx] = x > 0.0f ? x : alpha * x;
  }
}

/// GELU activation kernel (Gaussian Error Linear Unit)
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 *
/// x³)))
__global__ void gelu_kernel(const float *input, float *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = input[idx];
    // Constants for GELU approximation
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
    const float coef = 0.044715f;

    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coef * x_cubed);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

/// LayerNorm kernel - normalizes last dimension
/// input: [batch, hidden], output: [batch, hidden]
/// gamma, beta: [hidden]
__global__ void layernorm_kernel(const float *input, float *output,
                                 const float *gamma, const float *beta,
                                 int batch, int hidden, float eps) {
  int batch_idx = blockIdx.x;

  if (batch_idx < batch) {
    const float *in_row = input + batch_idx * hidden;
    float *out_row = output + batch_idx * hidden;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < hidden; ++i) {
      mean += in_row[i];
    }
    mean /= static_cast<float>(hidden);

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < hidden; ++i) {
      float diff = in_row[i] - mean;
      var += diff * diff;
    }
    var /= static_cast<float>(hidden);

    // Normalize and scale
    float inv_std = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      float normalized = (in_row[i] - mean) * inv_std;
      out_row[i] = normalized * gamma[i] + beta[i];
    }
  }
}

/// Softmax kernel for 2D input [batch, seq_len]
__global__ void softmax_kernel(const float *input, float *output, int batch,
                               int len) {
  int batch_idx = blockIdx.x;

  if (batch_idx < batch) {
    const float *in_row = input + batch_idx * len;
    float *out_row = output + batch_idx * len;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < len; ++i) {
      max_val = fmaxf(max_val, in_row[i]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
      out_row[i] = expf(in_row[i] - max_val);
      sum += out_row[i];
    }

    // Normalize
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      out_row[i] /= sum;
    }
  }
}

/// Element-wise addition kernel: C = A + B
__global__ void add_kernel(const float *A, const float *B, float *C,
                           size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

/// Element-wise subtraction kernel: C = A - B
__global__ void sub_kernel(const float *A, const float *B, float *C,
                           size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] - B[idx];
  }
}

/// Element-wise multiplication kernel: C = A * B
__global__ void mul_kernel(const float *A, const float *B, float *C,
                           size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] * B[idx];
  }
}

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

/// Naive matrix multiplication kernel (for small matrices)
/// C[M,N] = A[M,K] * B[K,N]
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

/// Tiled matrix multiplication using shared memory
/// C[M,N] = A[M,K] * B[K,N]
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  // Loop over tiles
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load A tile
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load B tile
    if (t * TILE_SIZE + threadIdx.y < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial product
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// ============================================================================
// Convolution Kernels
// ============================================================================

/// 2D Convolution kernel (NCHW format)
/// input: [N, C_in, H, W], weight: [C_out, C_in, K_h, K_w]
__global__ void conv2d_kernel(const float *input, const float *weight,
                              const float *bias, float *output, int N, int C_in,
                              int H, int W, int C_out, int K_h, int K_w,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int H_out, int W_out) {
  int n = blockIdx.z / C_out;
  int c = blockIdx.z % C_out;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < N && c < C_out && h < H_out && w < W_out) {
    float sum = 0.0f;

    for (int ci = 0; ci < C_in; ++ci) {
      for (int kh = 0; kh < K_h; ++kh) {
        for (int kw = 0; kw < K_w; ++kw) {
          int ih = h * stride_h - pad_h + kh;
          int iw = w * stride_w - pad_w + kw;

          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            int in_idx = ((n * C_in + ci) * H + ih) * W + iw;
            int w_idx = ((c * C_in + ci) * K_h + kh) * K_w + kw;
            sum += input[in_idx] * weight[w_idx];
          }
        }
      }
    }

    if (bias != nullptr) {
      sum += bias[c];
    }

    int out_idx = ((n * C_out + c) * H_out + h) * W_out + w;
    output[out_idx] = sum;
  }
}

// ============================================================================
// Pooling Kernels
// ============================================================================

/// Max pooling 2D kernel
__global__ void maxpool2d_kernel(const float *input, float *output, int N,
                                 int C, int H, int W, int K_h, int K_w,
                                 int stride_h, int stride_w, int H_out,
                                 int W_out) {
  int n = blockIdx.z / C;
  int c = blockIdx.z % C;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < N && c < C && h < H_out && w < W_out) {
    float max_val = -INFINITY;

    for (int kh = 0; kh < K_h; ++kh) {
      for (int kw = 0; kw < K_w; ++kw) {
        int ih = h * stride_h + kh;
        int iw = w * stride_w + kw;

        if (ih < H && iw < W) {
          int idx = ((n * C + c) * H + ih) * W + iw;
          max_val = fmaxf(max_val, input[idx]);
        }
      }
    }

    int out_idx = ((n * C + c) * H_out + h) * W_out + w;
    output[out_idx] = max_val;
  }
}

/// Average pooling 2D kernel
__global__ void avgpool2d_kernel(const float *input, float *output, int N,
                                 int C, int H, int W, int K_h, int K_w,
                                 int stride_h, int stride_w, int H_out,
                                 int W_out) {
  int n = blockIdx.z / C;
  int c = blockIdx.z % C;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < N && c < C && h < H_out && w < W_out) {
    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < K_h; ++kh) {
      for (int kw = 0; kw < K_w; ++kw) {
        int ih = h * stride_h + kh;
        int iw = w * stride_w + kw;

        if (ih < H && iw < W) {
          int idx = ((n * C + c) * H + ih) * W + iw;
          sum += input[idx];
          ++count;
        }
      }
    }

    int out_idx = ((n * C + c) * H_out + h) * W_out + w;
    output[out_idx] = sum / fmaxf(1.0f, static_cast<float>(count));
  }
}

// ============================================================================
// Reduction Kernels
// ============================================================================

/// Sum reduction kernel using shared memory
__global__ void sum_kernel(const float *input, float *output, size_t size) {
  __shared__ float sdata[BLOCK_SIZE];

  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data to shared memory
  sdata[tid] = (idx < size) ? input[idx] : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

/// Max reduction kernel
__global__ void max_kernel(const float *input, float *output, size_t size) {
  __shared__ float sdata[BLOCK_SIZE];

  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? input[idx] : -INFINITY;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    // Use atomicMax substitute for float (not directly supported)
    // This is simplified - real implementation uses atomicCAS
    output[blockIdx.x] = sdata[0];
  }
}

// ============================================================================
// Wrapper Functions (exported, no inline)
// ============================================================================

void relu_f32(float *data, size_t size) {
  int blocks = div_ceil(size, BLOCK_SIZE);
  relu_kernel<<<blocks, BLOCK_SIZE>>>(data, size);
}

void sigmoid_f32(float *data, size_t size) {
  int blocks = div_ceil(size, BLOCK_SIZE);
  sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(data, size);
}

void tanh_f32(float *data, size_t size) {
  int blocks = div_ceil(size, BLOCK_SIZE);
  tanh_kernel<<<blocks, BLOCK_SIZE>>>(data, size);
}

void add_f32(const float *A, const float *B, float *C, size_t size) {
  int blocks = div_ceil(size, BLOCK_SIZE);
  add_kernel<<<blocks, BLOCK_SIZE>>>(A, B, C, size);
}

void matmul_f32(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid(div_ceil(N, TILE_SIZE), div_ceil(M, TILE_SIZE));
  matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

void conv2d_f32(const float *input, const float *weight, const float *bias,
                float *output, int N, int C_in, int H, int W, int C_out,
                int K_h, int K_w, int stride_h, int stride_w, int pad_h,
                int pad_w) {
  int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

  dim3 block(16, 16);
  dim3 grid(div_ceil(W_out, 16), div_ceil(H_out, 16), N * C_out);
  conv2d_kernel<<<grid, block>>>(input, weight, bias, output, N, C_in, H, W,
                                 C_out, K_h, K_w, stride_h, stride_w, pad_h,
                                 pad_w, H_out, W_out);
}

void maxpool2d_f32(const float *input, float *output, int N, int C, int H,
                   int W, int K_h, int K_w, int stride_h, int stride_w) {
  int H_out = (H - K_h) / stride_h + 1;
  int W_out = (W - K_w) / stride_w + 1;

  dim3 block(16, 16);
  dim3 grid(div_ceil(W_out, 16), div_ceil(H_out, 16), N * C);
  maxpool2d_kernel<<<grid, block>>>(input, output, N, C, H, W, K_h, K_w,
                                    stride_h, stride_w, H_out, W_out);
}

// ============================================================================
// Transformer/BERT Kernel Wrappers
// ============================================================================

void gelu_f32(const float *input, float *output, size_t size) {
  int blocks = div_ceil(size, BLOCK_SIZE);
  gelu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
}

void layernorm_f32(const float *input, float *output, const float *gamma,
                   const float *beta, int batch, int hidden, float eps) {
  // One block per batch element
  layernorm_kernel<<<batch, BLOCK_SIZE>>>(input, output, gamma, beta, batch,
                                          hidden, eps);
}

void softmax_2d_f32(const float *input, float *output, int batch, int len) {
  softmax_kernel<<<batch, BLOCK_SIZE>>>(input, output, batch, len);
}

} // namespace cuda_kernels
} // namespace zenith

#endif // ZENITH_HAS_CUDA

#endif // ZENITH_CUDA_KERNELS_CU
