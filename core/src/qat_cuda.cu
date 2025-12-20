// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// CUDA Kernels for Quantization-Aware Training (QAT)
// Berdasarkan CetakBiru.md Fase 3: Pipeline Kuantisasi INT8
// Referensi: PyTorch FakeQuantize CUDA, TensorRT Q/DQ, Jacob et al. 2018

#ifndef ZENITH_QAT_CUDA_CU
#define ZENITH_QAT_CUDA_CU

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Utility for ceiling division
__host__ __device__ inline int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// Fake Quantization Forward Kernels
// ============================================================================

/// Fake quantize forward kernel (per-tensor)
/// Implements: x_dq = dequant(quant(x, scale, zp), scale, zp)
/// Reference: PyTorch torch/ao/quantization/fake_quantize.py
__global__ void fake_quantize_forward_kernel(const float *__restrict__ input,
                                             float *__restrict__ output,
                                             int size, float scale,
                                             int32_t zero_point, int64_t qmin,
                                             int64_t qmax) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // Quantize: q = round(x / scale) + zero_point
    float scaled = input[idx] / scale + static_cast<float>(zero_point);
    int64_t quantized = llrintf(scaled);

    // Clamp to quantization range
    quantized = max(qmin, min(qmax, quantized));

    // Dequantize: x_dq = (q - zero_point) * scale
    output[idx] =
        (static_cast<float>(quantized) - static_cast<float>(zero_point)) *
        scale;
  }
}

/// Fake quantize backward kernel with Straight-Through Estimator (STE)
/// Implements: grad_input = grad_output * mask
/// where mask = 1 if x in [qmin, qmax] range, 0 otherwise
/// Reference: Bengio et al. "Estimating Gradient for Stochastic Neurons"
__global__ void fake_quantize_backward_ste_kernel(
    const float *__restrict__ input, const float *__restrict__ grad_output,
    float *__restrict__ grad_input, int size, float scale, int32_t zero_point,
    int64_t qmin, int64_t qmax) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // Compute raw quantized value (before clipping)
    float scaled = input[idx] / scale + static_cast<float>(zero_point);
    int64_t quantized_raw = llrintf(scaled);

    // STE mask: 1 if within range, 0 otherwise
    float mask = (quantized_raw >= qmin && quantized_raw <= qmax) ? 1.0f : 0.0f;

    // Pass gradient through with mask
    grad_input[idx] = grad_output[idx] * mask;
  }
}

/// Per-channel fake quantize forward kernel
/// Used for weight quantization where each output channel has separate params
/// Reference: PyTorch per_channel_affine quantization
__global__ void fake_quantize_per_channel_forward_kernel(
    const float *__restrict__ input, float *__restrict__ output,
    const float *__restrict__ scales, const int32_t *__restrict__ zero_points,
    int num_channels, int channel_size, int64_t qmin, int64_t qmax) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = num_channels * channel_size;

  if (idx < total_size) {
    // Determine which channel this element belongs to
    int channel = idx / channel_size;

    float scale = scales[channel];
    int32_t zp = zero_points[channel];

    // Quantize
    float scaled = input[idx] / scale + static_cast<float>(zp);
    int64_t quantized = llrintf(scaled);
    quantized = max(qmin, min(qmax, quantized));

    // Dequantize
    output[idx] =
        (static_cast<float>(quantized) - static_cast<float>(zp)) * scale;
  }
}

/// Per-channel fake quantize backward with STE
__global__ void fake_quantize_per_channel_backward_ste_kernel(
    const float *__restrict__ input, const float *__restrict__ grad_output,
    float *__restrict__ grad_input, const float *__restrict__ scales,
    const int32_t *__restrict__ zero_points, int num_channels, int channel_size,
    int64_t qmin, int64_t qmax) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = num_channels * channel_size;

  if (idx < total_size) {
    int channel = idx / channel_size;

    float scale = scales[channel];
    int32_t zp = zero_points[channel];

    float scaled = input[idx] / scale + static_cast<float>(zp);
    int64_t quantized_raw = llrintf(scaled);

    float mask = (quantized_raw >= qmin && quantized_raw <= qmax) ? 1.0f : 0.0f;
    grad_input[idx] = grad_output[idx] * mask;
  }
}

// ============================================================================
// Observer Kernels (Min/Max Reduction)
// ============================================================================

/// Parallel min/max reduction kernel using shared memory
/// Computes global min and max of input array
__global__ void minmax_reduce_kernel(const float *__restrict__ input,
                                     float *__restrict__ min_out,
                                     float *__restrict__ max_out, int size) {

  __shared__ float s_min[BLOCK_SIZE];
  __shared__ float s_max[BLOCK_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize with extreme values
  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  // Grid-stride loop for large arrays
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    float val = input[i];
    local_min = fminf(local_min, val);
    local_max = fmaxf(local_max, val);
  }

  s_min[tid] = local_min;
  s_max[tid] = local_max;
  __syncthreads();

  // Block-level reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
    }
    __syncthreads();
  }

  // Write block result
  if (tid == 0) {
    atomicMin(reinterpret_cast<int *>(min_out), __float_as_int(s_min[0]));
    atomicMax(reinterpret_cast<int *>(max_out), __float_as_int(s_max[0]));
  }
}

/// Per-channel min/max reduction
__global__ void minmax_per_channel_reduce_kernel(
    const float *__restrict__ input, float *__restrict__ min_out,
    float *__restrict__ max_out, int num_channels, int channel_size) {

  int channel = blockIdx.x;
  if (channel >= num_channels)
    return;

  __shared__ float s_min[BLOCK_SIZE];
  __shared__ float s_max[BLOCK_SIZE];

  int tid = threadIdx.x;
  const float *channel_data = input + channel * channel_size;

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  for (int i = tid; i < channel_size; i += blockDim.x) {
    float val = channel_data[i];
    local_min = fminf(local_min, val);
    local_max = fmaxf(local_max, val);
  }

  s_min[tid] = local_min;
  s_max[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    min_out[channel] = s_min[0];
    max_out[channel] = s_max[0];
  }
}

// ============================================================================
// Batch Normalization Folding Kernel
// ============================================================================

/// Fold batch normalization into convolution/linear weights
/// W_folded = gamma * W / sqrt(var + eps)
/// b_folded = gamma * (b - mean) / sqrt(var + eps) + beta
__global__ void bn_fold_kernel(const float *__restrict__ weight,
                               const float *__restrict__ bias, // Can be nullptr
                               const float *__restrict__ bn_mean,
                               const float *__restrict__ bn_var,
                               const float *__restrict__ bn_gamma,
                               const float *__restrict__ bn_beta,
                               float *__restrict__ weight_out,
                               float *__restrict__ bias_out, int out_channels,
                               int weight_per_channel, float epsilon) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_weights = out_channels * weight_per_channel;

  if (idx < total_weights) {
    int channel = idx / weight_per_channel;

    float std_val = sqrtf(bn_var[channel] + epsilon);
    float scale = bn_gamma[channel] / std_val;

    // Fold weight
    weight_out[idx] = weight[idx] * scale;

    // Only thread 0 of each channel writes bias
    int local_idx = idx % weight_per_channel;
    if (local_idx == 0) {
      float orig_bias = (bias != nullptr) ? bias[channel] : 0.0f;
      bias_out[channel] =
          bn_gamma[channel] * (orig_bias - bn_mean[channel]) / std_val +
          bn_beta[channel];
    }
  }
}

// ============================================================================
// Quantization Parameter Computation Kernels
// ============================================================================

/// Compute symmetric quantization parameters from min/max
/// scale = abs_max / qmax, zero_point = 0
__global__ void compute_symmetric_qparams_kernel(
    const float *__restrict__ min_vals, const float *__restrict__ max_vals,
    float *__restrict__ scales, int32_t *__restrict__ zero_points,
    int num_channels, int64_t qmax) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_channels) {
    float abs_max = fmaxf(fabsf(min_vals[idx]), fabsf(max_vals[idx]));
    scales[idx] = fmaxf(abs_max / static_cast<float>(qmax), 1e-8f);
    zero_points[idx] = 0;
  }
}

/// Compute asymmetric quantization parameters from min/max
/// scale = (max - min) / (qmax - qmin), zero_point = round(qmin - min/scale)
__global__ void compute_asymmetric_qparams_kernel(
    const float *__restrict__ min_vals, const float *__restrict__ max_vals,
    float *__restrict__ scales, int32_t *__restrict__ zero_points,
    int num_channels, int64_t qmin, int64_t qmax) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_channels) {
    float range = max_vals[idx] - min_vals[idx];
    float scale = fmaxf(range / static_cast<float>(qmax - qmin), 1e-8f);
    int32_t zp = static_cast<int32_t>(
        roundf(static_cast<float>(qmin) - min_vals[idx] / scale));
    zp = max(static_cast<int32_t>(qmin), min(static_cast<int32_t>(qmax), zp));

    scales[idx] = scale;
    zero_points[idx] = zp;
  }
}

// ============================================================================
// Wrapper Functions (C++ callable)
// ============================================================================

extern "C" {

/// Fake quantize forward (per-tensor)
void fake_quantize_forward_cuda(const float *input, float *output, int size,
                                float scale, int32_t zero_point, int64_t qmin,
                                int64_t qmax, cudaStream_t stream) {

  int blocks = div_ceil(size, BLOCK_SIZE);
  fake_quantize_forward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      input, output, size, scale, zero_point, qmin, qmax);
}

/// Fake quantize backward with STE (per-tensor)
void fake_quantize_backward_ste_cuda(const float *input,
                                     const float *grad_output,
                                     float *grad_input, int size, float scale,
                                     int32_t zero_point, int64_t qmin,
                                     int64_t qmax, cudaStream_t stream) {

  int blocks = div_ceil(size, BLOCK_SIZE);
  fake_quantize_backward_ste_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      input, grad_output, grad_input, size, scale, zero_point, qmin, qmax);
}

/// Fake quantize forward (per-channel)
void fake_quantize_per_channel_forward_cuda(const float *input, float *output,
                                            const float *scales,
                                            const int32_t *zero_points,
                                            int num_channels, int channel_size,
                                            int64_t qmin, int64_t qmax,
                                            cudaStream_t stream) {

  int total_size = num_channels * channel_size;
  int blocks = div_ceil(total_size, BLOCK_SIZE);
  fake_quantize_per_channel_forward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      input, output, scales, zero_points, num_channels, channel_size, qmin,
      qmax);
}

/// Fake quantize backward with STE (per-channel)
void fake_quantize_per_channel_backward_ste_cuda(
    const float *input, const float *grad_output, float *grad_input,
    const float *scales, const int32_t *zero_points, int num_channels,
    int channel_size, int64_t qmin, int64_t qmax, cudaStream_t stream) {

  int total_size = num_channels * channel_size;
  int blocks = div_ceil(total_size, BLOCK_SIZE);
  fake_quantize_per_channel_backward_ste_kernel<<<blocks, BLOCK_SIZE, 0,
                                                  stream>>>(
      input, grad_output, grad_input, scales, zero_points, num_channels,
      channel_size, qmin, qmax);
}

/// Min/max observation (per-tensor)
void observe_minmax_cuda(const float *input, float *min_out, float *max_out,
                         int size, cudaStream_t stream) {

  // Initialize output with extreme values
  float init_min = FLT_MAX;
  float init_max = -FLT_MAX;
  cudaMemcpyAsync(min_out, &init_min, sizeof(float), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(max_out, &init_max, sizeof(float), cudaMemcpyHostToDevice,
                  stream);

  int blocks = min(div_ceil(size, BLOCK_SIZE), 1024);
  minmax_reduce_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, min_out,
                                                          max_out, size);
}

/// Min/max observation (per-channel)
void observe_minmax_per_channel_cuda(const float *input, float *min_out,
                                     float *max_out, int num_channels,
                                     int channel_size, cudaStream_t stream) {

  minmax_per_channel_reduce_kernel<<<num_channels, BLOCK_SIZE, 0, stream>>>(
      input, min_out, max_out, num_channels, channel_size);
}

/// Batch normalization folding
void bn_fold_cuda(const float *weight, const float *bias, const float *bn_mean,
                  const float *bn_var, const float *bn_gamma,
                  const float *bn_beta, float *weight_out, float *bias_out,
                  int out_channels, int weight_per_channel, float epsilon,
                  cudaStream_t stream) {

  int total_weights = out_channels * weight_per_channel;
  int blocks = div_ceil(total_weights, BLOCK_SIZE);
  bn_fold_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      weight, bias, bn_mean, bn_var, bn_gamma, bn_beta, weight_out, bias_out,
      out_channels, weight_per_channel, epsilon);
}

/// Compute symmetric quantization parameters
void compute_symmetric_qparams_cuda(const float *min_vals,
                                    const float *max_vals, float *scales,
                                    int32_t *zero_points, int num_channels,
                                    int64_t qmax, cudaStream_t stream) {

  int blocks = div_ceil(num_channels, BLOCK_SIZE);
  compute_symmetric_qparams_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      min_vals, max_vals, scales, zero_points, num_channels, qmax);
}

/// Compute asymmetric quantization parameters
void compute_asymmetric_qparams_cuda(const float *min_vals,
                                     const float *max_vals, float *scales,
                                     int32_t *zero_points, int num_channels,
                                     int64_t qmin, int64_t qmax,
                                     cudaStream_t stream) {

  int blocks = div_ceil(num_channels, BLOCK_SIZE);
  compute_asymmetric_qparams_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      min_vals, max_vals, scales, zero_points, num_channels, qmin, qmax);
}

} // extern "C"

#endif // ZENITH_QAT_CUDA_CU
