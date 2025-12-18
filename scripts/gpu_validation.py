#!/usr/bin/env python3
"""
Zenith GPU Validation Script
Jalankan dengan: python gpu_validation.py
"""

import subprocess
import sys
import os


def run_cmd(cmd, check=True):
    """Run command and return output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and not "warning" in result.stderr.lower():
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
    return result


def check_cuda():
    """Check CUDA availability."""
    print("=" * 60)
    print("CHECKING CUDA AVAILABILITY")
    print("=" * 60)

    run_cmd("nvidia-smi", check=False)
    run_cmd("nvcc --version", check=False)


def run_python_tests():
    """Run Python unit tests."""
    print("\n" + "=" * 60)
    print("RUNNING PYTHON UNIT TESTS")
    print("=" * 60)

    result = run_cmd("python -m pytest tests/ -v --tb=short -x", check=False)
    return result.returncode == 0


def write_and_compile_cuda_test():
    """Write and compile CUDA test with proper error checking."""
    print("\n" + "=" * 60)
    print("CUDA KERNEL VALIDATION")
    print("=" * 60)

    cuda_code = """
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        printf("CUDA Error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        return -1; \\
    } \\
} while(0)

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void fused_bias_relu(float* x, const float* bias, int n, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int c = idx % channels;
        float val = x[idx] + bias[c];
        x[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void layer_norm_kernel(
    float* output, const float* input, const float* gamma, const float* beta,
    int hidden_size, float eps
) {
    extern __shared__ float shared[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* row = input + batch_idx * hidden_size;
    float* out_row = output + batch_idx * hidden_size;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += row[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    
    float mean = shared[0] / hidden_size;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = row[i] - mean;
        var_sum += diff * diff;
    }
    shared[tid] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    
    float variance = shared[0] / hidden_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (row[i] - mean) * inv_std;
        out_row[i] = gamma[i] * normalized + beta[i];
    }
}

int main() {
    printf("Zenith CUDA Kernel Validation\\n");
    printf("==============================\\n\\n");
    
    // Test 1: Vector Add
    printf("Test 1: Vector Add\\n");
    {
        int n = 1024;
        float *h_a, *h_b, *h_c;
        float *d_a, *d_b, *d_c;
        
        h_a = (float*)malloc(n * sizeof(float));
        h_b = (float*)malloc(n * sizeof(float));
        h_c = (float*)malloc(n * sizeof(float));
        
        for (int i = 0; i < n; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
            h_c[i] = 0.0f;
        }
        
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
        
        vector_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool pass = true;
        for (int i = 0; i < n; i++) {
            if (fabsf(h_c[i] - 3.0f) > 1e-5f) {
                printf("  Mismatch at %d: expected 3.0, got %.6f\\n", i, h_c[i]);
                pass = false;
                break;
            }
        }
        printf("  Result: %s\\n\\n", pass ? "[PASS]" : "[FAIL]");
        
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
    }
    
    // Test 2: Fused Bias + ReLU
    printf("Test 2: Fused Bias + ReLU\\n");
    {
        int n = 1024;
        int channels = 64;
        float *h_x, *h_bias;
        float *d_x, *d_bias;
        
        h_x = (float*)malloc(n * sizeof(float));
        h_bias = (float*)malloc(channels * sizeof(float));
        
        for (int i = 0; i < n; i++) h_x[i] = -1.0f;
        for (int i = 0; i < channels; i++) h_bias[i] = 2.0f;
        
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, channels * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, channels * sizeof(float), cudaMemcpyHostToDevice));
        
        fused_bias_relu<<<(n + 255) / 256, 256>>>(d_x, d_bias, n, channels);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool pass = true;
        for (int i = 0; i < n; i++) {
            // -1 + 2 = 1, ReLU(1) = 1
            if (fabsf(h_x[i] - 1.0f) > 1e-5f) {
                printf("  Mismatch at %d: expected 1.0, got %.6f\\n", i, h_x[i]);
                pass = false;
                break;
            }
        }
        printf("  Result: %s\\n\\n", pass ? "[PASS]" : "[FAIL]");
        
        cudaFree(d_x); cudaFree(d_bias);
        free(h_x); free(h_bias);
    }
    
    // Test 3: LayerNorm
    printf("Test 3: LayerNorm\\n");
    {
        int batch = 4;
        int hidden = 256;
        int n = batch * hidden;
        
        float *h_input, *h_output, *h_gamma, *h_beta;
        float *d_in, *d_out, *d_gamma, *d_beta;
        
        h_input = (float*)malloc(n * sizeof(float));
        h_output = (float*)malloc(n * sizeof(float));
        h_gamma = (float*)malloc(hidden * sizeof(float));
        h_beta = (float*)malloc(hidden * sizeof(float));
        
        for (int i = 0; i < n; i++) h_input[i] = (float)(i % 10) / 10.0f;
        for (int i = 0; i < hidden; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }
        for (int i = 0; i < n; i++) h_output[i] = 0.0f;
        
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, hidden * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, hidden * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_in, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta, hidden * sizeof(float), cudaMemcpyHostToDevice));
        
        layer_norm_kernel<<<batch, 256, 256 * sizeof(float)>>>(d_out, d_in, d_gamma, d_beta, hidden, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_output, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Check first row: mean should be ~0, variance should be ~1
        float mean = 0, var = 0;
        for (int i = 0; i < hidden; i++) mean += h_output[i];
        mean /= hidden;
        for (int i = 0; i < hidden; i++) var += (h_output[i] - mean) * (h_output[i] - mean);
        var /= hidden;
        
        bool pass = (fabsf(mean) < 0.01f && fabsf(var - 1.0f) < 0.1f);
        printf("  Mean: %.6f (expected ~0)\\n", mean);
        printf("  Variance: %.6f (expected ~1)\\n", var);
        printf("  Result: %s\\n\\n", pass ? "[PASS]" : "[FAIL]");
        
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
        free(h_input); free(h_output); free(h_gamma); free(h_beta);
    }
    
    printf("==============================\\n");
    printf("CUDA Kernel Validation Complete\\n");
    
    return 0;
}
"""

    # Write CUDA file
    with open("test_cuda_fixed.cu", "w") as f:
        f.write(cuda_code)

    print("Compiling CUDA test...")
    result = run_cmd("nvcc -o test_cuda_fixed test_cuda_fixed.cu -O3", check=False)

    if result.returncode == 0:
        print("\nRunning CUDA test...")
        run_cmd("./test_cuda_fixed", check=False)
    else:
        print("Compilation failed!")
        return False

    return True


def main():
    """Main validation function."""
    print("\n" + "=" * 60)
    print("        ZENITH GPU VALIDATION")
    print("=" * 60)

    check_cuda()

    print("\n" + "-" * 60)
    run_python_tests()

    print("\n" + "-" * 60)
    write_and_compile_cuda_test()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
