# Zenith Speedup Roadmap: Achieving 2-7x Performance Gains

**Target**: 2-7x faster inference when complementing PyTorch, JAX, TensorFlow  
**Document Version**: 1.0  
**Date**: December 2024  
**Author**: Wahyu Ardiansyah

---

## Executive Summary

Untuk mencapai speedup 2-7x yang signifikan, Zenith perlu mengimplementasikan kombinasi dari teknik-teknik berikut:

| Technique | Expected Speedup | Implementation Priority |
|-----------|-----------------|------------------------|
| Flash Attention v2/v3 | 2-4x (attention ops) | **Critical** |
| Aggressive Kernel Fusion | 1.5-3x (overall) | **Critical** |
| INT8/FP8 Quantization | 2-4x (inference) | **High** |
| Tensor Cores Utilization | 2-8x (matmul/gemm) | **High** |
| Memory Access Optimization | 1.3-2x (bandwidth) | **Medium** |
| CUDA Graphs | 1.2-1.5x (latency) | **Medium** |
| Speculative Execution | 1.5-2x (LLMs) | **Future** |

---

## Current State Analysis

### What Zenith Already Has (Implemented):
1. **Basic Kernel Fusion** - Conv+BN+ReLU, Add+LayerNorm (core/src/fused_kernels.cu)
2. **cuBLAS Integration** - Attention with batched GEMM (cublas_attention.cu)
3. **Auto-tuner Framework** - Grid/Random/SA/GA search (optimization/autotuner.py)
4. **Mixed Precision Support** - FP16 kernels (fp16_kernels.cu)
5. **QAT Framework** - Quantization-aware training (optimization/qat.py)

### What's Missing for 2-7x Speedup:
1. **Flash Attention Implementation** - Memory-efficient attention
2. **Triton JIT Kernels** - Custom kernel generation
3. **Aggressive Graph-Level Fusion** - Beyond local patterns
4. **Tensor Core Kernels** - Explicit WMMA utilization
5. **INT8/FP8 Kernels** - True quantized execution
6. **CUDA Graphs** - Reduced kernel launch overhead
7. **Paged/Continuous Batching** - LLM serving optimizations

---

## Phase 1: Flash Attention (Target: 2-4x on Attention)

### Rationale
Attention is the bottleneck for Transformers. Standard attention is O(N^2) in memory.
Flash Attention reduces this to O(N) while also being faster due to reduced HBM access.

### Implementation Plan

```cpp
// core/include/zenith/flash_attention_v2.hpp

/// Flash Attention V2 Implementation
/// Based on: "FlashAttention-2: Faster Attention with Better Parallelism"
/// 
/// Key optimizations:
/// 1. Tiling - Process Q, K, V in blocks to fit in SRAM
/// 2. Recomputation - Don't store attention matrix
/// 3. Online Softmax - Compute softmax incrementally
/// 4. Work Partitioning - Different strategy from V1

void flash_attention_v2_forward(
    const __half* Q,  // [batch, heads, seq_len, head_dim]
    const __half* K,
    const __half* V,
    __half* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal = false  // For autoregressive models
);
```

### Kernel Design
```cpp
// Tiling strategy for Flash Attention V2
// Block sizes optimized for NVIDIA Ampere/Hopper

constexpr int Br = 64;   // Block rows for Q
constexpr int Bc = 64;   // Block cols for K/V
constexpr int Bd = 64;   // Head dimension (must match)

// Each thread block processes one (batch, head) pair
// Inner loop: iterate over K/V blocks, accumulate softmax numerator
```

### Expected Results
| Model | Seq Length | Vanilla Attention | Flash Attention V2 | Speedup |
|-------|------------|------------------|-------------------|---------|
| BERT-base | 512 | 8.0 ms | 2.5 ms | 3.2x |
| GPT-2 | 1024 | 35 ms | 10 ms | 3.5x |
| LLaMA-7B | 2048 | 280 ms | 70 ms | 4.0x |

---

## Phase 2: Aggressive Kernel Fusion (Target: 1.5-3x Overall)

### Current Fusion Patterns
```
Conv → BN → ReLU → FusedConvBNReLU
Add → LayerNorm → FusedAddLayerNorm
Linear → Bias → Act → FusedLinearAct
```

### New Fusion Patterns to Implement

```python
# 1. Full Transformer Block Fusion
"QKV_Projection_Fused"
  Before: Linear(Q) + Linear(K) + Linear(V) = 3 kernels
  After:  QKV_Linear = 1 kernel (3x fewer launches)

# 2. Multi-Head Attention Epilogue
"Attention_Epilogue_Fused"  
  Before: Softmax → Dropout → MatMul → Add → LayerNorm
  After:  Single fused kernel

# 3. FFN Block Fusion
"FFN_Fused"
  Before: Linear1 → GELU → Linear2 → Add → LayerNorm
  After:  2 kernels (GEMM cannot be fused with activations on GPU)

# 4. Batch Normalization Folding
  Fold BN weights into preceding Conv/Linear at compile time
```

### Implementation: Graph Pattern Matcher

```python
# zenith/optimization/aggressive_fusion.py

class TransformerBlockFusion(OptimizationPass):
    """
    Fuse entire transformer sub-blocks into optimized kernels.
    
    Patterns:
    1. QKV Projection: 3 Linear → 1 QKV Linear
    2. Attention: Q@K.T, Softmax, @V → FlashAttention  
    3. MLP: Linear-GELU-Linear → FusedMLP
    """
    
    FUSION_PATTERNS = [
        # Pattern: Three parallel Linear ops with same input
        FusionPattern(
            name="qkv_projection",
            match=[
                ("linear_q", "Linear", {"input": "x"}),
                ("linear_k", "Linear", {"input": "x"}),
                ("linear_v", "Linear", {"input": "x"}),
            ],
            fused_op="QKVLinear",
            speedup_estimate=2.5,
        ),
        # Pattern: GeLU sandwiched between lineears
        FusionPattern(
            name="gated_mlp",
            match=[
                ("up_proj", "Linear", {}),
                ("gate_proj", "Linear", {"input": "@same"}),
                ("act", "SiLU", {}),
                ("mult", "Mul", {}),
                ("down_proj", "Linear", {}),
            ],
            fused_op="GatedMLP",
            speedup_estimate=1.8,
        ),
    ]
```

---

## Phase 3: INT8/FP8 Quantization (Target: 2-4x)

### Current State
- QAT framework exists but outputs simulated quantization
- No actual INT8/FP8 kernel execution

### Required Implementation

```cpp
// core/include/zenith/int8_kernels.hpp

/// INT8 GEMM with Tensor Cores (NVIDIA Ampere+)
/// Uses INT8 inputs, INT32 accumulator, scaled to FP16/FP32 output
Status int8_gemm_tensor_core(
    const int8_t* A,        // Quantized input
    const int8_t* B,        // Quantized weight  
    float* C,               // Output (dequantized)
    int M, int N, int K,
    float scale_a,          // Input scale
    float scale_b,          // Weight scale
    float* bias = nullptr   // Optional bias
);

/// FP8 GEMM for NVIDIA Hopper (SM90+)
/// Uses E4M3 or E5M2 formats
Status fp8_gemm_hopper(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    __half* C,
    int M, int N, int K,
    float scale_a,
    float scale_b
);
```

### Quantization Flow

```
       PyTorch FP32 Model
              │
              ▼
    ┌─────────────────────┐
    │ Zenith Calibration  │ ← Run calibration dataset
    │ - Collect activation│   to find optimal scales
    │   statistics        │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Quantize Weights    │ ← Static quantization
    │ - Per-channel scale │
    │ - Symmetric/Asymm   │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Generate INT8 Graph │ ← Replace FP32 ops with
    │ - Insert Q/DQ nodes │   quantized versions
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Execute with INT8   │ ← Actual INT8 kernels
    │ Tensor Core Kernels │   (not simulation)
    └─────────────────────┘
```

### Expected Speedup

| Model | FP32 | INT8 | Speedup | Accuracy Drop |
|-------|------|------|---------|---------------|
| BERT-base | 8.0 ms | 2.5 ms | 3.2x | <0.1% |
| ResNet-50 | 4.2 ms | 1.4 ms | 3.0x | <0.3% |
| GPT-2 | 35 ms | 12 ms | 2.9x | <0.5% |

---

## Phase 4: Tensor Core Utilization (Target: 2-8x GEMM)

### Current State
- cuBLAS is used but without explicit Tensor Core control
- Default may not utilize Tensor Cores optimally

### Implementation: WMMA Kernels

```cpp
// core/src/wmma_kernels.cu
#include <mma.h>

using namespace nvcuda::wmma;

/// WMMA GEMM for FP16 with Tensor Cores
/// Fragment sizes: M=16, N=16, K=16 (Ampere)
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void wmma_gemm_fp16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Declare fragments
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += BLOCK_K) {
        load_matrix_sync(a_frag, A + warpM*16*K + k, K);
        load_matrix_sync(b_frag, B + k*N + warpN*16, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    store_matrix_sync(C + warpM*16*N + warpN*16, c_frag, N, mem_row_major);
}
```

### Tensor Core Computation Comparison

| Operation | Standard FP32 | Tensor Core FP16 | Speedup |
|-----------|--------------|------------------|---------|
| MatMul 4096x4096 | 12.5 ms | 1.56 ms | 8.0x |
| MatMul 1024x1024 | 0.45 ms | 0.12 ms | 3.75x |
| Attention (512 seq) | 8.0 ms | 2.5 ms | 3.2x |

---

## Phase 5: CUDA Graphs (Target: 1.2-1.5x Latency)

### Problem
Each kernel launch has ~5-20μs overhead. For a 12-layer BERT with 50+ kernels,
this adds 250-1000μs of pure overhead.

### Solution: CUDA Graphs

```cpp
// core/include/zenith/cuda_graph_executor.hpp

class CUDAGraphExecutor {
public:
    /// Capture a model's forward pass as a CUDA Graph
    /// Returns an executable graph that can be replayed
    Status capture(
        std::function<void()> forward_fn,
        void* static_inputs,
        size_t input_size
    );
    
    /// Execute the captured graph (ultra-low latency)
    Status replay(void* input_data);
    
private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    void* input_buffer_;
};
```

### Usage Pattern

```python
# First run: Capture the graph
with zenith.cuda_graph_capture() as graph:
    output = model(input_tensor)

# Subsequent runs: Replay (low latency)
for batch in dataloader:
    output = graph.replay(batch)  # ~5μs overhead vs ~500μs
```

---

## Phase 6: Memory Optimization (Target: 1.3-2x Bandwidth)

### Techniques

1. **Memory Pooling** - Reduce cudaMalloc overhead
2. **Tensor Layout Optimization** - NHWC vs NCHW based on operation
3. **In-place Operations** - Reduce memory allocations
4. **Gradient Checkpointing** - Trade compute for memory

### Implementation

```cpp
// core/include/zenith/memory_pool.hpp

class ZenithMemoryPool {
    /// Pre-allocate a large pool, then sub-allocate from it
    /// Eliminates cudaMalloc/Free latency (50-500μs each)
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    
    // Statistics
    size_t bytes_in_use() const;
    size_t peak_usage() const;
    size_t fragmentation() const;
};
```

---

## Phase 7: Triton JIT Integration (Target: Easy Custom Kernels)

### Why Triton?
- Write kernels in Python, compile to CUDA
- Automatic optimization (tiling, vectorization)
- 80-95% of expert CUDA performance

### Integration Plan

```python
# zenith/backends/triton_backend.py

import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Flash Attention in Triton - simpler than raw CUDA"""
    # ... kernel implementation
    
class TritonBackend:
    """Execute Zenith IR using Triton-compiled kernels"""
    
    def compile_graph(self, graph_ir: GraphIR) -> CompiledGraph:
        """Convert Zenith IR to Triton kernels"""
        pass
```

---

## Implementation Priority & Timeline

### Phase 1 (Month 1-2): Foundation for Big Speedup
- [ ] Flash Attention V2 kernel (CUDA)
- [ ] QKV Projection fusion
- [ ] WMMA GEMM kernels

**Expected Result**: 2-3x speedup on Transformers

### Phase 2 (Month 2-3): Quantization Runtime
- [ ] INT8 GEMM with Tensor Cores
- [ ] Calibration pipeline
- [ ] FP8 support for Hopper

**Expected Result**: Additional 1.5-2x from quantization

### Phase 3 (Month 3-4): Advanced Fusion & Graphs
- [ ] CUDA Graph executor
- [ ] Transformer block fusion
- [ ] Memory pool integration

**Expected Result**: Additional 1.2-1.5x from reduced overhead

### Phase 4 (Month 4-5): Triton & Polish
- [ ] Triton backend integration
- [ ] Auto-tuner for new kernels
- [ ] Comprehensive benchmarks

**Expected Result**: Easy extensibility, consistent performance

---

## Benchmark Targets

After full implementation, these are the target benchmarks:

| Model | Framework Only | + Zenith | Target Speedup |
|-------|---------------|----------|----------------|
| BERT-base (batch=1) | 8.0 ms | 1.5 ms | **5.3x** |
| BERT-base (batch=32) | 45 ms | 12 ms | **3.75x** |
| ResNet-50 (batch=1) | 4.2 ms | 1.2 ms | **3.5x** |
| ResNet-50 (batch=32) | 28 ms | 5.6 ms | **5.0x** |
| GPT-2 (1024 tokens) | 35 ms | 8 ms | **4.4x** |
| LLaMA-7B (512 tokens) | 280 ms | 45 ms | **6.2x** |

---

## Technical Prerequisites

### Hardware Requirements for Full Speedup
- **Minimum**: NVIDIA Ampere (RTX 3090, A100) for Tensor Cores
- **Recommended**: NVIDIA Hopper (H100) for FP8 support
- **VRAM**: 16GB+ for large models with optimization

### Software Requirements
- CUDA 12.0+ (for FP8)
- cuDNN 8.9+
- cuBLAS 12.0+
- Python 3.10+
- PyTorch 2.0+ (for torch.compile integration)

---

## Conclusion

Mencapai speedup 2-7x **sangat mungkin** dengan kombinasi:

1. **Flash Attention** → 2-4x on attention (40-60% of Transformer compute)
2. **Tensor Core kernels** → 2-8x on GEMM operations
3. **INT8 Quantization** → 2-4x overall
4. **Kernel Fusion** → 1.5-2x from reduced memory traffic
5. **CUDA Graphs** → 1.2-1.5x from reduced launch overhead

**Total Potential**: 2x × 1.5x × 1.3x = **3.9x average**, up to **7x peak**

The key is that all these optimizations are **multiplicative** when they target different bottlenecks. Zenith's architecture is already well-positioned to implement these - we just need to fill in the high-performance kernels.
