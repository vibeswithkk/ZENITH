# BLUEPRINT: Zenith 3X - Memperbaiki 7 Masalah Kritis

**Versi:** 1.0.0  
**Tanggal:** 21 Desember 2024  
**Status:** AKTIF  
**Tujuan:** Memperbaiki 7 masalah kritis untuk membuat Zenith 3X lebih powerful

---

## Executive Summary

Berdasarkan audit mendalam terhadap codebase Zenith, telah diidentifikasi **7 masalah kritis** yang mencegah Zenith dari mencapai potensi penuhnya. Blueprint ini menyediakan solusi berbasis riset dari sumber-sumber valid (TensorRT, ONNX Runtime, TVM, PyTorch 2.0, MLPerf, OpenTelemetry) yang selaras dengan arsitektur existing dan prinsip matematika/teori yang sudah ada di CetakBiru.md.

---

## Bagian 1: 7 Masalah Kritis yang Diidentifikasi

| # | Masalah | Dampak | Priority |
|---|---------|--------|----------|
| 1 | Zenith Runtime tidak ada | Kernels tidak dipanggil | CRITICAL |
| 2 | GraphIR → Kernel Dispatch terputus | Optimization sia-sia | CRITICAL |
| 3 | API tidak konsisten (7 cara berbeda) | User bingung | HIGH |
| 4 | Benchmark tidak representatif | Tidak bisa prove value | HIGH |
| 5 | Dokumentasi tidak lengkap | User tidak tahu cara pakai | MEDIUM |
| 6 | Error handling lemah | Silent failures | MEDIUM |
| 7 | Logging/Monitoring tidak ada | Tidak ada visibility | LOW-MED |

---

## Bagian 2: Riset dan Referensi

### 2.1 TensorRT Architecture (NVIDIA)
**Sumber:** NVIDIA TensorRT Developer Guide, abhik.xyz, massedcompute.com

**Pelajaran Kunci:**
1. **Build Phase vs Runtime Phase** - Optimasi dilakukan saat compile, runtime hanya eksekusi
2. **Kernel Auto-Tuning** - Profile multiple kernel implementations, pilih yang tercepat
3. **Layer Fusion** - Gabungkan Conv+BN+ReLU menjadi satu kernel
4. **CUDA Graphs** - Capture sequence of kernels, launch dengan satu API call
5. **Execution Context** - Holds dynamic network state during inference

**Relevansi untuk Zenith:**
- Zenith sudah punya kernels yang di-tune (cuda_kernels.cu, fp16_kernels.cu)
- MISSING: Runtime yang dispatch ke kernels tersebut
- MISSING: Execution context untuk state management

### 2.2 ONNX Runtime Execution Provider (Microsoft)
**Sumber:** onnxruntime.ai, becominghuman.ai

**Pelajaran Kunci:**
1. **Execution Provider (EP)** - Abstraksi antara graph dan hardware
2. **GetCapability()** - EP menyatakan operator apa yang didukung
3. **Priority-Based Assignment** - EP dengan prioritas tinggi dipilih dulu
4. **Fallback Mechanism** - CPU sebagai fallback default
5. **Stateless Kernels** - `const Compute()` untuk thread-safety

**Relevansi untuk Zenith:**
- Zenith sudah punya struktur mirip (backends/cuda_backend.py, backends/rocm_backend.py)
- MISSING: Capability query system
- MISSING: Proper fallback mechanism
- MISSING: Priority-based kernel selection

### 2.3 TVM Runtime Module (Apache)
**Sumber:** apache.org TVM docs, OSDI paper, d2l.ai

**Pelajaran Kunci:**
1. **Declarative Tensor Expression** - Decouple computation dari schedule
2. **AutoTVM** - Learning-based cost model untuk kernel selection
3. **PackedFunc** - Type-erased function interface untuk cross-language
4. **Lightweight Runtime** - < 1MB, deployable ke edge devices
5. **RPC for Remote Profiling** - Benchmark on target hardware

**Relevansi untuk Zenith:**
- Zenith sudah punya autotuner.py dengan konsep serupa
- MISSING: Integration antara autotuner dan runtime
- MISSING: PackedFunc-like interface

### 2.4 PyTorch 2.0 torch.compile (Meta)
**Sumber:** pytorch.org, medium.com, dev.to

**Pelajaran Kunci:**
1. **TorchDynamo** - Capture Python graph via bytecode analysis
2. **AOT Autograd** - Capture forward and backward passes
3. **TorchInductor** - Generate Triton/C++ kernels from graph
4. **Compilation Modes** - "default", "reduce-overhead", "max-autotune"
5. **Graph Breaks** - Handle when full graph cannot be captured
6. **Warm-up Strategy** - First calls for compilation, not inference

**Relevansi untuk Zenith:**
- Zenith sudah punya adapter yang mirip (pytorch_adapter.py)
- MISSING: Proper backend yang menggunakan Zenith kernels
- MISSING: Compilation caching

### 2.5 MLPerf Benchmark Standards (MLCommons)
**Sumber:** mlcommons.org, mlsys.org, github.com/mlcommons

**Pelajaran Kunci:**
1. **Four Scenarios:** Single-stream, Multi-stream, Offline, Server
2. **Metrics:** Latency (P90/P99), Throughput (samples/sec), Quality (accuracy)
3. **Quality Target:** Must meet 99% or 99.9% of reference accuracy
4. **Warm-up Runs:** Exclude compilation time from benchmarks
5. **Representative Models:** BERT, ResNet-50, GPT, LLaMA

**Relevansi untuk Zenith:**
- Zenith benchmarks harus mengikuti standar ini
- Current benchmarks: bert_fp16_full.py, production_benchmark.py
- MISSING: Proper scenario-based benchmarking
- MISSING: Quality target verification

### 2.6 Observability Best Practices (OpenTelemetry, Prometheus)
**Sumber:** opentelemetry.io, prometheus.io, medium.com

**Pelajaran Kunci:**
1. **Structured Logging** - JSON format dengan key-value pairs
2. **Three Pillars** - Traces, Metrics, Logs (correlated)
3. **Trace Context** - Span IDs untuk correlation
4. **Prometheus Metrics** - Counter, Gauge, Histogram, Summary
5. **Resource Attributes** - model.name, model.version, device

**Relevansi untuk Zenith:**
- Zenith sudah punya profiler.py (basic)
- MISSING: Structured logging system
- MISSING: Metrics export
- MISSING: Trace correlation

### 2.7 Numerical Precision Theory
**Sumber:** NVIDIA documentation, arxiv.org, mlr.press

**Pelajaran Kunci:**
1. **FP32** - 23 bits mantissa, wide dynamic range, high accuracy
2. **FP16** - 10 bits mantissa, narrower range, rounding errors
3. **BF16** - FP32-like range with FP16-like precision
4. **Error Bound:** |T(F)(x) - F(x)| / (|F(x)| + ε) ≤ δ
5. **Mixed Precision** - FP16 compute, FP32 accumulation
6. **Loss Scaling** - Scale gradients to prevent underflow

**Relevansi untuk Zenith:**
- CetakBiru.md Section 4.1 sudah mendefinisikan error bound
- Zenith sudah punya mixed_precision.py
- MISSING: Proper error bound verification in runtime
- MISSING: Automatic loss scaling

---

## Bagian 3: Arsitektur Solusi

### 3.1 Arsitektur Target (Setelah Blueprint)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  import zenith                                                       │   │
│  │  model = zenith.compile(pytorch_model, target="cuda", precision="fp16") │
│  │  output = model(input)  # FAST! Uses Zenith kernels                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           UNIFIED API LAYER                                 │
│  zenith/                                                                    │
│  ├── api.py          # zenith.compile() - ENTRY POINT                      │
│  ├── config.py       # [NEW] Global configuration                          │
│  └── logger.py       # [NEW] Structured logging                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ADAPTER LAYER (Existing)                            │
│  zenith/adapters/                                                           │
│  ├── pytorch_adapter.py   # PyTorch → GraphIR [UPDATE: connect to runtime] │
│  ├── tensorflow_adapter.py                                                  │
│  ├── jax_adapter.py                                                         │
│  └── onnx_adapter.py                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                       OPTIMIZATION LAYER (Existing)                         │
│  zenith/optimization/                                                       │
│  ├── fusion_pass.py       # Conv+BN+ReLU fusion                            │
│  ├── quantization.py      # INT8 quantization                              │
│  ├── mixed_precision.py   # FP16/BF16                                      │
│  ├── autotuner.py         # Kernel auto-tuning                             │
│  └── advanced_fusion.py   # Transformer fusion                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                      RUNTIME LAYER (NEW - Critical)                         │
│  zenith/runtime/                                                            │
│  ├── __init__.py                                                            │
│  ├── engine.py           # [NEW] ZenithEngine - main runtime               │
│  ├── executor.py         # [NEW] GraphExecutor - executes optimized graph  │
│  ├── dispatcher.py       # [NEW] KernelDispatcher - routes ops to kernels  │
│  ├── context.py          # [NEW] ExecutionContext - holds state            │
│  ├── memory_manager.py   # [NEW] GPU memory management                     │
│  └── cuda_graphs.py      # [NEW] CUDA Graphs for low-latency               │
├─────────────────────────────────────────────────────────────────────────────┤
│                       KERNEL LAYER (Existing - Fast)                        │
│  core/src/                                                                  │
│  ├── cuda_kernels.cu        # Linear, Conv, MatMul                         │
│  ├── fp16_kernels.cu        # FP16 with Tensor Cores                       │
│  ├── fused_kernels.cu       # Add+LayerNorm, Bias+ReLU                     │
│  ├── transformer_kernels.cu # Attention, GELU                              │
│  ├── flash_attention.cu     # Flash Attention                              │
│  └── flash_attention_v2.cu  # Flash Attention V2                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                      BACKEND LAYER (Existing)                               │
│  zenith/backends/                                                           │
│  ├── cuda_backend.py     # [UPDATE: register kernel capabilities]          │
│  ├── rocm_backend.py                                                        │
│  └── cpu_backend.py                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                          HARDWARE                                           │
│  NVIDIA GPU (CUDA) | AMD GPU (ROCm) | CPU (AVX/NEON)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Alur Eksekusi Baru

```
User: model = zenith.compile(pytorch_model, target="cuda")
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: CONVERSION (Existing)                                  │
│  PyTorchAdapter.from_model(model, sample_input)                 │
│  Output: GraphIR with nodes, inputs, outputs                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: OPTIMIZATION (Existing)                                │
│  PassManager.run(graph_ir)                                      │
│  - ConstantFolding                                              │
│  - DeadCodeElimination                                          │
│  - FusionPass (Conv+BN+ReLU, Add+LayerNorm)                    │
│  - LayoutOptimization                                           │
│  Output: Optimized GraphIR                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: COMPILATION (NEW - ZenithEngine)                       │
│  ZenithEngine.compile(optimized_ir, target)                     │
│  - Query backend capabilities                                   │
│  - Build execution plan                                         │
│  - Select kernels for each operation                            │
│  - Allocate GPU memory                                          │
│  - Create CUDA Graph if applicable                              │
│  Output: CompiledModel with execution plan                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: EXECUTION (NEW - GraphExecutor)                        │
│  User calls: output = model(input)                              │
│  GraphExecutor.run(input)                                       │
│  - Create ExecutionContext                                      │
│  - For each node in execution plan:                             │
│      KernelDispatcher.dispatch(node, context)                   │
│      → cuda.linear_fp16_gpu() / cuda.attention_fp16_gpu() / ... │
│  - Return output tensors                                        │
│  Output: Result tensor (FAST!)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bagian 4: Spesifikasi Teknis untuk Setiap Masalah

### 4.1 MASALAH #1: Zenith Runtime

**Problem Statement:**
GraphIR yang dioptimasi tidak dieksekusi menggunakan Zenith CUDA kernels. Eksekusi kembali ke framework asli (PyTorch/TensorFlow).

**Solusi: ZenithEngine + GraphExecutor**

**File Baru:** `zenith/runtime/engine.py`

```python
# Pseudocode - akan diimplementasikan dengan detail
class ZenithEngine:
    """
    Main compilation engine for Zenith.
    
    Inspired by:
    - TensorRT: Build phase creates optimized engine
    - ONNX Runtime: Execution provider abstraction
    - TVM: Module compilation and deployment
    """
    
    def __init__(self, backend: str = "cuda"):
        self.backend = backend
        self.kernel_registry = KernelRegistry()
        self.memory_manager = MemoryManager()
        
    def compile(self, graph_ir: GraphIR, config: CompileConfig) -> CompiledModel:
        """
        Compile GraphIR into executable model.
        
        Steps:
        1. Validate graph
        2. Query backend capabilities
        3. Build execution plan (topological sort)
        4. Select optimal kernel for each op
        5. Allocate memory
        6. Create CUDA Graph if applicable
        """
        # Step 1: Validate
        self._validate_graph(graph_ir)
        
        # Step 2: Query capabilities
        capabilities = self.backend.get_capabilities()
        
        # Step 3: Build execution plan
        execution_plan = self._build_execution_plan(graph_ir, capabilities)
        
        # Step 4: Select kernels
        for node in execution_plan.nodes:
            kernel = self.kernel_registry.get_kernel(
                node.op_type, 
                config.precision,
                node.input_shapes
            )
            node.assigned_kernel = kernel
            
        # Step 5: Allocate memory
        memory_plan = self.memory_manager.plan(execution_plan)
        
        # Step 6: Create CUDA Graph (optional)
        cuda_graph = None
        if config.use_cuda_graphs:
            cuda_graph = self._create_cuda_graph(execution_plan)
            
        return CompiledModel(
            graph_ir=graph_ir,
            execution_plan=execution_plan,
            memory_plan=memory_plan,
            cuda_graph=cuda_graph,
            engine=self
        )
```

**File Baru:** `zenith/runtime/executor.py`

```python
class GraphExecutor:
    """
    Executes compiled model using Zenith kernels.
    
    Inspired by:
    - TensorRT: Runtime execution with pre-selected kernels
    - ONNX Runtime: Stateless kernel execution
    """
    
    def __init__(self, compiled_model: CompiledModel):
        self.model = compiled_model
        self.dispatcher = KernelDispatcher()
        
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute the model with given inputs."""
        # Create execution context
        context = ExecutionContext(
            memory_plan=self.model.memory_plan,
            inputs=inputs
        )
        
        # Use CUDA Graph if available
        if self.model.cuda_graph is not None:
            return self._run_cuda_graph(context)
            
        # Execute node by node
        for node in self.model.execution_plan.nodes:
            self.dispatcher.dispatch(node, context)
            
        # Collect outputs
        return context.get_outputs()
```

**File Baru:** `zenith/runtime/dispatcher.py`

```python
class KernelDispatcher:
    """
    Dispatches operations to appropriate Zenith CUDA kernels.
    
    Maps GraphIR operations to cuda.* functions.
    """
    
    # Mapping dari op_type ke Zenith kernel
    KERNEL_MAP = {
        # Linear operations
        "MatMul": "cuda.matmul_gpu",
        "Gemm": "cuda.linear_gpu",
        "Linear": "cuda.linear_gpu",
        "LinearFP16": "cuda.linear_fp16_gpu",
        
        # Convolution
        "Conv": "cuda.conv2d_gpu",
        "ConvBnReLU": "cuda.conv_bn_relu_gpu",
        
        # Attention
        "Attention": "cuda.attention_gpu",
        "AttentionFP16": "cuda.attention_fp16_gpu",
        "FlashAttention": "cuda.flash_attention_gpu",
        
        # Normalization
        "LayerNormalization": "cuda.layernorm_gpu",
        "LayerNormFP16": "cuda.layernorm_fp16_gpu",
        "BatchNormalization": "cuda.batchnorm_gpu",
        
        # Activation
        "Relu": "cuda.relu_gpu",
        "Gelu": "cuda.gelu_gpu",
        "GeluFP16": "cuda.gelu_fp16_gpu",
        "Sigmoid": "cuda.sigmoid_gpu",
        "Softmax": "cuda.softmax_gpu",
        
        # Fused operations
        "FusedAddLayerNorm": "cuda.fused_add_layernorm_gpu",
        "FusedBiasReLU": "cuda.fused_bias_relu_gpu",
        "FusedBiasGeLU": "cuda.fused_bias_gelu_gpu",
        
        # Elementwise
        "Add": "cuda.add_gpu",
        "AddFP16": "cuda.add_fp16_gpu",
        "Mul": "cuda.mul_gpu",
        "Sub": "cuda.sub_gpu",
        
        # Reshape
        "Reshape": "cuda.reshape_gpu",
        "Transpose": "cuda.transpose_gpu",
        "TransposeForAttention": "cuda.transpose_for_attention",
    }
    
    def dispatch(self, node: ExecutionNode, context: ExecutionContext):
        """Dispatch node to appropriate kernel."""
        kernel_name = self.KERNEL_MAP.get(node.op_type)
        
        if kernel_name is None:
            raise UnsupportedOperationError(f"Op {node.op_type} not supported")
            
        # Get kernel function
        kernel_fn = self._get_kernel_function(kernel_name)
        
        # Get inputs from context
        inputs = [context.get_tensor(inp) for inp in node.inputs]
        
        # Get weights if needed
        weights = self._get_weights(node, context)
        
        # Execute kernel
        outputs = kernel_fn(*inputs, *weights)
        
        # Store outputs in context
        for i, output_name in enumerate(node.outputs):
            context.set_tensor(output_name, outputs[i] if isinstance(outputs, tuple) else outputs)
```

---

### 4.2 MASALAH #2: GraphIR → Kernel Dispatch

**Problem Statement:**
Tidak ada pemetaan antara operasi di GraphIR dengan Zenith CUDA kernels.

**Solusi: KernelRegistry dengan Capability Query**

**File Baru:** `zenith/runtime/kernel_registry.py`

```python
@dataclass
class KernelSpec:
    """Specification for a kernel."""
    name: str
    op_type: str
    precision: str  # "fp32", "fp16", "int8"
    input_constraints: dict  # shape constraints
    estimated_flops: callable  # function to estimate FLOPs
    kernel_fn: callable  # actual kernel function
    
class KernelRegistry:
    """
    Registry of all available Zenith kernels.
    
    Inspired by:
    - ONNX Runtime: OperatorRegistry
    - TVM: AutoTVM kernel database
    """
    
    def __init__(self):
        self._kernels: dict[str, list[KernelSpec]] = {}
        self._register_all_kernels()
        
    def _register_all_kernels(self):
        """Register all available CUDA kernels."""
        from zenith._zenith_core import cuda
        
        # Register Linear kernels
        self.register(KernelSpec(
            name="linear_fp32",
            op_type="Linear",
            precision="fp32",
            input_constraints={"min_batch": 1, "max_batch": 1024},
            estimated_flops=lambda m, n, k: 2 * m * n * k,
            kernel_fn=cuda.linear_gpu
        ))
        
        self.register(KernelSpec(
            name="linear_fp16",
            op_type="Linear",
            precision="fp16",
            input_constraints={"min_batch": 1, "max_batch": 1024},
            estimated_flops=lambda m, n, k: 2 * m * n * k,
            kernel_fn=cuda.linear_fp16_gpu
        ))
        
        # Register Attention kernels
        self.register(KernelSpec(
            name="attention_fp16",
            op_type="Attention",
            precision="fp16",
            input_constraints={"max_seq_len": 2048},
            estimated_flops=lambda b, h, s, d: 4 * b * h * s * s * d,
            kernel_fn=cuda.attention_fp16_gpu
        ))
        
        # ... register all other kernels
        
    def get_kernel(
        self, 
        op_type: str, 
        precision: str,
        input_shapes: list[tuple]
    ) -> KernelSpec:
        """Get optimal kernel for given operation."""
        candidates = self._kernels.get(op_type, [])
        
        # Filter by precision
        candidates = [k for k in candidates if k.precision == precision]
        
        # Filter by constraints
        candidates = [k for k in candidates if self._check_constraints(k, input_shapes)]
        
        if not candidates:
            raise NoKernelFoundError(f"No kernel for {op_type} with {precision}")
            
        # Select best kernel (could use autotuner results)
        return self._select_best(candidates, input_shapes)
```

---

### 4.3 MASALAH #3: API Tidak Konsisten

**Problem Statement:**
Ada 7 cara berbeda untuk menggunakan Zenith, membingungkan pengguna.

**Solusi: Unified API dengan backward compatibility**

**File Update:** `zenith/api.py`

```python
def compile(
    model: Any,
    target: str = "cuda",
    precision: str = "fp32",
    opt_level: int = 2,
    mode: str = "default",  # "default", "reduce-overhead", "max-autotune"
    **kwargs
) -> CompiledModel:
    """
    Compile and optimize a model for the target platform.
    
    This is THE SINGLE entry point for Zenith.
    
    Modes (inspired by torch.compile):
    - "default": Balanced compilation time and runtime performance
    - "reduce-overhead": Minimize CPU overhead, use CUDA Graphs
    - "max-autotune": Maximum tuning, longer compile time
    
    Example:
        import zenith
        
        # Simple usage
        model = zenith.compile(pytorch_model)
        output = model(input)
        
        # With options
        model = zenith.compile(
            pytorch_model,
            target="cuda",
            precision="fp16",
            mode="max-autotune"
        )
    """
    # Step 1: Convert to GraphIR (existing logic)
    graph_ir, original_model = _convert_to_graphir(model, kwargs.get("sample_input"))
    
    # Step 2: Optimize graph (existing logic, enhanced)
    optimized_ir = _optimize_graph(graph_ir, opt_level, precision)
    
    # Step 3: Compile with ZenithEngine (NEW!)
    from .runtime import ZenithEngine
    
    engine = ZenithEngine(backend=target)
    config = CompileConfig(
        precision=precision,
        mode=mode,
        use_cuda_graphs=(mode == "reduce-overhead"),
        auto_tune=(mode == "max-autotune"),
    )
    
    compiled = engine.compile(optimized_ir, config)
    
    # Step 4: Return callable model
    return compiled


# Backward compatible aliases
optimize = compile  # zenith.optimize() still works
```

**File Update:** `zenith/torch/__init__.py`

```python
def create_backend(
    target: str = "cuda",
    precision: str = "fp32",
    **kwargs
) -> callable:
    """
    Create torch.compile backend that uses Zenith kernels.
    
    This is now properly connected to ZenithEngine!
    """
    from ..runtime import ZenithEngine
    
    engine = ZenithEngine(backend=target)
    
    def zenith_backend(gm, example_inputs):
        # Convert FX Graph to GraphIR
        graph_ir = _fx_to_graphir(gm, example_inputs)
        
        # Optimize
        from ..optimization import optimize_graph
        optimized_ir, _ = optimize_graph(graph_ir, opt_level=2)
        
        # Compile with engine (connects to kernels!)
        config = CompileConfig(precision=precision)
        compiled = engine.compile(optimized_ir, config)
        
        # Return callable
        def forward(*args, **kwargs):
            inputs = _prepare_inputs(args, compiled.graph_ir.inputs)
            outputs = compiled.run(inputs)
            return _convert_outputs(outputs)
            
        return forward
        
    return zenith_backend
```

---

### 4.4 MASALAH #4: Benchmark Tidak Representatif

**Problem Statement:**
Benchmark saat ini tidak mengikuti standar industri (MLPerf).

**Solusi: MLPerf-style Benchmark Suite**

**File Baru:** `benchmarks/mlperf_suite.py`

```python
@dataclass
class BenchmarkConfig:
    """MLPerf-inspired benchmark configuration."""
    model_name: str
    batch_sizes: list[int]
    sequence_lengths: list[int]  # for transformers
    num_warmup: int = 10
    num_runs: int = 100
    quality_target: float = 0.99  # 99% of reference accuracy
    
    # MLPerf scenarios
    scenario: str = "single-stream"  # single-stream, multi-stream, offline, server
    
class ZenithBenchmark:
    """
    MLPerf-inspired benchmark suite for Zenith.
    
    Measures:
    - Latency (P50, P90, P99)
    - Throughput (samples/second)
    - Quality (accuracy vs reference)
    """
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        # 1. Load model
        model = self._load_model(config.model_name)
        
        # 2. Compile with Zenith
        zenith_model = zenith.compile(model, target="cuda", precision="fp16")
        
        # 3. Warmup (exclude from timing)
        for _ in range(config.num_warmup):
            _ = zenith_model(self._get_sample_input(config))
            
        # 4. Run benchmark based on scenario
        if config.scenario == "single-stream":
            result = self._run_single_stream(zenith_model, config)
        elif config.scenario == "offline":
            result = self._run_offline(zenith_model, config)
        elif config.scenario == "server":
            result = self._run_server(zenith_model, config)
            
        # 5. Verify quality
        accuracy = self._verify_accuracy(zenith_model, model)
        result.quality_passed = accuracy >= config.quality_target
        
        return result
        
    def _run_single_stream(self, model, config) -> BenchmarkResult:
        """Single-stream: measure per-query latency."""
        latencies = []
        
        for _ in range(config.num_runs):
            input_data = self._get_sample_input(config)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_data)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
            
        return BenchmarkResult(
            scenario="single-stream",
            latency_p50=np.percentile(latencies, 50),
            latency_p90=np.percentile(latencies, 90),
            latency_p99=np.percentile(latencies, 99),
            throughput=1000 / np.mean(latencies),  # queries/sec
        )
```

---

### 4.5 MASALAH #5: Dokumentasi Tidak Lengkap

**Solusi: Comprehensive Documentation**

**Struktur Dokumentasi Baru:**
```
docs/
├── README.md                    # Quick start
├── ARCHITECTURE.md              # System architecture (existing, update)
├── BLUEPRINT_ZENITH_3X.md       # This document
├── API_REFERENCE.md             # Complete API docs
├── USER_GUIDE/
│   ├── 01_installation.md
│   ├── 02_quick_start.md
│   ├── 03_pytorch_integration.md
│   ├── 04_tensorflow_integration.md
│   ├── 05_jax_integration.md
│   ├── 06_optimization_options.md
│   ├── 07_deployment.md
│   └── 08_troubleshooting.md
├── DEVELOPER_GUIDE/
│   ├── 01_architecture.md
│   ├── 02_adding_kernels.md
│   ├── 03_adding_operators.md
│   ├── 04_testing.md
│   └── 05_benchmarking.md
└── EXAMPLES/
    ├── bert_optimization.py
    ├── resnet_deployment.py
    └── llm_inference.py
```

---

### 4.6 MASALAH #6: Error Handling Lemah

**Solusi: Comprehensive Error System**

**File Baru:** `zenith/errors.py`

```python
class ZenithError(Exception):
    """Base class for Zenith errors."""
    pass

class CompilationError(ZenithError):
    """Error during model compilation."""
    pass

class UnsupportedOperationError(ZenithError):
    """Operation not supported by current backend."""
    def __init__(self, op_type: str, suggestions: list[str] = None):
        self.op_type = op_type
        self.suggestions = suggestions or []
        super().__init__(
            f"Operation '{op_type}' is not supported. "
            f"Suggestions: {', '.join(self.suggestions) if self.suggestions else 'None'}"
        )

class PrecisionError(ZenithError):
    """Error related to numerical precision."""
    def __init__(self, expected_tolerance: float, actual_error: float):
        self.expected_tolerance = expected_tolerance
        self.actual_error = actual_error
        super().__init__(
            f"Numerical precision violated: expected error < {expected_tolerance}, "
            f"got {actual_error}. Consider using higher precision."
        )

class KernelError(ZenithError):
    """Error during kernel execution."""
    pass

class MemoryError(ZenithError):
    """GPU memory allocation error."""
    pass
```

---

### 4.7 MASALAH #7: Logging/Monitoring

**Solusi: Structured Logging + Metrics**

**File Baru:** `zenith/observability/logger.py`

```python
import json
import logging
from dataclasses import dataclass, asdict
from enum import IntEnum

class Verbosity(IntEnum):
    SILENT = 0
    ERROR = 1  
    WARNING = 2  # Default production
    INFO = 3     # Default development
    DEBUG = 4

@dataclass
class LogEntry:
    """Structured log entry."""
    level: str
    message: str
    timestamp: str
    component: str  # "compiler", "runtime", "kernel"
    
    # Optional context
    model_name: str = None
    operation: str = None
    duration_ms: float = None
    memory_mb: float = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

class ZenithLogger:
    """
    Structured logger for Zenith.
    
    Inspired by:
    - OpenTelemetry structured logging
    - Google Cloud Logging best practices
    """
    
    _instance = None
    
    def __init__(self):
        self.verbosity = Verbosity.INFO
        self._handlers = []
        
    @classmethod
    def get(cls) -> 'ZenithLogger':
        if cls._instance is None:
            cls._instance = ZenithLogger()
        return cls._instance
        
    def set_verbosity(self, level: int):
        """Set verbosity level (0-4)."""
        self.verbosity = Verbosity(level)
        
    def info(self, message: str, **context):
        if self.verbosity >= Verbosity.INFO:
            self._emit(LogEntry(
                level="INFO",
                message=message,
                timestamp=datetime.now().isoformat(),
                component=context.get("component", "zenith"),
                **{k: v for k, v in context.items() if k != "component"}
            ))
            
    def compile_summary(self, stats: dict):
        """Log compilation summary (always shown at INFO level)."""
        if self.verbosity >= Verbosity.INFO:
            print(f"""
┌─────────────────────────────────────────────────────────┐
│ Zenith Compilation Complete                             │
├─────────────────────────────────────────────────────────┤
│ Model:      {stats.get('model_name', 'N/A'):<40} │
│ Target:     {stats.get('target', 'N/A'):<40} │
│ Precision:  {stats.get('precision', 'N/A'):<40} │
│ Time:       {stats.get('compile_time', 0):.2f}s{' ' * 33} │
│                                                         │
│ Optimizations Applied:                                  │
│   - Fused ops: {stats.get('fused_ops', 0):<38} │
│   - DCE removed: {stats.get('dce_removed', 0):<36} │
│   - Est. speedup: {stats.get('estimated_speedup', 1.0):.1f}x{' ' * 31} │
└─────────────────────────────────────────────────────────┘
            """)
```

**File Baru:** `zenith/observability/metrics.py`

```python
from dataclasses import dataclass
import time

@dataclass  
class InferenceMetrics:
    """Metrics for a single inference."""
    latency_ms: float
    memory_mb: float
    kernel_calls: int
    
class MetricsCollector:
    """
    Collects and exports metrics.
    
    Optional Prometheus integration.
    """
    
    def __init__(self):
        self._latency_histogram = []
        self._inference_count = 0
        self._error_count = 0
        
    def record_inference(self, metrics: InferenceMetrics):
        self._latency_histogram.append(metrics.latency_ms)
        self._inference_count += 1
        
    def get_summary(self) -> dict:
        if not self._latency_histogram:
            return {}
        return {
            "total_inferences": self._inference_count,
            "latency_p50_ms": np.percentile(self._latency_histogram, 50),
            "latency_p99_ms": np.percentile(self._latency_histogram, 99),
            "errors": self._error_count,
        }
```

---

## Bagian 5: Daftar Tugas Implementasi

### Phase 1: Runtime Core (2 minggu)
**Priority: CRITICAL**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 1.1 | Buat ZenithEngine | `zenith/runtime/engine.py` | 2 hari |
| 1.2 | Buat GraphExecutor | `zenith/runtime/executor.py` | 2 hari |
| 1.3 | Buat KernelDispatcher | `zenith/runtime/dispatcher.py` | 2 hari |
| 1.4 | Buat KernelRegistry | `zenith/runtime/kernel_registry.py` | 1 hari |
| 1.5 | Buat ExecutionContext | `zenith/runtime/context.py` | 1 hari |
| 1.6 | Buat MemoryManager | `zenith/runtime/memory_manager.py` | 2 hari |
| 1.7 | Register semua CUDA kernels | `zenith/runtime/kernel_registry.py` | 2 hari |
| 1.8 | Unit tests untuk runtime | `tests/python/test_runtime.py` | 2 hari |

### Phase 2: API Unification (1 minggu)
**Priority: HIGH**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 2.1 | Update zenith.compile() | `zenith/api.py` | 1 hari |
| 2.2 | Update ztorch.create_backend() | `zenith/torch/__init__.py` | 1 hari |
| 2.3 | Update ztf.compile() | `zenith/tensorflow/__init__.py` | 1 hari |
| 2.4 | Update zjax.compile() | `zenith/jax/__init__.py` | 1 hari |
| 2.5 | Integration tests | `tests/integration/` | 2 hari |

### Phase 3: Benchmarks (1 minggu)
**Priority: HIGH**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 3.1 | Buat MLPerf-style suite | `benchmarks/mlperf_suite.py` | 2 hari |
| 3.2 | BERT benchmark | `benchmarks/bert_mlperf.py` | 1 hari |
| 3.3 | ResNet benchmark | `benchmarks/resnet_mlperf.py` | 1 hari |
| 3.4 | Generate benchmark report | `benchmarks/report_generator.py` | 1 hari |
| 3.5 | Update docs/benchmarks | `docs/benchmarks/` | 1 hari |

### Phase 4: Observability (1 minggu)
**Priority: MEDIUM**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 4.1 | Buat structured logger | `zenith/observability/logger.py` | 1 hari |
| 4.2 | Buat metrics collector | `zenith/observability/metrics.py` | 1 hari |
| 4.3 | Integrate dengan runtime | Update runtime files | 1 hari |
| 4.4 | Add verbosity control | `zenith/__init__.py` | 0.5 hari |
| 4.5 | Update profiler | `zenith/optimization/profiler.py` | 1 hari |

### Phase 5: Error Handling (3 hari)
**Priority: MEDIUM**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 5.1 | Buat error hierarchy | `zenith/errors.py` | 0.5 hari |
| 5.2 | Add error handling di runtime | Update runtime files | 1 hari |
| 5.3 | Add helpful error messages | All files | 1 hari |
| 5.4 | Error tests | `tests/python/test_errors.py` | 0.5 hari |

### Phase 6: Documentation (4 hari)
**Priority: MEDIUM**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 6.1 | User Guide | `docs/USER_GUIDE/*.md` | 2 hari |
| 6.2 | API Reference | `docs/API_REFERENCE.md` | 1 hari |
| 6.3 | Examples | `docs/EXAMPLES/*.py` | 1 hari |

### Phase 7: Validation & Polish (3 hari)
**Priority: HIGH**

| # | Tugas | File | Estimasi |
|---|-------|------|----------|
| 7.1 | End-to-end testing | `tests/e2e/` | 1 hari |
| 7.2 | Performance validation | Run all benchmarks | 1 hari |
| 7.3 | Fix bugs and polish | Various | 1 hari |

---

## Bagian 6: Success Criteria

### Kriteria Keberhasilan:

| Metrik | Target | Cara Verifikasi |
|--------|--------|-----------------|
| **Speedup** | 2-7x vs PyTorch native | Run BERT/ResNet benchmarks |
| **API Consistency** | 1 primary way | `zenith.compile()` works everywhere |
| **Latency (BERT-base)** | < 5ms @ batch=1 | MLPerf single-stream |
| **Accuracy** | > 99% of reference | Compare outputs |
| **Coverage** | > 80% ops supported | Kernel registry count |
| **Documentation** | Complete user guide | Manual review |

---

## Bagian 7: Timeline

```
Week 1-2: Phase 1 (Runtime Core)
    └── ZenithEngine, GraphExecutor, KernelDispatcher DONE

Week 3: Phase 2 (API Unification)
    └── zenith.compile() properly connected

Week 4: Phase 3 (Benchmarks)
    └── MLPerf-style benchmarks running

Week 5: Phase 4-5 (Observability + Errors)
    └── Logging and error handling complete

Week 6: Phase 6-7 (Docs + Validation)
    └── Documentation and final testing

Total: 6 weeks
```

---

## Bagian 8: Risks dan Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Kernel bugs when integrated | High | Extensive unit tests, compare with PyTorch |
| Memory leaks | Medium | Memory profiling, CUDA memcheck |
| Performance regression | Medium | Continuous benchmarking |
| API breaking changes | Low | Deprecation warnings, docs |

---

## Appendix A: File Structure Setelah Implementasi

```
zenith/
├── __init__.py
├── api.py               # [UPDATE] zenith.compile()
├── config.py            # [NEW] Global config
├── errors.py            # [NEW] Error hierarchy
│
├── runtime/             # [NEW DIRECTORY]
│   ├── __init__.py
│   ├── engine.py        # ZenithEngine
│   ├── executor.py      # GraphExecutor
│   ├── dispatcher.py    # KernelDispatcher
│   ├── kernel_registry.py
│   ├── context.py       # ExecutionContext
│   ├── memory_manager.py
│   └── cuda_graphs.py
│
├── observability/       # [NEW DIRECTORY]
│   ├── __init__.py
│   ├── logger.py        # Structured logging
│   └── metrics.py       # Metrics collection
│
├── adapters/            # [EXISTING, minor updates]
│   ├── pytorch_adapter.py
│   ├── tensorflow_adapter.py
│   └── ...
│
├── optimization/        # [EXISTING]
│   ├── fusion_pass.py
│   ├── quantization.py
│   └── ...
│
├── backends/            # [EXISTING, minor updates]
│   ├── cuda_backend.py
│   └── ...
│
└── execution/           # [EXISTING, may deprecate in favor of runtime/]
    └── ...
```

---

**Dokumen ini akan diupdate seiring implementasi berlanjut.**

*End of Blueprint*
