name, memory.total [MiB], compute_cap
Tesla T4, 15360 MiB, 7.5 
Cloning into 'ZENITH'...
remote: Enumerating objects: 1358, done.
remote: Counting objects: 100% (171/171), done.
remote: Compressing objects: 100% (140/140), done.
remote: Total 1358 (delta 60), reused 131 (delta 28), pack-reused 1187 (from 1)
Receiving objects: 100% (1358/1358), 9.00 MiB | 7.49 MiB/s, done.
Resolving deltas: 100% (600/600), done.
/content/ZENITH/ZENITH
e631274 (HEAD -> main, origin/main, origin/HEAD) Fix UnsupportedOperationError: op_name -> op_type
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
  Building editable for pyzenith (pyproject.toml) ... done
Zenith version: 0.1.4
CUDA available: True

Exported symbols: 40
  - CompilationError
  - ConfigurationError
  - DataType
  - GraphIR
  - JAXAdapter
  - KernelError
  - Layout
  - MetricsServer
  - ModelConfig
  - Node
  - ONNXAdapter
  - PrecisionError
  - PrometheusExporter
  - PyTorchAdapter
  - Shape
  - Status
  - StatusCode
  - TensorDescriptor
  - TensorFlowAdapter
  - TritonBackend
  - TritonBackendConfig
  - UnsupportedOperationError
  - ValidationError
  - Verbosity
  - ZenithError
  - ZenithMemoryError
  - ZenithModelExporter
  - __author__
  - __version__
  - compile
  - dtype_size
  - dtype_to_string
  - export_to_onnx
  - export_to_torchscript
  - export_to_triton
  - get_version
  - is_native
  - optimize
  - set_verbosity
  - start_monitoring_server
GraphIR: test_graph
DataType: DataType.Float32
[OK] Core module working
[OK] Optimization passes instantiated
Original: float32 -> Quantized: int8
Scale: 0.032219, Zero point: 0
[OK] Quantization working
Registered operations: 29
Sample ops: ['Linear', 'MatMul', 'Gemm', 'Conv', 'Conv2d']
[OK] Runtime engine working
[INFO] [zenith] Test log message
Metrics collector: MetricsCollector
[OK] Observability working
MetricsServer: <class 'zenith.monitoring.server.MetricsServer'>
PrometheusExporter: <class 'zenith.monitoring.exporter.PrometheusExporter'>
start_monitoring_server: <function start_server at 0x7a16191b2700>
[OK] Monitoring module integrated
TritonBackend: <class 'zenith.serving.triton_backend.TritonBackend'>
ZenithModelExporter: <class 'zenith.serving.model_export.ZenithModelExporter'>
[OK] Serving module integrated
Caught: UnsupportedOperationError
Suggestions: 3
[OK] Error handling working
tests/python/test_runtime.py::TestKernelRegistry::test_kernel_spec_creation PASSED [ 54%]
tests/python/test_runtime.py::TestKernelRegistry::test_registry_register_and_query PASSED [ 56%]
tests/python/test_runtime.py::TestKernelRegistry::test_global_registry PASSED [ 59%]
tests/python/test_runtime.py::TestExecutionContext::test_context_creation PASSED [ 62%]
tests/python/test_runtime.py::TestExecutionContext::test_set_get_tensor PASSED [ 64%]
tests/python/test_runtime.py::TestExecutionContext::test_tensor_info PASSED [ 67%]
tests/python/test_runtime.py::TestExecutionContext::test_memory_tracking PASSED [ 70%]
tests/python/test_runtime.py::TestKernelDispatcher::test_dispatcher_creation PASSED [ 72%]
tests/python/test_runtime.py::TestKernelDispatcher::test_op_type_normalization PASSED [ 75%]
tests/python/test_runtime.py::TestMemoryManager::test_manager_creation PASSED [ 78%]
tests/python/test_runtime.py::TestMemoryManager::test_allocation PASSED  [ 81%]
tests/python/test_runtime.py::TestMemoryManager::test_memory_plan PASSED [ 83%]
tests/python/test_runtime.py::TestExecutionPlan::test_execution_plan_creation PASSED [ 86%]
tests/python/test_runtime.py::TestGraphExecutor::test_executor_creation PASSED [ 89%]
tests/python/test_runtime.py::TestZenithEngine::test_engine_creation PASSED [ 91%]
tests/python/test_runtime.py::TestZenithEngine::test_compile_config PASSED [ 94%]
tests/python/test_runtime.py::TestZenithEngine::test_list_supported_ops PASSED [ 97%]
tests/python/test_runtime.py::TestIntegration::test_import_runtime PASSED [100%]

============================== 37 passed in 1.31s ==============================
============================================================
TRANSFORMER BENCHMARK - FP32 vs FP16 (Tensor Core)
============================================================

Batch=8, Seq=128, D=768
FP32: 5.57 ms
FP16: 0.91 ms (Tensor Core)
Speedup: 6.15x

============================================================
ZENITH INTEGRATION TEST - FINAL SUMMARY
============================================================

Module Tests:
  [OK] Core (GraphIR, DataType)
  [OK] Optimization Passes
  [OK] Quantization
  [OK] Runtime (ZenithEngine)
  [OK] Observability (Logger/Metrics)
  [OK] Monitoring (Prometheus)
  [OK] Serving (Triton)
  [OK] Error Handling

Performance:
  FP32: 5.57 ms
  FP16: 0.91 ms
  Speedup: 6.15x

============================================================
ALL ZENITH MODULES WORKING CORRECTLY!
============================================================

