# Step 1: Check CPU info
!cat /proc/cpuinfo | grep 'model name' | head -1
model name	: Intel(R) Xeon(R) CPU @ 2.20GHz
cpu cores	: 1
avx
avx2
fma
sse4
1:32 AM
[2]
# Step 2: Clone ZENITH
!git clone https://github.com/vibeswithkk/ZENITH.git
Cloning into 'ZENITH'...
remote: Enumerating objects: 98, done.
remote: Counting objects: 100% (98/98), done.
remote: Compressing objects: 100% (83/83), done.
remote: Total 98 (delta 14), reused 93 (delta 9), pack-reused 0 (from 0)
Receiving objects: 100% (98/98), 7.73 MiB | 17.17 MiB/s, done.
Resolving deltas: 100% (14/14), done.
/content/ZENITH
1:32 AM
[3]
# Step 3: Install dependencies
!pip install numpy pytest onnx
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.0.2)
Requirement already satisfied: pytest in /usr/local/lib/python3.12/dist-packages (8.4.2)
Collecting onnx
  Downloading onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
Requirement already satisfied: iniconfig>=1 in /usr/local/lib/python3.12/dist-packages (from pytest) (2.3.0)
Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from pytest) (25.0)
Requirement already satisfied: pluggy<2,>=1.5 in /usr/local/lib/python3.12/dist-packages (from pytest) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in /usr/local/lib/python3.12/dist-packages (from pytest) (2.19.2)
Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (5.29.5)
Requirement already satisfied: typing_extensions>=4.7.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (4.15.0)
Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from onnx) (0.5.4)
Downloading onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (18.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.1/18.1 MB 40.3 MB/s eta 0:00:00
Installing collected packages: onnx
Successfully installed onnx-1.20.0
1:34 AM
[4]
# Step 4: Run full test suite (CPU only)
!python -m pytest tests/python/ -v --tb=short 2>&1 | tail -50
tests/python/test_onnx_adapter.py::TestTensorDescriptor::test_tensor_is_valid PASSED [ 63%]
tests/python/test_onnx_adapter.py::TestNodeOperations::test_node_creation PASSED [ 64%]
tests/python/test_onnx_adapter.py::TestNodeOperations::test_node_is_op PASSED [ 65%]
tests/python/test_onnx_adapter.py::TestNodeOperations::test_node_clone PASSED [ 66%]
tests/python/test_onnx_adapter.py::TestEndToEndConversion::test_simple_model_conversion PASSED [ 66%]
tests/python/test_onnx_adapter.py::TestEndToEndConversion::test_relu_model_conversion PASSED [ 67%]
tests/python/test_onnx_adapter.py::TestEndToEndConversion::test_multi_node_model_conversion PASSED [ 68%]
tests/python/test_onnx_adapter.py::TestEndToEndConversion::test_model_with_dynamic_shapes PASSED [ 69%]
tests/python/test_onnx_adapter.py::TestDataTypeConversion::test_dtype_mapping PASSED [ 70%]
tests/python/test_optimization.py::TestConstantFoldingPass::test_pass_name PASSED [ 70%]
tests/python/test_optimization.py::TestConstantFoldingPass::test_no_modification_without_constants PASSED [ 71%]
tests/python/test_optimization.py::TestConstantFoldingPass::test_fold_constant_operation PASSED [ 72%]
tests/python/test_optimization.py::TestConstantFoldingPass::test_non_foldable_op_unchanged PASSED [ 73%]
tests/python/test_optimization.py::TestDeadCodeEliminationPass::test_pass_name PASSED [ 73%]
tests/python/test_optimization.py::TestDeadCodeEliminationPass::test_no_dead_code PASSED [ 74%]
tests/python/test_optimization.py::TestDeadCodeEliminationPass::test_remove_dead_node PASSED [ 75%]
tests/python/test_optimization.py::TestDeadCodeEliminationPass::test_chain_dead_code_removal PASSED [ 76%]
tests/python/test_optimization.py::TestOperatorFusionPass::test_pass_name PASSED [ 76%]
tests/python/test_optimization.py::TestOperatorFusionPass::test_no_fusion_possible PASSED [ 77%]
tests/python/test_optimization.py::TestOperatorFusionPass::test_conv_relu_fusion PASSED [ 78%]
tests/python/test_optimization.py::TestOperatorFusionPass::test_no_fusion_with_multiple_consumers PASSED [ 79%]
tests/python/test_optimization.py::TestPassManager::test_add_pass PASSED [ 80%]
tests/python/test_optimization.py::TestPassManager::test_run_all_passes PASSED [ 80%]
tests/python/test_optimization.py::TestDefaultPassManager::test_create_default_manager PASSED [ 81%]
tests/python/test_optimization.py::TestDefaultPassManager::test_optimize_graph_level_0 PASSED [ 82%]
tests/python/test_optimization.py::TestDefaultPassManager::test_optimize_graph_level_2 PASSED [ 83%]
tests/python/test_phase3_quantization.py::TestAutotuner::test_search_space_definition PASSED [ 83%]
tests/python/test_phase3_quantization.py::TestAutotuner::test_search_space_sampling PASSED [ 84%]
tests/python/test_phase3_quantization.py::TestAutotuner::test_grid_search PASSED [ 85%]
tests/python/test_phase3_quantization.py::TestAutotuner::test_random_search PASSED [ 86%]
tests/python/test_phase3_quantization.py::TestAutotuner::test_tuning_cache PASSED [ 86%]
tests/python/test_phase3_quantization.py::TestAutotuner::test_kernel_autotuner PASSED [ 87%]
tests/python/test_phase3_quantization.py::TestMixedPrecision::test_precision_policy_fp16 PASSED [ 88%]
tests/python/test_phase3_quantization.py::TestMixedPrecision::test_precision_policy_bf16 PASSED [ 89%]
tests/python/test_phase3_quantization.py::TestMixedPrecision::test_dynamic_loss_scaler PASSED [ 90%]
tests/python/test_phase3_quantization.py::TestMixedPrecision::test_mixed_precision_manager_cast PASSED [ 90%]
tests/python/test_phase3_quantization.py::TestMixedPrecision::test_bf16_simulation PASSED [ 91%]
tests/python/test_phase3_quantization.py::TestMixedPrecision::test_precision_safety_check PASSED [ 92%]
tests/python/test_phase3_quantization.py::TestQuantization::test_quantization_params PASSED [ 93%]
tests/python/test_phase3_quantization.py::TestQuantization::test_minmax_calibrator PASSED [ 93%]
tests/python/test_phase3_quantization.py::TestQuantization::test_percentile_calibrator PASSED [ 94%]
tests/python/test_phase3_quantization.py::TestQuantization::test_quantizer_static PASSED [ 95%]
tests/python/test_phase3_quantization.py::TestQuantization::test_qat_simulator PASSED [ 96%]
tests/python/test_phase3_quantization.py::TestQuantization::test_quantization_error_measurement PASSED [ 96%]
tests/python/test_phase3_quantization.py::TestBackendAvailability::test_rocm_backend_unavailable PASSED [ 97%]
tests/python/test_phase3_quantization.py::TestBackendAvailability::test_oneapi_backend_unavailable PASSED [ 98%]
tests/python/test_phase3_quantization.py::TestEndToEndQuantization::test_full_quantization_pipeline PASSED [ 99%]
tests/python/test_phase3_quantization.py::TestEndToEndQuantization::test_dynamic_quantization PASSED [100%]

============================= 130 passed in 17.18s =============================
1:34 AM
[5]
# Step 5: Check C++ compiler for CPU backend
!g++ --version
g++ (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

1:34 AM
[6]
# Step 6: Check AVX2 support
avx2_test = '''
AVX2 test result: 3.000000
AVX2 Support: PASSED
1:34 AM
[7]
# Step 7: Check FMA support
fma_test = '''
FMA test result: 10.000000 (expected 10.0)
FMA Support: PASSED
1:34 AM
[8]
# Step 8: CPU MatMul benchmark (NumPy using BLAS)
import numpy as np
CPU MatMul Benchmark (NumPy/BLAS):
==================================================
Size 256x256: 2.68 ms, 12.5 GFLOPS
Size 512x512: 14.89 ms, 18.0 GFLOPS
Size 1024x1024: 90.20 ms, 23.8 GFLOPS
Size 2048x2048: 470.09 ms, 36.5 GFLOPS

CPU Benchmark: PASSED
1:34 AM
[9]
# Step 9: Test ZENITH optimization passes
import sys
Graph name: cpu_test
Passes applied: {'constant_folding': 0, 'dead_code_elimination': 0}
Optimization passes on CPU: PASSED
1:34 AM
[10]
# Step 10: Test Quantization (CPU-focused)
from zenith.optimization import Quantizer, QuantizationMode, CalibrationMethod
INT8 Quantization Results (CPU):
  fc1: dtype=int8, shape=(64, 128)
  fc2: dtype=int8, shape=(128, 10)

INT8 Quantization (CPU): PASSED
1:34 AM
[11]
# Step 11: Test Auto-tuner (CPU)
from zenith.optimization import KernelAutotuner, TuningConfig, SearchSpace
Best CPU params: {'tile_size': 256, 'unroll': 4}
Best time: 0.0005 ms
CPU Auto-tuner: PASSED
1:34 AM
[12]
# Step 12: Full test summary
!python -m pytest tests/python/ -v 2>&1 | grep -E '(passed|failed)' | tail -5
============================= 130 passed in 14.94s =============================