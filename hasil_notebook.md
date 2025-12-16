# Step 1: Verify GPU is available
!nvidia-smi
Tue Dec 16 18:27:00 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   75C    P0             34W /   70W |     172MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
1:27 AM
[14]
# Step 2: Clone ZENITH repository
!git clone https://github.com/vibeswithkk/ZENITH.git
Cloning into 'ZENITH'...
remote: Enumerating objects: 91, done.
remote: Counting objects: 100% (91/91), done.
remote: Compressing objects: 100% (77/77), done.
remote: Total 91 (delta 11), reused 88 (delta 8), pack-reused 0 (from 0)
Receiving objects: 100% (91/91), 7.72 MiB | 13.54 MiB/s, done.
Resolving deltas: 100% (11/11), done.
/content/ZENITH/ZENITH
1:27 AM
[15]
# Step 3: Install Python dependencies
!pip install numpy pytest onnx
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.0.2)
Requirement already satisfied: pytest in /usr/local/lib/python3.12/dist-packages (8.4.2)
Requirement already satisfied: onnx in /usr/local/lib/python3.12/dist-packages (1.20.0)
Requirement already satisfied: iniconfig>=1 in /usr/local/lib/python3.12/dist-packages (from pytest) (2.3.0)
Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from pytest) (25.0)
Requirement already satisfied: pluggy<2,>=1.5 in /usr/local/lib/python3.12/dist-packages (from pytest) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in /usr/local/lib/python3.12/dist-packages (from pytest) (2.19.2)
Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (5.29.5)
Requirement already satisfied: typing_extensions>=4.7.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (4.15.0)
Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from onnx) (0.5.4)
1:27 AM
[16]
# Step 4: Run Python unit tests
!python -m pytest tests/python/ -v --tb=short 2>&1 | tail -40
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

============================= 130 passed in 13.44s =============================
1:27 AM
[17]
# Step 5: Verify CUDA toolkit is available
!nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_02:18:23_PDT_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
1:27 AM
[18]
# Step 6: Test CUDA kernel compilation
# Create a simple test file to verify CUDA compilation works
CUDA test file created.
1:27 AM
[19]
# Step 7: Compile and run CUDA test
!nvcc -o cuda_test cuda_test.cu && ./cuda_test
CUDA Devices: 1
Device: Tesla T4
Compute Capability: 7.5
Total Memory: 15.83 GB
Multiprocessors: 40
Vector Add Test: FAILED
1:27 AM
[20]
# Step 8: Test ZENITH CUDA kernels compilation
# Compile the actual ZENITH CUDA kernels
ZENITH CUDA kernels found. Attempting compilation...
1:27 AM
[21]
# Step 9: Test Mixed Precision with PyTorch (uses CUDA)
import torch
PyTorch version: 2.9.0+cu126
CUDA available: True
CUDA device: Tesla T4
FP32 vs FP16 mean absolute diff: 0.008935
Mixed precision test: PASSED
1:27 AM
[22]
# Step 10: Test ZENITH Quantization Module
import sys
Quantization Results:
  layer1: dtype=int8, range=[-127, 117]
  layer2: dtype=int8, range=[-127, 124]

INT8 Quantization test: PASSED
1:27 AM
[23]
# Step 11: Test Auto-tuner with caching
from zenith.optimization import KernelAutotuner, TuningConfig, SearchSpace
Best params: {'tile': 16}
Best time: 0.0005 ms

Auto-tuner test: PASSED
1:27 AM
[24]
# Step 12: Full test suite summary
!python -m pytest tests/python/ -v 2>&1 | grep -E '(PASSED|FAILED|passed|failed)'
tests/python/test_advanced_optimization.py::TestConvBNFusion::test_compute_fused_weights_basic PASSED [  0%]
tests/python/test_advanced_optimization.py::TestConvBNFusion::test_compute_fused_weights_no_conv_bias PASSED [  1%]
tests/python/test_advanced_optimization.py::TestConvBNFusion::test_fusion_numerical_correctness PASSED [  2%]
tests/python/test_advanced_optimization.py::TestConvBNFusion::test_fusion_pass_pattern_matching PASSED [  3%]
tests/python/test_advanced_optimization.py::TestLayoutTransform::test_nhwc_to_nchw_transform PASSED [  3%]
tests/python/test_advanced_optimization.py::TestLayoutTransform::test_nchw_to_nhwc_transform PASSED [  4%]
tests/python/test_advanced_optimization.py::TestLayoutTransform::test_roundtrip_transform PASSED [  5%]
tests/python/test_advanced_optimization.py::TestLayoutTransform::test_layout_inference PASSED [  6%]
tests/python/test_advanced_optimization.py::TestLayoutTransform::test_backend_preferences PASSED [  6%]
tests/python/test_advanced_optimization.py::TestProfiler::test_profiler_basic PASSED [  7%]
tests/python/test_advanced_optimization.py::TestProfiler::test_profiler_export_json PASSED [  8%]
tests/python/test_advanced_optimization.py::TestProfiler::test_profiler_export_csv PASSED [  9%]
tests/python/test_advanced_optimization.py::TestProfiler::test_profiler_summary PASSED [ 10%]
tests/python/test_advanced_optimization.py::TestBenchmark::test_benchmark_basic PASSED [ 10%]
tests/python/test_advanced_optimization.py::TestBenchmark::test_benchmark_comparison PASSED [ 11%]
tests/python/test_advanced_optimization.py::TestBenchmark::test_benchmark_export PASSED [ 12%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_dimensions[4-4-4] PASSED [ 13%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_dimensions[8-16-8] PASSED [ 13%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_dimensions[16-8-32] PASSED [ 14%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_dimensions[32-64-16] PASSED [ 15%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_dimensions[64-64-64] PASSED [ 16%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_dimensions[128-128-128] PASSED [ 16%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_identity PASSED [ 17%]
tests/python/test_numerical_accuracy.py::TestMatMulAccuracy::test_matmul_zeros PASSED [ 18%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_relu_accuracy[16] PASSED [ 19%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_relu_accuracy[64] PASSED [ 20%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_relu_accuracy[256] PASSED [ 20%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_relu_accuracy[1024] PASSED [ 21%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_sigmoid_accuracy[16] PASSED [ 22%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_sigmoid_accuracy[64] PASSED [ 23%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_sigmoid_accuracy[256] PASSED [ 23%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_sigmoid_accuracy[1024] PASSED [ 24%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_tanh_accuracy[16] PASSED [ 25%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_tanh_accuracy[64] PASSED [ 26%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_tanh_accuracy[256] PASSED [ 26%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_tanh_accuracy[1024] PASSED [ 27%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_relu_edge_cases PASSED [ 28%]
tests/python/test_numerical_accuracy.py::TestActivationAccuracy::test_sigmoid_edge_cases PASSED [ 29%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_add_accuracy[16] PASSED [ 30%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_add_accuracy[64] PASSED [ 30%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_add_accuracy[256] PASSED [ 31%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_add_accuracy[1024] PASSED [ 32%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_sum_accuracy[16] PASSED [ 33%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_sum_accuracy[64] PASSED [ 33%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_sum_accuracy[256] PASSED [ 34%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_mean_accuracy[16] PASSED [ 35%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_mean_accuracy[64] PASSED [ 36%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_mean_accuracy[256] PASSED [ 36%]
tests/python/test_numerical_accuracy.py::TestElementWiseAccuracy::test_max_min_accuracy PASSED [ 37%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_basic[8] PASSED [ 38%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_basic[16] PASSED [ 39%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_basic[64] PASSED [ 40%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_basic[128] PASSED [ 40%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_sums_to_one PASSED [ 41%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_non_negative PASSED [ 42%]
tests/python/test_numerical_accuracy.py::TestSoftmaxAccuracy::test_softmax_monotonic PASSED [ 43%]
tests/python/test_numerical_accuracy.py::TestConv2DAccuracy::test_conv2d_identity_kernel PASSED [ 43%]
tests/python/test_numerical_accuracy.py::TestConv2DAccuracy::test_conv2d_simple PASSED [ 44%]
tests/python/test_numerical_accuracy.py::TestConv2DAccuracy::test_conv2d_with_padding PASSED [ 45%]
tests/python/test_numerical_accuracy.py::TestConv2DAccuracy::test_conv2d_multi_channel PASSED [ 46%]
tests/python/test_numerical_accuracy.py::TestPoolingAccuracy::test_maxpool2d_basic PASSED [ 46%]
tests/python/test_numerical_accuracy.py::TestPoolingAccuracy::test_maxpool2d_multi_channel PASSED [ 47%]
tests/python/test_numerical_accuracy.py::TestNumericalStability::test_matmul_large_values PASSED [ 48%]
tests/python/test_numerical_accuracy.py::TestNumericalStability::test_matmul_small_values PASSED [ 49%]
tests/python/test_numerical_accuracy.py::TestNumericalStability::test_softmax_numerical_stability PASSED [ 50%]
tests/python/test_onnx_adapter.py::TestONNXAdapterBasic::test_adapter_name PASSED [ 50%]
tests/python/test_onnx_adapter.py::TestONNXAdapterBasic::test_adapter_is_available_without_onnx PASSED [ 51%]
tests/python/test_onnx_adapter.py::TestONNXAdapterBasic::test_adapter_not_available_when_onnx_missing PASSED [ 52%]
tests/python/test_onnx_adapter.py::TestONNXAdapterConversion::test_from_model_with_model_proto PASSED [ 53%]
tests/python/test_onnx_adapter.py::TestONNXAdapterConversion::test_from_model_raises_for_invalid_type PASSED [ 53%]
tests/python/test_onnx_adapter.py::TestONNXAdapterConversion::test_from_model_with_string_path PASSED [ 54%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_default_construction PASSED [ 55%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_add_input PASSED [ 56%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_add_output PASSED [ 56%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_add_node PASSED [ 57%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_add_constant PASSED [ 58%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_find_nodes_by_op PASSED [ 59%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_validate_empty_graph PASSED [ 60%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_validate_no_inputs PASSED [ 60%]
tests/python/test_onnx_adapter.py::TestGraphIRConstruction::test_graphir_validate_success PASSED [ 61%]
tests/python/test_onnx_adapter.py::TestTensorDescriptor::test_tensor_descriptor_creation PASSED [ 62%]
tests/python/test_onnx_adapter.py::TestTensorDescriptor::test_tensor_size_bytes PASSED [ 63%]
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
============================= 130 passed in 10.67s =============================