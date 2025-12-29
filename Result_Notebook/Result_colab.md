  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 693.4/693.4 kB 23.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.1/133.1 kB 11.4 MB/s eta 0:00:00
Setup complete! Please restart runtime if this is first run.
PyTorch version: 2.9.0+cu126
Zenith version: 0.1.4
CUDA available: True
GPU: Tesla T4
Model created on: cuda
Total parameters: 421,642
Training samples: 60,000
Test samples: 10,000
Training with Pure PyTorch...
==================================================
Epoch 1: Train Loss=0.1591, Train Acc=95.06%, Test Loss=0.0460, Test Acc=98.50%
Epoch 2: Train Loss=0.0535, Train Acc=98.42%, Test Loss=0.0315, Test Acc=98.97%
Epoch 3: Train Loss=0.0388, Train Acc=98.80%, Test Loss=0.0275, Test Acc=99.09%
==================================================
Training complete in 64.86s
Final Test Accuracy: 99.09%
Benchmarking PyTorch Native Inference...
PyTorch Native: 8.019 +/- 0.020 ms/batch
  Range: [7.981, 8.049] ms
============================================================
APPLYING ZENITH OPTIMIZATION
============================================================

[Method 1] torch.compile with Zenith backend...
Model compiled with Zenith backend
Compiled Model Accuracy: 99.09%

[Method 2] Using @ztorch.compile decorator...
Decorated function output shape: torch.Size([1000, 10])
Benchmarking Zenith-Optimized Inference...
Zenith Optimized: 8.093 +/- 0.016 ms/batch
  Range: [8.072, 8.123] ms

============================================================
PERFORMANCE COMPARISON
============================================================

PyTorch Native:    8.019 ms/batch
Zenith Optimized:  8.093 ms/batch

Speedup: 0.99x
Improvement: -0.9%
============================================================
ONNX EXPORT WITH ZENITH
============================================================
W1221 10:26:32.581000 1422 torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 17 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
[torch.onnx] Obtain model graph for `MNISTClassifier([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `MNISTClassifier([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
WARNING:onnxscript.version_converter:The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 17).
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
ONNX model saved to: /tmp/mnist_classifier.onnx
ONNX model size: 1,688,649 bytes
ONNX validation: PASSED

Verifying ONNX model with ONNX Runtime...
ONNX Runtime output shape: (1, 10)
ONNX Runtime inference: PASSED
============================================================
ZENITH GRAPHIR CONVERSION
============================================================
Graph name: pytorch_exported_model
Input tensors: 1
Output tensors: 1

======================================================================
ZENITH x PYTORCH: MACHINE LEARNING DEMO SUMMARY
======================================================================

+------------------------------------------------------------------+
|                     WORKFLOW DEMONSTRATION                       |
+------------------------------------------------------------------+

  1. MODEL DEFINITION:   Pure PyTorch (nn.Module)           [CHECK]
  2. DATA LOADING:       Pure PyTorch (DataLoader)          [CHECK]
  3. TRAINING:           Pure PyTorch (optimizer.step())    [CHECK]
  4. OPTIMIZATION:       Zenith torch.compile backend       [CHECK]
  5. ONNX EXPORT:        Zenith to_onnx()                   [CHECK]
  6. GRAPHIR CONVERT:    Zenith PyTorchAdapter              [CHECK]

+------------------------------------------------------------------+
|                     KEY TAKEAWAYS                                |
+------------------------------------------------------------------+

  - Zenith does NOT replace PyTorch for training
  - Zenith ENHANCES inference performance
  - Zenith SIMPLIFIES production deployment (ONNX)
  - Zenith INTEGRATES seamlessly with existing code
  - Your PyTorch knowledge remains 100% applicable


Model Accuracy:     99.09%
Native Inference:   8.019 ms/batch
Zenith Inference:   8.093 ms/batch
Performance Gain:   0.99x faster

======================================================================
Zenith: Your PyTorch Companion for Production ML
======================================================================
