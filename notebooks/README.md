# Zenith Integration Test Notebooks

## integration_test_colab.ipynb

Comprehensive test for validating Zenith implementation on Google Colab with GPU.

### Tests Included:

1. **Hardware Backend Layer**
   - CUDA detection and memory operations
   - CPU backend fallback
   - Device properties querying

2. **Framework Trinity Adapters**
   - PyTorch adapter with torch.compile
   - TensorFlow adapter with tf.function
   - JAX adapter with jit compilation

3. **Memory Management**
   - Allocation/deallocation
   - Host-to-device transfers
   - Device-to-host transfers

4. **ONNX Export**
   - Model conversion to ONNX
   - ONNX validation

### How to Use:

1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/notebooks/integration_test_colab.ipynb)

2. Enable GPU runtime: Runtime > Change runtime type > GPU

3. Run all cells sequentially

### Expected Results (with GPU):

```
Hardware Backends:
  Available: ['cpu', 'cuda']
  CUDA: PASSED
  CPU:  PASSED

Framework Adapters:
  PyTorch:    PASSED
  TensorFlow: PASSED
  JAX:        PASSED
```
