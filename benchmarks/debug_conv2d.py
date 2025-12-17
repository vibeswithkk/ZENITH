# Debug script for conv2d - captures detailed error messages
# Run in Colab after build: python benchmarks/debug_conv2d.py

import sys

sys.path.insert(0, "build/python")

import numpy as np
import traceback

print("=" * 70)
print("ZENITH CONV2D DEBUG TEST")
print("=" * 70)

# Step 1: Check if CUDA is available
print("\n[1/4] Checking CUDA availability...")
try:
    from zenith._zenith_core import backends

    available = backends.list_available()
    print(f"  Available backends: {available}")
    print(f"  CUDA: {backends.is_cuda_available()}")
    print(f"  cuDNN: {backends.is_cudnn_available()}")
    if backends.is_cudnn_available():
        print(f"  cuDNN version: {backends.get_cudnn_version()}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Import CUDA module
print("\n[2/4] Importing CUDA module...")
try:
    from zenith._zenith_core import cuda

    print(f"  SUCCESS: cuda module imported")
    print(f"  has_cudnn: {cuda.has_cudnn()}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test basic operations first
print("\n[3/4] Testing basic CUDA operations...")

# Test ReLU first (simpler)
try:
    x = np.random.randn(1, 16, 4, 4).astype(np.float32)
    y = cuda.relu(x)
    print(f"  ReLU: PASS - input {x.shape} -> output {y.shape}")
except Exception as e:
    print(f"  ReLU: FAIL - {e}")
    traceback.print_exc()

# Test matmul
try:
    A = np.random.randn(32, 64).astype(np.float32)
    B = np.random.randn(64, 128).astype(np.float32)
    C = cuda.matmul(A, B)
    print(f"  MATMUL: PASS - {A.shape} @ {B.shape} = {C.shape}")
except Exception as e:
    print(f"  MATMUL: FAIL - {e}")
    traceback.print_exc()

# Step 4: Test Conv2D with various configurations
print("\n[4/4] Testing Conv2D...")

test_cases = [
    # (batch, in_ch, out_ch, H, W, K, stride, padding)
    (1, 3, 16, 8, 8, 3, 1, 1),  # Very small
    (1, 16, 16, 8, 8, 3, 1, 1),  # Small square
    (1, 16, 32, 14, 14, 3, 1, 1),  # Medium
    (1, 32, 64, 14, 14, 3, 1, 1),  # Larger
    (1, 3, 64, 28, 28, 3, 1, 1),  # Like your test
    (1, 64, 64, 28, 28, 3, 1, 1),  # Larger test
    (1, 3, 64, 224, 224, 7, 2, 3),  # ResNet conv1
]

for batch, in_ch, out_ch, H, W, K, stride, pad in test_cases:
    try:
        x = np.random.randn(batch, in_ch, H, W).astype(np.float32)
        w = np.random.randn(out_ch, in_ch, K, K).astype(np.float32)

        # Make sure arrays are contiguous
        x = np.ascontiguousarray(x)
        w = np.ascontiguousarray(w)

        print(
            f"\n  Testing: input[{batch},{in_ch},{H},{W}] * weight[{out_ch},{in_ch},{K},{K}] stride={stride} pad={pad}"
        )
        print(
            f"    x contiguous: {x.flags['C_CONTIGUOUS']}, w contiguous: {w.flags['C_CONTIGUOUS']}"
        )

        y = cuda.conv2d(x, w, stride=stride, padding=pad)

        # Expected output shape
        H_out = (H + 2 * pad - K) // stride + 1
        W_out = (W + 2 * pad - K) // stride + 1
        expected = (batch, out_ch, H_out, W_out)

        print(f"    PASS: output {y.shape}, expected {expected}")
        if y.shape != expected:
            print(f"    WARNING: Shape mismatch!")

    except Exception as e:
        print(f"    FAIL: {e}")
        traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
print("\nIf all tests fail, check:")
print("1. cuDNN version compatibility with CUDA 12.x")
print("2. GPU memory (run: nvidia-smi)")
print("3. Array memory layout (must be contiguous C-order)")
