# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Colab Validation Script

This script validates the installation of pyzenith and verifies CUDA acceleration
on Google Colab or any other GPU environment.
"""

import sys
import time
import numpy as np


def print_header(msg):
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def check_installation():
    print_header("Checking Installation")
    try:
        import zenith

        print(f"✅ pyzenith imported successfully")
        print(f"   Version: {zenith.__version__}")
        print(f"   Path: {zenith.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pyzenith: {e}")
        return False


def check_cuda():
    print_header("Checking CUDA Availability")

    # Check 1: ExecutionContext
    try:
        from zenith.execution import ExecutionContext

        ctx = ExecutionContext(device="cuda")
        print(f"   ExecutionContext created with device='cuda'")
    except Exception as e:
        print(f"⚠️  ExecutionContext creation failed: {e}")

    # Check 2: Native bindings
    try:
        from zenith import _zenith_core

        if hasattr(_zenith_core, "cuda"):
            print(f"✅ _zenith_core.cuda module found")
        else:
            print(f"⚠️  _zenith_core.cuda module NOT found")

        if hasattr(_zenith_core, "is_cuda_available"):
            available = _zenith_core.is_cuda_available()
            print(f"   _zenith_core.is_cuda_available(): {available}")
        else:
            # Try checking if cuda module is not None
            if hasattr(_zenith_core, "cuda") and _zenith_core.cuda is not None:
                print(f"   Implied availability via module presence")
            else:
                print(f"⚠️  Unable to determine availability via direct check")

    except ImportError:
        print(f"❌ _zenith_core native bindings NOT found")


def run_performance_test():
    print_header("Running Performance Test (CPU vs GPU)")

    from zenith.core import GraphIR, TensorDescriptor, Shape, DataType
    from zenith.execution import ONNXInterpreter

    # 1. Create a computation graph (MatMul + Relu)
    # C = Relu(MatMul(A, B))
    graph = GraphIR(name="benchmark_graph")

    M, K, N = 1024, 1024, 1024
    print(f"   Problem size: MatMul [{M}x{K}] * [{K}x{N}]")

    # Add inputs
    graph.add_input(TensorDescriptor("A", Shape([M, K]), DataType.Float32))
    graph.add_input(TensorDescriptor("B", Shape([K, N]), DataType.Float32))
    graph.add_output(TensorDescriptor("Y", Shape([M, N]), DataType.Float32))

    # Add nodes (Manual construction for test)
    # Note: In real usage, adapters handle this. We simulate the internal structure.
    from zenith.core import Node

    matmul_node = Node(
        name="MatMul_0",
        op_type="MatMul",
        inputs=["A", "B"],
        outputs=["temp"],
    )
    graph.add_node(matmul_node)

    relu_node = Node(
        name="Relu_0",
        op_type="Relu",
        inputs=["temp"],
        outputs=["Y"],
    )
    graph.add_node(relu_node)

    # Generate data
    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)

    # 2. Run on CPU
    print("\n   [Running on CPU]")
    try:
        interpreter_cpu = ONNXInterpreter(graph, device="cpu")
        start = time.perf_counter()
        res_cpu = interpreter_cpu(A=A_data, B=B_data)
        end = time.perf_counter()
        cpu_time = (end - start) * 1000
        print(f"   CPU Time: {cpu_time:.2f} ms")
    except Exception as e:
        print(f"❌ CPU run failed: {e}")
        return

    # 3. Run on GPU
    print("\n   [Running on GPU]")
    try:
        # Check if we can actually run on GPU
        from zenith import _zenith_core

        has_cuda = hasattr(_zenith_core, "cuda") and _zenith_core.cuda is not None

        if not has_cuda:
            print("⚠️  Skipping GPU test: CUDA not available")
            return

        interpreter_gpu = ONNXInterpreter(graph, device="cuda")

        # Warmup
        interpreter_gpu(A=A_data, B=B_data)

        # Timed run
        start = time.perf_counter()
        res_gpu = interpreter_gpu(A=A_data, B=B_data)
        if hasattr(_zenith_core.cuda, "sync"):
            _zenith_core.cuda.sync()
        end = time.perf_counter()
        gpu_time = (end - start) * 1000
        print(f"   GPU Time: {gpu_time:.2f} ms")

        # Speedup
        print(f"   Speedup: {cpu_time / gpu_time:.2f}x")

        # Verify correctness
        diff = np.max(np.abs(res_cpu["Y"] - res_gpu["Y"]))
        print(f"   Max Diff CPU vs GPU: {diff:.6f}")
        if diff < 1e-4:
            print("✅ Results Match")
        else:
            print("❌ Results Mismatch")

    except Exception as e:
        print(f"❌ GPU run failed: {e}")


if __name__ == "__main__":
    if not check_installation():
        sys.exit(1)

    check_cuda()
    run_performance_test()

    print_header("Test Complete")
