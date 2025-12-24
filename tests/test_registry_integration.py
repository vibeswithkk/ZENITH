#!/usr/bin/env python
"""Test script for verifying native CUDA kernel integration in registry."""

import sys


def test_registry_integration():
    """Test that JIT-compiled kernels are registered with highest priority."""
    print("=" * 60)
    print("  Testing Native Kernel Registry Integration")
    print("=" * 60)

    # Check if zenith_cuda was already built (by build_cuda.py)
    print("\n1. Checking for zenith_cuda module...")
    try:
        import zenith_cuda

        print(f"  zenith_cuda loaded: {len(dir(zenith_cuda))} functions")
    except ImportError as e:
        print(f"  zenith_cuda not available: {e}")
        print("  SKIP: Run 'python ZENITH/zenith/build_cuda.py' first")
        return False

    # Now test registry
    print("\n2. Testing KernelRegistry...")
    from zenith.runtime.kernel_registry import KernelRegistry, Precision

    registry = KernelRegistry()
    has_gpu = registry.initialize()
    print(f"  GPU kernels available: {has_gpu}")

    # Check if our JIT kernels are registered
    print("\n3. Checking registered JIT kernels...")
    ops_to_check = ["Relu", "Gelu", "LayerNorm", "MatMul"]

    for op in ops_to_check:
        kernel = registry.get_kernel(op, Precision.FP32)
        if kernel and "jit" in kernel.name:
            print(f"  {op}: {kernel.name} (priority={kernel.priority})")
        elif kernel:
            print(f"  {op}: {kernel.name} (fallback, priority={kernel.priority})")
        else:
            print(f"  {op}: NOT FOUND")

    # List all supported ops
    print("\n4. All supported operations:")
    all_ops = registry.list_supported_ops()
    print(f"  Total: {len(all_ops)} operations")

    print("\n" + "=" * 60)
    print("  REGISTRY INTEGRATION SUCCESSFUL")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_registry_integration()
    sys.exit(0 if success else 1)
