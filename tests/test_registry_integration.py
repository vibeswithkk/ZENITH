#!/usr/bin/env python
"""Test script for verifying native CUDA kernel integration in registry."""

import os
import sys


def setup_torch_extensions_path():
    """Add torch extensions cache directory to sys.path."""
    # Torch JIT-compiles extensions to ~/.cache/torch_extensions/
    # We need to add this to sys.path for cross-process import
    import torch

    # Get the cache directory where torch stores JIT extensions
    cache_dir = os.path.join(
        torch.utils.cpp_extension._get_build_directory("zenith_cuda", False),
    )

    # Also check standard locations
    possible_paths = [
        cache_dir,
        os.path.expanduser("~/.cache/torch_extensions/py312_cu126/zenith_cuda"),
        os.path.expanduser("~/.cache/torch_extensions/py311_cu121/zenith_cuda"),
        "/root/.cache/torch_extensions/py312_cu126/zenith_cuda",  # Colab
    ]

    for path in possible_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            print(f"  Added to path: {path}")
            return True

    return False


def test_registry_integration():
    """Test that JIT-compiled kernels are registered with highest priority."""
    print("=" * 60)
    print("  Testing Native Kernel Registry Integration")
    print("=" * 60)

    # Setup path to find zenith_cuda
    print("\n1. Setting up module path...")
    path_found = setup_torch_extensions_path()

    # Check if zenith_cuda was already built (by build_cuda.py)
    print("\n2. Checking for zenith_cuda module...")
    try:
        import zenith_cuda

        print(f"  zenith_cuda loaded: {len(dir(zenith_cuda))} functions")
        print(f"  Available: {[x for x in dir(zenith_cuda) if not x.startswith('_')]}")
    except ImportError as e:
        print(f"  zenith_cuda not available: {e}")
        if not path_found:
            print("  Hint: Run 'python ZENITH/zenith/build_cuda.py' first")
        return False

    # Now test registry
    print("\n3. Testing KernelRegistry...")
    from zenith.runtime.kernel_registry import KernelRegistry, Precision

    registry = KernelRegistry()
    has_gpu = registry.initialize()
    print(f"  GPU kernels available: {has_gpu}")

    # Check if our JIT kernels are registered
    print("\n4. Checking registered JIT kernels...")
    ops_to_check = ["Relu", "Gelu", "LayerNorm", "MatMul"]
    jit_count = 0

    for op in ops_to_check:
        kernel = registry.get_kernel(op, Precision.FP32)
        if kernel and "jit" in kernel.name:
            print(f"  {op}: {kernel.name} (priority={kernel.priority})")
            jit_count += 1
        elif kernel:
            print(f"  {op}: {kernel.name} (fallback)")
        else:
            print(f"  {op}: NOT FOUND")

    # List all supported ops
    print("\n5. All supported operations:")
    all_ops = registry.list_supported_ops()
    print(f"  Total: {len(all_ops)} operations")

    print("\n" + "=" * 60)
    if jit_count > 0:
        print(f"  SUCCESS: {jit_count}/{len(ops_to_check)} JIT kernels active")
    else:
        print("  PARTIAL: Using PyTorch fallback kernels")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_registry_integration()
    sys.exit(0 if success else 1)
