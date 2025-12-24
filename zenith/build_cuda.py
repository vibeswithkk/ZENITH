#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Build script for Zenith Native CUDA Extension.

Usage (in Colab):
    !pip install torch --quiet  # Ensure torch is installed
    !python ZENITH/zenith/build_cuda.py

This will compile the native CUDA kernels and install the extension.
"""

import os
import sys
import subprocess


def ensure_torch():
    """Ensure PyTorch is available (bypass zenith.torch conflict)."""
    import importlib.util
    import importlib

    # Check if real torch is available (not zenith.torch)
    spec = importlib.util.find_spec("torch")

    # If torch spec points to our zenith/torch, it's not the real one
    if spec is None or (spec.origin and "zenith" in spec.origin):
        print("PyTorch not found or shadowed. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch", "--quiet"]
        )
        # Clear cached module
        if "torch" in sys.modules:
            del sys.modules["torch"]

    # Force reimport
    import torch as _torch

    return _torch


def build_cuda_extension():
    """Build the CUDA extension using torch.utils.cpp_extension."""
    torch = ensure_torch()
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    print("=" * 60)
    print("  Building Zenith Native CUDA Extension")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Get paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_src = os.path.join(root_dir, "core", "src")
    bindings = os.path.join(root_dir, "core", "bindings")
    include_dir = os.path.join(root_dir, "core", "include")

    # Source files
    sources = [
        os.path.join(bindings, "cuda_bindings.cpp"),
        os.path.join(core_src, "cuda_kernels.cu"),
        os.path.join(core_src, "flash_attention.cu"),
        os.path.join(core_src, "fused_kernels.cu"),
    ]

    # Check sources exist
    for src in sources:
        if not os.path.exists(src):
            print(f"WARNING: Source not found: {src}")

    print(f"\nCompiling {len(sources)} source files...")

    try:
        # Build with JIT compilation
        zenith_cuda = load(
            name="zenith_cuda",
            sources=sources,
            extra_include_paths=[include_dir],
            extra_cflags=["-O3", "-DZENITH_HAS_CUDA"],
            extra_cuda_cflags=[
                "-O3",
                "-DZENITH_HAS_CUDA",
                "--use_fast_math",
                "-gencode=arch=compute_70,code=sm_70",  # Volta
                "-gencode=arch=compute_75,code=sm_75",  # Turing (T4)
                "-gencode=arch=compute_80,code=sm_80",  # Ampere
            ],
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("  BUILD SUCCESSFUL")
        print("=" * 60)

        # Test
        print("\nTesting kernels...")
        x = torch.randn(2, 256, device="cuda")
        y = zenith_cuda.relu(x)
        print(f"  relu: OK (output shape: {y.shape})")

        y = zenith_cuda.gelu(x)
        print(f"  gelu: OK (output shape: {y.shape})")

        gamma = torch.ones(256, device="cuda")
        beta = torch.zeros(256, device="cuda")
        y = zenith_cuda.layernorm(x, gamma, beta)
        print(f"  layernorm: OK (output shape: {y.shape})")

        A = torch.randn(64, 128, device="cuda")
        B = torch.randn(128, 256, device="cuda")
        C = zenith_cuda.matmul(A, B)
        print(f"  matmul: OK (output shape: {C.shape})")

        print("\nAll kernels verified!")
        return True

    except Exception as e:
        print(f"\nBUILD FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = build_cuda_extension()
    sys.exit(0 if success else 1)
