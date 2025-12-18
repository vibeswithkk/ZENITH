#!/usr/bin/env python3
"""
Zenith CUDA Build & Test Script for Google Colab

Run this entire cell in Google Colab with GPU runtime enabled.
Make sure to: Runtime > Change runtime type > T4 GPU

Usage in Colab:
1. Enable GPU runtime
2. Copy and paste this entire script into a cell
3. Run the cell
"""

# ==============================================================================
# STEP 1: Check GPU Availability
# ==============================================================================
print("=" * 60)
print("STEP 1: Checking GPU Availability")
print("=" * 60)

import subprocess

result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print(f"GPU Found: {result.stdout.strip()}")
else:
    print(
        "ERROR: No GPU found! Enable GPU runtime: Runtime > Change runtime type > T4 GPU"
    )
    raise SystemExit(1)

result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print(
    f"CUDA Compiler: {result.stdout.split('release')[-1].split(',')[0].strip() if result.returncode == 0 else 'Not found'}"
)

# ==============================================================================
# STEP 2: Clone Repository
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 2: Cloning Zenith Repository")
print("=" * 60)

import os

os.chdir("/content")

# Remove if exists
subprocess.run(["rm", "-rf", "ZENITH"], check=False)

# Clone
result = subprocess.run(
    ["git", "clone", "https://github.com/vibeswithkk/ZENITH.git"],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(f"Clone error: {result.stderr}")
    raise SystemExit(1)
print("Repository cloned successfully!")

os.chdir("/content/ZENITH")

# ==============================================================================
# STEP 3: Install Dependencies
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: Installing Dependencies")
print("=" * 60)

subprocess.run(["pip", "install", "-q", "pybind11", "numpy", "pytest"], check=True)
print("Dependencies installed!")

# Check pybind11
result = subprocess.run(
    ["python3", "-c", 'import pybind11; print(f"pybind11: {pybind11.__version__}")'],
    capture_output=True,
    text=True,
)
print(result.stdout.strip())

# ==============================================================================
# STEP 4: Configure CMake with CUDA
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4: Configuring CMake with CUDA")
print("=" * 60)

subprocess.run(["mkdir", "-p", "build"], check=True)
os.chdir("/content/ZENITH/build")

cmake_result = subprocess.run(
    [
        "cmake",
        "..",
        "-DZENITH_ENABLE_CUDA=ON",
        "-DZENITH_BUILD_PYTHON=ON",
        "-DZENITH_BUILD_TESTS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=75",  # Tesla T4 architecture
    ],
    capture_output=True,
    text=True,
)

print(cmake_result.stdout)
if cmake_result.returncode != 0:
    print(f"CMake Error: {cmake_result.stderr}")
    raise SystemExit(1)

# ==============================================================================
# STEP 5: Build Zenith Core
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building Zenith Core with CUDA")
print("=" * 60)

# Get number of CPUs
import multiprocessing

num_cpus = multiprocessing.cpu_count()
print(f"Building with {num_cpus} parallel jobs...")

build_result = subprocess.run(
    ["cmake", "--build", ".", f"-j{num_cpus}"], capture_output=True, text=True
)
print(build_result.stdout)
if build_result.returncode != 0:
    print(f"Build Error: {build_result.stderr}")
    raise SystemExit(1)

print("Build completed successfully!")

# ==============================================================================
# STEP 6: Install Python Module
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 6: Installing Python Module")
print("=" * 60)

os.chdir("/content/ZENITH")

# Copy the built module
import shutil
import glob

so_files = glob.glob("build/python/zenith/_zenith_core*.so")
if so_files:
    shutil.copy(so_files[0], "zenith/")
    print(f"Copied: {so_files[0]} -> zenith/")
else:
    print("ERROR: Built module not found!")
    raise SystemExit(1)

# Verify import
result = subprocess.run(
    [
        "python3",
        "-c",
        """
import sys
sys.path.insert(0, "/content/ZENITH")
from zenith._zenith_core import backends, kernels
print("Module imported successfully!")
print(f"Available backends: {backends.list_available()}")
print(f"CUDA available: {backends.is_cuda_available()}")
""",
    ],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(f"Import Error: {result.stderr}")

# ==============================================================================
# STEP 7: Run Tests
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 7: Running Python Tests")
print("=" * 60)

os.chdir("/content/ZENITH")
test_result = subprocess.run(
    ["python3", "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
    capture_output=True,
    text=True,
    timeout=300,
)
# Print last 50 lines
lines = test_result.stdout.strip().split("\n")
print("\n".join(lines[-50:]))

# ==============================================================================
# STEP 8: Test CUDA Operations
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 8: Testing CUDA Operations")
print("=" * 60)

cuda_test_code = """
import sys
sys.path.insert(0, "/content/ZENITH")
import numpy as np

try:
    from zenith._zenith_core import cuda, backends
    
    if backends.is_cuda_available():
        print("CUDA Backend: AVAILABLE")
        
        # Test CUDA MatMul
        A = np.random.randn(256, 128).astype(np.float32)
        B = np.random.randn(128, 64).astype(np.float32)
        
        C_cuda = cuda.matmul(A, B)
        C_numpy = np.matmul(A, B)
        
        max_diff = np.abs(C_cuda - C_numpy).max()
        print(f"CUDA MatMul Test: max_diff = {max_diff:.2e}")
        print(f"Result: {'PASS' if max_diff < 1e-4 else 'FAIL'}")
        
        # Sync
        cuda.sync()
        print("CUDA sync: OK")
    else:
        print("CUDA Backend: NOT AVAILABLE (check build)")
        
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Error: {e}")
"""

result = subprocess.run(
    ["python3", "-c", cuda_test_code], capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(f"Stderr: {result.stderr}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("BUILD COMPLETE!")
print("=" * 60)
print("""
You can now use Zenith in this Colab session:

    import sys
    sys.path.insert(0, "/content/ZENITH")
    from zenith._zenith_core import cuda, kernels, backends
    
    # Check backends
    print(backends.list_available())
    
    # CUDA MatMul
    import numpy as np
    A = np.random.randn(1024, 512).astype(np.float32)
    B = np.random.randn(512, 768).astype(np.float32)
    C = cuda.matmul(A, B)
    cuda.sync()
""")
