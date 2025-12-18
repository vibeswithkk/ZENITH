#!/bin/bash
# Zenith CUDA Build Script for Google Colab
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0
#
# Usage: Run this script in a Colab cell with:
#   !bash build_cuda.sh

set -e

echo "=============================================="
echo "Zenith CUDA Build Script"
echo "=============================================="

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Make sure you're using a GPU runtime."
    exit 1
fi

echo "CUDA found: $(nvcc --version | head -1)"

# Install build dependencies
echo "Installing dependencies..."
pip install -q pybind11 numpy

# Create build directory
mkdir -p build
cd build

# Configure with CUDA
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DZENITH_BUILD_PYTHON=ON \
    -DZENITH_ENABLE_CUDA=ON \
    -DZENITH_BUILD_TESTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# Build
echo "Building..."
make -j$(nproc) zenith_core _zenith_core

# Copy module to python path
echo "Installing module..."
cp python/zenith/_zenith_core*.so ../zenith/ 2>/dev/null || \
cp lib/_zenith_core*.so ../zenith/ 2>/dev/null || \
find . -name "_zenith_core*.so" -exec cp {} ../zenith/ \;

cd ..

echo ""
echo "=============================================="
echo "Build complete! Test with:"
echo "  python -c 'from zenith._zenith_core import backends; print(backends.list_available())'"
echo "=============================================="
