"""
Zenith Backends Module.

Provides utilities for checking hardware availability.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""


def is_cpu_available() -> bool:
    """Check if CPU backend is available."""
    return True


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    try:
        # Try CuPy
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        pass

    try:
        # Try PyTorch
        import torch

        return torch.cuda.is_available()
    except Exception:
        pass

    try:
        # Try direct CUDA check
        import ctypes

        cuda = ctypes.CDLL("libcudart.so")
        count = ctypes.c_int()
        cuda.cudaGetDeviceCount(ctypes.byref(count))
        return count.value > 0
    except Exception:
        return False


def is_rocm_available() -> bool:
    """Check if ROCm (AMD GPU) is available."""
    try:
        import torch

        return torch.cuda.is_available() and "rocm" in torch.__config__.show().lower()
    except Exception:
        return False


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    try:
        import jax

        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


def get_available_backends() -> list[str]:
    """Get list of available backends."""
    backends = ["cpu"]
    if is_cuda_available():
        backends.append("cuda")
    if is_rocm_available():
        backends.append("rocm")
    if is_tpu_available():
        backends.append("tpu")
    return backends


def get_cuda_device_count() -> int:
    """Get number of CUDA devices."""
    if not is_cuda_available():
        return 0

    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pass

    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def get_cuda_device_name(device_id: int = 0) -> str:
    """Get CUDA device name."""
    if not is_cuda_available():
        return ""

    try:
        import cupy

        with cupy.cuda.Device(device_id):
            props = cupy.cuda.runtime.getDeviceProperties(device_id)
            return props["name"].decode()
    except Exception:
        pass

    try:
        import torch

        return torch.cuda.get_device_name(device_id)
    except Exception:
        return ""
