# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Hardware Backend Package

Enterprise-grade hardware abstraction layer supporting:
- CUDA (NVIDIA GPUs)
- ROCm/HIP (AMD GPUs)
- oneAPI/SYCL (Intel GPUs/CPUs)
- CPU (Universal fallback)

Architecture follows CetakBiru.md Section 3.2 (Line 214-239):
- Hardware Abstraction Layer (HAL) design
- Uniform interface for all backends
- Dynamic backend registration
- Automatic fallback chain

Usage:
    import zenith.backends as zb

    # Check availability
    print(zb.is_cuda_available())
    print(zb.is_rocm_available())
    print(zb.is_oneapi_available())

    # Get device
    device = zb.get_device("cuda:0")

    # List available devices
    print(zb.list_devices())

    # Set default device
    zb.set_device("cuda:0")
"""

from .base import (
    BackendError,
    BackendExecutionError,
    BackendMemoryError,
    BackendNotAvailableError,
    BackendType,
    BaseBackend,
    CPUBackend,
    DataType,
    DeviceProperties,
)
from .cuda_backend import CUDABackend
from .oneapi_backend import OneAPIBackend
from .registry import (
    BackendRegistry,
    DeviceManager,
    get_current_device,
    get_device,
    list_devices,
    set_device,
    synchronize,
)
from .rocm_backend import ROCmBackend


def is_cpu_available() -> bool:
    """Check if CPU backend is available (always True)."""
    return True


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    try:
        backend = CUDABackend()
        return backend.is_available()
    except Exception:
        return False


def is_rocm_available() -> bool:
    """Check if ROCm (AMD GPU) is available."""
    try:
        backend = ROCmBackend()
        return backend.is_available()
    except Exception:
        return False


def is_oneapi_available() -> bool:
    """Check if Intel oneAPI is available."""
    try:
        backend = OneAPIBackend()
        return backend.is_available()
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
    """Get list of available backend names."""
    backends = ["cpu"]
    if is_cuda_available():
        backends.append("cuda")
    if is_rocm_available():
        backends.append("rocm")
    if is_oneapi_available():
        backends.append("oneapi")
    if is_tpu_available():
        backends.append("tpu")
    return backends


def get_cuda_device_count() -> int:
    """Get number of CUDA devices."""
    try:
        backend = CUDABackend()
        if backend.is_available():
            return backend.get_device_count()
    except Exception:
        pass
    return 0


def get_rocm_device_count() -> int:
    """Get number of ROCm devices."""
    try:
        backend = ROCmBackend()
        if backend.is_available():
            return backend.get_device_count()
    except Exception:
        pass
    return 0


def get_oneapi_device_count() -> int:
    """Get number of oneAPI devices."""
    try:
        backend = OneAPIBackend()
        if backend.is_available():
            return backend.get_device_count()
    except Exception:
        pass
    return 0


def get_cuda_device_name(device_id: int = 0) -> str:
    """Get CUDA device name."""
    try:
        backend = CUDABackend(device_id=device_id)
        if backend.is_available():
            props = backend.get_device_properties()
            return props.name
    except Exception:
        pass
    return ""


def get_rocm_device_name(device_id: int = 0) -> str:
    """Get ROCm device name."""
    try:
        backend = ROCmBackend(device_id=device_id)
        if backend.is_available():
            props = backend.get_device_properties()
            return props.name
    except Exception:
        pass
    return ""


def get_oneapi_device_name(device_id: int = 0) -> str:
    """Get oneAPI device name."""
    try:
        backend = OneAPIBackend(device_id=device_id)
        if backend.is_available():
            props = backend.get_device_properties()
            return props.name
    except Exception:
        pass
    return ""


def create_backend(device: str) -> BaseBackend:
    """
    Create a backend for the specified device.

    Args:
        device: Device string (e.g., "cuda:0", "rocm:0", "cpu").

    Returns:
        Initialized backend instance.

    Raises:
        BackendNotAvailableError: If backend is not available.
    """
    backend = get_device(device)
    if backend is None:
        raise BackendNotAvailableError(f"Backend not available: {device}")
    return backend


__all__ = [
    # Base classes
    "BaseBackend",
    "BackendType",
    "DataType",
    "DeviceProperties",
    # Exceptions
    "BackendError",
    "BackendNotAvailableError",
    "BackendMemoryError",
    "BackendExecutionError",
    # Backend implementations
    "CPUBackend",
    "CUDABackend",
    "ROCmBackend",
    "OneAPIBackend",
    # Registry
    "BackendRegistry",
    "DeviceManager",
    # Device management functions
    "get_device",
    "set_device",
    "get_current_device",
    "list_devices",
    "synchronize",
    "create_backend",
    # Availability checks
    "is_cpu_available",
    "is_cuda_available",
    "is_rocm_available",
    "is_oneapi_available",
    "is_tpu_available",
    "get_available_backends",
    # Device counts
    "get_cuda_device_count",
    "get_rocm_device_count",
    "get_oneapi_device_count",
    # Device names
    "get_cuda_device_name",
    "get_rocm_device_name",
    "get_oneapi_device_name",
]
