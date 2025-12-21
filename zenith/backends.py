# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Backends Module - Compatibility Layer

This module provides backward compatibility with the original backends.py API.
All functionality is now implemented in the zenith.backends package.

For new code, use:
    from zenith.backends import (
        is_cuda_available,
        get_device,
        CUDABackend,
        ROCmBackend,
        OneAPIBackend,
    )
"""

# Re-export everything from the backends package
from zenith.backends import (
    is_cpu_available,
    is_cuda_available,
    is_rocm_available,
    is_oneapi_available,
    is_tpu_available,
    get_available_backends,
    get_cuda_device_count,
    get_cuda_device_name,
    get_rocm_device_count,
    get_rocm_device_name,
    get_oneapi_device_count,
    get_oneapi_device_name,
    get_device,
    set_device,
    get_current_device,
    list_devices,
    synchronize,
    create_backend,
    # Classes
    BaseBackend,
    BackendType,
    DataType,
    DeviceProperties,
    CPUBackend,
    CUDABackend,
    ROCmBackend,
    OneAPIBackend,
    BackendRegistry,
    DeviceManager,
    # Exceptions
    BackendError,
    BackendNotAvailableError,
    BackendMemoryError,
    BackendExecutionError,
)

__all__ = [
    # Availability checks
    "is_cpu_available",
    "is_cuda_available",
    "is_rocm_available",
    "is_oneapi_available",
    "is_tpu_available",
    "get_available_backends",
    # Device counts
    "get_cuda_device_count",
    "get_cuda_device_name",
    "get_rocm_device_count",
    "get_rocm_device_name",
    "get_oneapi_device_count",
    "get_oneapi_device_name",
    # Device management
    "get_device",
    "set_device",
    "get_current_device",
    "list_devices",
    "synchronize",
    "create_backend",
    # Classes
    "BaseBackend",
    "BackendType",
    "DataType",
    "DeviceProperties",
    "CPUBackend",
    "CUDABackend",
    "ROCmBackend",
    "OneAPIBackend",
    "BackendRegistry",
    "DeviceManager",
    # Exceptions
    "BackendError",
    "BackendNotAvailableError",
    "BackendMemoryError",
    "BackendExecutionError",
]
