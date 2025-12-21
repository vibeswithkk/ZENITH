# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
ROCm Backend Implementation

Provides AMD GPU support through HIP runtime.
Detection priority: PyTorch ROCm > hip-python > ctypes fallback.

Architecture follows CetakBiru.md Section 3.2 Line 571:
- Backend ROCm for AMD GPU acceleration
- MIOpen integration for deep learning ops
- rocBLAS for linear algebra

Note: HIP provides a CUDA-compatible API that works on AMD GPUs.
"""

import ctypes
import logging
from typing import Any, Optional, Union

from .base import (
    BackendError,
    BackendMemoryError,
    BackendType,
    BaseBackend,
    DataType,
    DeviceProperties,
)

logger = logging.getLogger("zenith.backends.rocm")


def _detect_rocm_via_torch() -> bool:
    """Detect ROCm through PyTorch."""
    try:
        import torch

        # Check if PyTorch was built with ROCm
        if hasattr(torch, "__config__"):
            config = torch.__config__.show()
            if "rocm" in config.lower() or "hip" in config.lower():
                if torch.cuda.is_available():
                    return True
        # Alternative check via version string
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return True
    except Exception:
        pass
    return False


def _detect_rocm_via_hip() -> bool:
    """Detect ROCm through direct HIP library."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int(0)
        result = hip.hipGetDeviceCount(ctypes.byref(count))
        return result == 0 and count.value > 0
    except Exception:
        pass
    return False


class ROCmBackend(BaseBackend):
    """
    ROCm backend for AMD GPU execution.

    Implements the Backend interface using AMD HIP API.
    Mirrors the C++ ROCmBackend in core/include/zenith/rocm_backend.hpp.

    Detection Strategy:
    1. Check for PyTorch built with ROCm support
    2. Check for hip-python package
    3. Fallback to ctypes libamdhip64 binding

    Key Differences from CUDA:
    - Wavefront size is 64 (vs 32 warp size in NVIDIA)
    - Uses MIOpen instead of cuDNN
    - Uses rocBLAS instead of cuBLAS
    """

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._torch = None
        self._hip = None
        self._backend_lib: Optional[str] = None
        self._device_allocations: dict[int, Any] = {}

    @property
    def name(self) -> str:
        return "rocm"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.ROCM

    def is_available(self) -> bool:
        """Check if ROCm is available."""
        # Try PyTorch ROCm
        if _detect_rocm_via_torch():
            try:
                import torch

                self._torch = torch
                self._backend_lib = "torch"
                device_count = torch.cuda.device_count()
                if device_count > 0 and self._device_id < device_count:
                    return True
            except Exception:
                pass

        # Try direct HIP library
        if _detect_rocm_via_hip():
            try:
                self._hip = ctypes.CDLL("libamdhip64.so")
                self._backend_lib = "hip"
                count = ctypes.c_int(0)
                self._hip.hipGetDeviceCount(ctypes.byref(count))
                if count.value > 0 and self._device_id < count.value:
                    return True
            except Exception:
                pass

        return False

    def _do_initialize(self) -> bool:
        """Initialize ROCm/HIP context on the specified device."""
        if self._backend_lib == "torch":
            try:
                self._torch.cuda.set_device(self._device_id)
                self._torch.cuda.init()
                return True
            except Exception as e:
                logger.error(f"PyTorch ROCm init failed: {e}")
                return False

        elif self._backend_lib == "hip":
            try:
                result = self._hip.hipSetDevice(self._device_id)
                if result != 0:
                    logger.error(f"hipSetDevice failed with code {result}")
                    return False
                return True
            except Exception as e:
                logger.error(f"HIP init failed: {e}")
                return False

        return False

    def _do_cleanup(self) -> None:
        """Cleanup ROCm resources."""
        for ptr in list(self._device_allocations.keys()):
            try:
                self._free_device_memory(ptr)
            except Exception as e:
                logger.warning(f"Failed to free ROCm memory {ptr}: {e}")
        self._device_allocations.clear()

    def get_device_count(self) -> int:
        """Get number of AMD GPU devices."""
        if self._backend_lib == "torch":
            return self._torch.cuda.device_count()
        elif self._backend_lib == "hip":
            count = ctypes.c_int(0)
            self._hip.hipGetDeviceCount(ctypes.byref(count))
            return count.value
        return 0

    def get_device_properties(self) -> DeviceProperties:
        """Get ROCm device properties."""
        if not self.is_available():
            return DeviceProperties(is_available=False)

        if self._backend_lib == "torch":
            try:
                props = self._torch.cuda.get_device_properties(self._device_id)
                free, total = self._torch.cuda.mem_get_info(self._device_id)

                return DeviceProperties(
                    name=props.name,
                    vendor="AMD",
                    backend_type=BackendType.ROCM,
                    device_id=self._device_id,
                    total_memory=total,
                    free_memory=free,
                    compute_capability=(props.major, props.minor),
                    max_threads_per_block=props.max_threads_per_multi_processor,
                    warp_size=64,  # AMD wavefront size
                    multiprocessor_count=props.multi_processor_count,
                    supports_fp16=True,  # MI-series supports FP16
                    supports_bf16=props.major >= 9,  # MI300 series
                    supports_fp64=True,
                    is_available=True,
                )
            except Exception as e:
                logger.warning(f"ROCm torch device props failed: {e}")

        elif self._backend_lib == "hip":
            try:
                # hipDeviceProp_t structure (simplified)
                class HipDeviceProp(ctypes.Structure):
                    _fields_ = [
                        ("name", ctypes.c_char * 256),
                        ("totalGlobalMem", ctypes.c_size_t),
                        ("sharedMemPerBlock", ctypes.c_size_t),
                        ("regsPerBlock", ctypes.c_int),
                        ("warpSize", ctypes.c_int),
                        ("maxThreadsPerBlock", ctypes.c_int),
                        ("maxThreadsDim", ctypes.c_int * 3),
                        ("maxGridSize", ctypes.c_int * 3),
                        ("clockRate", ctypes.c_int),
                        ("memoryClockRate", ctypes.c_int),
                        ("memoryBusWidth", ctypes.c_int),
                        ("totalConstMem", ctypes.c_size_t),
                        ("major", ctypes.c_int),
                        ("minor", ctypes.c_int),
                        ("multiProcessorCount", ctypes.c_int),
                    ]

                props = HipDeviceProp()
                result = self._hip.hipGetDeviceProperties(
                    ctypes.byref(props), self._device_id
                )
                if result == 0:
                    free = ctypes.c_size_t(0)
                    total = ctypes.c_size_t(0)
                    self._hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))

                    return DeviceProperties(
                        name=props.name.decode("utf-8").strip("\x00"),
                        vendor="AMD",
                        backend_type=BackendType.ROCM,
                        device_id=self._device_id,
                        total_memory=total.value,
                        free_memory=free.value,
                        compute_capability=(props.major, props.minor),
                        max_threads_per_block=props.maxThreadsPerBlock,
                        warp_size=props.warpSize,
                        multiprocessor_count=props.multiProcessorCount,
                        supports_fp16=True,
                        supports_bf16=props.major >= 9,
                        supports_fp64=True,
                        is_available=True,
                    )
            except Exception as e:
                logger.warning(f"HIP device props failed: {e}")

        return DeviceProperties(
            name=f"AMD GPU {self._device_id}",
            vendor="AMD",
            backend_type=BackendType.ROCM,
            device_id=self._device_id,
            warp_size=64,
            is_available=True,
        )

    def allocate(self, size_bytes: int) -> int:
        """Allocate device memory on AMD GPU."""
        if not self._initialized:
            self.initialize()

        if size_bytes <= 0:
            raise BackendMemoryError(f"Invalid allocation size: {size_bytes}")

        try:
            if self._backend_lib == "torch":
                tensor = self._torch.empty(
                    size_bytes,
                    dtype=self._torch.uint8,
                    device=f"cuda:{self._device_id}",
                )
                ptr = tensor.data_ptr()
                self._device_allocations[ptr] = tensor
                return ptr

            elif self._backend_lib == "hip":
                ptr = ctypes.c_void_p()
                result = self._hip.hipMalloc(ctypes.byref(ptr), size_bytes)
                if result != 0:
                    raise BackendMemoryError(f"hipMalloc failed: {result}")
                self._device_allocations[ptr.value] = ptr.value
                return ptr.value

        except BackendMemoryError:
            raise
        except Exception as e:
            raise BackendMemoryError(f"ROCm allocation failed: {e}") from e

        raise BackendMemoryError("No ROCm backend available")

    def _free_device_memory(self, ptr: int) -> None:
        """Internal: free device memory."""
        if self._backend_lib == "torch":
            if ptr in self._device_allocations:
                del self._device_allocations[ptr]

        elif self._backend_lib == "hip":
            if ptr in self._device_allocations:
                self._hip.hipFree(ctypes.c_void_p(ptr))
                del self._device_allocations[ptr]

    def deallocate(self, ptr: int) -> None:
        """Free device memory."""
        if ptr not in self._device_allocations:
            logger.warning(f"Attempt to free unknown ROCm pointer: {ptr}")
            return
        self._free_device_memory(ptr)

    def copy_to_device(
        self,
        dst: int,
        src: Union[int, bytes, Any],
        size_bytes: int,
    ) -> None:
        """Copy data from host to AMD GPU device."""
        if not self._initialized:
            raise BackendError("Backend not initialized")

        try:
            if self._backend_lib == "torch":
                if dst in self._device_allocations:
                    dst_tensor = self._device_allocations[dst]
                    if hasattr(src, "numpy"):
                        src_tensor = self._torch.from_numpy(src).to(
                            dtype=self._torch.uint8
                        )
                    elif isinstance(src, bytes):
                        import numpy as np

                        arr = np.frombuffer(src, dtype=np.uint8)
                        src_tensor = self._torch.from_numpy(arr.copy()).to(
                            dtype=self._torch.uint8
                        )
                    else:
                        raise BackendError(f"Unsupported source: {type(src)}")
                    dst_tensor[:size_bytes].copy_(src_tensor[:size_bytes])

            elif self._backend_lib == "hip":
                # hipMemcpyHostToDevice = 1
                if isinstance(src, bytes):
                    src_buf = (ctypes.c_char * len(src)).from_buffer_copy(src)
                    self._hip.hipMemcpy(
                        ctypes.c_void_p(dst),
                        src_buf,
                        size_bytes,
                        1,
                    )
                elif hasattr(src, "ctypes"):
                    self._hip.hipMemcpy(
                        ctypes.c_void_p(dst),
                        ctypes.c_void_p(src.ctypes.data),
                        size_bytes,
                        1,
                    )

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"ROCm copy_to_device failed: {e}") from e

    def copy_to_host(
        self,
        dst: Union[int, bytearray, Any],
        src: int,
        size_bytes: int,
    ) -> None:
        """Copy data from AMD GPU device to host."""
        if not self._initialized:
            raise BackendError("Backend not initialized")

        try:
            if self._backend_lib == "torch":
                if src in self._device_allocations:
                    src_tensor = self._device_allocations[src]
                    host_tensor = src_tensor[:size_bytes].cpu()
                    if hasattr(dst, "view"):
                        import numpy as np

                        dst_view = dst.view(np.uint8)
                        dst_view[:size_bytes] = host_tensor.numpy()
                    elif isinstance(dst, bytearray):
                        dst[:size_bytes] = host_tensor.numpy().tobytes()

            elif self._backend_lib == "hip":
                # hipMemcpyDeviceToHost = 2
                if isinstance(dst, bytearray):
                    buf = (ctypes.c_char * size_bytes)()
                    self._hip.hipMemcpy(
                        buf,
                        ctypes.c_void_p(src),
                        size_bytes,
                        2,
                    )
                    dst[:size_bytes] = bytes(buf)
                elif hasattr(dst, "ctypes"):
                    self._hip.hipMemcpy(
                        ctypes.c_void_p(dst.ctypes.data),
                        ctypes.c_void_p(src),
                        size_bytes,
                        2,
                    )

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"ROCm copy_to_host failed: {e}") from e

    def synchronize(self) -> None:
        """Synchronize AMD GPU device."""
        if not self._initialized:
            return

        if self._backend_lib == "torch":
            self._torch.cuda.synchronize(self._device_id)
        elif self._backend_lib == "hip":
            self._hip.hipDeviceSynchronize()

    def supports_dtype(self, dtype: DataType) -> bool:
        """Check ROCm dtype support."""
        props = self.get_device_properties()

        if dtype == DataType.Float16:
            return props.supports_fp16
        elif dtype == DataType.BFloat16:
            return props.supports_bf16
        elif dtype == DataType.Float64:
            return props.supports_fp64

        return dtype in {
            DataType.Float32,
            DataType.Int32,
            DataType.Int64,
            DataType.Int8,
            DataType.UInt8,
        }
