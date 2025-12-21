# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Intel oneAPI Backend Implementation

Provides Intel GPU/CPU acceleration through SYCL runtime.
Detection priority: dpctl > intel-extension-for-pytorch > ctypes.

Architecture follows CetakBiru.md Section 3.2 Line 571:
- Backend oneAPI for Intel GPU acceleration
- oneDNN integration for deep learning ops
- oneMKL for linear algebra

oneAPI supports:
- Intel Data Center GPU Max Series (Ponte Vecchio)
- Intel Arc GPUs (Alchemist)
- Intel integrated GPUs
- Intel Xeon CPUs with AVX-512
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

logger = logging.getLogger("zenith.backends.oneapi")


def _detect_oneapi_via_dpctl() -> bool:
    """Detect oneAPI through dpctl (Data Parallel Control)."""
    try:
        import dpctl

        devices = dpctl.get_devices()
        return len(devices) > 0
    except Exception:
        pass
    return False


def _detect_oneapi_via_torch() -> bool:
    """Detect oneAPI through Intel Extension for PyTorch (IPEX)."""
    try:
        import intel_extension_for_pytorch as ipex
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:
        pass
    return False


def _detect_oneapi_via_level_zero() -> bool:
    """Detect oneAPI through Level Zero library."""
    try:
        ze = ctypes.CDLL("libze_loader.so")
        # zeInit
        result = ze.zeInit(0)
        if result == 0:
            return True
    except Exception:
        pass
    return False


class OneAPIBackend(BaseBackend):
    """
    Intel oneAPI/SYCL backend for GPU and CPU execution.

    Implements the Backend interface using Intel SYCL runtime.
    Mirrors the C++ OneAPIBackend in core/include/zenith/oneapi_backend.hpp.

    Detection Strategy:
    1. Check for dpctl (Data Parallel Control library)
    2. Check for Intel Extension for PyTorch (IPEX) with XPU support
    3. Fallback to Level Zero API through ctypes

    Device Types:
    - GPU: Intel discrete/integrated GPUs
    - CPU: Intel CPUs with SYCL support
    - ACCELERATOR: FPGA or other accelerators
    """

    class DeviceSelector:
        """Device type selector for oneAPI."""

        GPU = "gpu"
        CPU = "cpu"
        ACCELERATOR = "accelerator"
        ANY = "any"

    def __init__(
        self,
        device_id: int = 0,
        device_type: str = "gpu",
    ):
        super().__init__(device_id)
        self._device_type = device_type
        self._dpctl = None
        self._torch = None
        self._ipex = None
        self._level_zero = None
        self._backend_lib: Optional[str] = None
        self._device_allocations: dict[int, Any] = {}
        self._sycl_device = None
        self._sycl_queue = None

    @property
    def name(self) -> str:
        return "oneapi"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.ONEAPI

    def is_available(self) -> bool:
        """Check if oneAPI is available."""
        # Try dpctl
        if _detect_oneapi_via_dpctl():
            try:
                import dpctl

                self._dpctl = dpctl
                self._backend_lib = "dpctl"

                # Get devices of requested type
                if self._device_type == "gpu":
                    devices = dpctl.get_devices(device_type=dpctl.device_type.gpu)
                elif self._device_type == "cpu":
                    devices = dpctl.get_devices(device_type=dpctl.device_type.cpu)
                else:
                    devices = dpctl.get_devices()

                if len(devices) > self._device_id:
                    return True
            except Exception as e:
                logger.debug(f"dpctl detection failed: {e}")

        # Try Intel Extension for PyTorch
        if _detect_oneapi_via_torch():
            try:
                import intel_extension_for_pytorch as ipex
                import torch

                self._ipex = ipex
                self._torch = torch
                self._backend_lib = "ipex"
                device_count = torch.xpu.device_count()
                if device_count > self._device_id:
                    return True
            except Exception as e:
                logger.debug(f"IPEX detection failed: {e}")

        # Try Level Zero
        if _detect_oneapi_via_level_zero():
            try:
                self._level_zero = ctypes.CDLL("libze_loader.so")
                self._backend_lib = "level_zero"
                return True
            except Exception as e:
                logger.debug(f"Level Zero detection failed: {e}")

        return False

    def _do_initialize(self) -> bool:
        """Initialize SYCL context on the specified device."""
        if self._backend_lib == "dpctl":
            try:
                import dpctl

                if self._device_type == "gpu":
                    devices = dpctl.get_devices(device_type=dpctl.device_type.gpu)
                elif self._device_type == "cpu":
                    devices = dpctl.get_devices(device_type=dpctl.device_type.cpu)
                else:
                    devices = dpctl.get_devices()

                if len(devices) > self._device_id:
                    self._sycl_device = devices[self._device_id]
                    self._sycl_queue = dpctl.SyclQueue(self._sycl_device)
                    return True
                return False
            except Exception as e:
                logger.error(f"dpctl init failed: {e}")
                return False

        elif self._backend_lib == "ipex":
            try:
                self._torch.xpu.set_device(self._device_id)
                return True
            except Exception as e:
                logger.error(f"IPEX init failed: {e}")
                return False

        elif self._backend_lib == "level_zero":
            try:
                # Level Zero initialization already done in detection
                return True
            except Exception as e:
                logger.error(f"Level Zero init failed: {e}")
                return False

        return False

    def _do_cleanup(self) -> None:
        """Cleanup oneAPI resources."""
        for ptr in list(self._device_allocations.keys()):
            try:
                self._free_device_memory(ptr)
            except Exception as e:
                logger.warning(f"Failed to free oneAPI memory {ptr}: {e}")
        self._device_allocations.clear()

        if self._sycl_queue is not None:
            self._sycl_queue = None
        if self._sycl_device is not None:
            self._sycl_device = None

    def get_device_count(self) -> int:
        """Get number of oneAPI devices."""
        if self._backend_lib == "dpctl":
            try:
                import dpctl

                if self._device_type == "gpu":
                    return len(dpctl.get_devices(device_type=dpctl.device_type.gpu))
                elif self._device_type == "cpu":
                    return len(dpctl.get_devices(device_type=dpctl.device_type.cpu))
                return len(dpctl.get_devices())
            except Exception:
                pass

        elif self._backend_lib == "ipex":
            try:
                return self._torch.xpu.device_count()
            except Exception:
                pass

        return 1 if self.is_available() else 0

    def get_device_properties(self) -> DeviceProperties:
        """Get oneAPI device properties."""
        if not self.is_available():
            return DeviceProperties(is_available=False)

        if self._backend_lib == "dpctl":
            try:
                import dpctl

                if self._device_type == "gpu":
                    devices = dpctl.get_devices(device_type=dpctl.device_type.gpu)
                elif self._device_type == "cpu":
                    devices = dpctl.get_devices(device_type=dpctl.device_type.cpu)
                else:
                    devices = dpctl.get_devices()

                if len(devices) > self._device_id:
                    dev = devices[self._device_id]
                    return DeviceProperties(
                        name=dev.name,
                        vendor=dev.vendor,
                        backend_type=BackendType.ONEAPI,
                        device_id=self._device_id,
                        total_memory=dev.global_mem_size,
                        free_memory=dev.global_mem_size,  # Approximation
                        max_work_group_size=dev.max_work_group_size,
                        multiprocessor_count=dev.max_compute_units,
                        supports_fp16=dev.has_aspect_fp16,
                        supports_fp64=dev.has_aspect_fp64,
                        supports_bf16=False,  # Check specific device
                        is_available=True,
                    )
            except Exception as e:
                logger.warning(f"dpctl device props failed: {e}")

        elif self._backend_lib == "ipex":
            try:
                props = self._torch.xpu.get_device_properties(self._device_id)
                return DeviceProperties(
                    name=props.name if hasattr(props, "name") else "Intel XPU",
                    vendor="Intel",
                    backend_type=BackendType.ONEAPI,
                    device_id=self._device_id,
                    total_memory=props.total_memory
                    if hasattr(props, "total_memory")
                    else 0,
                    supports_fp16=True,
                    supports_fp64=True,
                    is_available=True,
                )
            except Exception as e:
                logger.warning(f"IPEX device props failed: {e}")

        return DeviceProperties(
            name=f"Intel {self._device_type.upper()} {self._device_id}",
            vendor="Intel",
            backend_type=BackendType.ONEAPI,
            device_id=self._device_id,
            is_available=True,
        )

    def allocate(self, size_bytes: int) -> int:
        """Allocate device memory on Intel GPU."""
        if not self._initialized:
            self.initialize()

        if size_bytes <= 0:
            raise BackendMemoryError(f"Invalid allocation size: {size_bytes}")

        try:
            if self._backend_lib == "dpctl":
                import dpctl.memory as dpmem

                mem = dpmem.MemoryUSMDevice(size_bytes, queue=self._sycl_queue)
                ptr = mem._pointer
                self._device_allocations[ptr] = mem
                return ptr

            elif self._backend_lib == "ipex":
                tensor = self._torch.empty(
                    size_bytes,
                    dtype=self._torch.uint8,
                    device=f"xpu:{self._device_id}",
                )
                ptr = tensor.data_ptr()
                self._device_allocations[ptr] = tensor
                return ptr

            elif self._backend_lib == "level_zero":
                # Level Zero memory allocation is complex
                # Return CPU fallback for now
                buffer = bytearray(size_bytes)
                ptr = id(buffer)
                self._device_allocations[ptr] = buffer
                return ptr

        except Exception as e:
            raise BackendMemoryError(f"oneAPI allocation failed: {e}") from e

        raise BackendMemoryError("No oneAPI backend available")

    def _free_device_memory(self, ptr: int) -> None:
        """Internal: free device memory."""
        if ptr in self._device_allocations:
            del self._device_allocations[ptr]

    def deallocate(self, ptr: int) -> None:
        """Free device memory."""
        if ptr not in self._device_allocations:
            logger.warning(f"Attempt to free unknown oneAPI pointer: {ptr}")
            return
        self._free_device_memory(ptr)

    def copy_to_device(
        self,
        dst: int,
        src: Union[int, bytes, Any],
        size_bytes: int,
    ) -> None:
        """Copy data from host to Intel device."""
        if not self._initialized:
            raise BackendError("Backend not initialized")

        try:
            if self._backend_lib == "dpctl":
                import dpctl.memory as dpmem

                if dst in self._device_allocations:
                    dst_mem = self._device_allocations[dst]
                    if isinstance(src, bytes):
                        dst_mem.copy_from_host(src[:size_bytes])
                    elif hasattr(src, "tobytes"):
                        dst_mem.copy_from_host(src.tobytes()[:size_bytes])

            elif self._backend_lib == "ipex":
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

            elif self._backend_lib == "level_zero":
                if dst in self._device_allocations:
                    buf = self._device_allocations[dst]
                    if isinstance(src, bytes):
                        buf[:size_bytes] = src[:size_bytes]

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"oneAPI copy_to_device failed: {e}") from e

    def copy_to_host(
        self,
        dst: Union[int, bytearray, Any],
        src: int,
        size_bytes: int,
    ) -> None:
        """Copy data from Intel device to host."""
        if not self._initialized:
            raise BackendError("Backend not initialized")

        try:
            if self._backend_lib == "dpctl":
                if src in self._device_allocations:
                    src_mem = self._device_allocations[src]
                    data = src_mem.copy_to_host()
                    if isinstance(dst, bytearray):
                        dst[:size_bytes] = data[:size_bytes]
                    elif hasattr(dst, "view"):
                        import numpy as np

                        dst_view = dst.view(np.uint8)
                        dst_view[:size_bytes] = np.frombuffer(
                            data[:size_bytes], dtype=np.uint8
                        )

            elif self._backend_lib == "ipex":
                if src in self._device_allocations:
                    src_tensor = self._device_allocations[src]
                    host_tensor = src_tensor[:size_bytes].cpu()
                    if hasattr(dst, "view"):
                        import numpy as np

                        dst_view = dst.view(np.uint8)
                        dst_view[:size_bytes] = host_tensor.numpy()
                    elif isinstance(dst, bytearray):
                        dst[:size_bytes] = host_tensor.numpy().tobytes()

            elif self._backend_lib == "level_zero":
                if src in self._device_allocations:
                    buf = self._device_allocations[src]
                    if isinstance(dst, bytearray):
                        dst[:size_bytes] = buf[:size_bytes]

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"oneAPI copy_to_host failed: {e}") from e

    def synchronize(self) -> None:
        """Synchronize Intel device."""
        if not self._initialized:
            return

        if self._backend_lib == "dpctl" and self._sycl_queue is not None:
            self._sycl_queue.wait()
        elif self._backend_lib == "ipex":
            self._torch.xpu.synchronize(self._device_id)

    def supports_dtype(self, dtype: DataType) -> bool:
        """Check oneAPI dtype support."""
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
