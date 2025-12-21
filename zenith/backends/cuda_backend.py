# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
CUDA Backend Implementation

Provides CUDA/NVIDIA GPU support through available Python libraries.
Follows the detection priority: cupy > torch > ctypes fallback.

Architecture follows CetakBiru.md Section 3.2:
- Hardware Abstraction Layer for CUDA Runtime
- cuDNN/cuBLAS integration through wrapper libraries
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

logger = logging.getLogger("zenith.backends.cuda")


class CUDABackend(BaseBackend):
    """
    CUDA backend for NVIDIA GPU execution.

    Detection Strategy:
    1. Check for CuPy (preferred - full control)
    2. Check for PyTorch CUDA
    3. Fallback to ctypes libcudart binding

    Memory Management:
    - Uses the available library's allocation functions
    - Tracks allocations for cleanup
    """

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._cupy = None
        self._torch = None
        self._cudart = None
        self._backend_lib: Optional[str] = None
        self._device_allocations: dict[int, Any] = {}

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA

    def is_available(self) -> bool:
        """Check if CUDA is available through any supported method."""
        # Try CuPy
        try:
            import cupy

            device_count = cupy.cuda.runtime.getDeviceCount()
            if device_count > 0:
                self._cupy = cupy
                self._backend_lib = "cupy"
                return True
        except Exception:
            pass

        # Try PyTorch
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                self._backend_lib = "torch"
                return True
        except Exception:
            pass

        # Try direct CUDA runtime
        try:
            cudart = ctypes.CDLL("libcudart.so")
            count = ctypes.c_int(0)
            result = cudart.cudaGetDeviceCount(ctypes.byref(count))
            if result == 0 and count.value > 0:
                self._cudart = cudart
                self._backend_lib = "ctypes"
                return True
        except Exception:
            pass

        return False

    def _do_initialize(self) -> bool:
        """Initialize CUDA context on the specified device."""
        if self._backend_lib == "cupy":
            try:
                self._cupy.cuda.Device(self._device_id).use()
                return True
            except Exception as e:
                logger.error(f"CuPy device init failed: {e}")
                return False

        elif self._backend_lib == "torch":
            try:
                self._torch.cuda.set_device(self._device_id)
                return True
            except Exception as e:
                logger.error(f"Torch CUDA init failed: {e}")
                return False

        elif self._backend_lib == "ctypes":
            try:
                result = self._cudart.cudaSetDevice(self._device_id)
                return result == 0
            except Exception as e:
                logger.error(f"CUDA runtime init failed: {e}")
                return False

        return False

    def _do_cleanup(self) -> None:
        """Cleanup CUDA resources."""
        for ptr in list(self._device_allocations.keys()):
            try:
                self._free_device_memory(ptr)
            except Exception as e:
                logger.warning(f"Failed to free CUDA memory {ptr}: {e}")
        self._device_allocations.clear()

    def get_device_count(self) -> int:
        """Get number of CUDA devices."""
        if self._backend_lib == "cupy":
            return self._cupy.cuda.runtime.getDeviceCount()
        elif self._backend_lib == "torch":
            return self._torch.cuda.device_count()
        elif self._backend_lib == "ctypes":
            count = ctypes.c_int(0)
            self._cudart.cudaGetDeviceCount(ctypes.byref(count))
            return count.value
        return 0

    def get_device_properties(self) -> DeviceProperties:
        """Get CUDA device properties."""
        if not self.is_available():
            return DeviceProperties(is_available=False)

        if self._backend_lib == "cupy":
            try:
                with self._cupy.cuda.Device(self._device_id):
                    props = self._cupy.cuda.runtime.getDeviceProperties(self._device_id)
                    free, total = self._cupy.cuda.runtime.memGetInfo()
                    return DeviceProperties(
                        name=props["name"].decode()
                        if isinstance(props["name"], bytes)
                        else props["name"],
                        vendor="NVIDIA",
                        backend_type=BackendType.CUDA,
                        device_id=self._device_id,
                        total_memory=total,
                        free_memory=free,
                        compute_capability=(props["major"], props["minor"]),
                        max_threads_per_block=props["maxThreadsPerBlock"],
                        warp_size=props["warpSize"],
                        multiprocessor_count=props["multiProcessorCount"],
                        supports_fp16=props["major"] >= 6,
                        supports_bf16=props["major"] >= 8,
                        supports_fp64=True,
                        is_available=True,
                    )
            except Exception as e:
                logger.warning(f"CuPy device props failed: {e}")

        elif self._backend_lib == "torch":
            try:
                props = self._torch.cuda.get_device_properties(self._device_id)
                free, total = self._torch.cuda.mem_get_info(self._device_id)
                return DeviceProperties(
                    name=props.name,
                    vendor="NVIDIA",
                    backend_type=BackendType.CUDA,
                    device_id=self._device_id,
                    total_memory=total,
                    free_memory=free,
                    compute_capability=(props.major, props.minor),
                    max_threads_per_block=props.max_threads_per_multi_processor,
                    warp_size=props.warp_size if hasattr(props, "warp_size") else 32,
                    multiprocessor_count=props.multi_processor_count,
                    supports_fp16=props.major >= 6,
                    supports_bf16=props.major >= 8,
                    supports_fp64=True,
                    is_available=True,
                )
            except Exception as e:
                logger.warning(f"Torch device props failed: {e}")

        return DeviceProperties(
            name=f"CUDA Device {self._device_id}",
            vendor="NVIDIA",
            backend_type=BackendType.CUDA,
            device_id=self._device_id,
            is_available=True,
        )

    def allocate(self, size_bytes: int) -> int:
        """Allocate device memory."""
        if not self._initialized:
            self.initialize()

        if size_bytes <= 0:
            raise BackendMemoryError(f"Invalid allocation size: {size_bytes}")

        try:
            if self._backend_lib == "cupy":
                mem = self._cupy.cuda.alloc(size_bytes)
                ptr = int(mem.ptr)
                self._device_allocations[ptr] = mem
                return ptr

            elif self._backend_lib == "torch":
                tensor = self._torch.empty(
                    size_bytes,
                    dtype=self._torch.uint8,
                    device=f"cuda:{self._device_id}",
                )
                ptr = tensor.data_ptr()
                self._device_allocations[ptr] = tensor
                return ptr

            elif self._backend_lib == "ctypes":
                ptr = ctypes.c_void_p()
                result = self._cudart.cudaMalloc(ctypes.byref(ptr), size_bytes)
                if result != 0:
                    raise BackendMemoryError(f"cudaMalloc failed: {result}")
                self._device_allocations[ptr.value] = ptr.value
                return ptr.value

        except Exception as e:
            raise BackendMemoryError(f"CUDA allocation failed: {e}") from e

        raise BackendMemoryError("No CUDA backend available")

    def _free_device_memory(self, ptr: int) -> None:
        """Internal: free device memory."""
        if self._backend_lib == "cupy":
            if ptr in self._device_allocations:
                del self._device_allocations[ptr]

        elif self._backend_lib == "torch":
            if ptr in self._device_allocations:
                del self._device_allocations[ptr]

        elif self._backend_lib == "ctypes":
            if ptr in self._device_allocations:
                self._cudart.cudaFree(ctypes.c_void_p(ptr))
                del self._device_allocations[ptr]

    def deallocate(self, ptr: int) -> None:
        """Free device memory."""
        if ptr not in self._device_allocations:
            logger.warning(f"Attempt to free unknown pointer: {ptr}")
            return
        self._free_device_memory(ptr)

    def copy_to_device(
        self,
        dst: int,
        src: Union[int, bytes, Any],
        size_bytes: int,
    ) -> None:
        """Copy data from host to CUDA device."""
        if not self._initialized:
            raise BackendError("Backend not initialized")

        try:
            if self._backend_lib == "cupy":
                if hasattr(src, "ctypes"):  # numpy array
                    import numpy as np

                    src_ptr = src.ctypes.data
                    self._cupy.cuda.runtime.memcpy(
                        dst,
                        src_ptr,
                        size_bytes,
                        1,  # cudaMemcpyHostToDevice
                    )
                elif isinstance(src, bytes):
                    import numpy as np

                    arr = np.frombuffer(src, dtype=np.uint8)
                    src_ptr = arr.ctypes.data
                    self._cupy.cuda.runtime.memcpy(dst, src_ptr, size_bytes, 1)

            elif self._backend_lib == "torch":
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

            elif self._backend_lib == "ctypes":
                if isinstance(src, bytes):
                    src_ptr = ctypes.c_char_p(src)
                else:
                    src_ptr = ctypes.c_void_p(src)
                self._cudart.cudaMemcpy(ctypes.c_void_p(dst), src_ptr, size_bytes, 1)

        except Exception as e:
            raise BackendError(f"CUDA copy_to_device failed: {e}") from e

    def copy_to_host(
        self,
        dst: Union[int, bytearray, Any],
        src: int,
        size_bytes: int,
    ) -> None:
        """Copy data from CUDA device to host."""
        if not self._initialized:
            raise BackendError("Backend not initialized")

        try:
            if self._backend_lib == "cupy":
                if hasattr(dst, "ctypes"):  # numpy array
                    dst_ptr = dst.ctypes.data
                    self._cupy.cuda.runtime.memcpy(
                        dst_ptr,
                        src,
                        size_bytes,
                        2,  # cudaMemcpyDeviceToHost
                    )
                elif isinstance(dst, bytearray):
                    import numpy as np

                    temp = np.empty(size_bytes, dtype=np.uint8)
                    self._cupy.cuda.runtime.memcpy(temp.ctypes.data, src, size_bytes, 2)
                    dst[:size_bytes] = temp.tobytes()

            elif self._backend_lib == "torch":
                if src in self._device_allocations:
                    src_tensor = self._device_allocations[src]
                    host_tensor = src_tensor[:size_bytes].cpu()
                    if hasattr(dst, "view"):
                        import numpy as np

                        dst_view = dst.view(np.uint8)
                        dst_view[:size_bytes] = host_tensor.numpy()
                    elif isinstance(dst, bytearray):
                        dst[:size_bytes] = host_tensor.numpy().tobytes()

            elif self._backend_lib == "ctypes":
                if isinstance(dst, bytearray):
                    buf = (ctypes.c_char * size_bytes)()
                    self._cudart.cudaMemcpy(buf, ctypes.c_void_p(src), size_bytes, 2)
                    dst[:size_bytes] = bytes(buf)
                else:
                    dst_ptr = ctypes.c_void_p(dst)
                    self._cudart.cudaMemcpy(
                        dst_ptr, ctypes.c_void_p(src), size_bytes, 2
                    )

        except Exception as e:
            raise BackendError(f"CUDA copy_to_host failed: {e}") from e

    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        if not self._initialized:
            return

        if self._backend_lib == "cupy":
            self._cupy.cuda.Stream.null.synchronize()
        elif self._backend_lib == "torch":
            self._torch.cuda.synchronize(self._device_id)
        elif self._backend_lib == "ctypes":
            self._cudart.cudaDeviceSynchronize()

    def supports_dtype(self, dtype: DataType) -> bool:
        """Check CUDA dtype support."""
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
