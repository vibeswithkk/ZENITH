# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Backend Base Classes

This module defines the abstract interfaces for hardware backends,
matching the C++ Backend class contract defined in core/include/zenith/backend.hpp.

Architecture follows CetakBiru.md Section 3.2 (Line 214-239):
- Hardware Abstraction Layer (HAL) design
- Uniform interface for CUDA, ROCm, SYCL, CPU backends
- Memory management and kernel dispatch abstraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union
import logging

logger = logging.getLogger("zenith.backends")


class BackendType(Enum):
    """Enumeration of supported backend types."""

    CPU = auto()
    CUDA = auto()
    ROCM = auto()
    ONEAPI = auto()
    TPU = auto()
    METAL = auto()
    VULKAN = auto()


class DataType(Enum):
    """Data types supported by backends."""

    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    BFloat16 = "bfloat16"
    Int8 = "int8"
    Int16 = "int16"
    Int32 = "int32"
    Int64 = "int64"
    UInt8 = "uint8"
    Bool = "bool"


@dataclass
class DeviceProperties:
    """Hardware device properties.

    Mirrors the information available from C++ backend device queries.
    """

    name: str = "Unknown Device"
    vendor: str = "Unknown"
    backend_type: BackendType = BackendType.CPU
    device_id: int = 0
    total_memory: int = 0
    free_memory: int = 0
    compute_capability: tuple[int, int] = (0, 0)
    max_threads_per_block: int = 1024
    max_work_group_size: int = 256
    warp_size: int = 32
    multiprocessor_count: int = 1
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_fp64: bool = True
    driver_version: str = ""
    is_available: bool = False


@dataclass
class AllocationInfo:
    """Information about a memory allocation."""

    ptr: int  # Memory address as integer
    size_bytes: int
    backend_type: BackendType
    device_id: int
    is_host_accessible: bool = False


class BackendError(Exception):
    """Base exception for backend errors."""

    pass


class BackendNotAvailableError(BackendError):
    """Raised when a backend is not available."""

    pass


class BackendMemoryError(BackendError):
    """Raised when memory allocation fails."""

    pass


class BackendExecutionError(BackendError):
    """Raised when execution fails."""

    pass


class BaseBackend(ABC):
    """
    Abstract base class for all hardware backends.

    This class defines the interface that all backend implementations must
    follow, matching the C++ Backend class in core/include/zenith/backend.hpp.

    Contract:
    - name(): Return unique identifier string
    - is_available(): Return True only if backend can execute
    - allocate()/deallocate(): Memory management
    - copy_to_device()/copy_to_host(): Data transfer
    - synchronize(): Wait for pending operations
    - execute(): Run computation graph

    Thread Safety: Implementations must be thread-safe.
    """

    def __init__(self, device_id: int = 0):
        """Initialize backend.

        Args:
            device_id: Device index for multi-device systems.
        """
        self._device_id = device_id
        self._initialized = False
        self._allocations: dict[int, AllocationInfo] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this backend.

        Returns:
            String identifier (e.g., "cpu", "cuda", "rocm", "oneapi").
        """
        pass

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type enum."""
        pass

    @property
    def device_id(self) -> int:
        """Return the device ID."""
        return self._device_id

    @property
    def is_initialized(self) -> bool:
        """Check if backend has been initialized."""
        return self._initialized

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        This method must NOT raise exceptions. It should return False
        if the backend cannot be used for any reason.

        Returns:
            True if backend is available and functional.
        """
        pass

    def initialize(self) -> bool:
        """Initialize the backend.

        Must be called before any other operations.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        if not self.is_available():
            logger.warning(f"Backend {self.name} is not available")
            return False

        try:
            success = self._do_initialize()
            self._initialized = success
            if success:
                logger.info(f"Backend {self.name}:{self._device_id} initialized")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    def _do_initialize(self) -> bool:
        """Backend-specific initialization. Override in subclasses."""
        return True

    def cleanup(self) -> None:
        """Release all resources held by this backend."""
        if not self._initialized:
            return

        # Free any tracked allocations
        for ptr in list(self._allocations.keys()):
            try:
                self.deallocate(ptr)
            except Exception as e:
                logger.warning(f"Failed to deallocate {ptr}: {e}")

        self._do_cleanup()
        self._initialized = False
        logger.debug(f"Backend {self.name}:{self._device_id} cleaned up")

    def _do_cleanup(self) -> None:
        """Backend-specific cleanup. Override in subclasses."""
        pass

    @abstractmethod
    def get_device_properties(self) -> DeviceProperties:
        """Get properties of the current device.

        Returns:
            DeviceProperties object with hardware information.
        """
        pass

    def get_device_count(self) -> int:
        """Get number of devices available for this backend.

        Returns:
            Number of available devices.
        """
        return 1 if self.is_available() else 0

    @abstractmethod
    def allocate(self, size_bytes: int) -> int:
        """Allocate memory on this backend's device.

        Args:
            size_bytes: Number of bytes to allocate.

        Returns:
            Memory address as integer, or 0 on failure.

        Raises:
            BackendMemoryError: If allocation fails.
        """
        pass

    @abstractmethod
    def deallocate(self, ptr: int) -> None:
        """Free memory allocated by this backend.

        Args:
            ptr: Memory address returned by allocate().
        """
        pass

    @abstractmethod
    def copy_to_device(
        self,
        dst: int,
        src: Union[int, bytes, "numpy.ndarray"],
        size_bytes: int,
    ) -> None:
        """Copy data from host to device.

        Args:
            dst: Destination device memory address.
            src: Source data (host pointer, bytes, or numpy array).
            size_bytes: Number of bytes to copy.

        Raises:
            BackendError: If copy fails.
        """
        pass

    @abstractmethod
    def copy_to_host(
        self,
        dst: Union[int, bytearray, "numpy.ndarray"],
        src: int,
        size_bytes: int,
    ) -> None:
        """Copy data from device to host.

        Args:
            dst: Destination host buffer (pointer, bytearray, or numpy array).
            src: Source device memory address.
            size_bytes: Number of bytes to copy.

        Raises:
            BackendError: If copy fails.
        """
        pass

    def synchronize(self) -> None:
        """Wait for all pending operations to complete.

        Blocking call. Should be called before reading results.
        """
        pass

    def supports_dtype(self, dtype: DataType) -> bool:
        """Check if backend supports a specific data type.

        Args:
            dtype: Data type to check.

        Returns:
            True if the data type is supported.
        """
        basic_types = {
            DataType.Float32,
            DataType.Float64,
            DataType.Int32,
            DataType.Int64,
        }
        return dtype in basic_types

    def get_memory_info(self) -> tuple[int, int]:
        """Get device memory information.

        Returns:
            Tuple of (free_bytes, total_bytes).
        """
        props = self.get_device_properties()
        return (props.free_memory, props.total_memory)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"<{self.__class__.__name__}({self.name}:{self._device_id}, {status})>"


class CPUBackend(BaseBackend):
    """
    CPU backend implementation.

    Uses standard memory operations. Always available as fallback.
    Implements the base interface for host-side execution.
    """

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._allocations: dict[int, bytes] = {}

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CPU

    def is_available(self) -> bool:
        return True

    def get_device_properties(self) -> DeviceProperties:
        import platform
        import os

        try:
            import psutil

            total_mem = psutil.virtual_memory().total
            free_mem = psutil.virtual_memory().available
        except ImportError:
            total_mem = 0
            free_mem = 0

        cpu_count = os.cpu_count() or 1

        return DeviceProperties(
            name=platform.processor() or "CPU",
            vendor=platform.system(),
            backend_type=BackendType.CPU,
            device_id=0,
            total_memory=total_mem,
            free_memory=free_mem,
            compute_capability=(0, 0),
            max_threads_per_block=cpu_count,
            max_work_group_size=cpu_count,
            warp_size=1,
            multiprocessor_count=cpu_count,
            supports_fp16=True,
            supports_bf16=False,
            supports_fp64=True,
            driver_version=platform.python_version(),
            is_available=True,
        )

    def allocate(self, size_bytes: int) -> int:
        """Allocate memory on CPU."""
        if size_bytes <= 0:
            raise BackendMemoryError(f"Invalid allocation size: {size_bytes}")

        try:
            buffer = bytearray(size_bytes)
            ptr = id(buffer)
            self._allocations[ptr] = buffer
            return ptr
        except MemoryError as e:
            raise BackendMemoryError(f"CPU allocation failed: {e}") from e

    def deallocate(self, ptr: int) -> None:
        """Free CPU memory."""
        if ptr in self._allocations:
            del self._allocations[ptr]

    def copy_to_device(
        self,
        dst: int,
        src: Union[int, bytes, Any],
        size_bytes: int,
    ) -> None:
        """Copy data (CPU to CPU is a memcpy)."""
        if dst not in self._allocations:
            raise BackendError(f"Invalid destination pointer: {dst}")

        dst_buffer = self._allocations[dst]

        if isinstance(src, bytes):
            dst_buffer[:size_bytes] = src[:size_bytes]
        elif isinstance(src, bytearray):
            dst_buffer[:size_bytes] = src[:size_bytes]
        elif hasattr(src, "tobytes"):  # numpy array
            data = src.tobytes()
            dst_buffer[:size_bytes] = data[:size_bytes]
        elif isinstance(src, int) and src in self._allocations:
            src_buffer = self._allocations[src]
            dst_buffer[:size_bytes] = src_buffer[:size_bytes]
        else:
            raise BackendError(f"Unsupported source type: {type(src)}")

    def copy_to_host(
        self,
        dst: Union[int, bytearray, Any],
        src: int,
        size_bytes: int,
    ) -> None:
        """Copy data from device (CPU) to host."""
        if src not in self._allocations:
            raise BackendError(f"Invalid source pointer: {src}")

        src_buffer = self._allocations[src]

        if isinstance(dst, bytearray):
            dst[:size_bytes] = src_buffer[:size_bytes]
        elif hasattr(dst, "view"):  # numpy array
            import numpy as np

            view = dst.view(np.uint8)
            view[:size_bytes] = np.frombuffer(
                bytes(src_buffer[:size_bytes]), dtype=np.uint8
            )
        elif isinstance(dst, int) and dst in self._allocations:
            dst_buffer = self._allocations[dst]
            dst_buffer[:size_bytes] = src_buffer[:size_bytes]
        else:
            raise BackendError(f"Unsupported destination type: {type(dst)}")

    def synchronize(self) -> None:
        """No-op for CPU (synchronous execution)."""
        pass
