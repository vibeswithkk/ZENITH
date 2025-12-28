# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith JAX Memory Management Module.

Provides unified memory management for JAX arrays with:
- Activation storage and retrieval
- Memory pooling and reuse
- CPU offloading for large tensors
- Memory profiling and monitoring

Technical Foundation:
--------------------
Based on:
- NVIDIA Megatron-LM activation memory management patterns
- PyTorch CUDA caching allocator concepts
- JAX device_put/device_get for memory movement

Memory Model:
------------
JAX arrays are immutable and managed by XLA. This module provides:
1. Layer-level activation caching for training
2. Memory budget enforcement via eviction policies
3. CPU offloading for activations exceeding threshold

Thread Safety:
-------------
All public methods are thread-safe using RLock for distributed training.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("zenith.jax.memory_manager")


def _get_jax():
    """Lazy import of JAX with error handling."""
    try:
        import jax

        return jax
    except ImportError as e:
        raise ImportError(
            "JAX is required for memory management. "
            "Install with: pip install jax jaxlib"
        ) from e


def _get_jnp():
    """Lazy import of jax.numpy."""
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for memory management. "
            "Install with: pip install jax jaxlib"
        ) from e


class EvictionPolicy(Enum):
    """
    Policy for evicting arrays from memory when under pressure.

    LRU: Least Recently Used - evict oldest accessed
    LFU: Least Frequently Used - evict least accessed
    SIZE: Size Priority - evict largest arrays first
    FIFO: First In First Out - evict oldest stored
    """

    LRU = "lru"
    LFU = "lfu"
    SIZE = "size"
    FIFO = "fifo"


class DeviceType(Enum):
    """Device types for memory allocation."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass
class JAXMemoryConfig:
    """
    Configuration for JAX memory management.

    Attributes:
        max_memory_bytes: Maximum memory budget (None = unlimited)
        eviction_policy: Policy for evicting arrays under memory pressure
        enable_offloading: Allow offloading to CPU when memory is full
        offload_threshold_bytes: Size threshold for automatic offloading
        enable_profiling: Track detailed memory statistics
        preallocate_fraction: Fraction of memory to preallocate (0.0-1.0)
    """

    max_memory_bytes: Optional[int] = None
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_offloading: bool = False
    offload_threshold_bytes: int = 100 * 1024 * 1024  # 100 MB
    enable_profiling: bool = False
    preallocate_fraction: float = 0.9


@dataclass
class ArrayMetadata:
    """
    Metadata for a stored JAX array.

    Tracks array properties and access patterns for memory management.
    """

    layer_id: int
    shape: Tuple[int, ...]
    dtype: Any
    device: str
    size_bytes: int
    creation_time: float
    access_count: int = 0
    last_access_time: float = 0.0
    is_checkpoint: bool = False
    is_offloaded: bool = False
    original_device: Optional[str] = None

    def update_access(self) -> None:
        """Update access statistics for LRU/LFU policies."""
        self.access_count += 1
        self.last_access_time = time.time()

    def __post_init__(self):
        """Initialize timing fields."""
        if self.last_access_time == 0.0:
            self.last_access_time = self.creation_time


@dataclass
class MemoryStats:
    """Runtime memory statistics."""

    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    total_allocated_bytes: int = 0
    total_freed_bytes: int = 0
    store_count: int = 0
    retrieve_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    offload_count: int = 0
    prefetch_count: int = 0


def compute_array_size(arr) -> int:
    """
    Compute size of a JAX array in bytes.

    Args:
        arr: JAX array, numpy array, or pytree containing arrays

    Returns:
        Total size in bytes
    """
    if hasattr(arr, "nbytes"):
        return int(arr.nbytes)

    if hasattr(arr, "size") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        element_size = dtype.itemsize if hasattr(dtype, "itemsize") else 4
        return int(arr.size * element_size)

    if isinstance(arr, (list, tuple)):
        return sum(compute_array_size(x) for x in arr)

    if isinstance(arr, dict):
        return sum(compute_array_size(v) for v in arr.values())

    return 8


def get_device_string(arr) -> str:
    """Get device string for a JAX array or numpy array."""
    if hasattr(arr, "devices"):
        try:
            devices = arr.devices()
            if devices:
                device = next(iter(devices))
                return str(device)
        except (TypeError, AttributeError):
            pass

    if hasattr(arr, "device"):
        device_attr = getattr(arr, "device", None)
        if callable(device_attr):
            try:
                return str(device_attr())
            except Exception:
                pass
        elif device_attr is not None:
            return str(device_attr)

    return "cpu"


class JAXActivationStore:
    """
    Memory pool for efficient JAX array storage and retrieval.

    Implements configurable eviction policies and memory tracking.
    Thread-safe for use in multi-device training.

    Design based on:
    - NVIDIA Megatron-LM activation memory management
    - PyTorch CUDA caching allocator concepts

    Example:
        store = JAXActivationStore(max_memory_bytes=1024 * 1024 * 1024)  # 1GB

        # Store activation
        store.store(layer_id=0, array=hidden_states, is_checkpoint=True)

        # Retrieve activation
        retrieved = store.retrieve(layer_id=0)

        # Get statistics
        print(store.statistics)
    """

    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        enable_profiling: bool = False,
    ):
        """
        Initialize activation store.

        Args:
            max_memory_bytes: Maximum memory budget (None = unlimited)
            eviction_policy: Policy for evicting arrays
            enable_profiling: Track detailed timing statistics
        """
        self._store: Dict[int, Any] = {}
        self._metadata: Dict[int, ArrayMetadata] = {}
        self._access_order: OrderedDict[int, float] = OrderedDict()
        self._max_memory = max_memory_bytes
        self._current_memory = 0
        self._eviction_policy = eviction_policy
        self._enable_profiling = enable_profiling
        self._lock = threading.RLock()
        self._stats = MemoryStats()

    def store(
        self,
        layer_id: int,
        array: Any,
        is_checkpoint: bool = False,
    ) -> bool:
        """
        Store a JAX array.

        Args:
            layer_id: Unique identifier for the layer
            array: The JAX array to store
            is_checkpoint: Whether this is a checkpoint (protected from eviction)

        Returns:
            True if stored successfully, False if eviction failed
        """
        with self._lock:
            size_bytes = compute_array_size(array)

            if hasattr(array, "shape"):
                shape = tuple(array.shape)
            else:
                shape = ()

            if hasattr(array, "dtype"):
                dtype = array.dtype
            else:
                dtype = None

            device = get_device_string(array)

            if layer_id in self._store:
                old_size = self._metadata[layer_id].size_bytes
                self._current_memory -= old_size
                self._stats.total_freed_bytes += old_size

            if self._max_memory is not None:
                while (
                    self._current_memory + size_bytes > self._max_memory
                    and len(self._store) > 0
                ):
                    if not self._evict_one():
                        return False

            self._store[layer_id] = array
            self._metadata[layer_id] = ArrayMetadata(
                layer_id=layer_id,
                shape=shape,
                dtype=dtype,
                device=device,
                size_bytes=size_bytes,
                creation_time=time.time(),
                is_checkpoint=is_checkpoint,
            )
            self._access_order[layer_id] = time.time()
            self._current_memory += size_bytes

            self._stats.store_count += 1
            self._stats.total_allocated_bytes += size_bytes
            self._stats.current_memory_bytes = self._current_memory
            self._stats.peak_memory_bytes = max(
                self._stats.peak_memory_bytes, self._current_memory
            )

            return True

    def retrieve(self, layer_id: int) -> Optional[Any]:
        """
        Retrieve a JAX array.

        Args:
            layer_id: Unique identifier for the layer

        Returns:
            The array if found, None otherwise
        """
        with self._lock:
            self._stats.retrieve_count += 1

            if layer_id in self._store:
                self._metadata[layer_id].update_access()
                self._access_order.move_to_end(layer_id)
                self._stats.hit_count += 1
                return self._store[layer_id]
            else:
                self._stats.miss_count += 1
                return None

    def remove(self, layer_id: int) -> Optional[Any]:
        """
        Remove and return a JAX array.

        Args:
            layer_id: Unique identifier for the layer

        Returns:
            The array if found, None otherwise
        """
        with self._lock:
            if layer_id in self._store:
                array = self._store.pop(layer_id)
                metadata = self._metadata.pop(layer_id)
                self._access_order.pop(layer_id, None)
                self._current_memory -= metadata.size_bytes
                self._stats.total_freed_bytes += metadata.size_bytes
                self._stats.current_memory_bytes = self._current_memory
                return array
            return None

    def _evict_one(self) -> bool:
        """
        Evict one array based on policy.

        Returns:
            True if eviction was successful, False if no evictable items
        """
        candidates = [
            (lid, meta)
            for lid, meta in self._metadata.items()
            if not meta.is_checkpoint
        ]

        if not candidates:
            return False

        if self._eviction_policy == EvictionPolicy.LRU:
            victim_id = min(candidates, key=lambda x: x[1].last_access_time)[0]
        elif self._eviction_policy == EvictionPolicy.LFU:
            victim_id = min(candidates, key=lambda x: x[1].access_count)[0]
        elif self._eviction_policy == EvictionPolicy.SIZE:
            victim_id = max(candidates, key=lambda x: x[1].size_bytes)[0]
        elif self._eviction_policy == EvictionPolicy.FIFO:
            victim_id = min(candidates, key=lambda x: x[1].creation_time)[0]
        else:
            victim_id = candidates[0][0]

        victim_meta = self._metadata[victim_id]
        self._store.pop(victim_id)
        self._metadata.pop(victim_id)
        self._access_order.pop(victim_id, None)
        self._current_memory -= victim_meta.size_bytes

        self._stats.eviction_count += 1
        self._stats.total_freed_bytes += victim_meta.size_bytes
        self._stats.current_memory_bytes = self._current_memory

        return True

    def clear(self) -> int:
        """
        Clear all stored arrays.

        Returns:
            Number of arrays cleared
        """
        with self._lock:
            count = len(self._store)

            self._stats.total_freed_bytes += self._current_memory

            self._store.clear()
            self._metadata.clear()
            self._access_order.clear()
            self._current_memory = 0
            self._stats.current_memory_bytes = 0

            return count

    def contains(self, layer_id: int) -> bool:
        """Check if layer_id exists in store."""
        with self._lock:
            return layer_id in self._store

    def __len__(self) -> int:
        """Return number of stored arrays."""
        return len(self._store)

    def __contains__(self, layer_id: int) -> bool:
        """Support 'in' operator."""
        return self.contains(layer_id)

    @property
    def memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._current_memory

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in megabytes."""
        return self._current_memory / (1024 * 1024)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                "current_memory_bytes": self._current_memory,
                "current_memory_mb": self._current_memory / (1024 * 1024),
                "peak_memory_bytes": self._stats.peak_memory_bytes,
                "peak_memory_mb": self._stats.peak_memory_bytes / (1024 * 1024),
                "stored_count": len(self._store),
                "checkpoint_count": sum(
                    1 for m in self._metadata.values() if m.is_checkpoint
                ),
                "store_count": self._stats.store_count,
                "retrieve_count": self._stats.retrieve_count,
                "hit_count": self._stats.hit_count,
                "miss_count": self._stats.miss_count,
                "hit_rate": (
                    self._stats.hit_count / max(1, self._stats.retrieve_count)
                ),
                "eviction_count": self._stats.eviction_count,
                "total_allocated_bytes": self._stats.total_allocated_bytes,
                "total_freed_bytes": self._stats.total_freed_bytes,
            }


class JAXMemoryManager:
    """
    High-level memory manager for JAX training.

    Provides:
    - Activation store with configurable policies
    - CPU offloading for large activations
    - Memory monitoring and profiling
    - Automatic memory cleanup

    Example:
        config = JAXMemoryConfig(
            max_memory_bytes=8 * 1024 * 1024 * 1024,  # 8GB
            enable_offloading=True,
            offload_threshold_bytes=500 * 1024 * 1024,  # 500MB
        )

        manager = JAXMemoryManager(config)

        # Store activation
        manager.store(layer_id=0, array=hidden_states)

        # Get with automatic offloading
        retrieved = manager.retrieve(layer_id=0)

        # Monitor memory
        print(f"Memory: {manager.memory_usage_mb:.2f} MB")
    """

    def __init__(self, config: Optional[JAXMemoryConfig] = None):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration (uses defaults if None)
        """
        self._config = config if config else JAXMemoryConfig()
        self._store = JAXActivationStore(
            max_memory_bytes=self._config.max_memory_bytes,
            eviction_policy=self._config.eviction_policy,
            enable_profiling=self._config.enable_profiling,
        )
        self._offloaded: Dict[int, Any] = {}
        self._offload_metadata: Dict[int, ArrayMetadata] = {}
        self._lock = threading.RLock()
        self._stats = MemoryStats()

    @property
    def config(self) -> JAXMemoryConfig:
        """Get current configuration."""
        return self._config

    @property
    def memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._store.memory_usage

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in megabytes."""
        return self._store.memory_usage_mb

    def store(
        self,
        layer_id: int,
        array: Any,
        is_checkpoint: bool = False,
        allow_offload: bool = True,
    ) -> bool:
        """
        Store a JAX array with optional CPU offloading.

        Args:
            layer_id: Unique identifier for the layer
            array: The JAX array to store
            is_checkpoint: Whether this is a checkpoint
            allow_offload: Allow automatic CPU offloading

        Returns:
            True if stored successfully
        """
        jax = _get_jax()

        size_bytes = compute_array_size(array)

        if (
            self._config.enable_offloading
            and allow_offload
            and size_bytes >= self._config.offload_threshold_bytes
        ):
            return self._offload_to_cpu(layer_id, array)

        return self._store.store(layer_id, array, is_checkpoint=is_checkpoint)

    def retrieve(self, layer_id: int, prefetch: bool = True) -> Optional[Any]:
        """
        Retrieve a JAX array, prefetching from CPU if offloaded.

        Args:
            layer_id: Unique identifier for the layer
            prefetch: Whether to move back to accelerator if offloaded

        Returns:
            The array if found, None otherwise
        """
        jax = _get_jax()

        result = self._store.retrieve(layer_id)
        if result is not None:
            return result

        with self._lock:
            if layer_id in self._offloaded:
                array = self._offloaded[layer_id]

                if prefetch:
                    metadata = self._offload_metadata.get(layer_id)
                    if metadata and metadata.original_device:
                        try:
                            devices = jax.devices(metadata.original_device)
                            if devices:
                                array = jax.device_put(array, devices[0])
                                self._stats.prefetch_count += 1
                        except Exception:
                            pass

                return array

        return None

    def remove(self, layer_id: int) -> Optional[Any]:
        """
        Remove and return a stored array.

        Args:
            layer_id: Unique identifier for the layer

        Returns:
            The array if found, None otherwise
        """
        result = self._store.remove(layer_id)
        if result is not None:
            return result

        with self._lock:
            if layer_id in self._offloaded:
                array = self._offloaded.pop(layer_id)
                self._offload_metadata.pop(layer_id, None)
                return array

        return None

    def _offload_to_cpu(self, layer_id: int, array: Any) -> bool:
        """
        Offload array to CPU memory.

        Args:
            layer_id: Layer identifier
            array: Array to offload

        Returns:
            True if successful
        """
        jax = _get_jax()

        try:
            original_device = get_device_string(array)

            cpu_devices = jax.devices("cpu")
            if not cpu_devices:
                return self._store.store(layer_id, array, is_checkpoint=False)

            cpu_array = jax.device_put(array, cpu_devices[0])

            with self._lock:
                self._offloaded[layer_id] = cpu_array
                self._offload_metadata[layer_id] = ArrayMetadata(
                    layer_id=layer_id,
                    shape=tuple(array.shape) if hasattr(array, "shape") else (),
                    dtype=array.dtype if hasattr(array, "dtype") else None,
                    device="cpu",
                    size_bytes=compute_array_size(array),
                    creation_time=time.time(),
                    is_offloaded=True,
                    original_device=original_device,
                )
                self._stats.offload_count += 1

            return True

        except Exception as e:
            logger.warning(f"Failed to offload to CPU: {e}")
            return self._store.store(layer_id, array, is_checkpoint=False)

    def offload(self, array: Any) -> Any:
        """
        Manually offload an array to CPU.

        Args:
            array: Array to offload

        Returns:
            CPU array
        """
        jax = _get_jax()

        try:
            cpu_devices = jax.devices("cpu")
            if cpu_devices:
                return jax.device_put(array, cpu_devices[0])
        except Exception as e:
            logger.warning(f"Failed to offload: {e}")

        return array

    def prefetch(self, array: Any, device: str = "gpu") -> Any:
        """
        Prefetch an array to accelerator.

        Args:
            array: Array to prefetch
            device: Target device type

        Returns:
            Array on target device
        """
        jax = _get_jax()

        try:
            devices = jax.devices(device)
            if devices:
                return jax.device_put(array, devices[0])
        except Exception as e:
            logger.warning(f"Failed to prefetch to {device}: {e}")

        return array

    def clear(self) -> int:
        """
        Clear all stored arrays.

        Returns:
            Total number of arrays cleared
        """
        count = self._store.clear()

        with self._lock:
            count += len(self._offloaded)
            self._offloaded.clear()
            self._offload_metadata.clear()

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        store_stats = self._store.statistics

        with self._lock:
            offloaded_size = sum(m.size_bytes for m in self._offload_metadata.values())

            return {
                **store_stats,
                "offloaded_count": len(self._offloaded),
                "offloaded_bytes": offloaded_size,
                "offloaded_mb": offloaded_size / (1024 * 1024),
                "offload_operations": self._stats.offload_count,
                "prefetch_operations": self._stats.prefetch_count,
                "total_managed_bytes": store_stats["current_memory_bytes"]
                + offloaded_size,
                "total_managed_mb": (
                    store_stats["current_memory_bytes"] + offloaded_size
                )
                / (1024 * 1024),
            }


def get_device_memory_info() -> Dict[str, Any]:
    """
    Get memory information for available devices.

    Returns:
        Dict with device memory information
    """
    jax = _get_jax()

    info = {
        "devices": [],
        "total_available": 0,
    }

    try:
        for device in jax.devices():
            device_info = {
                "id": str(device),
                "kind": device.platform,
            }

            if hasattr(device, "memory_stats"):
                try:
                    stats = device.memory_stats()
                    if stats:
                        device_info["memory"] = stats
                except Exception:
                    pass

            info["devices"].append(device_info)

    except Exception as e:
        logger.warning(f"Failed to get device info: {e}")

    return info


__all__ = [
    "EvictionPolicy",
    "DeviceType",
    "JAXMemoryConfig",
    "ArrayMetadata",
    "MemoryStats",
    "JAXActivationStore",
    "JAXMemoryManager",
    "compute_array_size",
    "get_device_string",
    "get_device_memory_info",
]
