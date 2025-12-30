# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
GPU Metrics Collector for Zenith

Provides real-time GPU metrics using NVIDIA Management Library (pynvml).
Falls back gracefully when pynvml is not available.

Example:
    from zenith.observability import gpu_metrics

    if gpu_metrics.is_available():
        stats = gpu_metrics.get_current()
        print(f"GPU Utilization: {stats['utilization_percent']}%")
        print(f"Memory Used: {stats['memory_used_mb']} MB")
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import threading

# Attempt to import pynvml, gracefully handle if not available
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None


@dataclass
class GPUStats:
    """
    GPU statistics snapshot.

    Attributes:
        device_index: GPU device index
        name: GPU device name
        utilization_percent: GPU core utilization (0-100)
        memory_used_mb: Memory currently in use (MB)
        memory_total_mb: Total GPU memory (MB)
        memory_free_mb: Available GPU memory (MB)
        temperature_celsius: GPU temperature in Celsius
        power_draw_watts: Current power consumption (Watts)
        power_limit_watts: Power limit setting (Watts)
    """

    device_index: int
    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_index": self.device_index,
            "name": self.name,
            "utilization_percent": self.utilization_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_utilization_percent": round(
                (self.memory_used_mb / self.memory_total_mb) * 100, 2
            )
            if self.memory_total_mb > 0
            else 0,
            "temperature_celsius": self.temperature_celsius,
            "power_draw_watts": self.power_draw_watts,
            "power_limit_watts": self.power_limit_watts,
        }


class GPUMetricsCollector:
    """
    Collects GPU metrics using NVIDIA Management Library.

    Thread-safe singleton pattern ensures consistent access across the framework.

    Example:
        collector = GPUMetricsCollector.get()
        if collector.is_available():
            stats = collector.get_stats(device_index=0)
            print(f"GPU Temp: {stats.temperature_celsius}C")
    """

    _instance: Optional["GPUMetricsCollector"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize NVML if not already done."""
        if self._initialized:
            return

        self._available = False
        self._device_count = 0
        self._handles: Dict[int, Any] = {}

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._device_count = pynvml.nvmlDeviceGetCount()
                for i in range(self._device_count):
                    self._handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
                self._available = True
            except Exception:
                self._available = False

        GPUMetricsCollector._initialized = True

    @classmethod
    def get(cls) -> "GPUMetricsCollector":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None and cls._instance._available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        cls._instance = None
        cls._initialized = False

    def is_available(self) -> bool:
        """Check if GPU metrics collection is available."""
        return self._available

    def get_device_count(self) -> int:
        """Get number of available GPU devices."""
        return self._device_count

    def get_stats(self, device_index: int = 0) -> Optional[GPUStats]:
        """
        Get current GPU statistics for specified device.

        Args:
            device_index: GPU device index (default: 0)

        Returns:
            GPUStats object or None if not available
        """
        if not self._available:
            return None

        if device_index not in self._handles:
            return None

        handle = self._handles[device_index]

        try:
            # Device name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_percent = float(utilization.gpu)

            # Memory
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            memory_free_mb = memory.free / (1024 * 1024)

            # Temperature (may not be available on all GPUs)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = None

            # Power (may not be available on all GPUs)
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                power_draw = None
                power_limit = None

            return GPUStats(
                device_index=device_index,
                name=name,
                utilization_percent=util_percent,
                memory_used_mb=round(memory_used_mb, 2),
                memory_total_mb=round(memory_total_mb, 2),
                memory_free_mb=round(memory_free_mb, 2),
                temperature_celsius=temp,
                power_draw_watts=round(power_draw, 2) if power_draw else None,
                power_limit_watts=round(power_limit, 2) if power_limit else None,
            )

        except Exception:
            return None

    def get_all_stats(self) -> Dict[int, GPUStats]:
        """
        Get statistics for all available GPU devices.

        Returns:
            Dictionary mapping device index to GPUStats
        """
        result = {}
        for i in range(self._device_count):
            stats = self.get_stats(i)
            if stats:
                result[i] = stats
        return result


# Module-level convenience functions
_collector: Optional[GPUMetricsCollector] = None


def is_available() -> bool:
    """Check if GPU metrics are available."""
    global _collector
    if _collector is None:
        _collector = GPUMetricsCollector.get()
    return _collector.is_available()


def get_current(device_index: int = 0) -> Optional[Dict[str, Any]]:
    """
    Get current GPU metrics as dictionary.

    Args:
        device_index: GPU device index (default: 0)

    Returns:
        Dictionary with GPU stats or None if not available
    """
    global _collector
    if _collector is None:
        _collector = GPUMetricsCollector.get()

    stats = _collector.get_stats(device_index)
    return stats.to_dict() if stats else None


def get_memory_info(device_index: int = 0) -> Optional[Dict[str, float]]:
    """
    Get GPU memory information.

    Returns:
        Dictionary with used_mb, total_mb, free_mb, utilization_percent
    """
    stats = get_current(device_index)
    if not stats:
        return None

    return {
        "used_mb": stats["memory_used_mb"],
        "total_mb": stats["memory_total_mb"],
        "free_mb": stats["memory_free_mb"],
        "utilization_percent": stats["memory_utilization_percent"],
    }


def get_utilization(device_index: int = 0) -> Optional[float]:
    """Get GPU core utilization percentage (0-100)."""
    stats = get_current(device_index)
    return stats["utilization_percent"] if stats else None
