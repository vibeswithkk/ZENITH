# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Backend Registry and Device Manager

Provides unified device management and backend registration following
CetakBiru.md Section 3.2 Line 247 - Kernel Dispatch and Selection.

Features:
- Dynamic backend registration
- Device string parsing ("cuda:0", "rocm:1", "oneapi:gpu")
- Fallback chain management
- Thread-safe singleton registry
"""

import logging
import threading
from typing import Dict, List, Optional, Type, Union

from .base import BackendType, BaseBackend, CPUBackend, DeviceProperties

logger = logging.getLogger("zenith.backends.registry")


class BackendRegistry:
    """
    Singleton registry for hardware backends.

    Mirrors the C++ BackendRegistry in core/include/zenith/backend.hpp.

    Thread Safety: All operations are protected by a lock.
    """

    _instance: Optional["BackendRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "BackendRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._backends: Dict[str, BaseBackend] = {}
        self._backend_classes: Dict[str, Type[BaseBackend]] = {}
        self._fallback_chain: List[str] = ["cuda", "rocm", "oneapi", "cpu"]
        self._default_backend: Optional[str] = None
        self._initialized = True

        # Auto-register built-in backends
        self._register_builtin_backends()

    def _register_builtin_backends(self) -> None:
        """Register built-in backend classes."""
        from .cuda_backend import CUDABackend
        from .rocm_backend import ROCmBackend
        from .oneapi_backend import OneAPIBackend

        self._backend_classes["cpu"] = CPUBackend
        self._backend_classes["cuda"] = CUDABackend
        self._backend_classes["rocm"] = ROCmBackend
        self._backend_classes["oneapi"] = OneAPIBackend

    def register_backend(self, backend: BaseBackend) -> None:
        """
        Register a backend instance.

        Args:
            backend: Backend instance to register.
        """
        with self._lock:
            key = f"{backend.name}:{backend.device_id}"
            self._backends[key] = backend
            logger.debug(f"Registered backend: {key}")

    def register_backend_class(
        self,
        name: str,
        backend_class: Type[BaseBackend],
    ) -> None:
        """
        Register a backend class for lazy instantiation.

        Args:
            name: Backend name (e.g., "cuda", "rocm").
            backend_class: Backend class to register.
        """
        with self._lock:
            self._backend_classes[name] = backend_class
            logger.debug(f"Registered backend class: {name}")

    def get(
        self,
        device: str,
        auto_init: bool = True,
    ) -> Optional[BaseBackend]:
        """
        Get a backend by device string.

        Args:
            device: Device string (e.g., "cuda:0", "rocm", "cpu").
            auto_init: Automatically initialize the backend if not ready.

        Returns:
            Backend instance or None if not available.
        """
        backend_name, device_id = self._parse_device_string(device)

        with self._lock:
            key = f"{backend_name}:{device_id}"

            # Check if already instantiated
            if key in self._backends:
                backend = self._backends[key]
                if auto_init and not backend.is_initialized:
                    backend.initialize()
                return backend

            # Try to create from class
            if backend_name in self._backend_classes:
                try:
                    backend_class = self._backend_classes[backend_name]
                    backend = backend_class(device_id=device_id)
                    self._backends[key] = backend
                    if auto_init and backend.is_available():
                        backend.initialize()
                    return backend
                except Exception as e:
                    logger.warning(f"Failed to create {key}: {e}")

        return None

    def get_default(self, auto_init: bool = True) -> Optional[BaseBackend]:
        """
        Get the default backend (first available in fallback chain).

        Args:
            auto_init: Automatically initialize the backend.

        Returns:
            First available backend or None.
        """
        if self._default_backend:
            backend = self.get(self._default_backend, auto_init)
            if backend and backend.is_available():
                return backend

        # Try fallback chain
        for backend_name in self._fallback_chain:
            backend = self.get(f"{backend_name}:0", auto_init)
            if backend and backend.is_available():
                self._default_backend = f"{backend_name}:0"
                return backend

        return None

    def set_default(self, device: str) -> None:
        """Set the default backend."""
        self._default_backend = device

    def set_fallback_chain(self, chain: List[str]) -> None:
        """
        Set the fallback chain for backend selection.

        Args:
            chain: List of backend names in order of preference.
        """
        self._fallback_chain = chain

    def list_backends(self) -> List[str]:
        """Get list of registered backend names."""
        with self._lock:
            return list(self._backend_classes.keys())

    def list_available(self) -> List[str]:
        """Get list of available backends."""
        available = []
        for name in self._backend_classes:
            try:
                backend = self.get(f"{name}:0", auto_init=False)
                if backend and backend.is_available():
                    available.append(name)
            except Exception:
                pass
        return available

    def get_all_devices(self) -> List[str]:
        """Get list of all available device strings."""
        devices = []
        for name in self.list_available():
            try:
                backend = self.get(f"{name}:0", auto_init=False)
                if backend:
                    count = backend.get_device_count()
                    for i in range(count):
                        devices.append(f"{name}:{i}")
            except Exception:
                pass
        return devices

    @staticmethod
    def _parse_device_string(device: str) -> tuple[str, int]:
        """
        Parse device string into backend name and device ID.

        Args:
            device: Device string (e.g., "cuda:0", "rocm", "oneapi:gpu").

        Returns:
            Tuple of (backend_name, device_id).
        """
        if ":" in device:
            parts = device.split(":", 1)
            backend_name = parts[0].lower()
            try:
                device_id = int(parts[1])
            except ValueError:
                # Handle "oneapi:gpu" style
                device_id = 0
        else:
            backend_name = device.lower()
            device_id = 0

        return backend_name, device_id

    def cleanup_all(self) -> None:
        """Cleanup all registered backends."""
        with self._lock:
            for backend in self._backends.values():
                try:
                    backend.cleanup()
                except Exception as e:
                    logger.warning(f"Cleanup failed for {backend.name}: {e}")
            self._backends.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.cleanup_all()
                cls._instance = None


class DeviceManager:
    """
    High-level device management interface.

    Provides convenient methods for device selection and memory operations.
    """

    def __init__(self):
        self._registry = BackendRegistry()
        self._current_device: Optional[str] = None

    def get_device(self, device: str) -> Optional[BaseBackend]:
        """
        Get a backend for the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "rocm").

        Returns:
            Initialized backend or None.
        """
        return self._registry.get(device, auto_init=True)

    def set_device(self, device: str) -> bool:
        """
        Set the current default device.

        Args:
            device: Device string.

        Returns:
            True if device was set successfully.
        """
        backend = self.get_device(device)
        if backend and backend.is_available():
            self._current_device = device
            self._registry.set_default(device)
            return True
        return False

    def get_current_device(self) -> Optional[BaseBackend]:
        """Get the current device backend."""
        if self._current_device:
            return self.get_device(self._current_device)
        return self._registry.get_default()

    def list_devices(self) -> List[str]:
        """List all available devices."""
        return self._registry.get_all_devices()

    def get_device_properties(
        self,
        device: Optional[str] = None,
    ) -> DeviceProperties:
        """
        Get properties of a device.

        Args:
            device: Device string, or None for current device.

        Returns:
            DeviceProperties object.
        """
        if device:
            backend = self.get_device(device)
        else:
            backend = self.get_current_device()

        if backend:
            return backend.get_device_properties()
        return DeviceProperties(is_available=False)

    def is_available(self, backend_name: str) -> bool:
        """
        Check if a backend type is available.

        Args:
            backend_name: Backend name (e.g., "cuda", "rocm").

        Returns:
            True if backend is available.
        """
        backend = self._registry.get(f"{backend_name}:0", auto_init=False)
        return backend is not None and backend.is_available()

    def synchronize(self, device: Optional[str] = None) -> None:
        """
        Synchronize a device.

        Args:
            device: Device string, or None for current device.
        """
        if device:
            backend = self.get_device(device)
        else:
            backend = self.get_current_device()

        if backend:
            backend.synchronize()


# Module-level convenience functions
_device_manager: Optional[DeviceManager] = None


def _get_device_manager() -> DeviceManager:
    """Get or create the global DeviceManager."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device(device: str) -> Optional[BaseBackend]:
    """Get a backend for the specified device."""
    return _get_device_manager().get_device(device)


def set_device(device: str) -> bool:
    """Set the current default device."""
    return _get_device_manager().set_device(device)


def get_current_device() -> Optional[BaseBackend]:
    """Get the current device backend."""
    return _get_device_manager().get_current_device()


def list_devices() -> List[str]:
    """List all available devices."""
    return _get_device_manager().list_devices()


def synchronize(device: Optional[str] = None) -> None:
    """Synchronize a device."""
    _get_device_manager().synchronize(device)
