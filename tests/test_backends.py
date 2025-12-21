# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Test Suite for Hardware Backends

Tests for:
- Base backend interface
- CPU backend (always available)
- CUDA backend (when available)
- ROCm backend (when available)
- oneAPI backend (when available)
- Backend registry
- Device manager
"""

import pytest
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def registry():
    """Get fresh backend registry."""
    from zenith.backends.registry import BackendRegistry

    BackendRegistry.reset()
    return BackendRegistry()


@pytest.fixture
def cpu_backend():
    """Create CPU backend."""
    from zenith.backends import CPUBackend

    return CPUBackend()


@pytest.fixture
def cuda_backend():
    """Create CUDA backend if available."""
    from zenith.backends import CUDABackend, is_cuda_available

    if not is_cuda_available():
        pytest.skip("CUDA not available")
    return CUDABackend()


@pytest.fixture
def rocm_backend():
    """Create ROCm backend if available."""
    from zenith.backends import ROCmBackend, is_rocm_available

    if not is_rocm_available():
        pytest.skip("ROCm not available")
    return ROCmBackend()


@pytest.fixture
def oneapi_backend():
    """Create oneAPI backend if available."""
    from zenith.backends import OneAPIBackend, is_oneapi_available

    if not is_oneapi_available():
        pytest.skip("oneAPI not available")
    return OneAPIBackend()


# =============================================================================
# Base Backend Interface Tests
# =============================================================================


class TestBaseBackendInterface:
    """Tests for base backend interface compliance."""

    def test_backend_has_name(self, cpu_backend):
        """All backends must have a name property."""
        assert hasattr(cpu_backend, "name")
        assert isinstance(cpu_backend.name, str)
        assert len(cpu_backend.name) > 0

    def test_backend_has_backend_type(self, cpu_backend):
        """All backends must have a backend_type property."""
        from zenith.backends import BackendType

        assert hasattr(cpu_backend, "backend_type")
        assert isinstance(cpu_backend.backend_type, BackendType)

    def test_backend_has_is_available(self, cpu_backend):
        """All backends must have is_available method."""
        assert hasattr(cpu_backend, "is_available")
        result = cpu_backend.is_available()
        assert isinstance(result, bool)

    def test_backend_has_device_id(self, cpu_backend):
        """All backends must have device_id property."""
        assert hasattr(cpu_backend, "device_id")
        assert isinstance(cpu_backend.device_id, int)

    def test_backend_has_memory_methods(self, cpu_backend):
        """All backends must have memory management methods."""
        assert hasattr(cpu_backend, "allocate")
        assert hasattr(cpu_backend, "deallocate")
        assert hasattr(cpu_backend, "copy_to_device")
        assert hasattr(cpu_backend, "copy_to_host")

    def test_backend_has_synchronize(self, cpu_backend):
        """All backends must have synchronize method."""
        assert hasattr(cpu_backend, "synchronize")

    def test_backend_context_manager(self, cpu_backend):
        """Backends should support context manager protocol."""
        from zenith.backends import CPUBackend

        with CPUBackend() as backend:
            assert backend.is_initialized

    def test_backend_repr(self, cpu_backend):
        """Backends should have meaningful repr."""
        repr_str = repr(cpu_backend)
        assert "cpu" in repr_str.lower()


# =============================================================================
# CPU Backend Tests
# =============================================================================


class TestCPUBackend:
    """Tests for CPU backend."""

    def test_cpu_always_available(self, cpu_backend):
        """CPU backend should always be available."""
        assert cpu_backend.is_available() is True

    def test_cpu_name(self, cpu_backend):
        """CPU backend should have correct name."""
        assert cpu_backend.name == "cpu"

    def test_cpu_device_properties(self, cpu_backend):
        """CPU backend should return valid device properties."""
        props = cpu_backend.get_device_properties()
        assert props.is_available is True
        assert props.vendor is not None

    def test_cpu_allocate_deallocate(self, cpu_backend):
        """CPU backend should allocate and deallocate memory."""
        size = 1024
        ptr = cpu_backend.allocate(size)
        assert ptr != 0
        cpu_backend.deallocate(ptr)

    def test_cpu_allocate_zero_fails(self, cpu_backend):
        """Allocating zero bytes should fail."""
        from zenith.backends import BackendMemoryError

        with pytest.raises(BackendMemoryError):
            cpu_backend.allocate(0)

    def test_cpu_allocate_negative_fails(self, cpu_backend):
        """Allocating negative bytes should fail."""
        from zenith.backends import BackendMemoryError

        with pytest.raises(BackendMemoryError):
            cpu_backend.allocate(-100)

    def test_cpu_copy_bytes(self, cpu_backend):
        """CPU backend should copy bytes to and from memory."""
        size = 256
        ptr = cpu_backend.allocate(size)

        # Copy to device
        src_data = bytes(range(256))
        cpu_backend.copy_to_device(ptr, src_data, size)

        # Copy back
        dst_data = bytearray(size)
        cpu_backend.copy_to_host(dst_data, ptr, size)

        assert bytes(dst_data) == src_data
        cpu_backend.deallocate(ptr)

    def test_cpu_copy_numpy(self, cpu_backend):
        """CPU backend should copy numpy arrays."""
        size = 100 * 4  # 100 float32s
        ptr = cpu_backend.allocate(size)

        # Create numpy array
        src_arr = np.arange(100, dtype=np.float32)
        cpu_backend.copy_to_device(ptr, src_arr, size)

        # Copy back
        dst_arr = np.zeros(100, dtype=np.float32)
        cpu_backend.copy_to_host(dst_arr, ptr, size)

        np.testing.assert_array_equal(dst_arr, src_arr)
        cpu_backend.deallocate(ptr)

    def test_cpu_synchronize(self, cpu_backend):
        """CPU synchronize should be a no-op."""
        cpu_backend.synchronize()  # Should not raise


# =============================================================================
# CUDA Backend Tests
# =============================================================================


class TestCUDABackend:
    """Tests for CUDA backend (skipped if not available)."""

    def test_cuda_name(self, cuda_backend):
        """CUDA backend should have correct name."""
        assert cuda_backend.name == "cuda"

    def test_cuda_device_properties(self, cuda_backend):
        """CUDA backend should return valid device properties."""
        props = cuda_backend.get_device_properties()
        assert props.is_available is True
        assert props.vendor == "NVIDIA"
        assert props.total_memory > 0

    def test_cuda_device_count(self, cuda_backend):
        """CUDA backend should report device count."""
        count = cuda_backend.get_device_count()
        assert count >= 1

    def test_cuda_initialize(self, cuda_backend):
        """CUDA backend should initialize."""
        assert cuda_backend.initialize() is True
        assert cuda_backend.is_initialized is True

    def test_cuda_allocate_deallocate(self, cuda_backend):
        """CUDA backend should allocate device memory."""
        cuda_backend.initialize()
        size = 1024 * 1024  # 1 MB
        ptr = cuda_backend.allocate(size)
        assert ptr != 0
        cuda_backend.deallocate(ptr)

    def test_cuda_memory_transfer(self, cuda_backend):
        """CUDA backend should transfer data."""
        cuda_backend.initialize()
        size = 1000 * 4  # 1000 float32s

        ptr = cuda_backend.allocate(size)
        src = np.random.randn(1000).astype(np.float32)
        dst = np.zeros(1000, dtype=np.float32)

        cuda_backend.copy_to_device(ptr, src, size)
        cuda_backend.synchronize()
        cuda_backend.copy_to_host(dst, ptr, size)

        np.testing.assert_array_almost_equal(dst, src, decimal=5)
        cuda_backend.deallocate(ptr)


# =============================================================================
# ROCm Backend Tests
# =============================================================================


class TestROCmBackend:
    """Tests for ROCm backend (skipped if not available)."""

    def test_rocm_name(self, rocm_backend):
        """ROCm backend should have correct name."""
        assert rocm_backend.name == "rocm"

    def test_rocm_device_properties(self, rocm_backend):
        """ROCm backend should return valid device properties."""
        props = rocm_backend.get_device_properties()
        assert props.is_available is True
        assert props.vendor == "AMD"
        assert props.warp_size == 64  # AMD wavefront size

    def test_rocm_device_count(self, rocm_backend):
        """ROCm backend should report device count."""
        count = rocm_backend.get_device_count()
        assert count >= 1


# =============================================================================
# oneAPI Backend Tests
# =============================================================================


class TestOneAPIBackend:
    """Tests for oneAPI backend (skipped if not available)."""

    def test_oneapi_name(self, oneapi_backend):
        """oneAPI backend should have correct name."""
        assert oneapi_backend.name == "oneapi"

    def test_oneapi_device_properties(self, oneapi_backend):
        """oneAPI backend should return valid device properties."""
        props = oneapi_backend.get_device_properties()
        assert props.is_available is True


# =============================================================================
# Backend Registry Tests
# =============================================================================


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_registry_singleton(self):
        """Registry should be a singleton."""
        from zenith.backends.registry import BackendRegistry

        reg1 = BackendRegistry()
        reg2 = BackendRegistry()
        assert reg1 is reg2

    def test_registry_list_backends(self, registry):
        """Registry should list available backend types."""
        backends = registry.list_backends()
        assert "cpu" in backends
        assert "cuda" in backends
        assert "rocm" in backends
        assert "oneapi" in backends

    def test_registry_get_cpu(self, registry):
        """Registry should return CPU backend."""
        backend = registry.get("cpu:0")
        assert backend is not None
        assert backend.name == "cpu"

    def test_registry_get_default(self, registry):
        """Registry should return a default backend."""
        backend = registry.get_default()
        assert backend is not None
        assert backend.is_available()

    def test_registry_parse_device_string(self, registry):
        """Registry should parse device strings correctly."""
        assert registry._parse_device_string("cuda:0") == ("cuda", 0)
        assert registry._parse_device_string("rocm:1") == ("rocm", 1)
        assert registry._parse_device_string("cpu") == ("cpu", 0)
        assert registry._parse_device_string("CUDA:2") == ("cuda", 2)


# =============================================================================
# Device Manager Tests
# =============================================================================


class TestDeviceManager:
    """Tests for device manager."""

    def test_device_manager_get_device(self):
        """DeviceManager should get devices."""
        from zenith.backends.registry import DeviceManager

        dm = DeviceManager()
        device = dm.get_device("cpu:0")
        assert device is not None
        assert device.name == "cpu"

    def test_device_manager_list_devices(self):
        """DeviceManager should list available devices."""
        from zenith.backends.registry import DeviceManager

        dm = DeviceManager()
        devices = dm.list_devices()
        assert "cpu:0" in devices

    def test_device_manager_set_device(self):
        """DeviceManager should set current device."""
        from zenith.backends.registry import DeviceManager

        dm = DeviceManager()
        result = dm.set_device("cpu:0")
        assert result is True
        current = dm.get_current_device()
        assert current is not None
        assert current.name == "cpu"


# =============================================================================
# Module-level API Tests
# =============================================================================


class TestModuleLevelAPI:
    """Tests for module-level convenience functions."""

    def test_is_cpu_available(self):
        """is_cpu_available should return True."""
        from zenith.backends import is_cpu_available

        assert is_cpu_available() is True

    def test_is_cuda_available_type(self):
        """is_cuda_available should return bool."""
        from zenith.backends import is_cuda_available

        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_is_rocm_available_type(self):
        """is_rocm_available should return bool."""
        from zenith.backends import is_rocm_available

        result = is_rocm_available()
        assert isinstance(result, bool)

    def test_is_oneapi_available_type(self):
        """is_oneapi_available should return bool."""
        from zenith.backends import is_oneapi_available

        result = is_oneapi_available()
        assert isinstance(result, bool)

    def test_get_available_backends(self):
        """get_available_backends should return list with cpu."""
        from zenith.backends import get_available_backends

        backends = get_available_backends()
        assert isinstance(backends, list)
        assert "cpu" in backends

    def test_list_devices(self):
        """list_devices should return device strings."""
        from zenith.backends import list_devices

        devices = list_devices()
        assert isinstance(devices, list)
        assert any("cpu" in d for d in devices)

    def test_get_device(self):
        """get_device should return backend."""
        from zenith.backends import get_device

        device = get_device("cpu:0")
        assert device is not None

    def test_create_backend(self):
        """create_backend should return backend."""
        from zenith.backends import create_backend

        backend = create_backend("cpu:0")
        assert backend is not None
        assert backend.name == "cpu"

    def test_create_backend_invalid_raises(self):
        """create_backend with invalid should raise."""
        from zenith.backends import (
            BackendNotAvailableError,
            create_backend,
        )

        with pytest.raises(BackendNotAvailableError):
            create_backend("nonexistent:0")


# =============================================================================
# Data Type Tests
# =============================================================================


class TestDataTypes:
    """Tests for data type support."""

    def test_cpu_supports_float32(self, cpu_backend):
        """CPU should support float32."""
        from zenith.backends import DataType

        assert cpu_backend.supports_dtype(DataType.Float32) is True

    def test_cpu_supports_int32(self, cpu_backend):
        """CPU should support int32."""
        from zenith.backends import DataType

        assert cpu_backend.supports_dtype(DataType.Int32) is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_pointer_deallocate(self, cpu_backend):
        """Deallocating invalid pointer should not crash."""
        cpu_backend.deallocate(123456789)  # Should log warning, not crash

    def test_copy_invalid_destination(self, cpu_backend):
        """Copy to invalid destination should raise."""
        from zenith.backends import BackendError

        with pytest.raises(BackendError):
            cpu_backend.copy_to_device(999999, b"test", 4)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBackendIntegration:
    """Integration tests across backends."""

    def test_fallback_chain(self):
        """Test fallback chain from CUDA to CPU."""
        from zenith.backends import get_device

        # Try to get any available GPU, fall back to CPU
        device = None
        for backend_name in ["cuda", "rocm", "oneapi", "cpu"]:
            device = get_device(f"{backend_name}:0")
            if device and device.is_available():
                break

        assert device is not None
        assert device.is_available()

    def test_multiple_backends_coexist(self):
        """Multiple backends can be instantiated."""
        from zenith.backends import CPUBackend, CUDABackend

        cpu = CPUBackend()
        cuda = CUDABackend()

        assert cpu.is_available()
        # CUDA may or may not be available
        assert isinstance(cuda.is_available(), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
