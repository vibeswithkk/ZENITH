"""
Hardware Profiling Module

Provides hardware detection and profiling utilities for auto-tuning:
- CUDA device detection and capability querying
- ROCm device detection
- CPU feature detection (SIMD, cache sizes)
- Hardware-aware constraint generation

Based on CetakBiru Section 6.3 Hardware Abstraction requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeviceType(Enum):
    """Hardware device types."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    UNKNOWN = "unknown"


class CPUVendor(Enum):
    """CPU vendor identification."""

    INTEL = "intel"
    AMD = "amd"
    ARM = "arm"
    APPLE = "apple"
    UNKNOWN = "unknown"


@dataclass
class CacheInfo:
    """CPU cache information."""

    l1_data_kb: int = 0
    l1_instruction_kb: int = 0
    l2_kb: int = 0
    l3_kb: int = 0
    cache_line_bytes: int = 64


@dataclass
class SIMDFeatures:
    """SIMD instruction set support."""

    sse: bool = False
    sse2: bool = False
    sse3: bool = False
    ssse3: bool = False
    sse4_1: bool = False
    sse4_2: bool = False
    avx: bool = False
    avx2: bool = False
    avx512f: bool = False
    avx512vnni: bool = False
    neon: bool = False
    sve: bool = False

    @property
    def max_vector_width(self) -> int:
        """Get maximum SIMD vector width in bits."""
        if self.avx512f:
            return 512
        elif self.avx or self.avx2:
            return 256
        elif self.sse or self.sse2:
            return 128
        elif self.neon:
            return 128
        elif self.sve:
            return 256  # Variable, using common width
        return 64


@dataclass
class CPUInfo:
    """CPU hardware information."""

    vendor: CPUVendor = CPUVendor.UNKNOWN
    model_name: str = ""
    num_cores: int = 1
    num_threads: int = 1
    frequency_mhz: float = 0.0
    cache: CacheInfo = field(default_factory=CacheInfo)
    simd: SIMDFeatures = field(default_factory=SIMDFeatures)
    architecture: str = ""

    @property
    def supports_vnni(self) -> bool:
        """Check if CPU supports VNNI for INT8 acceleration."""
        return self.simd.avx512vnni


@dataclass
class CUDADeviceInfo:
    """CUDA device information."""

    device_id: int = 0
    name: str = ""
    compute_capability: tuple[int, int] = (0, 0)
    total_memory_mb: int = 0
    multiprocessor_count: int = 0
    max_threads_per_block: int = 0
    max_block_dim: tuple[int, int, int] = (0, 0, 0)
    max_grid_dim: tuple[int, int, int] = (0, 0, 0)
    warp_size: int = 32
    shared_memory_per_block_kb: int = 0
    clock_rate_mhz: float = 0.0
    memory_clock_rate_mhz: float = 0.0
    memory_bus_width_bits: int = 0

    @property
    def supports_tensor_cores(self) -> bool:
        """Check if device supports Tensor Cores (Volta+)."""
        return self.compute_capability >= (7, 0)

    @property
    def supports_fp16(self) -> bool:
        """Check if device has good FP16 support (Pascal+)."""
        return self.compute_capability >= (6, 0)

    @property
    def supports_bf16(self) -> bool:
        """Check if device supports BF16 (Ampere+)."""
        return self.compute_capability >= (8, 0)

    @property
    def supports_int8_tensor_cores(self) -> bool:
        """Check if device supports INT8 Tensor Cores (Turing+)."""
        return self.compute_capability >= (7, 5)

    @property
    def theoretical_flops(self) -> float:
        """Estimate theoretical peak FLOPS (FP32)."""
        # Cores per SM varies by architecture
        cores_per_sm = 64  # Approximate for modern architectures
        return (
            2.0 * self.multiprocessor_count * cores_per_sm * self.clock_rate_mhz * 1e6
        )


@dataclass
class ROCmDeviceInfo:
    """ROCm (AMD GPU) device information."""

    device_id: int = 0
    name: str = ""
    gcn_arch: str = ""
    total_memory_mb: int = 0
    compute_units: int = 0
    max_work_group_size: int = 0
    wavefront_size: int = 64


@dataclass
class HardwareInfo:
    """Complete hardware profile."""

    cpu: CPUInfo = field(default_factory=CPUInfo)
    cuda_devices: list[CUDADeviceInfo] = field(default_factory=list)
    rocm_devices: list[ROCmDeviceInfo] = field(default_factory=list)
    system_memory_mb: int = 0
    os_name: str = ""
    os_version: str = ""

    @property
    def has_cuda(self) -> bool:
        """Check if CUDA is available."""
        return len(self.cuda_devices) > 0

    @property
    def has_rocm(self) -> bool:
        """Check if ROCm is available."""
        return len(self.rocm_devices) > 0

    @property
    def primary_device_type(self) -> DeviceType:
        """Get the primary accelerator type."""
        if self.has_cuda:
            return DeviceType.CUDA
        elif self.has_rocm:
            return DeviceType.ROCM
        return DeviceType.CPU

    def get_default_cuda_device(self) -> CUDADeviceInfo | None:
        """Get the default (first) CUDA device."""
        return self.cuda_devices[0] if self.cuda_devices else None


def detect_cpu_info() -> CPUInfo:
    """Detect CPU information."""
    info = CPUInfo()

    # Get architecture
    info.architecture = platform.machine()

    # Get core/thread count
    try:
        info.num_cores = os.cpu_count() or 1
        info.num_threads = info.num_cores  # Simplified
    except Exception:
        pass

    # Detect vendor from architecture or processor name
    arch = info.architecture.lower()
    if "arm" in arch or "aarch" in arch:
        info.vendor = CPUVendor.ARM
        info.simd.neon = True
    elif platform.system() == "Darwin" and "arm" in arch:
        info.vendor = CPUVendor.APPLE
        info.simd.neon = True
    else:
        # x86 - try to detect vendor
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "GenuineIntel" in cpuinfo:
                    info.vendor = CPUVendor.INTEL
                elif "AuthenticAMD" in cpuinfo:
                    info.vendor = CPUVendor.AMD

                # Parse model name
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        info.model_name = line.split(":")[1].strip()
                        break

                # Parse flags for SIMD features
                for line in cpuinfo.split("\n"):
                    if "flags" in line:
                        flags = line.lower()
                        info.simd.sse = "sse" in flags
                        info.simd.sse2 = "sse2" in flags
                        info.simd.sse3 = "sse3" in flags
                        info.simd.ssse3 = "ssse3" in flags
                        info.simd.sse4_1 = "sse4_1" in flags
                        info.simd.sse4_2 = "sse4_2" in flags
                        info.simd.avx = "avx" in flags
                        info.simd.avx2 = "avx2" in flags
                        info.simd.avx512f = "avx512f" in flags
                        info.simd.avx512vnni = "avx512vnni" in flags
                        break
        except Exception:
            pass

    return info


def detect_cuda_devices() -> list[CUDADeviceInfo]:
    """Detect CUDA devices using nvidia-smi."""
    devices = []

    try:
        # Try to get device info from nvidia-smi
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    device = CUDADeviceInfo()
                    device.device_id = int(parts[0])
                    device.name = parts[1]
                    device.total_memory_mb = int(parts[2])
                    cc_str = parts[3]
                    if "." in cc_str:
                        major, minor = cc_str.split(".")
                        device.compute_capability = (int(major), int(minor))
                    devices.append(device)
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    # Try to get more details from Python CUDA bindings if available
    try:
        import pycuda.driver as cuda

        cuda.init()
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            if i < len(devices):
                devices[i].multiprocessor_count = dev.get_attribute(
                    cuda.device_attribute.MULTIPROCESSOR_COUNT
                )
                devices[i].max_threads_per_block = dev.get_attribute(
                    cuda.device_attribute.MAX_THREADS_PER_BLOCK
                )
                devices[i].shared_memory_per_block_kb = (
                    dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
                    // 1024
                )
    except ImportError:
        pass
    except Exception:
        pass

    return devices


def detect_rocm_devices() -> list[ROCmDeviceInfo]:
    """Detect ROCm (AMD GPU) devices."""
    devices = []

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            # Parse ROCm-SMI output
            lines = result.stdout.strip().split("\n")
            device_id = 0
            for line in lines:
                if "GPU" in line and ":" in line:
                    device = ROCmDeviceInfo()
                    device.device_id = device_id
                    device.name = line.split(":")[-1].strip()
                    device.wavefront_size = 64  # Default for AMD
                    devices.append(device)
                    device_id += 1
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return devices


def detect_system_memory() -> int:
    """Detect total system memory in MB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Value is in kB
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        pass

    # Fallback for macOS
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass

    return 0


def detect_hardware() -> HardwareInfo:
    """Detect complete hardware profile."""
    info = HardwareInfo()

    # OS info
    info.os_name = platform.system()
    info.os_version = platform.release()

    # CPU
    info.cpu = detect_cpu_info()

    # Memory
    info.system_memory_mb = detect_system_memory()

    # GPUs
    info.cuda_devices = detect_cuda_devices()
    info.rocm_devices = detect_rocm_devices()

    return info


# =============================================================================
# Hardware Constraints for Auto-Tuning
# =============================================================================


@dataclass
class HardwareConstraints:
    """Constraints for kernel parameter selection based on hardware."""

    # Block/thread limits
    max_threads_per_block: int = 1024
    max_block_dim_x: int = 1024
    max_block_dim_y: int = 1024
    max_block_dim_z: int = 64
    max_grid_dim_x: int = 2147483647
    max_grid_dim_y: int = 65535
    max_grid_dim_z: int = 65535

    # Memory limits
    max_shared_memory_per_block_kb: int = 48
    max_registers_per_block: int = 65536

    # Vector widths
    preferred_vector_width: int = 4
    max_vector_width: int = 8

    # Tile sizes (common valid values)
    valid_tile_sizes: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    valid_unroll_factors: list[int] = field(default_factory=lambda: [1, 2, 4, 8])


def get_constraints_for_device(device_type: DeviceType) -> HardwareConstraints:
    """Get hardware constraints for a specific device type."""
    constraints = HardwareConstraints()

    if device_type == DeviceType.CUDA:
        # CUDA-specific constraints
        constraints.max_threads_per_block = 1024
        constraints.max_shared_memory_per_block_kb = 48  # Conservative
        constraints.preferred_vector_width = 4
        constraints.max_vector_width = 4  # float4

    elif device_type == DeviceType.ROCM:
        # ROCm-specific constraints
        constraints.max_threads_per_block = 1024
        constraints.max_shared_memory_per_block_kb = 64  # LDS for GCN
        constraints.preferred_vector_width = 4

    elif device_type == DeviceType.CPU:
        # CPU-specific constraints
        constraints.max_threads_per_block = 1  # N/A for CPU
        constraints.preferred_vector_width = 8  # AVX256 / 32-bit
        constraints.max_vector_width = 16  # AVX512 / 32-bit

    return constraints


def filter_search_space_by_hardware(
    param_name: str,
    values: list[int],
    constraints: HardwareConstraints,
) -> list[int]:
    """Filter parameter values based on hardware constraints."""
    if "tile" in param_name.lower():
        return [v for v in values if v in constraints.valid_tile_sizes]

    if "unroll" in param_name.lower():
        return [v for v in values if v in constraints.valid_unroll_factors]

    if "block" in param_name.lower() or "thread" in param_name.lower():
        return [v for v in values if v <= constraints.max_threads_per_block]

    if "vector" in param_name.lower():
        return [v for v in values if v <= constraints.max_vector_width]

    return values


# =============================================================================
# Singleton Hardware Profile
# =============================================================================

_cached_hardware_info: HardwareInfo | None = None


def get_hardware_info() -> HardwareInfo:
    """Get cached hardware information."""
    global _cached_hardware_info
    if _cached_hardware_info is None:
        _cached_hardware_info = detect_hardware()
    return _cached_hardware_info


def refresh_hardware_info() -> HardwareInfo:
    """Refresh and return hardware information."""
    global _cached_hardware_info
    _cached_hardware_info = detect_hardware()
    return _cached_hardware_info
