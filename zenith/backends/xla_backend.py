# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith XLA Backend

Provides XLA compilation and execution for JAX-based workflows.
Extends BaseBackend to integrate with Zenith's backend registry.

Features:
- Direct XLA compilation from GraphIR
- HLO generation and caching
- Device placement (CPU, GPU, TPU)
- Compiled function caching
- Integration with JAX ecosystem

Example:
    from zenith.backends.xla_backend import XLABackend

    backend = XLABackend(device="cuda:0")
    compiled = backend.compile(graph_ir, config)
    output = backend.execute(compiled, inputs)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .base import BaseBackend

logger = logging.getLogger("zenith.backends.xla")


# Lazy imports for optional JAX dependency
def _get_jax():
    """Lazy import of JAX."""
    try:
        import jax

        return jax
    except ImportError as e:
        raise ImportError(
            "JAX is required for XLA backend. Install with: pip install jax jaxlib"
        ) from e


def _get_jnp():
    """Lazy import of jax.numpy."""
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for XLA backend. Install with: pip install jax jaxlib"
        ) from e


class XLADeviceType(Enum):
    """XLA device types."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    AUTO = "auto"


@dataclass
class XLACompileConfig:
    """Configuration for XLA compilation.

    Attributes:
        device: Target device (cpu, gpu, tpu, auto)
        precision: Precision mode (fp32, fp16, bf16)
        enable_xla_optimization: Enable XLA optimizations
        enable_fusion: Enable operator fusion
        enable_memory_optimization: Enable memory optimizations
        cache_compiled: Cache compiled functions
        profile: Enable profiling
        debug: Enable debug mode
    """

    device: str = "auto"
    precision: str = "fp32"
    enable_xla_optimization: bool = True
    enable_fusion: bool = True
    enable_memory_optimization: bool = True
    cache_compiled: bool = True
    profile: bool = False
    debug: bool = False

    # Advanced options
    xla_flags: Dict[str, Any] = field(default_factory=dict)
    donation_argnums: Tuple[int, ...] = ()
    static_argnums: Tuple[int, ...] = ()


@dataclass
class XLACompilationResult:
    """Result of XLA compilation.

    Attributes:
        compiled_fn: The compiled JAX function
        hlo_module: The HLO module (if available)
        device: Target device
        input_shapes: Expected input shapes
        output_shapes: Expected output shapes
        compile_time_ms: Compilation time in milliseconds
        cache_key: Cache key for this compilation
    """

    compiled_fn: Any
    hlo_module: Optional[Any] = None
    device: str = "auto"
    input_shapes: Optional[List[Tuple[int, ...]]] = None
    output_shapes: Optional[List[Tuple[int, ...]]] = None
    compile_time_ms: float = 0.0
    cache_key: Optional[str] = None


@dataclass
class XLAExecutionStats:
    """Statistics for XLA execution.

    Attributes:
        total_executions: Total number of executions
        total_time_ms: Total execution time
        avg_time_ms: Average execution time
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
    """

    total_executions: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class XLABackend(BaseBackend):
    """XLA Backend for Zenith.

    Provides XLA compilation and execution for JAX-based workflows.
    Integrates with Zenith's backend registry and GraphIR.

    Example:
        backend = XLABackend(device="cuda:0")

        @backend.compile
        def forward(x, params):
            return model(x, params)

        output = forward(x, params)
    """

    name = "xla"

    def __init__(
        self,
        device: str = "auto",
        config: Optional[XLACompileConfig] = None,
    ):
        """Initialize XLA backend.

        Args:
            device: Target device (auto, cpu, gpu, cuda:N, tpu)
            config: Compilation configuration
        """
        super().__init__()

        self._config = config or XLACompileConfig(device=device)
        self._device = self._resolve_device(device)

        # Compilation cache
        self._compile_cache: Dict[str, XLACompilationResult] = {}
        self._cache_lock = threading.Lock()

        # Statistics
        self._stats = XLAExecutionStats()
        self._stats_lock = threading.Lock()

        # Lazy JAX import
        self._jax = None
        self._jnp = None

        logger.info(f"XLA Backend initialized with device: {self._device}")

    @property
    def jax(self):
        """Lazy JAX import."""
        if self._jax is None:
            self._jax = _get_jax()
        return self._jax

    @property
    def jnp(self):
        """Lazy jax.numpy import."""
        if self._jnp is None:
            self._jnp = _get_jnp()
        return self._jnp

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device.

        Args:
            device: Device string (auto, cpu, gpu, cuda:N, tpu)

        Returns:
            Resolved device string
        """
        if device == "auto":
            # Try to detect best available device
            try:
                jax = _get_jax()
                devices = jax.devices()

                # Prefer GPU/TPU over CPU
                for d in devices:
                    if d.platform in ("gpu", "tpu"):
                        return d.platform

                return "cpu"
            except ImportError:
                return "cpu"

        # Normalize cuda:N to gpu
        if device.startswith("cuda"):
            return "gpu"

        return device

    def _get_jax_device(self):
        """Get JAX device object."""
        jax = self.jax

        if self._device == "gpu":
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                return gpu_devices[0]
        elif self._device == "tpu":
            tpu_devices = jax.devices("tpu")
            if tpu_devices:
                return tpu_devices[0]

        return jax.devices("cpu")[0]

    def _compute_cache_key(
        self,
        fn: Callable,
        args_shapes: List[Tuple[int, ...]],
        config: XLACompileConfig,
    ) -> str:
        """Compute cache key for compiled function.

        Args:
            fn: Function to compile
            args_shapes: Shapes of input arguments
            config: Compilation config

        Returns:
            Cache key string
        """
        # Combine function identity and shapes
        fn_id = f"{fn.__module__}.{fn.__qualname__}"
        shapes_str = str(args_shapes)
        config_str = f"{config.device}_{config.precision}_{config.enable_fusion}"

        key_str = f"{fn_id}_{shapes_str}_{config_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def compile(
        self,
        fn: Callable,
        config: Optional[XLACompileConfig] = None,
    ) -> Callable:
        """Compile a function with XLA.

        This method JIT-compiles the function using JAX's XLA backend
        with the specified configuration.

        Args:
            fn: Function to compile
            config: Optional compilation config (uses default if None)

        Returns:
            Compiled function

        Example:
            @backend.compile
            def forward(x, params):
                return model(x, params)
        """
        compile_config = config or self._config
        jax = self.jax

        # Get device
        device = self._get_jax_device()

        # Create JIT-compiled version
        jit_options = {}

        if compile_config.static_argnums:
            jit_options["static_argnums"] = compile_config.static_argnums

        if compile_config.donation_argnums:
            jit_options["donate_argnums"] = compile_config.donation_argnums

        # Apply JIT
        compiled_fn = jax.jit(fn, **jit_options)

        # Wrap with device placement
        def wrapped_fn(*args, **kwargs):
            # Move inputs to device
            args = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, device) if hasattr(x, "shape") else x, args
            )
            return compiled_fn(*args, **kwargs)

        return wrapped_fn

    def compile_with_cache(
        self,
        fn: Callable,
        example_args: Sequence[Any],
        config: Optional[XLACompileConfig] = None,
    ) -> XLACompilationResult:
        """Compile with caching based on input shapes.

        Args:
            fn: Function to compile
            example_args: Example arguments for shape inference
            config: Compilation config

        Returns:
            XLACompilationResult with cached compiled function
        """
        compile_config = config or self._config
        jax = self.jax

        # Extract shapes
        def get_shape(x):
            if hasattr(x, "shape"):
                return tuple(x.shape)
            return None

        args_shapes = [get_shape(a) for a in example_args]

        # Compute cache key
        cache_key = self._compute_cache_key(fn, args_shapes, compile_config)

        # Check cache
        with self._cache_lock:
            if cache_key in self._compile_cache:
                with self._stats_lock:
                    self._stats.cache_hits += 1
                return self._compile_cache[cache_key]

        # Compile
        start_time = time.time()
        compiled_fn = self.compile(fn, compile_config)
        compile_time_ms = (time.time() - start_time) * 1000

        # Create result
        result = XLACompilationResult(
            compiled_fn=compiled_fn,
            device=self._device,
            input_shapes=args_shapes,
            compile_time_ms=compile_time_ms,
            cache_key=cache_key,
        )

        # Cache result
        if compile_config.cache_compiled:
            with self._cache_lock:
                self._compile_cache[cache_key] = result

        with self._stats_lock:
            self._stats.cache_misses += 1

        return result

    def execute(
        self,
        compiled: XLACompilationResult,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a compiled function.

        Args:
            compiled: Compilation result
            *args: Input arguments
            **kwargs: Keyword arguments

        Returns:
            Function output
        """
        jax = self.jax

        start_time = time.time()
        result = compiled.compiled_fn(*args, **kwargs)

        # Block until ready for accurate timing
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

        exec_time_ms = (time.time() - start_time) * 1000

        # Update stats
        with self._stats_lock:
            self._stats.total_executions += 1
            self._stats.total_time_ms += exec_time_ms
            self._stats.avg_time_ms = (
                self._stats.total_time_ms / self._stats.total_executions
            )

        return result

    def lower_to_hlo(
        self,
        fn: Callable,
        example_args: Sequence[Any],
    ) -> str:
        """Lower function to HLO text representation.

        Args:
            fn: Function to lower
            example_args: Example arguments for tracing

        Returns:
            HLO text representation
        """
        jax = self.jax

        # JIT and lower
        jitted = jax.jit(fn)
        lowered = jitted.lower(*example_args)

        # Get HLO text
        try:
            hlo_text = lowered.as_text()
        except AttributeError:
            # Older JAX versions
            hlo_text = str(lowered.compiler_ir())

        return hlo_text

    def get_hlo_module(
        self,
        fn: Callable,
        example_args: Sequence[Any],
    ) -> Any:
        """Get HLO module for a function.

        Args:
            fn: Function to lower
            example_args: Example arguments

        Returns:
            HLO module object
        """
        jax = self.jax

        jitted = jax.jit(fn)
        lowered = jitted.lower(*example_args)

        try:
            return lowered.compiler_ir()
        except AttributeError:
            return lowered

    def profile_execution(
        self,
        fn: Callable,
        args: Sequence[Any],
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """Profile function execution.

        Args:
            fn: Function to profile
            args: Input arguments
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        jax = self.jax

        # Compile
        compiled = self.compile(fn)

        # Warmup
        for _ in range(warmup_runs):
            result = compiled(*args)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()

        # Profile
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = compiled(*args)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            times.append((time.time() - start) * 1000)

        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2],
            "total_runs": num_runs,
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices.

        Returns:
            Dictionary with device information
        """
        jax = self.jax

        devices = jax.devices()

        info = {
            "default_backend": jax.default_backend(),
            "selected_device": self._device,
            "devices": [],
        }

        for d in devices:
            info["devices"].append(
                {
                    "id": d.id,
                    "platform": d.platform,
                    "device_kind": str(d.device_kind)
                    if hasattr(d, "device_kind")
                    else "unknown",
                }
            )

        return info

    def supports_bfloat16(self) -> bool:
        """Check if device supports BF16.

        Returns:
            True if BF16 is supported
        """
        # BF16 is supported on TPU and Ampere+ GPUs
        if self._device == "tpu":
            return True

        if self._device == "gpu":
            # Check for Ampere+ (compute capability >= 8.0)
            try:
                jax = self.jax
                device = self._get_jax_device()

                # Try to get compute capability
                if hasattr(device, "compute_capability"):
                    major, minor = device.compute_capability
                    return major >= 8

                # Fallback: assume modern GPUs support it
                return True
            except Exception:
                return True

        return False

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        with self._cache_lock:
            self._compile_cache.clear()
        logger.info("XLA compilation cache cleared")

    @property
    def stats(self) -> XLAExecutionStats:
        """Get execution statistics."""
        with self._stats_lock:
            return XLAExecutionStats(
                total_executions=self._stats.total_executions,
                total_time_ms=self._stats.total_time_ms,
                avg_time_ms=self._stats.avg_time_ms,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
            )

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        with self._stats_lock:
            self._stats = XLAExecutionStats()

    # BaseBackend interface methods

    def is_available(self) -> bool:
        """Check if backend is available."""
        try:
            _get_jax()
            return True
        except ImportError:
            return False

    def get_name(self) -> str:
        """Get backend name."""
        return self.name

    def get_device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def backend_type(self):
        """Return the backend type enum."""
        from .base import BackendType

        if self._device == "tpu":
            return BackendType.TPU
        elif self._device == "gpu":
            return BackendType.CUDA
        return BackendType.CPU

    def get_device_properties(self):
        """Get properties of the current device."""
        from .base import DeviceProperties, BackendType

        if self._device == "tpu":
            bt = BackendType.TPU
        elif self._device == "gpu":
            bt = BackendType.CUDA
        else:
            bt = BackendType.CPU

        return DeviceProperties(
            name=f"XLA {self._device.upper()} Device",
            vendor="XLA",
            backend_type=bt,
            is_available=self.is_available(),
            supports_fp16=True,
            supports_bf16=self.supports_bfloat16(),
        )

    def allocate(self, size_bytes: int) -> int:
        """Allocate memory on this backend's device.

        Note: XLA manages memory automatically via JAX.
        This returns a placeholder value.
        """
        # XLA/JAX manages memory automatically
        # Return a placeholder to satisfy the interface
        return 0

    def deallocate(self, ptr: int) -> None:
        """Free memory allocated by this backend.

        Note: XLA manages memory automatically via JAX.
        """
        # XLA/JAX manages memory automatically, no-op
        pass

    def copy_to_device(self, data, ptr: int) -> None:
        """Copy data to device.

        Note: Use jax.device_put() instead for XLA.
        """
        # XLA uses jax.device_put, this is a no-op placeholder
        pass

    def copy_to_host(self, ptr: int, size_bytes: int):
        """Copy data from device to host.

        Note: Use numpy conversion instead for XLA.
        """
        # XLA uses numpy conversion, return None as placeholder
        return None


# Convenience functions


def create_xla_backend(
    device: str = "auto",
    precision: str = "fp32",
    **kwargs,
) -> XLABackend:
    """Create XLA backend with configuration.

    Args:
        device: Target device
        precision: Precision mode
        **kwargs: Additional config options

    Returns:
        Configured XLABackend instance
    """
    config = XLACompileConfig(
        device=device,
        precision=precision,
        **kwargs,
    )
    return XLABackend(device=device, config=config)


def xla_jit(
    fn: Optional[Callable] = None,
    *,
    device: str = "auto",
    static_argnums: Tuple[int, ...] = (),
    donate_argnums: Tuple[int, ...] = (),
) -> Callable:
    """Decorator for XLA JIT compilation.

    Args:
        fn: Function to compile
        device: Target device
        static_argnums: Static argument indices
        donate_argnums: Donated argument indices

    Returns:
        Compiled function

    Example:
        @xla_jit(device="gpu")
        def forward(params, x):
            return model(params, x)
    """

    def decorator(func: Callable) -> Callable:
        config = XLACompileConfig(
            device=device,
            static_argnums=static_argnums,
            donation_argnums=donate_argnums,
        )
        backend = XLABackend(device=device, config=config)
        return backend.compile(func, config)

    if fn is not None:
        return decorator(fn)
    return decorator


# Register backend
def register_xla_backend():
    """Register XLA backend with Zenith backend registry."""
    try:
        from .registry import register_backend

        register_backend("xla", XLABackend)
        logger.info("XLA backend registered")
    except ImportError:
        logger.warning("Could not register XLA backend: registry not available")
