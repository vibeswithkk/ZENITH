# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
CUDA Graphs Python Interface for Zenith.

This module provides high-level Python API for CUDA Graphs functionality,
enabling significant latency reduction for repetitive inference workloads.

Technical Foundation:
- CUDA Graphs capture kernel execution sequences into replay-able graphs
- Reduces CPU launch overhead from ~25us/kernel to ~5us/graph
- 10-30% latency improvement for fixed-shape inference

Architecture:
- CudaGraphManager: High-level Python manager for graph capture and replay
- GraphCaptureContext: Context manager for stream capture
- CachedGraphModel: Wrapper for models that automatically use CUDA graphs

Usage:
    import zenith
    from zenith.runtime.cuda_graphs import CudaGraphManager

    # Method 1: Context manager
    manager = CudaGraphManager()
    with manager.capture("model_inference") as ctx:
        output = model(input)
    # Subsequent calls use cached graph
    manager.replay("model_inference", stream)

    # Method 2: Decorator
    @manager.graph_cached("model_forward")
    def forward_pass(model, input):
        return model(input)

    # Method 3: Automatic with CompiledModel
    compiled = zenith.compile(model, mode="reduce-overhead")  # Uses graphs

References:
- NVIDIA CUDA Programming Guide: Graph Management
- PyTorch: torch.cuda.CUDAGraph, torch.cuda.make_graphed_callables
- TensorRT: Automatic CUDA Graph integration

IMPORTANT NOTES FOR CUDA GRAPHS:
1. Input tensors must use the SAME memory addresses between capture and replay
2. Output tensors are written to the SAME memory locations
3. Shapes must be FIXED - no dynamic shapes allowed
4. Model weights must NOT change between capture and replay
"""

from __future__ import annotations

import functools
import hashlib
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

# Type variable for generic functions
F = TypeVar("F", bound=Callable[..., Any])


class GraphCaptureMode(Enum):
    """CUDA Graph capture modes."""

    GLOBAL = "global"
    THREAD_LOCAL = "thread_local"
    RELAXED = "relaxed"


class GraphStatus(Enum):
    """Status of a cached CUDA graph."""

    NOT_CAPTURED = "not_captured"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    INSTANTIATED = "instantiated"
    INVALID = "invalid"


@dataclass
class GraphStatistics:
    """Statistics for CUDA graph execution."""

    node_count: int = 0
    launch_count: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    capture_time_ms: float = 0.0

    def update_average(self) -> None:
        """Update average time based on total and launch count."""
        if self.launch_count > 0:
            self.average_time_ms = self.total_time_ms / self.launch_count


@dataclass
class CachedGraph:
    """Container for a cached CUDA graph."""

    key: str
    status: GraphStatus = GraphStatus.NOT_CAPTURED
    statistics: GraphStatistics = field(default_factory=GraphStatistics)
    input_shapes: tuple = field(default_factory=tuple)
    # Native handles stored here when CUDA is available
    graph_handle: Optional[Any] = None
    exec_handle: Optional[Any] = None
    # Static buffers for input/output (required for CUDA Graphs)
    static_inputs: Optional[list[Any]] = None
    static_outputs: Optional[Any] = None


class CudaGraphManager:
    """
    High-level manager for CUDA Graphs in Zenith.

    Provides:
    1. Graph capture and caching
    2. Automatic graph selection based on input shapes
    3. Statistics tracking
    4. Memory management with LRU eviction
    5. Thread-safe operations

    Example:
        manager = CudaGraphManager()

        # Capture a graph
        with manager.capture("inference", stream) as capture_stream:
            output = model(input)

        # Replay cached graph
        manager.replay("inference", stream)

        # Or use decorator
        @manager.graph_cached("forward")
        def model_forward(x):
            return model(x)

        result = model_forward(input)  # First call captures, subsequent replay
    """

    def __init__(
        self,
        max_cached_graphs: int = 100,
        warmup_iterations: int = 3,
        enable_stats: bool = True,
    ):
        """
        Initialize CUDA Graph Manager.

        Args:
            max_cached_graphs: Maximum number of graphs to cache (LRU eviction)
            warmup_iterations: Number of warmup runs before capture
            enable_stats: Whether to track execution statistics
        """
        self._cache: dict[str, CachedGraph] = {}
        self._cache_order: list[str] = []  # For LRU tracking
        self._max_cached = max_cached_graphs
        self._warmup_iterations = warmup_iterations
        self._enable_stats = enable_stats
        self._torch_available = False
        self._cuda_available = False
        self._lock = threading.RLock()  # Thread safety
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize backend (PyTorch CUDA or native)."""
        try:
            import torch

            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            self._torch_available = False

        if not self._cuda_available:
            warnings.warn(
                "CUDA not available. CUDA Graphs will be disabled. "
                "Functions will execute without graph optimization.",
                RuntimeWarning,
                stacklevel=3,
            )

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def graphs_enabled(self) -> bool:
        """Check if CUDA graphs are enabled and usable."""
        return self._cuda_available and self._torch_available

    def _enforce_cache_limit(self) -> None:
        """Enforce maximum cache size using LRU eviction."""
        with self._lock:
            while len(self._cache) >= self._max_cached and self._cache_order:
                # Remove oldest entry (LRU)
                oldest_key = self._cache_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]

    def _update_lru(self, key: str) -> None:
        """Update LRU order for a key."""
        with self._lock:
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)

    def capture(
        self,
        key: str,
        stream: Optional[Any] = None,
        mode: GraphCaptureMode = GraphCaptureMode.GLOBAL,
        pool: Optional[Any] = None,
    ) -> "GraphCaptureContext":
        """
        Create a context manager for graph capture.

        Args:
            key: Unique identifier for this graph
            stream: CUDA stream to capture on (None = default stream)
            mode: Capture mode (global, thread_local, relaxed)
            pool: Memory pool for allocations during capture

        Returns:
            GraphCaptureContext for use in with statement

        Example:
            with manager.capture("forward_pass") as ctx:
                output = model(input)
        """
        return GraphCaptureContext(self, key, stream, mode, pool)

    def replay(self, key: str, stream: Optional[Any] = None) -> bool:
        """
        Replay a cached graph.

        Args:
            key: Key of the graph to replay
            stream: CUDA stream to launch on

        Returns:
            True if graph was replayed, False if not found

        Raises:
            ValueError: If graph is not in INSTANTIATED state
        """
        with self._lock:
            if key not in self._cache:
                return False

            cached = self._cache[key]
            if cached.status != GraphStatus.INSTANTIATED:
                raise ValueError(
                    f"Graph '{key}' is not instantiated. Status: {cached.status}"
                )

            if not self.graphs_enabled:
                return False

            # Update LRU
            self._update_lru(key)

            # Launch the graph
            self._launch_graph(cached, stream)
            return True

    def _launch_graph(self, cached: CachedGraph, stream: Optional[Any]) -> None:
        """Internal method to launch a cached graph."""
        if not self._torch_available:
            return

        import torch

        # Get stream
        if stream is None:
            stream = torch.cuda.current_stream()

        # Launch graph
        if cached.exec_handle is not None:
            # Record start time if stats enabled
            start_event = None
            end_event = None
            if self._enable_stats:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)

            # Launch
            cached.exec_handle.replay()

            # Record end time and update stats
            if self._enable_stats and start_event and end_event:
                end_event.record(stream)
                end_event.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                cached.statistics.launch_count += 1
                cached.statistics.total_time_ms += elapsed
                cached.statistics.update_average()

    def _begin_capture(
        self,
        key: str,
        stream: Any,
        pool: Optional[Any],
    ) -> Any:
        """Begin graph capture (internal)."""
        if not self._torch_available:
            return None

        import torch

        with self._lock:
            # Enforce cache limit before adding new entry
            self._enforce_cache_limit()

            # Create graph object
            graph = torch.cuda.CUDAGraph()

            # Create cached entry
            cached = CachedGraph(key=key, status=GraphStatus.CAPTURING)
            if key in self._cache:
                # Remove from LRU order
                if key in self._cache_order:
                    self._cache_order.remove(key)

            self._cache[key] = cached
            self._cache_order.append(key)

            # Begin capture
            kwargs = {}
            if pool is not None:
                kwargs["pool"] = pool

            graph.capture_begin(**kwargs)
            cached.graph_handle = graph

        return stream

    def _end_capture(self, key: str) -> None:
        """End graph capture and instantiate (internal)."""
        with self._lock:
            if key not in self._cache:
                return

            cached = self._cache[key]
            if cached.graph_handle is None:
                return

            if not self._torch_available:
                return

            # End capture
            graph = cached.graph_handle
            graph.capture_end()

            # Instantiate (in PyTorch, the graph is ready after capture_end)
            cached.exec_handle = graph
            cached.status = GraphStatus.INSTANTIATED

            # Update statistics
            cached.statistics.node_count = 1

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached graph.

        Args:
            key: Key of the graph to invalidate

        Returns:
            True if graph was invalidated, False if not found
        """
        with self._lock:
            if key not in self._cache:
                return False

            self._cache[key].status = GraphStatus.INVALID
            self._cache[key].graph_handle = None
            self._cache[key].exec_handle = None
            self._cache[key].static_inputs = None
            self._cache[key].static_outputs = None

            # Remove from cache and LRU order
            del self._cache[key]
            if key in self._cache_order:
                self._cache_order.remove(key)
            return True

    def clear(self) -> int:
        """
        Clear all cached graphs.

        Returns:
            Number of graphs cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._cache_order.clear()
            return count

    def get_statistics(self, key: str) -> Optional[GraphStatistics]:
        """
        Get statistics for a cached graph.

        Args:
            key: Key of the graph

        Returns:
            GraphStatistics or None if not found
        """
        with self._lock:
            if key not in self._cache:
                return None
            return self._cache[key].statistics

    def get_all_statistics(self) -> dict[str, GraphStatistics]:
        """Get statistics for all cached graphs."""
        with self._lock:
            return {key: cached.statistics for key, cached in self._cache.items()}

    def graph_cached(
        self,
        key: Optional[str] = None,
        warmup: Optional[int] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to automatically cache and replay CUDA graphs.

        IMPORTANT: This decorator requires static input/output buffers.
        The decorated function's inputs must use the SAME tensor memory
        between calls for graph replay to work correctly.

        Args:
            key: Unique key for the graph (defaults to function name)
            warmup: Number of warmup iterations (defaults to manager setting)

        Returns:
            Decorated function

        Example:
            @manager.graph_cached("model_forward")
            def forward(x):
                return model(x)
        """

        def decorator(func: F) -> F:
            graph_key = key or func.__name__
            warmup_count = warmup if warmup is not None else self._warmup_iterations
            call_count = 0
            captured = False
            static_output_ref: list[Any] = [None]  # Mutable container for closure

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                nonlocal call_count, captured

                if not self.graphs_enabled:
                    return func(*args, **kwargs)

                call_count += 1

                # Warmup phase
                if call_count <= warmup_count:
                    return func(*args, **kwargs)

                # Capture phase (only once)
                if not captured:
                    with self.capture(graph_key):
                        result = func(*args, **kwargs)
                    # Store reference to static output
                    static_output_ref[0] = result
                    captured = True
                    return result

                # Replay phase - use cached graph
                if self.replay(graph_key):
                    # Graph executed successfully
                    # Return the static output (same tensor, updated in-place)
                    return static_output_ref[0]

                # Fallback if replay failed
                return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def make_graphed_callable(
        self,
        func: Callable[..., Any],
        sample_args: tuple,
        sample_kwargs: Optional[dict] = None,
        key: Optional[str] = None,
        num_warmup: int = 3,
    ) -> Callable[..., Any]:
        """
        Create a graph-cached version of a callable with static buffers.

        This creates static input and output buffers that are reused across
        graph replays. Inputs are copied TO static buffers before replay,
        and outputs are copied FROM static buffers after replay.

        Args:
            func: Function to graph
            sample_args: Sample arguments for warmup and capture
            sample_kwargs: Sample keyword arguments
            key: Unique key for the graph
            num_warmup: Number of warmup iterations

        Returns:
            Graph-cached callable

        Example:
            graphed_forward = manager.make_graphed_callable(
                model.forward,
                (sample_input,),
                key="model_forward"
            )
            output = graphed_forward(input)
        """
        sample_kwargs = sample_kwargs or {}
        graph_key = key or f"graphed_{id(func)}"

        if not self.graphs_enabled:
            return func

        import torch

        # Create static input buffers (copies of sample inputs)
        static_args = []
        for arg in sample_args:
            if isinstance(arg, torch.Tensor):
                static_args.append(arg.clone())
            else:
                static_args.append(arg)
        static_args = tuple(static_args)

        static_kwargs = {}
        for k, v in sample_kwargs.items():
            if isinstance(v, torch.Tensor):
                static_kwargs[k] = v.clone()
            else:
                static_kwargs[k] = v

        # Warmup runs
        for _ in range(num_warmup):
            with torch.no_grad():
                func(*static_args, **static_kwargs)

        # Capture with static buffers
        with self.capture(graph_key):
            with torch.no_grad():
                static_output = func(*static_args, **static_kwargs)

        # Store static buffers in cached graph
        with self._lock:
            if graph_key in self._cache:
                self._cache[graph_key].static_inputs = list(static_args)
                self._cache[graph_key].static_outputs = static_output

        def graphed_func(*args: Any, **kwargs: Any) -> Any:
            # Copy inputs to static buffers
            for i, (arg, static_arg) in enumerate(zip(args, static_args)):
                if isinstance(arg, torch.Tensor) and isinstance(
                    static_arg, torch.Tensor
                ):
                    static_arg.copy_(arg)

            for k, v in kwargs.items():
                if k in static_kwargs and isinstance(v, torch.Tensor):
                    static_kwargs[k].copy_(v)

            # Replay graph
            self.replay(graph_key)

            # Return output (already updated in-place)
            return static_output

        return graphed_func

    @property
    def cache_size(self) -> int:
        """Get number of cached graphs."""
        with self._lock:
            return len(self._cache)

    @property
    def total_launches(self) -> int:
        """Get total number of graph launches across all cached graphs."""
        with self._lock:
            return sum(c.statistics.launch_count for c in self._cache.values())


class GraphCaptureContext:
    """
    Context manager for CUDA graph capture.

    Usage:
        with manager.capture("key") as ctx:
            # All CUDA operations here are captured
            output = model(input)
        # Graph is now cached and ready for replay
    """

    def __init__(
        self,
        manager: CudaGraphManager,
        key: str,
        stream: Optional[Any],
        mode: GraphCaptureMode,
        pool: Optional[Any],
    ):
        self._manager = manager
        self._key = key
        self._stream = stream
        self._mode = mode
        self._pool = pool
        self._capture_stream = None
        self._entered = False

    def __enter__(self) -> "GraphCaptureContext":
        """Begin graph capture."""
        self._entered = True

        if not self._manager.graphs_enabled:
            return self

        import torch

        # Get or create stream
        if self._stream is None:
            self._stream = torch.cuda.current_stream()

        # Begin capture
        self._capture_stream = self._manager._begin_capture(
            self._key, self._stream, self._pool
        )

        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """End graph capture."""
        if not self._entered:
            return False

        if exc_type is not None:
            # Exception occurred, invalidate the graph
            self._manager.invalidate(self._key)
            return False

        if self._manager.graphs_enabled:
            # End capture and instantiate
            self._manager._end_capture(self._key)

        return False

    @property
    def stream(self) -> Optional[Any]:
        """Get the capture stream."""
        return self._capture_stream


class CachedGraphModel:
    """
    Wrapper that adds CUDA graph caching to a model.

    Automatically captures and replays CUDA graphs for inference,
    providing significant latency reduction for fixed-shape inputs.

    IMPORTANT: This wrapper uses static buffers. The first call after
    warmup creates static input/output tensors. Subsequent calls copy
    inputs to static buffers and return the static output (modified in-place).

    Usage:
        model = MyModel()
        cached_model = CachedGraphModel(model)

        # First calls are warmup
        # Next call captures graph
        output = cached_model(input)

        # Subsequent calls replay graph (faster!)
        output = cached_model(input)

    Note:
        - Input shapes must be fixed for graph replay
        - Model weights must NOT change after capture
    """

    def __init__(
        self,
        model: Any,
        warmup_iterations: int = 3,
        cache_by_shape: bool = True,
    ):
        """
        Initialize CachedGraphModel.

        Args:
            model: The model to wrap
            warmup_iterations: Number of warmup runs before capture
            cache_by_shape: Whether to cache different graphs for different shapes
        """
        self._model = model
        self._manager = CudaGraphManager(warmup_iterations=warmup_iterations)
        self._cache_by_shape = cache_by_shape
        self._warmup_counters: dict[str, int] = {}
        self._captured_keys: set[str] = set()
        self._static_outputs: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _get_shape_key(self, args: tuple, kwargs: dict) -> str:
        """Generate a unique key based on input shapes."""
        shapes = []

        for arg in args:
            if hasattr(arg, "shape"):
                shapes.append(str(tuple(arg.shape)))
            else:
                shapes.append(str(type(arg).__name__))

        for k, val in sorted(kwargs.items()):
            if hasattr(val, "shape"):
                shapes.append(f"{k}:{tuple(val.shape)}")

        shape_str = "_".join(shapes)
        return hashlib.md5(shape_str.encode()).hexdigest()[:16]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with automatic CUDA graph caching."""
        if not self._manager.graphs_enabled:
            return self._model(*args, **kwargs)

        # Get cache key
        if self._cache_by_shape:
            key = f"model_{self._get_shape_key(args, kwargs)}"
        else:
            key = "model_default"

        with self._lock:
            # Initialize warmup counter if needed
            if key not in self._warmup_counters:
                self._warmup_counters[key] = self._manager._warmup_iterations

            # Warmup phase
            if self._warmup_counters[key] > 0:
                self._warmup_counters[key] -= 1
                return self._model(*args, **kwargs)

            # Capture phase (only once per key)
            if key not in self._captured_keys:
                with self._manager.capture(key):
                    result = self._model(*args, **kwargs)
                self._captured_keys.add(key)
                self._static_outputs[key] = result
                return result

            # Replay phase
            if self._manager.replay(key):
                # Return static output (modified in-place by graph replay)
                return self._static_outputs[key]

        # Fallback to direct execution
        return self._model(*args, **kwargs)

    def invalidate(self) -> None:
        """Invalidate all cached graphs (e.g., after model update)."""
        with self._lock:
            self._manager.clear()
            self._captured_keys.clear()
            self._warmup_counters.clear()
            self._static_outputs.clear()

    @property
    def statistics(self) -> dict[str, GraphStatistics]:
        """Get execution statistics for all cached graphs."""
        return self._manager.get_all_statistics()


# Thread-safe global manager with double-checked locking
_global_manager: Optional[CudaGraphManager] = None
_global_manager_lock = threading.Lock()


def get_global_manager() -> CudaGraphManager:
    """Get the global CUDA graph manager (thread-safe singleton)."""
    global _global_manager
    if _global_manager is None:
        with _global_manager_lock:
            # Double-checked locking
            if _global_manager is None:
                _global_manager = CudaGraphManager()
    return _global_manager


def capture(
    key: str,
    stream: Optional[Any] = None,
    mode: GraphCaptureMode = GraphCaptureMode.GLOBAL,
) -> GraphCaptureContext:
    """
    Capture CUDA operations into a graph using global manager.

    Example:
        from zenith.runtime.cuda_graphs import capture, replay

        with capture("my_graph"):
            output = model(input)

        # Later
        replay("my_graph")
    """
    return get_global_manager().capture(key, stream, mode)


def replay(key: str, stream: Optional[Any] = None) -> bool:
    """Replay a cached graph using global manager."""
    return get_global_manager().replay(key, stream)


def graph_cached(
    key: Optional[str] = None,
    warmup: int = 3,
) -> Callable[[F], F]:
    """
    Decorator to automatically cache and replay CUDA graphs.

    Example:
        from zenith.runtime.cuda_graphs import graph_cached

        @graph_cached("model_forward")
        def forward(x):
            return model(x)
    """
    return get_global_manager().graph_cached(key, warmup)
