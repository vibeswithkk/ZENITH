# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Native Gradient Checkpointing Implementation (Phase 2).

This module provides a native implementation of gradient checkpointing that
does not rely on torch.utils.checkpoint. It implements the rematerialization
logic directly using torch.autograd.Function.

Technical Foundation:
--------------------
Based on Chen et al., 2016 ("Training Deep Nets with Sublinear Memory Cost"):

    Memory Complexity: O(sqrt(N)) vs O(N) for standard backprop
    Compute Overhead: O(N * sqrt(N)) vs O(N) for standard

The implementation stores only checkpoint activations and recomputes
intermediate activations during the backward pass.

Mathematical Model:
------------------
For a network with N sequential layers:
- Standard: Store all N activations = O(N) memory
- Sqrt checkpointing: Store sqrt(N) checkpoints, recompute segments
  - Memory: O(sqrt(N) + sqrt(N)) = O(sqrt(N))
  - Compute: Forward + Backward + Recompute = O(N) + O(N) + O(sqrt(N) * sqrt(N)) = O(N)
  - Total time overhead: ~33% (empirical)

Key Classes:
-----------
- CheckpointFunction: Custom autograd.Function for rematerialization
- ActivationStore: Memory pool for efficient activation management
- OptimalCheckpointSelector: DP-based optimal checkpoint selection
- NativeCheckpointer: High-level API for native checkpointing

References:
----------
1. Chen et al., 2016: "Training Deep Nets with Sublinear Memory Cost"
2. Jain et al., 2020: "Checkmate: Breaking the Memory Wall"
3. PyTorch autograd.Function documentation
4. NVIDIA Megatron-LM activation checkpointing
"""

from __future__ import annotations

import functools
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

# Type variables
T = TypeVar("T")


def _get_torch():
    """Lazy import of torch with error handling."""
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for native gradient checkpointing. "
            "Install with: pip install torch"
        ) from e


class EvictionPolicy(Enum):
    """Policy for evicting activations from memory when under pressure."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SIZE_PRIORITY = "size_priority"  # Evict largest first
    RECOMPUTE_COST = "recompute_cost"  # Evict cheapest to recompute


class RematerializationStrategy(Enum):
    """Strategy for when to rematerialize activations."""

    ON_DEMAND = "on_demand"  # Recompute only when needed in backward
    PREFETCH = "prefetch"  # Recompute slightly ahead of time
    BATCH = "batch"  # Recompute in batches for efficiency


@dataclass
class ActivationMetadata:
    """Metadata for a stored activation tensor."""

    layer_id: int
    shape: Tuple[int, ...]
    dtype: Any
    device: Any
    size_bytes: int
    creation_time: float
    access_count: int = 0
    last_access_time: float = 0.0
    recompute_cost_ms: float = 0.0
    is_checkpoint: bool = False

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()


class ActivationStore:
    """
    Memory pool for efficient activation storage and retrieval.

    Implements configurable eviction policies and memory tracking.
    Thread-safe for use in distributed training.

    Design based on:
    - NVIDIA Megatron-LM activation memory management
    - PyTorch CUDA caching allocator concepts
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
            eviction_policy: Policy for evicting activations
            enable_profiling: Track detailed timing statistics
        """
        self._store: Dict[int, Any] = {}  # layer_id -> tensor
        self._metadata: Dict[int, ActivationMetadata] = {}
        self._access_order: OrderedDict[int, float] = OrderedDict()
        self._max_memory = max_memory_bytes
        self._current_memory = 0
        self._eviction_policy = eviction_policy
        self._enable_profiling = enable_profiling
        self._lock = threading.RLock()

        # Profiling statistics
        self._stats = {
            "store_count": 0,
            "eviction_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "total_stored_bytes": 0,
            "total_evicted_bytes": 0,
        }

    def store(
        self,
        layer_id: int,
        tensor: Any,
        is_checkpoint: bool = False,
        recompute_cost_ms: float = 0.0,
    ) -> bool:
        """
        Store an activation tensor.

        Args:
            layer_id: Unique identifier for the layer
            tensor: The activation tensor to store
            is_checkpoint: Whether this is a checkpoint (protected from eviction)
            recompute_cost_ms: Estimated cost to recompute this activation

        Returns:
            True if stored successfully, False if eviction failed
        """
        torch = _get_torch()

        with self._lock:
            # Calculate tensor size
            if isinstance(tensor, torch.Tensor):
                size_bytes = tensor.numel() * tensor.element_size()
                shape = tuple(tensor.shape)
                dtype = tensor.dtype
                device = tensor.device
            else:
                # For non-tensor types, estimate size
                size_bytes = 8  # Minimal size
                shape = ()
                dtype = None
                device = None

            # Check memory budget and evict if necessary
            if self._max_memory is not None:
                while (
                    self._current_memory + size_bytes > self._max_memory
                    and len(self._store) > 0
                ):
                    if not self._evict_one():
                        return False

            # Store the tensor
            self._store[layer_id] = tensor
            self._metadata[layer_id] = ActivationMetadata(
                layer_id=layer_id,
                shape=shape,
                dtype=dtype,
                device=device,
                size_bytes=size_bytes,
                creation_time=time.time(),
                recompute_cost_ms=recompute_cost_ms,
                is_checkpoint=is_checkpoint,
            )
            self._access_order[layer_id] = time.time()
            self._current_memory += size_bytes

            # Update statistics
            self._stats["store_count"] += 1
            self._stats["total_stored_bytes"] += size_bytes

            return True

    def retrieve(self, layer_id: int) -> Optional[Any]:
        """
        Retrieve an activation tensor.

        Args:
            layer_id: Unique identifier for the layer

        Returns:
            The tensor if found, None otherwise
        """
        with self._lock:
            if layer_id in self._store:
                # Update access statistics
                self._metadata[layer_id].update_access()
                self._access_order.move_to_end(layer_id)
                self._stats["hit_count"] += 1
                return self._store[layer_id]
            else:
                self._stats["miss_count"] += 1
                return None

    def remove(self, layer_id: int) -> Optional[Any]:
        """
        Remove and return an activation tensor.

        Args:
            layer_id: Unique identifier for the layer

        Returns:
            The tensor if found, None otherwise
        """
        with self._lock:
            if layer_id in self._store:
                tensor = self._store.pop(layer_id)
                metadata = self._metadata.pop(layer_id)
                self._access_order.pop(layer_id, None)
                self._current_memory -= metadata.size_bytes
                return tensor
            return None

    def _evict_one(self) -> bool:
        """
        Evict one activation based on policy.

        Returns:
            True if eviction was successful, False if no evictable items
        """
        # Find candidate for eviction (skip checkpoints)
        candidates = [
            (lid, meta)
            for lid, meta in self._metadata.items()
            if not meta.is_checkpoint
        ]

        if not candidates:
            return False

        # Select victim based on policy
        if self._eviction_policy == EvictionPolicy.LRU:
            victim_id = min(candidates, key=lambda x: x[1].last_access_time)[0]
        elif self._eviction_policy == EvictionPolicy.LFU:
            victim_id = min(candidates, key=lambda x: x[1].access_count)[0]
        elif self._eviction_policy == EvictionPolicy.SIZE_PRIORITY:
            victim_id = max(candidates, key=lambda x: x[1].size_bytes)[0]
        elif self._eviction_policy == EvictionPolicy.RECOMPUTE_COST:
            victim_id = min(candidates, key=lambda x: x[1].recompute_cost_ms)[0]
        else:
            victim_id = candidates[0][0]

        # Evict
        victim_meta = self._metadata[victim_id]
        self._store.pop(victim_id)
        self._metadata.pop(victim_id)
        self._access_order.pop(victim_id, None)
        self._current_memory -= victim_meta.size_bytes

        # Update statistics
        self._stats["eviction_count"] += 1
        self._stats["total_evicted_bytes"] += victim_meta.size_bytes

        return True

    def clear(self) -> int:
        """
        Clear all stored activations.

        Returns:
            Number of activations cleared
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._metadata.clear()
            self._access_order.clear()
            self._current_memory = 0
            return count

    @property
    def memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._current_memory

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["current_memory_bytes"] = self._current_memory
            stats["stored_count"] = len(self._store)
            stats["checkpoint_count"] = sum(
                1 for m in self._metadata.values() if m.is_checkpoint
            )
            return stats


class OptimalCheckpointSelector:
    """
    Optimal checkpoint selection using dynamic programming.

    Based on Chen et al., 2016 algorithm for finding optimal checkpoint
    positions that minimize peak memory while respecting compute budget.

    Mathematical Model:
    ------------------
    Given N layers with memory costs m[i] and compute costs c[i],
    find checkpoint set S that minimizes:
        max_{segment} sum(m[i] for i in segment)
    Subject to:
        sum(c[i] for recomputed i) <= lambda * sum(c[i])
    """

    def __init__(
        self,
        num_layers: int,
        memory_costs: Optional[List[float]] = None,
        compute_costs: Optional[List[float]] = None,
        lambda_tolerance: float = 0.33,
    ):
        """
        Initialize checkpoint selector.

        Args:
            num_layers: Total number of layers
            memory_costs: Memory cost per layer (None = uniform)
            compute_costs: Compute cost per layer (None = uniform)
            lambda_tolerance: Maximum compute overhead (0.33 = 33%)
        """
        self._num_layers = num_layers
        self._memory_costs = memory_costs or [1.0] * num_layers
        self._compute_costs = compute_costs or [1.0] * num_layers
        self._lambda = lambda_tolerance
        self._cached_solution: Optional[List[int]] = None

    def select_checkpoints_sqrt(self) -> List[int]:
        """
        Select checkpoints using sqrt(N) heuristic.

        Returns:
            List of layer indices to checkpoint
        """
        if self._num_layers == 0:
            return []

        if self._num_layers == 1:
            return [0]

        interval = max(1, int(math.sqrt(self._num_layers)))
        return list(range(0, self._num_layers, interval))

    def select_checkpoints_dp(self) -> List[int]:
        """
        Select checkpoints using dynamic programming.

        Finds optimal checkpoint positions that minimize peak memory
        while respecting compute overhead budget.

        Time Complexity: O(N^2)
        Space Complexity: O(N)

        Returns:
            List of layer indices to checkpoint
        """
        if self._cached_solution is not None:
            return self._cached_solution

        n = self._num_layers
        if n <= 2:
            return list(range(n))

        # dp[i] = minimum peak memory to process layers 0..i with i checkpointed
        # prev[i] = previous checkpoint for optimal solution ending at i
        INF = float("inf")

        # Prefix sums for efficient range queries
        mem_prefix = [0.0]
        for m in self._memory_costs:
            mem_prefix.append(mem_prefix[-1] + m)

        comp_prefix = [0.0]
        for c in self._compute_costs:
            comp_prefix.append(comp_prefix[-1] + c)

        def segment_memory(i: int, j: int) -> float:
            """Memory needed for segment [i, j)."""
            return mem_prefix[j] - mem_prefix[i]

        def segment_compute(i: int, j: int) -> float:
            """Compute cost for segment [i, j)."""
            return comp_prefix[j] - comp_prefix[i]

        total_compute = comp_prefix[n]
        # max_recompute is the EXTRA compute we're willing to do on top of baseline
        # Baseline = forward + backward = 2 * total_compute
        # With checkpointing, we add recomputation overhead
        max_extra_recompute = self._lambda * total_compute

        dp = [INF] * (n + 1)
        prev = [-1] * (n + 1)
        overhead = [0.0] * (n + 1)  # Tracks extra recomputation

        dp[0] = 0
        overhead[0] = 0

        for j in range(1, n + 1):
            for i in range(j):
                seg_mem = segment_memory(i, j)

                # The recomputation overhead for a segment is:
                # - If i > 0 (not the first checkpoint), we need to recompute
                #   from checkpoint i to j during backward pass
                # - This is the EXTRA cost beyond what we'd normally do
                # For checkpointing: overhead = (segment_length - 1) * avg_cost
                # Simplified: each segment except the first layer needs recompute
                segment_length = j - i
                if segment_length > 1:
                    # We recompute all but the first layer of the segment
                    extra_cost = segment_compute(i + 1, j)
                else:
                    extra_cost = 0.0

                new_overhead = overhead[i] + extra_cost

                if new_overhead <= max_extra_recompute:
                    new_peak = max(dp[i], seg_mem)
                    if new_peak < dp[j]:
                        dp[j] = new_peak
                        prev[j] = i
                        overhead[j] = new_overhead

        # Reconstruct solution
        checkpoints = []
        pos = n
        while pos > 0:
            if prev[pos] < 0:
                # No valid solution found within budget
                # Fall back to sqrt heuristic
                return self.select_checkpoints_sqrt()
            checkpoints.append(prev[pos])
            pos = prev[pos]

        checkpoints = sorted(set(cp for cp in checkpoints if cp >= 0))
        self._cached_solution = checkpoints
        return checkpoints

    def select_checkpoints(self, method: str = "sqrt") -> List[int]:
        """
        Select checkpoints using specified method.

        Args:
            method: "sqrt" for heuristic, "dp" for optimal

        Returns:
            List of layer indices to checkpoint
        """
        if method == "sqrt":
            return self.select_checkpoints_sqrt()
        elif method == "dp":
            return self.select_checkpoints_dp()
        else:
            raise ValueError(f"Unknown method: {method}")


class CheckpointFunction:
    """
    Native checkpoint function using torch.autograd.Function.

    This is the core rematerialization engine that:
    1. During forward: Executes function but only saves checkpoint activations
    2. During backward: Recomputes intermediate activations from checkpoints

    Thread Safety: Uses thread-local storage for nested checkpointing.
    """

    _local = threading.local()

    @classmethod
    def _get_function_class(cls) -> type:
        """
        Create and return the autograd.Function class.

        This is done lazily to avoid importing torch at module load time.
        """
        if not hasattr(cls._local, "function_class"):
            torch = _get_torch()

            class _CheckpointFunctionImpl(torch.autograd.Function):
                """
                Internal autograd.Function implementation.

                This class handles the forward/backward logic for checkpointing.
                """

                @staticmethod
                def forward(
                    ctx: Any,
                    run_function: Callable,
                    preserve_rng_state: bool,
                    *args: Any,
                ) -> Any:
                    """
                    Forward pass with activation checkpointing.

                    Args:
                        ctx: Autograd context
                        run_function: The function to checkpoint
                        preserve_rng_state: Whether to save RNG state
                        *args: Arguments to run_function

                    Returns:
                        Output of run_function
                    """
                    # Separate tensor and non-tensor arguments
                    ctx.run_function = run_function
                    ctx.preserve_rng_state = preserve_rng_state

                    # Save which inputs are tensors (for backward)
                    ctx.tensor_indices = []
                    ctx.non_tensor_args = []
                    tensor_inputs = []

                    for i, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor):
                            ctx.tensor_indices.append(i)
                            tensor_inputs.append(arg)
                        else:
                            ctx.non_tensor_args.append((i, arg))

                    ctx.num_args = len(args)

                    # Save input tensors for backward recomputation
                    ctx.save_for_backward(*tensor_inputs)

                    # Save RNG states if needed
                    if preserve_rng_state:
                        ctx.cpu_rng_state = torch.get_rng_state()
                        if torch.cuda.is_available():
                            ctx.cuda_rng_state = torch.cuda.get_rng_state()
                        else:
                            ctx.cuda_rng_state = None

                    # Execute forward pass WITHOUT gradient tracking
                    # This is the key: we don't store intermediate activations
                    with torch.no_grad():
                        outputs = run_function(*args)

                    return outputs

                @staticmethod
                def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
                    """
                    Backward pass with rematerialization.

                    This recomputes the forward pass to get intermediate
                    activations, then computes gradients normally.

                    Args:
                        ctx: Autograd context
                        *grad_outputs: Gradients of outputs

                    Returns:
                        Tuple of gradients for each input
                    """
                    # Retrieve saved tensors
                    tensor_inputs = ctx.saved_tensors

                    # Reconstruct full argument list
                    args = [None] * ctx.num_args

                    tensor_idx = 0
                    for i in ctx.tensor_indices:
                        args[i] = tensor_inputs[tensor_idx]
                        tensor_idx += 1

                    for i, val in ctx.non_tensor_args:
                        args[i] = val

                    # Restore RNG state if preserved
                    if ctx.preserve_rng_state:
                        rng_state_backup = torch.get_rng_state()
                        torch.set_rng_state(ctx.cpu_rng_state)
                        if ctx.cuda_rng_state is not None:
                            cuda_backup = torch.cuda.get_rng_state()
                            torch.cuda.set_rng_state(ctx.cuda_rng_state)

                    # Enable gradient computation for recomputation
                    # This is the REMATERIALIZATION step
                    detached_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            detached = arg.detach()
                            detached.requires_grad = arg.requires_grad
                            detached_args.append(detached)
                        else:
                            detached_args.append(arg)

                    with torch.enable_grad():
                        outputs = ctx.run_function(*detached_args)

                    # Restore RNG state
                    if ctx.preserve_rng_state:
                        torch.set_rng_state(rng_state_backup)
                        if ctx.cuda_rng_state is not None:
                            torch.cuda.set_rng_state(cuda_backup)

                    # Ensure outputs is a tuple for consistent handling
                    if not isinstance(outputs, tuple):
                        outputs = (outputs,)
                    if not isinstance(grad_outputs, tuple):
                        grad_outputs = (grad_outputs,)

                    # Filter outputs that require grad
                    outputs_with_grad = []
                    grads_with_outputs = []
                    for out, grad in zip(outputs, grad_outputs):
                        if isinstance(out, torch.Tensor) and out.requires_grad:
                            outputs_with_grad.append(out)
                            grads_with_outputs.append(grad)

                    if not outputs_with_grad:
                        # No outputs require grad
                        return (None, None) + tuple(None for _ in args)

                    # Compute gradients
                    input_tensors = [
                        arg
                        for arg in detached_args
                        if isinstance(arg, torch.Tensor) and arg.requires_grad
                    ]

                    if not input_tensors:
                        return (None, None) + tuple(None for _ in args)

                    grads = torch.autograd.grad(
                        outputs_with_grad,
                        input_tensors,
                        grads_with_outputs,
                        allow_unused=True,
                    )

                    # Map gradients back to original argument positions
                    result_grads = [
                        None,
                        None,
                    ]  # For run_function and preserve_rng_state
                    grad_idx = 0

                    for i, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor) and arg.requires_grad:
                            result_grads.append(
                                grads[grad_idx] if grad_idx < len(grads) else None
                            )
                            grad_idx += 1
                        else:
                            result_grads.append(None)

                    return tuple(result_grads)

            cls._local.function_class = _CheckpointFunctionImpl

        return cls._local.function_class

    @classmethod
    def apply(
        cls,
        function: Callable[..., T],
        *args: Any,
        preserve_rng_state: bool = True,
    ) -> T:
        """
        Apply checkpointing to a function.

        Args:
            function: Function to checkpoint
            *args: Arguments to the function
            preserve_rng_state: Whether to preserve RNG state

        Returns:
            Output of the function
        """
        func_class = cls._get_function_class()
        return func_class.apply(function, preserve_rng_state, *args)


def native_checkpoint(
    function: Callable[..., T],
    *args: Any,
    preserve_rng_state: bool = True,
    debug: bool = False,
    **kwargs: Any,
) -> T:
    """
    Apply native gradient checkpointing to a function.

    This is the primary API for Phase 2 native checkpointing.
    Unlike Phase 1 which wraps torch.utils.checkpoint, this
    implements rematerialization directly.

    Args:
        function: Function to apply checkpointing to
        *args: Positional arguments to the function
        preserve_rng_state: Whether to preserve RNG state for reproducibility
        debug: Enable debug logging
        **kwargs: Keyword arguments to the function

    Returns:
        Output of the function

    Example:
        def expensive_layer(x):
            x = self.attention(x)
            x = self.ffn(x)
            return x

        # Instead of storing all intermediate activations,
        # only input is stored and forward is recomputed during backward
        output = native_checkpoint(expensive_layer, input_tensor)
    """
    torch = _get_torch()

    # Handle kwargs by creating a partial function
    if kwargs:
        function = functools.partial(function, **kwargs)

    # If no tensors require grad, skip checkpointing
    has_grad = any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args)

    if not has_grad or not torch.is_grad_enabled():
        return function(*args)

    return CheckpointFunction.apply(
        function, *args, preserve_rng_state=preserve_rng_state
    )


def native_checkpoint_sequential(
    functions: List[Callable],
    segments: int = 0,
    input_tensor: Any = None,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
) -> Any:
    """
    Checkpoint a sequence of functions/layers.

    Divides the sequence into segments and checkpoints each segment.
    Activations are only stored at segment boundaries.

    Args:
        functions: List of callables to apply sequentially
        segments: Number of segments (0 = auto = sqrt(N))
        input_tensor: Input to first function (required)
        preserve_rng_state: Whether to preserve RNG state
        use_reentrant: Whether to use reentrant checkpointing

    Returns:
        Output of the last function

    Raises:
        ValueError: If input_tensor is None or functions is empty

    Example:
        layers = [layer1, layer2, layer3, layer4]
        # Will create sqrt(4) = 2 segments
        output = native_checkpoint_sequential(layers, input_tensor=x)
    """
    if input_tensor is None:
        raise ValueError("input_tensor is required")

    n = len(functions)
    if n == 0:
        return input_tensor

    # Auto-calculate segments using sqrt heuristic
    if segments <= 0:
        segments = max(1, int(math.sqrt(n)))

    segments = min(segments, n)

    # Calculate segment boundaries
    segment_size = n // segments
    remainder = n % segments

    boundaries = [0]
    pos = 0
    for i in range(segments):
        size = segment_size + (1 if i < remainder else 0)
        pos += size
        boundaries.append(pos)

    def run_segment(start: int, end: int, x: Any) -> Any:
        """Run a segment of functions."""
        for i in range(start, end):
            x = functions[i](x)
        return x

    # Execute each segment with checkpointing
    result = input_tensor
    for i in range(segments):
        start = boundaries[i]
        end = boundaries[i + 1]

        segment_fn = functools.partial(run_segment, start, end)
        result = native_checkpoint(
            segment_fn,
            result,
            preserve_rng_state=preserve_rng_state,
        )

    return result


class NativeCheckpointer:
    """
    High-level manager for native gradient checkpointing.

    Provides:
    - Automatic checkpoint selection
    - Memory tracking and profiling
    - Integration with ActivationStore for memory management

    Example:
        checkpointer = NativeCheckpointer(policy="sqrt")

        # Wrap a module
        checkpointer.wrap_module(model.transformer)

        # Or use manually
        for i, layer in enumerate(model.layers):
            if checkpointer.should_checkpoint(i):
                x = checkpointer.checkpoint(layer, x)
            else:
                x = layer(x)
    """

    def __init__(
        self,
        policy: str = "sqrt",
        max_memory_mb: Optional[float] = None,
        lambda_tolerance: float = 0.33,
        enable_profiling: bool = False,
    ):
        """
        Initialize native checkpointer.

        Args:
            policy: Checkpoint selection policy ("sqrt" or "dp")
            max_memory_mb: Maximum memory budget in MB (None = unlimited)
            lambda_tolerance: Maximum compute overhead (0.33 = 33%)
            enable_profiling: Enable timing profiling
        """
        self._policy = policy
        self._lambda = lambda_tolerance
        self._enable_profiling = enable_profiling

        max_bytes = int(max_memory_mb * 1024 * 1024) if max_memory_mb else None
        self._store = ActivationStore(
            max_memory_bytes=max_bytes,
            enable_profiling=enable_profiling,
        )

        self._selector: Optional[OptimalCheckpointSelector] = None
        self._checkpoint_positions: set[int] = set()
        self._num_layers: int = 0
        self._lock = threading.RLock()

        # Profiling
        self._forward_times: List[float] = []
        self._backward_times: List[float] = []

    def configure(
        self,
        num_layers: int,
        memory_costs: Optional[List[float]] = None,
        compute_costs: Optional[List[float]] = None,
    ) -> None:
        """
        Configure checkpointer for a specific model.

        Args:
            num_layers: Total number of layers
            memory_costs: Per-layer memory costs (None = uniform)
            compute_costs: Per-layer compute costs (None = uniform)
        """
        with self._lock:
            self._num_layers = num_layers
            self._selector = OptimalCheckpointSelector(
                num_layers=num_layers,
                memory_costs=memory_costs,
                compute_costs=compute_costs,
                lambda_tolerance=self._lambda,
            )
            positions = self._selector.select_checkpoints(method=self._policy)
            self._checkpoint_positions = set(positions)

    def should_checkpoint(self, layer_idx: int) -> bool:
        """
        Check if a layer should be checkpointed.

        Args:
            layer_idx: Index of the layer

        Returns:
            True if the layer should be checkpointed
        """
        if self._num_layers == 0:
            # Auto-configure with sqrt heuristic if not configured
            return (
                layer_idx == 0 or layer_idx % max(1, int(math.sqrt(layer_idx + 1))) == 0
            )

        return layer_idx in self._checkpoint_positions

    def checkpoint(
        self,
        function: Callable[..., T],
        *args: Any,
        preserve_rng_state: bool = True,
        **kwargs: Any,
    ) -> T:
        """
        Apply checkpointing to a function.

        Args:
            function: Function to checkpoint
            *args: Positional arguments
            preserve_rng_state: Preserve RNG state
            **kwargs: Keyword arguments

        Returns:
            Function output
        """
        if self._enable_profiling:
            start = time.time()

        result = native_checkpoint(
            function,
            *args,
            preserve_rng_state=preserve_rng_state,
            **kwargs,
        )

        if self._enable_profiling:
            self._forward_times.append(time.time() - start)

        return result

    def checkpoint_sequential(
        self,
        functions: List[Callable],
        input_tensor: Any,
        segments: int = 0,
    ) -> Any:
        """
        Checkpoint a sequence of functions.

        Args:
            functions: List of functions
            input_tensor: Input tensor
            segments: Number of segments (0 = auto)

        Returns:
            Output tensor
        """
        return native_checkpoint_sequential(
            functions,
            segments=segments,
            input_tensor=input_tensor,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpointing statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._store.statistics.copy()
        stats["policy"] = self._policy
        stats["num_layers"] = self._num_layers
        stats["num_checkpoints"] = len(self._checkpoint_positions)
        stats["checkpoint_positions"] = sorted(self._checkpoint_positions)

        if self._forward_times:
            stats["avg_forward_time_ms"] = (
                sum(self._forward_times) / len(self._forward_times) * 1000
            )

        return stats

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._store.clear()
            self._forward_times.clear()
            self._backward_times.clear()


# Convenience function for creating a managed checkpointer
_global_checkpointer: Optional[NativeCheckpointer] = None
_global_checkpointer_lock = threading.Lock()


def get_native_checkpointer(
    policy: str = "sqrt",
    max_memory_mb: Optional[float] = None,
) -> NativeCheckpointer:
    """
    Get the global native checkpointer (singleton).

    Args:
        policy: Checkpoint selection policy
        max_memory_mb: Maximum memory budget

    Returns:
        Global NativeCheckpointer instance
    """
    global _global_checkpointer

    if _global_checkpointer is None:
        with _global_checkpointer_lock:
            if _global_checkpointer is None:
                _global_checkpointer = NativeCheckpointer(
                    policy=policy,
                    max_memory_mb=max_memory_mb,
                )

    return _global_checkpointer
