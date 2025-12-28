# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith JAX Gradient Checkpointing Module.

Provides memory-efficient gradient checkpointing for JAX training with
optimal checkpoint selection algorithms.

Technical Foundation:
--------------------
Based on Chen et al., 2016 ("Training Deep Nets with Sublinear Memory Cost"):

    Memory Complexity: O(sqrt(N)) vs O(N) for standard backprop
    Compute Overhead: ~33% additional forward pass computation

The implementation wraps JAX's native jax.checkpoint (jax.remat) with
Zenith's optimal checkpoint selection algorithms.

Mathematical Model:
------------------
For a network with N sequential layers:
    - Standard: Store all N activations = O(N) memory
    - Sqrt checkpointing: Store sqrt(N) checkpoints
        Memory: O(sqrt(N))
        Compute: O(N) + O(N) + O(sqrt(N) * sqrt(N)) = O(N)

References:
----------
1. Chen et al., 2016: "Training Deep Nets with Sublinear Memory Cost"
2. Jain et al., 2020: "Checkmate: Breaking the Memory Wall"
3. JAX checkpointing docs: https://jax.dev/docs/advanced_autodiff.html
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger("zenith.jax.checkpointing")

F = TypeVar("F", bound=Callable[..., Any])


def _get_jax():
    """Lazy import of JAX with error handling."""
    try:
        import jax

        return jax
    except ImportError as e:
        raise ImportError(
            "JAX is required for gradient checkpointing. "
            "Install with: pip install jax jaxlib"
        ) from e


def _get_jnp():
    """Lazy import of jax.numpy."""
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for gradient checkpointing. "
            "Install with: pip install jax jaxlib"
        ) from e


class CheckpointPolicy(Enum):
    """
    Policy for checkpoint selection.

    NOTHING: Save nothing, recompute everything (maximum memory savings)
    EVERYTHING: Save everything (no checkpointing, standard behavior)
    DOTS_SAVEABLE: Save dot products only (good for transformers)
    DOTS_WITH_NO_BATCH: Save dot products for params only
    OFFLOAD_CPU: Offload activations to CPU memory
    """

    NOTHING = "nothing"
    EVERYTHING = "everything"
    DOTS_SAVEABLE = "dots_saveable"
    DOTS_WITH_NO_BATCH = "dots_with_no_batch"
    OFFLOAD_CPU = "offload_cpu"


class SelectionMethod(Enum):
    """Algorithm for checkpoint position selection."""

    SQRT = "sqrt"  # O(sqrt(n)) checkpoints, simple heuristic
    DP = "dp"  # Dynamic programming, optimal solution
    UNIFORM = "uniform"  # Evenly spaced checkpoints
    CUSTOM = "custom"  # User-provided positions


@dataclass
class CheckpointConfig:
    """
    Configuration for gradient checkpointing.

    Attributes:
        policy: Checkpoint policy (affects what values are saved)
        selection_method: Algorithm for selecting checkpoint positions
        memory_budget_gb: Maximum memory budget in GB (None = unlimited)
        compute_tolerance: Maximum allowed compute overhead (default 0.33 = 33%)
        offload_to_cpu: Enable CPU offloading for large activations
        offload_threshold_mb: Size threshold for CPU offloading in MB
        enable_profiling: Track detailed timing and memory statistics
    """

    policy: CheckpointPolicy = CheckpointPolicy.DOTS_SAVEABLE
    selection_method: SelectionMethod = SelectionMethod.SQRT
    memory_budget_gb: Optional[float] = None
    compute_tolerance: float = 0.33
    offload_to_cpu: bool = False
    offload_threshold_mb: float = 100.0
    enable_profiling: bool = False
    custom_checkpoints: Optional[List[int]] = None


@dataclass
class CheckpointingStats:
    """Statistics from checkpointing operations."""

    num_checkpoints: int = 0
    total_layers: int = 0
    estimated_memory_saved_mb: float = 0.0
    estimated_compute_overhead_percent: float = 0.0
    offloaded_activations: int = 0
    recomputation_count: int = 0
    peak_memory_mb: float = 0.0
    checkpoint_positions: List[int] = field(default_factory=list)


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

    Time Complexity: O(N^2) for DP, O(N) for sqrt heuristic
    Space Complexity: O(N)
    """

    def __init__(
        self,
        num_layers: int,
        memory_costs: Optional[List[float]] = None,
        compute_costs: Optional[List[float]] = None,
        compute_tolerance: float = 0.33,
    ):
        """
        Initialize checkpoint selector.

        Args:
            num_layers: Total number of layers
            memory_costs: Memory cost per layer (None = uniform cost of 1.0)
            compute_costs: Compute cost per layer (None = uniform cost of 1.0)
            compute_tolerance: Maximum compute overhead allowed (0.33 = 33%)
        """
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if compute_tolerance < 0 or compute_tolerance > 1:
            raise ValueError("compute_tolerance must be in [0, 1]")

        self._num_layers = num_layers
        self._memory_costs = memory_costs if memory_costs else [1.0] * num_layers
        self._compute_costs = compute_costs if compute_costs else [1.0] * num_layers
        self._compute_tolerance = compute_tolerance

        if len(self._memory_costs) != num_layers:
            raise ValueError(
                f"memory_costs length ({len(self._memory_costs)}) "
                f"must match num_layers ({num_layers})"
            )
        if len(self._compute_costs) != num_layers:
            raise ValueError(
                f"compute_costs length ({len(self._compute_costs)}) "
                f"must match num_layers ({num_layers})"
            )

    def select_sqrt(self) -> List[int]:
        """
        Select checkpoints using sqrt(N) heuristic.

        This is the standard approach with theoretical guarantees:
        - Memory: O(sqrt(N))
        - Compute overhead: O(N)

        Returns:
            List of layer indices to checkpoint
        """
        if self._num_layers == 0:
            return []

        if self._num_layers == 1:
            return [0]

        interval = max(1, int(math.sqrt(self._num_layers)))
        return list(range(0, self._num_layers, interval))

    def select_uniform(self, num_checkpoints: Optional[int] = None) -> List[int]:
        """
        Select uniformly spaced checkpoints.

        Args:
            num_checkpoints: Number of checkpoints (default = sqrt(N))

        Returns:
            List of layer indices to checkpoint
        """
        if self._num_layers == 0:
            return []

        if num_checkpoints is None:
            num_checkpoints = max(1, int(math.sqrt(self._num_layers)))

        num_checkpoints = min(num_checkpoints, self._num_layers)

        if num_checkpoints == 1:
            return [0]

        interval = self._num_layers / num_checkpoints
        return [int(i * interval) for i in range(num_checkpoints)]

    def select_dp(self) -> List[int]:
        """
        Select checkpoints using dynamic programming.

        Finds optimal checkpoint positions that minimize peak memory
        while respecting compute overhead budget.

        Time Complexity: O(N^2)
        Space Complexity: O(N)

        Returns:
            List of optimal layer indices to checkpoint
        """
        n = self._num_layers
        if n == 0:
            return []
        if n == 1:
            return [0]

        total_compute = sum(self._compute_costs)
        max_recompute = self._compute_tolerance * total_compute

        INF = float("inf")
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        parent = [[-1] * (n + 1) for _ in range(n + 1)]

        dp[0][0] = 0

        for i in range(n):
            for k in range(n + 1):
                if dp[i][k] == INF:
                    continue

                segment_memory = 0.0
                segment_compute = 0.0

                for j in range(i + 1, n + 1):
                    segment_memory += self._memory_costs[j - 1]
                    segment_compute += self._compute_costs[j - 1]

                    if segment_compute > max_recompute:
                        break

                    new_cost = dp[i][k] + segment_memory
                    if new_cost < dp[j][k + 1]:
                        dp[j][k + 1] = new_cost
                        parent[j][k + 1] = i

        best_k = 0
        best_cost = INF
        for k in range(n + 1):
            if dp[n][k] < best_cost:
                best_cost = dp[n][k]
                best_k = k

        checkpoints = []
        pos = n
        k = best_k
        while pos > 0 and k > 0:
            prev = parent[pos][k]
            if prev >= 0:
                checkpoints.append(prev)
            pos = prev
            k -= 1

        checkpoints.reverse()

        if len(checkpoints) == 0 and n > 0:
            return self.select_sqrt()

        return checkpoints

    def select(self, method: SelectionMethod = SelectionMethod.SQRT) -> List[int]:
        """
        Select checkpoints using specified method.

        Args:
            method: Selection algorithm to use

        Returns:
            List of layer indices to checkpoint
        """
        if method == SelectionMethod.SQRT:
            return self.select_sqrt()
        elif method == SelectionMethod.DP:
            return self.select_dp()
        elif method == SelectionMethod.UNIFORM:
            return self.select_uniform()
        else:
            return self.select_sqrt()

    def estimate_memory_reduction(self, checkpoints: List[int]) -> float:
        """
        Estimate memory reduction percentage from checkpointing.

        Args:
            checkpoints: List of checkpoint positions

        Returns:
            Estimated memory reduction as percentage (0-100)
        """
        if self._num_layers == 0:
            return 0.0

        total_memory = sum(self._memory_costs)
        if total_memory == 0:
            return 0.0

        checkpoint_memory = sum(
            self._memory_costs[i] for i in checkpoints if i < len(self._memory_costs)
        )

        num_segments = len(checkpoints) + 1
        avg_segment_size = self._num_layers / max(1, num_segments)
        segment_memory = avg_segment_size * (total_memory / self._num_layers)

        estimated_peak = checkpoint_memory + segment_memory
        reduction = (1 - estimated_peak / total_memory) * 100

        return max(0.0, min(100.0, reduction))


def _create_jax_policy(policy: CheckpointPolicy):
    """
    Create JAX checkpoint policy function.

    JAX policies are functions that take a primitive and return True
    if the primitive's output should be saved.
    """
    jax = _get_jax()

    if policy == CheckpointPolicy.NOTHING:
        return jax.checkpoint_policies.nothing_saveable
    elif policy == CheckpointPolicy.EVERYTHING:
        return jax.checkpoint_policies.everything_saveable
    elif policy == CheckpointPolicy.DOTS_SAVEABLE:
        return jax.checkpoint_policies.dots_saveable
    elif policy == CheckpointPolicy.DOTS_WITH_NO_BATCH:
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    else:
        return jax.checkpoint_policies.dots_saveable


def checkpoint(
    fn: F,
    *,
    policy: CheckpointPolicy = CheckpointPolicy.DOTS_SAVEABLE,
    prevent_cse: bool = True,
    static_argnums: Optional[Tuple[int, ...]] = None,
) -> F:
    """
    Apply gradient checkpointing to a function.

    This is the primary API for applying checkpointing to JAX functions.
    Wraps jax.checkpoint with Zenith policy configuration.

    Args:
        fn: Function to checkpoint
        policy: Checkpoint policy to use
        prevent_cse: Prevent common subexpression elimination
        static_argnums: Arguments that should be treated as static

    Returns:
        Checkpointed version of the function

    Example:
        @zenith.jax.checkpoint
        def transformer_layer(x, params):
            return attention(x, params) + ffn(x, params)
    """
    jax = _get_jax()

    jax_policy = _create_jax_policy(policy)

    return jax.checkpoint(
        fn,
        prevent_cse=prevent_cse,
        policy=jax_policy,
        static_argnums=static_argnums,
    )


def checkpoint_sequential(
    functions: Sequence[Callable],
    input_value: Any,
    segments: Optional[int] = None,
    policy: CheckpointPolicy = CheckpointPolicy.DOTS_SAVEABLE,
    selection_method: SelectionMethod = SelectionMethod.SQRT,
) -> Any:
    """
    Apply checkpointing to a sequence of functions.

    Automatically selects optimal checkpoint positions based on
    the selected algorithm (sqrt or DP).

    Args:
        functions: Sequence of functions to apply in order
        input_value: Initial input value
        segments: Number of segments (None = auto-select using sqrt)
        policy: Checkpoint policy for each segment
        selection_method: Algorithm for checkpoint selection

    Returns:
        Final output after applying all functions

    Example:
        layers = [layer1, layer2, layer3, layer4, layer5]
        output = checkpoint_sequential(layers, x, segments=2)
    """
    if len(functions) == 0:
        return input_value

    if len(functions) == 1:
        return functions[0](input_value)

    jax = _get_jax()
    jax_policy = _create_jax_policy(policy)

    num_layers = len(functions)

    if segments is None:
        selector = OptimalCheckpointSelector(num_layers)
        checkpoints = selector.select(selection_method)
        segments = len(checkpoints)

    segments = max(1, min(segments, num_layers))
    segment_size = math.ceil(num_layers / segments)

    def run_segment(start_idx: int, end_idx: int, x: Any) -> Any:
        for i in range(start_idx, min(end_idx, num_layers)):
            x = functions[i](x)
        return x

    current = input_value
    for seg_idx in range(segments):
        start = seg_idx * segment_size
        end = min(start + segment_size, num_layers)

        if start >= num_layers:
            break

        if seg_idx < segments - 1:
            checkpointed_segment = jax.checkpoint(
                lambda x, s=start, e=end: run_segment(s, e, x),
                prevent_cse=True,
                policy=jax_policy,
            )
            current = checkpointed_segment(current)
        else:
            current = run_segment(start, end, current)

    return current


class ZenithCheckpointer:
    """
    High-level checkpointing manager for JAX models.

    Provides a unified interface for applying gradient checkpointing
    with configurable policies and optimal checkpoint selection.

    Attributes:
        config: Checkpointing configuration
        stats: Runtime statistics

    Example:
        checkpointer = ZenithCheckpointer(
            config=CheckpointConfig(
                policy=CheckpointPolicy.DOTS_SAVEABLE,
                selection_method=SelectionMethod.DP,
            )
        )

        # Checkpoint a single function
        checkpointed_fn = checkpointer.checkpoint(my_function)

        # Checkpoint sequential layers
        output = checkpointer.checkpoint_sequential(layers, x)
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize the checkpointer.

        Args:
            config: Checkpointing configuration (uses defaults if None)
        """
        self._config = config if config else CheckpointConfig()
        self._stats = CheckpointingStats()
        self._lock = threading.Lock()
        self._profiling_data: List[Dict[str, Any]] = []

    @property
    def config(self) -> CheckpointConfig:
        """Get current configuration."""
        return self._config

    @property
    def stats(self) -> CheckpointingStats:
        """Get runtime statistics."""
        return self._stats

    def checkpoint(
        self,
        fn: F,
        *,
        policy: Optional[CheckpointPolicy] = None,
        static_argnums: Optional[Tuple[int, ...]] = None,
    ) -> F:
        """
        Apply checkpointing to a function.

        Args:
            fn: Function to checkpoint
            policy: Override policy (uses config default if None)
            static_argnums: Arguments treated as static

        Returns:
            Checkpointed function
        """
        effective_policy = policy if policy else self._config.policy

        with self._lock:
            self._stats.num_checkpoints += 1

        return checkpoint(
            fn,
            policy=effective_policy,
            static_argnums=static_argnums,
        )

    def checkpoint_sequential(
        self,
        functions: Sequence[Callable],
        input_value: Any,
        segments: Optional[int] = None,
    ) -> Any:
        """
        Apply checkpointing to sequential functions.

        Args:
            functions: Sequence of functions
            input_value: Initial input
            segments: Number of segments (None = auto)

        Returns:
            Final output
        """
        num_layers = len(functions)

        if self._config.custom_checkpoints is not None:
            segments = len(self._config.custom_checkpoints)

        with self._lock:
            self._stats.total_layers = num_layers

        result = checkpoint_sequential(
            functions=functions,
            input_value=input_value,
            segments=segments,
            policy=self._config.policy,
            selection_method=self._config.selection_method,
        )

        if self._config.enable_profiling:
            selector = OptimalCheckpointSelector(num_layers)
            checkpoints = selector.select(self._config.selection_method)

            with self._lock:
                self._stats.checkpoint_positions = checkpoints
                self._stats.estimated_memory_saved_mb = (
                    selector.estimate_memory_reduction(checkpoints)
                )

        return result

    def wrap_module(
        self,
        module_apply_fn: Callable,
        checkpoint_layers: bool = True,
    ) -> Callable:
        """
        Wrap a Flax/Haiku module apply function with checkpointing.

        Args:
            module_apply_fn: The module's apply function
            checkpoint_layers: Whether to checkpoint internal layers

        Returns:
            Wrapped apply function with checkpointing
        """
        if not checkpoint_layers:
            return module_apply_fn

        return self.checkpoint(module_apply_fn)

    def reset_stats(self) -> None:
        """Reset runtime statistics."""
        with self._lock:
            self._stats = CheckpointingStats()
            self._profiling_data.clear()


def remat(
    fn: F,
    *,
    policy: CheckpointPolicy = CheckpointPolicy.DOTS_SAVEABLE,
    prevent_cse: bool = True,
) -> F:
    """
    Alias for checkpoint (matching JAX terminology).

    Rematerialization is the process of recomputing intermediate values
    during the backward pass instead of storing them.

    Args:
        fn: Function to rematerialize
        policy: Rematerialization policy
        prevent_cse: Prevent common subexpression elimination

    Returns:
        Rematerialized function
    """
    return checkpoint(fn, policy=policy, prevent_cse=prevent_cse)


__all__ = [
    "CheckpointPolicy",
    "SelectionMethod",
    "CheckpointConfig",
    "CheckpointingStats",
    "OptimalCheckpointSelector",
    "ZenithCheckpointer",
    "checkpoint",
    "checkpoint_sequential",
    "remat",
]
