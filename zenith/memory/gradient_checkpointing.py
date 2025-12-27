# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Gradient Checkpointing Module for Zenith (Phase 1 - PyTorch Integration).

This module provides memory-efficient training by selectively storing activations
and recomputing them during backward pass. Based on Chen et al., 2016.

Technical Foundation:
---------------------
Standard backpropagation requires O(N) memory for N layers, storing all
intermediate activations. Gradient checkpointing reduces this to O(sqrt(N))
by only storing checkpoint activations and recomputing intermediate values.

Mathematical Model (from CetakBiru.md Section 4.2):
    Minimize: M_peak(S)
    Subject to: T_recomputation(S) <= (1 + lambda) * T_baseline

    Where:
    - S = subset of nodes selected as checkpoints
    - M_peak = peak memory usage
    - lambda = time tolerance factor (e.g., 0.2 for 20% overhead)

Memory Complexity:
    - Standard:    O(N) memory, O(N) compute
    - Checkpointed: O(sqrt(N)) memory, O(N * sqrt(N)) compute
    - Segment-k:   O(N/k + k) memory, O(N + N/k) compute

References:
-----------
1. Chen et al., 2016: "Training Deep Nets with Sublinear Memory Cost"
2. Jain et al., 2020: "Checkmate: Breaking the Memory Wall"
3. PyTorch torch.utils.checkpoint documentation
4. NVIDIA Megatron-LM activation checkpointing

Usage:
------
    import zenith.memory as zmem

    # Method 1: Wrap a function
    output = zmem.checkpoint(expensive_function, input1, input2)

    # Method 2: Wrap sequential layers
    output = zmem.checkpoint_sequential(layers, num_segments, input)

    # Method 3: Decorator
    @zmem.auto_checkpoint(policy="sqrt")
    class MyModel(nn.Module):
        ...

    # Method 4: Context manager
    with zmem.CheckpointingContext(enabled=True):
        output = model(input)
"""

from __future__ import annotations

import functools
import math
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, TypeVar, Union
import weakref

# Type variables
T = TypeVar("T")
ModuleT = TypeVar("ModuleT")


class CheckpointPolicy(Enum):
    """Policy for automatic checkpoint selection.

    NONE: No checkpointing (standard training).
    SQRT: Checkpoint every sqrt(N) layers (optimal for memory).
    SEGMENT: Divide into K equal segments, checkpoint boundaries.
    EVERY_N: Checkpoint every N layers.
    SELECTIVE: Only checkpoint specified layer types.
    MEMORY_AWARE: Dynamic selection based on layer memory usage.
    """

    NONE = "none"
    SQRT = "sqrt"
    SEGMENT = "segment"
    EVERY_N = "every_n"
    SELECTIVE = "selective"
    MEMORY_AWARE = "memory_aware"


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing.

    Attributes:
        enabled: Whether checkpointing is enabled.
        policy: Checkpoint selection policy.
        num_segments: Number of segments for SEGMENT policy.
        every_n: Checkpoint interval for EVERY_N policy.
        checkpoint_layers: Layer types to checkpoint for SELECTIVE policy.
        preserve_rng_state: Whether to preserve RNG state during recomputation.
        use_reentrant: Use reentrant checkpointing (PyTorch < 2.0 compatibility).
        memory_efficient: Use memory-efficient implementation when available.
        lambda_tolerance: Time overhead tolerance factor (0.2 = 20% slower OK).
        debug: Enable debug mode with detailed logging.
    """

    enabled: bool = True
    policy: CheckpointPolicy = CheckpointPolicy.SQRT
    num_segments: int = 0  # 0 = auto-calculate
    every_n: int = 1
    checkpoint_layers: tuple[str, ...] = ()
    preserve_rng_state: bool = True
    use_reentrant: bool = False  # Non-reentrant is safer, PyTorch 2.0+
    memory_efficient: bool = True
    lambda_tolerance: float = 0.2
    debug: bool = False


# Thread-local context for checkpointing settings
_checkpoint_context = threading.local()


def _get_context_config() -> Optional[CheckpointConfig]:
    """Get the current checkpointing context configuration."""
    return getattr(_checkpoint_context, "config", None)


def _set_context_config(config: Optional[CheckpointConfig]) -> None:
    """Set the current checkpointing context configuration."""
    _checkpoint_context.config = config


class CheckpointingContext:
    """Context manager for enabling/disabling gradient checkpointing.

    This allows temporary control over checkpointing behavior within a scope.

    Example:
        # Enable checkpointing for this scope
        with CheckpointingContext(enabled=True, policy=CheckpointPolicy.SQRT):
            loss = model(input)
            loss.backward()

        # Disable checkpointing for validation
        with CheckpointingContext(enabled=False):
            output = model(input)
    """

    def __init__(
        self,
        enabled: bool = True,
        policy: CheckpointPolicy = CheckpointPolicy.SQRT,
        **kwargs: Any,
    ):
        """Initialize checkpointing context.

        Args:
            enabled: Whether checkpointing is enabled in this context.
            policy: Checkpoint selection policy.
            **kwargs: Additional config options (see CheckpointConfig).
        """
        self._new_config = CheckpointConfig(enabled=enabled, policy=policy, **kwargs)
        self._old_config: Optional[CheckpointConfig] = None

    def __enter__(self) -> "CheckpointingContext":
        """Enter the checkpointing context."""
        self._old_config = _get_context_config()
        _set_context_config(self._new_config)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit the checkpointing context."""
        _set_context_config(self._old_config)
        return False

    @property
    def config(self) -> CheckpointConfig:
        """Get the configuration for this context."""
        return self._new_config


def _get_torch():
    """Lazy import of torch with comprehensive error handling."""
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for gradient checkpointing. "
            "Install with: pip install torch"
        ) from e


def _get_checkpoint_module():
    """Get torch.utils.checkpoint module."""
    torch = _get_torch()
    return torch.utils.checkpoint


def _has_non_reentrant_checkpoint() -> bool:
    """Check if non-reentrant checkpointing is available (PyTorch 1.11+)."""
    torch = _get_torch()
    version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    return version >= (1, 11)


def checkpoint(
    function: Callable[..., T],
    *args: Any,
    preserve_rng_state: bool = True,
    use_reentrant: Optional[bool] = None,
    context_fn: Optional[Callable[[], Any]] = None,
    determinism_check: str = "default",
    debug: bool = False,
    **kwargs: Any,
) -> T:
    """Apply gradient checkpointing to a function.

    This wraps torch.utils.checkpoint.checkpoint with Zenith enhancements:
    - Automatic detection of PyTorch version capabilities
    - Context-aware configuration
    - Memory usage estimation
    - Debug logging

    Args:
        function: Function to checkpoint. Must be deterministic.
        *args: Arguments to pass to the function.
        preserve_rng_state: Save and restore RNG state during recomputation.
        use_reentrant: Use reentrant checkpointing. False is safer but
            requires PyTorch >= 1.11. None = auto-detect.
        context_fn: Function returning context manager for recomputation.
        determinism_check: How to check for determinism ("default", "none").
        debug: Enable debug output.
        **kwargs: Additional arguments for the function.

    Returns:
        Output of the function, with gradients computed via checkpointing.

    Example:
        def bottleneck(x, conv1, conv2, conv3):
            x = F.relu(conv1(x))
            x = F.relu(conv2(x))
            x = conv3(x)
            return x

        # Without checkpointing: stores all intermediate activations
        out = bottleneck(x, conv1, conv2, conv3)

        # With checkpointing: stores only input/output, recomputes intermediate
        out = checkpoint(bottleneck, x, conv1, conv2, conv3)

    Note:
        The function must be deterministic. Non-deterministic operations
        (like dropout) require preserve_rng_state=True.
    """
    # Check context configuration
    ctx_config = _get_context_config()
    if ctx_config is not None and not ctx_config.enabled:
        # Checkpointing disabled in context, run function directly
        return function(*args, **kwargs)

    torch = _get_torch()
    cp_module = _get_checkpoint_module()

    # Determine reentrant mode
    if use_reentrant is None:
        if ctx_config is not None:
            use_reentrant = ctx_config.use_reentrant
        else:
            # Default to non-reentrant if available (safer)
            use_reentrant = not _has_non_reentrant_checkpoint()

    # Build checkpoint kwargs
    cp_kwargs: dict[str, Any] = {
        "preserve_rng_state": preserve_rng_state,
    }

    # Add non-reentrant specific options if supported
    if _has_non_reentrant_checkpoint():
        cp_kwargs["use_reentrant"] = use_reentrant

        if not use_reentrant:
            # Non-reentrant specific options (PyTorch 2.0+)
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_version >= (2, 0):
                if context_fn is not None:
                    cp_kwargs["context_fn"] = context_fn
                if determinism_check != "default":
                    cp_kwargs["determinism_check"] = determinism_check

    # Debug logging
    if debug or (ctx_config is not None and ctx_config.debug):
        _log_checkpoint_call(function, args, kwargs)

    # Handle kwargs by creating a partial function
    if kwargs:
        function = functools.partial(function, **kwargs)

    # Call PyTorch checkpoint
    return cp_module.checkpoint(function, *args, **cp_kwargs)


def _log_checkpoint_call(function: Callable, args: tuple, kwargs: dict) -> None:
    """Log checkpoint call for debugging."""
    import logging

    logger = logging.getLogger("zenith.memory.checkpoint")
    func_name = getattr(function, "__name__", str(function))
    arg_shapes = []
    torch = _get_torch()

    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg_shapes.append(f"Tensor{tuple(arg.shape)}")
        else:
            arg_shapes.append(type(arg).__name__)

    logger.debug(f"Checkpointing {func_name} with args: {', '.join(arg_shapes)}")


def checkpoint_sequential(
    functions: Sequence[Callable],
    segments: int = 0,
    input: Any = None,
    preserve_rng_state: bool = True,
    use_reentrant: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """Checkpoint a sequence of functions/modules.

    Divides the sequence into segments and checkpoints each segment boundary.
    This is optimal for sequential models like ResNet.

    Memory complexity: O(N/segments + segments)
    Compute overhead: O(N/segments) recomputation

    Args:
        functions: Sequence of callables (e.g., nn.ModuleList).
        segments: Number of segments. 0 = auto-calculate (sqrt(N)).
        input: Input to the first function.
        preserve_rng_state: Whether to preserve RNG state.
        use_reentrant: Whether to use reentrant checkpointing.
        **kwargs: Additional arguments.

    Returns:
        Output of the last function in the sequence.

    Example:
        class ResNet(nn.Module):
            def __init__(self):
                self.layers = nn.ModuleList([
                    BasicBlock(64, 64) for _ in range(16)
                ])

            def forward(self, x):
                # Checkpoint every 4 layers (4 segments for 16 layers)
                return checkpoint_sequential(self.layers, 4, x)
    """
    if input is None:
        raise ValueError("input argument is required")

    n_functions = len(functions)
    if n_functions == 0:
        return input

    ctx_config = _get_context_config()
    if ctx_config is not None and not ctx_config.enabled:
        # Checkpointing disabled, run directly
        result = input
        for func in functions:
            result = func(result)
        return result

    # Auto-calculate segments using sqrt heuristic
    if segments <= 0:
        if ctx_config is not None and ctx_config.num_segments > 0:
            segments = ctx_config.num_segments
        else:
            # Optimal: sqrt(N) segments for O(sqrt(N)) memory
            segments = max(1, int(math.sqrt(n_functions)))

    # Ensure segments doesn't exceed number of functions
    segments = min(segments, n_functions)

    if use_reentrant is None:
        if ctx_config is not None:
            use_reentrant = ctx_config.use_reentrant
        else:
            use_reentrant = not _has_non_reentrant_checkpoint()

    torch = _get_torch()
    cp_module = _get_checkpoint_module()

    # Use PyTorch's checkpoint_sequential
    cp_kwargs = {"preserve_rng_state": preserve_rng_state}
    if _has_non_reentrant_checkpoint():
        cp_kwargs["use_reentrant"] = use_reentrant

    # Convert to nn.Sequential if needed
    if not isinstance(functions, torch.nn.Sequential):
        functions = torch.nn.Sequential(*functions)

    return cp_module.checkpoint_sequential(functions, segments, input, **cp_kwargs)


def checkpoint_wrapper(
    module: ModuleT,
    checkpoint_fn: Callable = checkpoint,
    **checkpoint_kwargs: Any,
) -> ModuleT:
    """Wrap a module to use gradient checkpointing in forward pass.

    This modifies the module's forward method to use checkpointing.
    Useful for applying checkpointing to specific layers.

    Args:
        module: The nn.Module to wrap.
        checkpoint_fn: The checkpoint function to use.
        **checkpoint_kwargs: Arguments to pass to checkpoint function.

    Returns:
        The modified module (same object).

    Example:
        class TransformerBlock(nn.Module):
            ...

        # Wrap the block to use checkpointing
        block = checkpoint_wrapper(TransformerBlock())

        # Now forward() will automatically checkpoint
        output = block(input)
    """
    torch = _get_torch()

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(module)}")

    original_forward = module.forward

    @functools.wraps(original_forward)
    def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
        # Only checkpoint during training
        if module.training:
            return checkpoint_fn(original_forward, *args, **checkpoint_kwargs, **kwargs)
        return original_forward(*args, **kwargs)

    # Store original for potential unwrapping
    module._zenith_original_forward = original_forward  # type: ignore
    module.forward = checkpointed_forward  # type: ignore

    return module


def unwrap_checkpoint(module: ModuleT) -> ModuleT:
    """Remove checkpoint wrapper from a module.

    Args:
        module: Module previously wrapped with checkpoint_wrapper.

    Returns:
        The module with original forward restored.
    """
    if hasattr(module, "_zenith_original_forward"):
        module.forward = module._zenith_original_forward  # type: ignore
        delattr(module, "_zenith_original_forward")
    return module


class SegmentCheckpointer:
    """Manages checkpointing for a sequence of operations.

    This class provides fine-grained control over which operations
    are checkpointed within a larger computation.

    Example:
        checkpointer = SegmentCheckpointer(num_checkpoints=4)

        def forward(x):
            for i, layer in enumerate(self.layers):
                if checkpointer.should_checkpoint(i, len(self.layers)):
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
            return x
    """

    def __init__(
        self,
        num_checkpoints: int = 0,
        policy: CheckpointPolicy = CheckpointPolicy.SQRT,
        checkpoint_layers: Optional[set[int]] = None,
    ):
        """Initialize segment checkpointer.

        Args:
            num_checkpoints: Number of checkpoints to use (0 = auto).
            policy: Policy for selecting checkpoint positions.
            checkpoint_layers: Explicit set of layer indices to checkpoint.
        """
        self._num_checkpoints = num_checkpoints
        self._policy = policy
        self._checkpoint_layers = checkpoint_layers
        self._cached_positions: dict[int, set[int]] = {}

    def _compute_checkpoint_positions(self, total_layers: int) -> set[int]:
        """Compute which layer positions should be checkpointed."""
        if self._checkpoint_layers is not None:
            return self._checkpoint_layers

        if total_layers in self._cached_positions:
            return self._cached_positions[total_layers]

        positions: set[int] = set()

        if self._policy == CheckpointPolicy.NONE:
            pass  # No checkpoints
        elif self._policy == CheckpointPolicy.SQRT:
            # Checkpoint every sqrt(N) layers
            interval = max(1, int(math.sqrt(total_layers)))
            positions = {i for i in range(0, total_layers, interval)}
        elif self._policy == CheckpointPolicy.SEGMENT:
            # Divide into segments, checkpoint boundaries
            num_segments = self._num_checkpoints or max(1, int(math.sqrt(total_layers)))
            segment_size = total_layers / num_segments
            positions = {int(i * segment_size) for i in range(num_segments)}
        elif self._policy == CheckpointPolicy.EVERY_N:
            n = self._num_checkpoints if self._num_checkpoints > 0 else 1
            positions = {i for i in range(0, total_layers, n)}

        self._cached_positions[total_layers] = positions
        return positions

    def should_checkpoint(self, layer_idx: int, total_layers: int) -> bool:
        """Determine if a layer should be checkpointed.

        Args:
            layer_idx: Index of the current layer.
            total_layers: Total number of layers.

        Returns:
            True if this layer should be checkpointed.
        """
        positions = self._compute_checkpoint_positions(total_layers)
        return layer_idx in positions


class ModuleCheckpointer:
    """Applies checkpointing to specific module types within a model.

    This is useful for selectively checkpointing memory-intensive layers
    like attention or large linear layers.

    Example:
        # Checkpoint all attention and MLP layers
        checkpointer = ModuleCheckpointer(
            target_types=(nn.MultiheadAttention, nn.Linear),
            min_params=1_000_000  # Only checkpoint layers with 1M+ params
        )
        checkpointer.apply(model)
    """

    def __init__(
        self,
        target_types: tuple[type, ...] = (),
        target_names: tuple[str, ...] = (),
        min_params: int = 0,
        max_depth: Optional[int] = None,
        exclude_types: tuple[type, ...] = (),
        exclude_names: tuple[str, ...] = (),
    ):
        """Initialize module checkpointer.

        Args:
            target_types: Module types to checkpoint.
            target_names: Module names (substrings) to checkpoint.
            min_params: Minimum parameter count to checkpoint.
            max_depth: Maximum nesting depth to apply (None = unlimited).
            exclude_types: Module types to never checkpoint.
            exclude_names: Module names to never checkpoint.
        """
        self._target_types = target_types
        self._target_names = target_names
        self._min_params = min_params
        self._max_depth = max_depth
        self._exclude_types = exclude_types
        self._exclude_names = exclude_names
        self._wrapped_modules: weakref.WeakSet = weakref.WeakSet()

    def _should_checkpoint(
        self,
        name: str,
        module: Any,
        depth: int,
    ) -> bool:
        """Determine if a module should be checkpointed."""
        torch = _get_torch()

        # Check exclusions first
        if isinstance(module, self._exclude_types):
            return False

        for exclude_name in self._exclude_names:
            if exclude_name in name:
                return False

        # Check depth
        if self._max_depth is not None and depth > self._max_depth:
            return False

        # Check target types
        if self._target_types and isinstance(module, self._target_types):
            pass  # Match
        elif self._target_names:
            if not any(target in name for target in self._target_names):
                return False
        elif not self._target_types:
            return False  # No targets specified, skip

        # Check parameter count
        if self._min_params > 0:
            param_count = sum(p.numel() for p in module.parameters())
            if param_count < self._min_params:
                return False

        return True

    def apply(self, model: Any) -> int:
        """Apply checkpointing to matching modules.

        Args:
            model: The root module to process.

        Returns:
            Number of modules wrapped.
        """
        count = 0
        self._apply_recursive(model, "", 0, count)
        return len(self._wrapped_modules)

    def _apply_recursive(
        self,
        module: Any,
        name: str,
        depth: int,
        count: int,
    ) -> None:
        """Recursively apply checkpointing."""
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if self._should_checkpoint(full_name, child, depth):
                checkpoint_wrapper(child)
                self._wrapped_modules.add(child)

            # Recurse into children
            self._apply_recursive(child, full_name, depth + 1, count)

    def remove(self) -> int:
        """Remove checkpointing from all wrapped modules.

        Returns:
            Number of modules unwrapped.
        """
        count = 0
        for module in list(self._wrapped_modules):
            unwrap_checkpoint(module)
            count += 1
        self._wrapped_modules.clear()
        return count


def auto_checkpoint(
    policy: Union[CheckpointPolicy, str] = CheckpointPolicy.SQRT,
    num_segments: int = 0,
    target_types: tuple[type, ...] = (),
    min_params: int = 0,
    **config_kwargs: Any,
) -> Callable[[type], type]:
    """Decorator to automatically apply checkpointing to a module class.

    This decorator modifies the class's forward method to use checkpointing
    based on the specified policy.

    Args:
        policy: Checkpoint selection policy.
        num_segments: Number of segments (for SEGMENT policy).
        target_types: Module types to checkpoint (for SELECTIVE policy).
        min_params: Minimum parameters for checkpointing.
        **config_kwargs: Additional config options.

    Returns:
        Decorated class.

    Example:
        @auto_checkpoint(policy=CheckpointPolicy.SQRT)
        class MyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    TransformerBlock() for _ in range(12)
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
    """
    if isinstance(policy, str):
        policy = CheckpointPolicy(policy)

    def decorator(cls: type) -> type:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            # Apply checkpointing after initialization
            if target_types:
                checkpointer = ModuleCheckpointer(
                    target_types=target_types,
                    min_params=min_params,
                )
                checkpointer.apply(self)
                self._zenith_checkpointer = checkpointer

        cls.__init__ = new_init  # type: ignore

        # Store config for reference
        cls._zenith_checkpoint_config = CheckpointConfig(
            policy=policy,
            num_segments=num_segments,
            **config_kwargs,
        )

        return cls

    return decorator


@dataclass
class MemoryStats:
    """Memory statistics for checkpointing analysis."""

    total_activations_mb: float = 0.0
    checkpointed_mb: float = 0.0
    stored_mb: float = 0.0
    estimated_peak_mb: float = 0.0
    memory_savings_pct: float = 0.0
    recompute_overhead_pct: float = 0.0


def estimate_memory_savings(
    model: Any,
    sample_input: Any,
    policy: CheckpointPolicy = CheckpointPolicy.SQRT,
    num_segments: int = 0,
) -> MemoryStats:
    """Estimate memory savings from checkpointing.

    This function calculates theoretical memory savings based on the
    checkpointing policy and model structure.

    Args:
        model: The model to analyze.
        sample_input: Sample input for profiling (unused in current impl).
        policy: Checkpointing policy to simulate.
        num_segments: Number of segments (for SEGMENT policy).

    Returns:
        MemoryStats with savings estimates.

    Note:
        This provides a theoretical estimate based on published research.
        Actual savings depend on model architecture and activation sizes.
    """
    torch = _get_torch()
    stats = MemoryStats()

    # Count total layers/modules
    try:
        total_layers = sum(1 for _ in model.modules())
    except (AttributeError, TypeError):
        # Model doesn't have modules() method, try to get layer count another way
        total_layers = 1

    if total_layers <= 1:
        # No meaningful checkpointing for single layer
        stats.memory_savings_pct = 0.0
        stats.recompute_overhead_pct = 0.0
        return stats

    # Calculate checkpoint positions based on policy
    if policy == CheckpointPolicy.SQRT:
        num_checkpoints = max(1, int(math.sqrt(total_layers)))
    elif policy == CheckpointPolicy.SEGMENT:
        num_checkpoints = (
            num_segments if num_segments > 0 else int(math.sqrt(total_layers))
        )
    elif policy == CheckpointPolicy.EVERY_N:
        num_checkpoints = max(1, total_layers // max(1, num_segments))
    elif policy == CheckpointPolicy.NONE:
        stats.memory_savings_pct = 0.0
        stats.recompute_overhead_pct = 0.0
        return stats
    else:
        num_checkpoints = total_layers

    # Ensure valid range
    num_checkpoints = max(1, min(num_checkpoints, total_layers))

    # Mathematical model for memory savings:
    # Standard: O(N) activations stored
    # With k checkpoints: O(N/k + k) activations stored
    # Optimal k = sqrt(N) gives O(sqrt(N)) memory

    # Memory reduction factor: stored_with_cp / stored_without_cp
    # For sqrt checkpointing: sqrt(N) / N = 1/sqrt(N)
    reduction_factor = num_checkpoints / total_layers

    # Memory savings percentage
    # savings = 1 - (memory_with_cp / memory_without_cp)
    # For theoretical sqrt model: 1 - 1/sqrt(N)
    stats.memory_savings_pct = (1 - math.sqrt(reduction_factor)) * 100

    # Recompute overhead: each segment is recomputed during backward
    # For sqrt(N) checkpoints, we recompute sqrt(N) segments of size sqrt(N)
    # Total recompute = N (same as forward), so overhead ~ 33% of total time
    stats.recompute_overhead_pct = min(
        100.0, (1 / max(1, math.sqrt(num_checkpoints))) * 100
    )

    # Warn if CUDA not available (actual memory profiling not possible)
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA not available. Memory savings are theoretical estimates only.",
            RuntimeWarning,
            stacklevel=2,
        )

    return stats


def get_memory_stats() -> dict[str, float]:
    """Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats in MB.
    """
    torch = _get_torch()

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }
