# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
FX Graph Optimizer for torch.compile Backend

This module applies pattern-based optimizations to PyTorch FX GraphModules,
enabling Zenith to replace standard operations with optimized implementations.

Key Features:
- Uses torch.fx.subgraph_rewriter for pattern matching
- Applies multiple optimization passes
- Graceful fallback on pattern match failure
"""

import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("zenith.optimization.fx_optimizer")

# Check for torch availability
_HAS_TORCH = False
_HAS_FX = False
torch = None

try:
    import torch as _torch

    torch = _torch
    _HAS_TORCH = True

    # Check for FX and subgraph_rewriter
    if hasattr(torch, "fx"):
        from torch.fx import subgraph_rewriter

        _HAS_FX = True
except ImportError:
    pass


@dataclass
class OptimizationStats:
    """Statistics from FX graph optimization."""

    total_patterns_checked: int = 0
    patterns_matched: int = 0
    patterns_replaced: int = 0
    patterns_failed: int = 0
    passes_applied: list[str] = field(default_factory=list)


class FXOptimizer:
    """
    Optimizes PyTorch FX GraphModules using pattern replacement.

    This optimizer uses torch.fx.subgraph_rewriter to find and replace
    subgraphs matching specific patterns with optimized implementations.

    Example:
        optimizer = FXOptimizer()
        optimized_gm = optimizer.optimize(gm, example_inputs)
    """

    def __init__(
        self,
        enable_attention: bool = True,
        enable_activation: bool = True,
        enable_normalization: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the FX optimizer.

        Args:
            enable_attention: Enable attention pattern replacement
            enable_activation: Enable activation pattern replacement
            enable_normalization: Enable normalization pattern replacement
            verbose: Enable verbose logging
        """
        self.enable_attention = enable_attention
        self.enable_activation = enable_activation
        self.enable_normalization = enable_normalization
        self.verbose = verbose
        self.stats = OptimizationStats()

        if not _HAS_FX:
            logger.warning("PyTorch FX not available, optimization disabled")

    def optimize(self, gm: Any, example_inputs: Optional[list] = None) -> Any:
        """
        Apply all enabled optimizations to the GraphModule.

        Args:
            gm: PyTorch FX GraphModule
            example_inputs: Optional example inputs for tracing

        Returns:
            Optimized GraphModule
        """
        if not _HAS_FX:
            logger.debug("FX not available, returning original graph")
            return gm

        self.stats = OptimizationStats()

        try:
            from .fx_patterns import (
                get_attention_patterns,
                get_activation_patterns,
                get_normalization_patterns,
            )

            # Apply attention patterns
            if self.enable_attention:
                for pattern in get_attention_patterns():
                    if pattern.enabled:
                        gm = self._apply_pattern(gm, pattern)

            # Apply activation patterns
            if self.enable_activation:
                for pattern in get_activation_patterns():
                    if pattern.enabled:
                        gm = self._apply_pattern(gm, pattern)

            # Apply normalization patterns
            if self.enable_normalization:
                for pattern in get_normalization_patterns():
                    if pattern.enabled:
                        gm = self._apply_pattern(gm, pattern)

            # Recompile the graph
            gm.recompile()

            if self.verbose:
                logger.info(f"FX optimization complete: {self.stats}")

        except Exception as e:
            logger.warning(f"FX optimization failed: {e}")
            # Return original on failure

        return gm

    def _apply_pattern(self, gm: Any, pattern: Any) -> Any:
        """
        Apply a single pattern replacement to the GraphModule.

        Args:
            gm: GraphModule to optimize
            pattern: FXPattern to apply

        Returns:
            Modified GraphModule
        """
        self.stats.total_patterns_checked += 1

        try:
            # Use PyTorch's subgraph_rewriter
            replaced = subgraph_rewriter.replace_pattern(
                gm,
                pattern.pattern_fn,
                pattern.replacement_fn,
            )

            if replaced:
                self.stats.patterns_matched += 1
                self.stats.patterns_replaced += len(replaced)
                self.stats.passes_applied.append(pattern.name)

                if self.verbose:
                    logger.info(
                        f"Pattern '{pattern.name}' matched {len(replaced)} times"
                    )

        except Exception as e:
            self.stats.patterns_failed += 1
            if self.verbose:
                logger.debug(f"Pattern '{pattern.name}' failed: {e}")

        return gm

    def get_stats(self) -> OptimizationStats:
        """Get optimization statistics."""
        return self.stats


def optimize_fx_graph(
    gm: Any,
    example_inputs: Optional[list] = None,
    **kwargs,
) -> Any:
    """
    Convenience function to optimize an FX GraphModule.

    Args:
        gm: PyTorch FX GraphModule
        example_inputs: Optional example inputs
        **kwargs: Additional options passed to FXOptimizer

    Returns:
        Optimized GraphModule
    """
    optimizer = FXOptimizer(**kwargs)
    return optimizer.optimize(gm, example_inputs)


def is_fx_available() -> bool:
    """Check if FX optimization is available."""
    return _HAS_FX
