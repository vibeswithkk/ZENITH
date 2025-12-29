# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit tests for FX Graph Pattern Matching

Tests the pattern detection and replacement functionality
in zenith.optimization.fx_patterns and fx_optimizer.
"""

import pytest


class TestFXPatterns:
    """Tests for FX pattern definitions."""

    def test_pattern_registry_available(self):
        """Test that pattern registry functions are available."""
        from zenith.optimization.fx_patterns import (
            get_attention_patterns,
            get_activation_patterns,
            get_normalization_patterns,
            get_all_patterns,
        )

        assert callable(get_attention_patterns)
        assert callable(get_activation_patterns)
        assert callable(get_normalization_patterns)
        assert callable(get_all_patterns)

    def test_get_all_patterns_returns_list(self):
        """Test that get_all_patterns returns a non-empty list."""
        from zenith.optimization.fx_patterns import get_all_patterns

        patterns = get_all_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_pattern_structure(self):
        """Test that patterns have required attributes."""
        from zenith.optimization.fx_patterns import get_all_patterns, FXPattern

        patterns = get_all_patterns()
        for pattern in patterns:
            assert isinstance(pattern, FXPattern)
            assert hasattr(pattern, "name")
            assert hasattr(pattern, "pattern_fn")
            assert hasattr(pattern, "replacement_fn")
            assert hasattr(pattern, "priority")
            assert hasattr(pattern, "enabled")
            assert callable(pattern.pattern_fn)
            assert callable(pattern.replacement_fn)

    def test_attention_patterns_exist(self):
        """Test that attention patterns are registered."""
        from zenith.optimization.fx_patterns import get_attention_patterns

        patterns = get_attention_patterns()
        assert len(patterns) >= 1
        names = [p.name for p in patterns]
        assert "sdpa_to_flash" in names

    def test_activation_patterns_exist(self):
        """Test that activation patterns are registered."""
        from zenith.optimization.fx_patterns import get_activation_patterns

        patterns = get_activation_patterns()
        assert len(patterns) >= 1
        names = [p.name for p in patterns]
        assert "gelu_tanh_to_native" in names

    def test_normalization_patterns_exist(self):
        """Test that normalization patterns are registered."""
        from zenith.optimization.fx_patterns import get_normalization_patterns

        patterns = get_normalization_patterns()
        assert len(patterns) >= 1
        names = [p.name for p in patterns]
        assert "manual_layernorm_to_native" in names

    def test_zenith_cuda_availability_check(self):
        """Test that CUDA availability check function exists."""
        from zenith.optimization.fx_patterns import is_zenith_cuda_available

        result = is_zenith_cuda_available()
        assert isinstance(result, bool)


class TestFXOptimizer:
    """Tests for FX graph optimizer."""

    def test_optimizer_import(self):
        """Test that FXOptimizer can be imported."""
        from zenith.optimization.fx_optimizer import FXOptimizer

        assert FXOptimizer is not None

    def test_optimizer_initialization(self):
        """Test FXOptimizer initialization with default parameters."""
        from zenith.optimization.fx_optimizer import FXOptimizer

        optimizer = FXOptimizer()
        assert optimizer.enable_attention is True
        assert optimizer.enable_activation is True
        assert optimizer.enable_normalization is True

    def test_optimizer_custom_initialization(self):
        """Test FXOptimizer initialization with custom parameters."""
        from zenith.optimization.fx_optimizer import FXOptimizer

        optimizer = FXOptimizer(
            enable_attention=False,
            enable_activation=True,
            enable_normalization=False,
            verbose=True,
        )
        assert optimizer.enable_attention is False
        assert optimizer.enable_activation is True
        assert optimizer.enable_normalization is False
        assert optimizer.verbose is True

    def test_optimize_fx_graph_function(self):
        """Test that optimize_fx_graph convenience function exists."""
        from zenith.optimization.fx_optimizer import optimize_fx_graph

        assert callable(optimize_fx_graph)

    def test_is_fx_available_function(self):
        """Test that is_fx_available function exists and returns bool."""
        from zenith.optimization.fx_optimizer import is_fx_available

        result = is_fx_available()
        assert isinstance(result, bool)

    def test_optimization_stats_structure(self):
        """Test OptimizationStats dataclass structure."""
        from zenith.optimization.fx_optimizer import OptimizationStats

        stats = OptimizationStats()
        assert stats.total_patterns_checked == 0
        assert stats.patterns_matched == 0
        assert stats.patterns_replaced == 0
        assert stats.patterns_failed == 0
        assert isinstance(stats.passes_applied, list)


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="PyTorch required",
)
class TestFXPatternsWithTorch:
    """Tests requiring PyTorch."""

    def test_sdpa_pattern_callable(self):
        """Test that SDPA pattern function works."""
        import torch
        from zenith.optimization.fx_patterns import sdpa_pattern

        # Create test tensors
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)

        result = sdpa_pattern(q, k, v)
        assert result.shape == v.shape

    def test_gelu_pattern_callable(self):
        """Test that GELU pattern function works."""
        import torch
        from zenith.optimization.fx_patterns import gelu_tanh_pattern

        x = torch.randn(2, 64)
        result = gelu_tanh_pattern(x)
        assert result.shape == x.shape

    def test_layernorm_pattern_callable(self):
        """Test that LayerNorm pattern function works."""
        import torch
        from zenith.optimization.fx_patterns import layernorm_pattern

        x = torch.randn(2, 64)
        weight = torch.ones(64)
        bias = torch.zeros(64)

        result = layernorm_pattern(x, weight, bias)
        assert result.shape == x.shape

    def test_replacement_numerical_accuracy(self):
        """Test that replacements produce similar results to patterns."""
        import torch
        from zenith.optimization.fx_patterns import (
            gelu_tanh_pattern,
            zenith_gelu_replacement,
        )

        x = torch.randn(2, 64)

        pattern_result = gelu_tanh_pattern(x)
        replacement_result = zenith_gelu_replacement(x)

        # Allow small numerical difference
        diff = (pattern_result - replacement_result).abs().max()
        assert diff < 1e-5, f"GELU replacement differs by {diff}"
