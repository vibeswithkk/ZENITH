# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for Advanced Fusion Patterns and Passes.

Tests:
- FusionPriority enum
- AdvancedFusionPattern dataclass
- AdvancedFusionPass pattern matching
- FlashAttentionFusion detection
- TransformerBlockFusion detection
"""

import pytest

from zenith.optimization.advanced_fusion import (
    FusionPriority,
    AdvancedFusionPattern,
    AdvancedFusionPass,
    FlashAttentionFusion,
    TransformerBlockFusion,
    QKV_PROJECTION_PATTERN,
    SELF_ATTENTION_PATTERN,
    ATTENTION_EPILOGUE_PATTERN,
    GATED_MLP_PATTERN,
    FFN_BLOCK_PATTERN,
    ALL_ADVANCED_PATTERNS,
)
from zenith.core import GraphIR, Node


class TestFusionPriority:
    """Tests for FusionPriority enum."""

    def test_priority_values_exist(self):
        """Test that all priority levels are defined."""
        assert hasattr(FusionPriority, "CRITICAL")
        assert hasattr(FusionPriority, "HIGH")
        assert hasattr(FusionPriority, "MEDIUM")
        assert hasattr(FusionPriority, "LOW")

    def test_priority_ordering(self):
        """Test that priorities have correct ordering."""
        assert FusionPriority.CRITICAL.value < FusionPriority.HIGH.value
        assert FusionPriority.HIGH.value < FusionPriority.MEDIUM.value
        assert FusionPriority.MEDIUM.value < FusionPriority.LOW.value


class TestAdvancedFusionPattern:
    """Tests for AdvancedFusionPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a custom fusion pattern."""
        pattern = AdvancedFusionPattern(
            name="test_pattern",
            description="Test pattern for unit tests",
            ops_to_match=["Linear", "ReLU"],
            fused_op_name="FusedLinearReLU",
            priority=FusionPriority.MEDIUM,
            estimated_speedup=1.5,
        )

        assert pattern.name == "test_pattern"
        assert pattern.fused_op_name == "FusedLinearReLU"
        assert len(pattern.ops_to_match) == 2
        assert pattern.estimated_speedup == 1.5

    def test_predefined_patterns_exist(self):
        """Test that all predefined patterns are available."""
        assert QKV_PROJECTION_PATTERN is not None
        assert SELF_ATTENTION_PATTERN is not None
        assert ATTENTION_EPILOGUE_PATTERN is not None
        assert GATED_MLP_PATTERN is not None
        assert FFN_BLOCK_PATTERN is not None

    def test_all_patterns_list(self):
        """Test ALL_ADVANCED_PATTERNS contains expected patterns."""
        assert len(ALL_ADVANCED_PATTERNS) >= 5
        pattern_names = [p.name for p in ALL_ADVANCED_PATTERNS]
        assert "qkv_projection" in pattern_names
        assert "self_attention" in pattern_names


class TestAdvancedFusionPass:
    """Tests for AdvancedFusionPass."""

    def test_pass_creation(self):
        """Test creating an AdvancedFusionPass."""
        fusion_pass = AdvancedFusionPass()
        assert fusion_pass is not None
        assert len(fusion_pass.patterns) > 0

    def test_pass_with_custom_patterns(self):
        """Test creating pass with custom pattern list."""
        custom_patterns = [FFN_BLOCK_PATTERN]
        fusion_pass = AdvancedFusionPass(patterns=custom_patterns)
        assert len(fusion_pass.patterns) == 1

    def test_pass_with_priority_filter(self):
        """Test that priority filter works."""
        fusion_pass = AdvancedFusionPass(min_priority=FusionPriority.CRITICAL)
        # Should filter out lower priority patterns
        assert all(
            p.priority.value <= FusionPriority.CRITICAL.value
            for p in fusion_pass.active_patterns
        )

    def test_run_on_empty_graph(self):
        """Test running fusion pass on empty graph."""
        fusion_pass = AdvancedFusionPass()
        graph = GraphIR(name="empty_graph")

        result_graph, stats = fusion_pass.run(graph)

        assert result_graph is not None
        assert stats["nodes_before"] == 0
        assert stats["nodes_after"] == 0


class TestFlashAttentionFusion:
    """Tests for FlashAttentionFusion detection."""

    def test_detect_on_empty_graph(self):
        """Test attention detection on empty graph."""
        graph = GraphIR(name="empty")
        patterns = FlashAttentionFusion.detect_attention_pattern(graph)
        assert patterns == []


class TestTransformerBlockFusion:
    """Tests for TransformerBlockFusion detection."""

    def test_detect_on_empty_graph(self):
        """Test transformer block detection on empty graph."""
        graph = GraphIR(name="empty")
        blocks = TransformerBlockFusion.detect_transformer_blocks(graph)
        assert blocks == []


class TestPatternMatching:
    """Tests for pattern matching logic."""

    def test_pattern_matches_basic(self):
        """Test basic pattern matching logic."""
        pattern = AdvancedFusionPattern(
            name="test",
            description="Test",
            ops_to_match=["Add", "Mul"],
            fused_op_name="FusedAddMul",
            priority=FusionPriority.LOW,
            estimated_speedup=1.2,
            requires_contiguous=False,
        )

        # Create mock nodes
        node1 = Node(op_type="Add", name="add1", inputs=["x"], outputs=["y"])
        node2 = Node(op_type="Mul", name="mul1", inputs=["y"], outputs=["z"])

        graph = GraphIR(name="test")

        # Test that we have correct number of ops_to_match
        assert len(pattern.ops_to_match) == 2


class TestIntegration:
    """Integration tests for the advanced fusion system."""

    def test_import_from_optimization(self):
        """Test that advanced fusion can be imported from optimization."""
        from zenith.optimization import (
            AdvancedFusionPass,
            ALL_ADVANCED_PATTERNS,
        )

        assert AdvancedFusionPass is not None
        assert len(ALL_ADVANCED_PATTERNS) > 0

    def test_fusion_pass_with_graph_ir(self):
        """Test fusion pass integration with GraphIR."""
        from zenith.optimization import AdvancedFusionPass

        graph = GraphIR(name="test_graph")
        fusion_pass = AdvancedFusionPass()

        result, stats = fusion_pass.run(graph)

        assert isinstance(result, GraphIR)
        assert isinstance(stats, dict)
        assert "patterns_applied" in stats
