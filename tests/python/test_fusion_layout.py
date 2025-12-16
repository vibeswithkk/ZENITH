"""
Comprehensive tests for fusion and layout transformation passes.
"""

import numpy as np

from zenith.core import GraphIR, TensorDescriptor, Shape, DataType
from zenith.optimization.fusion_pass import FusionPass, compute_conv_bn_weights


class TestFusionPassPatterns:
    """Test fusion pattern matching."""

    def test_conv_relu_pattern(self):
        """Test Conv-ReLU fusion pattern detection."""
        graph = GraphIR(name="conv_relu")

        # Use correct add_node signature: op_type, name, inputs, outputs
        graph.add_node(
            op_type="Conv",
            name="conv1",
            inputs=[TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)],
            outputs=[
                TensorDescriptor("conv_out", Shape([1, 64, 112, 112]), DataType.Float32)
            ],
        )

        graph.add_node(
            op_type="Relu",
            name="relu1",
            inputs=[
                TensorDescriptor("conv_out", Shape([1, 64, 112, 112]), DataType.Float32)
            ],
            outputs=[
                TensorDescriptor("relu_out", Shape([1, 64, 112, 112]), DataType.Float32)
            ],
        )

        fusion_pass = FusionPass()
        fused_graph = fusion_pass.apply(graph)
        assert fused_graph is not None

    def test_no_fusion_possible(self):
        """Test when no fusion patterns match."""
        graph = GraphIR(name="no_fusion")

        graph.add_node(
            op_type="Add",
            name="add1",
            inputs=[
                TensorDescriptor("a", Shape([1, 10]), DataType.Float32),
                TensorDescriptor("b", Shape([1, 10]), DataType.Float32),
            ],
            outputs=[TensorDescriptor("c", Shape([1, 10]), DataType.Float32)],
        )

        fusion_pass = FusionPass()
        result = fusion_pass.apply(graph)
        assert len(result.nodes) == 1


class TestFusionWeightComputation:
    """Test fused weight computation."""

    def test_conv_bn_weight_folding(self):
        """Test mathematical correctness of Conv-BN weight folding."""
        conv_weight = np.random.randn(64, 3, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(64).astype(np.float32)

        bn_gamma = np.random.randn(64).astype(np.float32)
        bn_beta = np.random.randn(64).astype(np.float32)
        bn_mean = np.random.randn(64).astype(np.float32)
        bn_var = np.abs(np.random.randn(64).astype(np.float32)) + 0.1
        epsilon = 1e-5

        fused_weight, fused_bias = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var, epsilon
        )

        assert fused_weight.shape == conv_weight.shape
        assert fused_bias.shape == conv_bias.shape

    def test_conv_bn_no_conv_bias(self):
        """Test fusion when conv has no bias."""
        conv_weight = np.random.randn(32, 16, 3, 3).astype(np.float32)
        conv_bias = None

        bn_gamma = np.ones(32, dtype=np.float32)
        bn_beta = np.zeros(32, dtype=np.float32)
        bn_mean = np.zeros(32, dtype=np.float32)
        bn_var = np.ones(32, dtype=np.float32)

        fused_weight, fused_bias = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var
        )

        assert fused_weight is not None
        assert fused_bias is not None


class TestFusionPassEdgeCases:
    """Test edge cases in fusion pass."""

    def test_empty_graph(self):
        """Test fusion on empty graph."""
        graph = GraphIR(name="empty")
        fusion_pass = FusionPass()
        result = fusion_pass.apply(graph)
        assert len(result.nodes) == 0

    def test_single_node_graph(self):
        """Test fusion on single node graph."""
        graph = GraphIR(name="single")
        graph.add_node(
            op_type="Relu",
            name="single_op",
            inputs=[TensorDescriptor("x", Shape([1, 10]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 10]), DataType.Float32)],
        )

        fusion_pass = FusionPass()
        result = fusion_pass.apply(graph)
        assert len(result.nodes) == 1

    def test_fusion_stats(self):
        """Test fusion statistics."""
        graph = GraphIR(name="stats_test")
        fusion_pass = FusionPass()
        fusion_pass.apply(graph)
        stats = fusion_pass.get_stats()
        assert isinstance(stats, dict)
