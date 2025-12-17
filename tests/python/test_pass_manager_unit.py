"""
PassManager and Optimization Passes Unit Tests

Comprehensive unit testing for optimization system as specified in CetakBiru 5.3:
- PassManager: Pass registration, execution, statistics
- FusionPass: Pattern detection, weight folding
- LayoutPass: Layout transformations
- Individual pass operations
"""

import pytest
import numpy as np

from zenith.core import GraphIR, TensorDescriptor, Shape, DataType
from zenith.optimization import (
    PassManager,
    FusionPass,
    MixedPrecisionManager,
    PrecisionPolicy,
    Profiler,
    compute_conv_bn_weights,
)
from zenith.optimization.layout_pass import LayoutTransformPass as LayoutPass


class TestPassManager:
    """Unit tests for PassManager."""

    def test_pass_manager_creation(self):
        """Test PassManager creation."""
        pm = PassManager()
        assert pm is not None

    def test_pass_manager_add_pass(self):
        """Test adding passes with actual pass objects."""
        from zenith.optimization import ConstantFoldingPass, DeadCodeEliminationPass

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(DeadCodeEliminationPass())
        # Should not raise

    def test_pass_manager_add_multiple(self):
        """Test adding multiple passes."""
        from zenith.optimization import ConstantFoldingPass, OperatorFusionPass

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(OperatorFusionPass())
        # Should not raise

    def test_pass_manager_run_empty_graph(self):
        """Test running on empty graph."""
        from zenith.optimization import ConstantFoldingPass

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        graph = GraphIR(name="empty")
        result, stats = pm.run(graph)
        assert result is not None

    def test_pass_manager_run_with_nodes(self):
        """Test running on graph with nodes."""
        from zenith.optimization import DeadCodeEliminationPass

        pm = PassManager()
        pm.add_pass(DeadCodeEliminationPass())

        graph = GraphIR(name="test")
        graph.add_node(
            op_type="Add",
            name="add1",
            inputs=[TensorDescriptor("x", Shape([1, 10]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 10]), DataType.Float32)],
        )
        graph.add_input(TensorDescriptor("x", Shape([1, 10]), DataType.Float32))
        graph.add_output(TensorDescriptor("y", Shape([1, 10]), DataType.Float32))

        result, stats = pm.run(graph)
        assert result is not None

    def test_pass_manager_run_returns_stats(self):
        """Test that run returns statistics."""
        from zenith.optimization import ConstantFoldingPass

        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())

        graph = GraphIR(name="test")
        result, stats = pm.run(graph)
        assert isinstance(stats, dict)


class TestFusionPass:
    """Unit tests for FusionPass."""

    def test_fusion_pass_creation(self):
        """Test FusionPass creation."""
        fp = FusionPass()
        assert fp is not None

    def test_fusion_pass_apply_empty_graph(self):
        """Test applying to empty graph."""
        fp = FusionPass()
        graph = GraphIR(name="empty")
        result = fp.apply(graph)
        assert result is not None
        assert len(result.nodes) == 0

    def test_fusion_pass_apply_single_node(self):
        """Test applying to single node graph."""
        fp = FusionPass()
        graph = GraphIR(name="single")
        graph.add_node(
            op_type="Relu",
            name="relu1",
            inputs=[TensorDescriptor("x", Shape([1, 10]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 10]), DataType.Float32)],
        )
        result = fp.apply(graph)
        assert len(result.nodes) == 1

    def test_fusion_pass_get_stats(self):
        """Test getting fusion statistics."""
        fp = FusionPass()
        graph = GraphIR(name="test")
        fp.apply(graph)
        stats = fp.get_stats()
        assert isinstance(stats, dict)


class TestLayoutPass:
    """Unit tests for LayoutPass."""

    def test_layout_pass_creation(self):
        """Test LayoutPass creation."""
        lp = LayoutPass()
        assert lp is not None

    def test_layout_pass_apply_empty_graph(self):
        """Test applying to empty graph."""
        lp = LayoutPass()
        graph = GraphIR(name="empty")
        result = lp.apply(graph)
        assert result is not None

    def test_layout_pass_get_stats(self):
        """Test getting layout statistics."""
        lp = LayoutPass()
        graph = GraphIR(name="test")
        lp.apply(graph)
        stats = lp.get_stats()
        assert isinstance(stats, dict)


class TestConvBNWeightFolding:
    """Unit tests for Conv-BN weight folding."""

    def test_compute_conv_bn_weights_basic(self):
        """Test basic weight folding."""
        conv_weight = np.random.randn(64, 3, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(64).astype(np.float32)
        bn_gamma = np.ones(64, dtype=np.float32)
        bn_beta = np.zeros(64, dtype=np.float32)
        bn_mean = np.zeros(64, dtype=np.float32)
        bn_var = np.ones(64, dtype=np.float32)

        fused_w, fused_b = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var
        )

        assert fused_w.shape == conv_weight.shape
        assert fused_b.shape == conv_bias.shape

    def test_compute_conv_bn_weights_no_bias(self):
        """Test weight folding without conv bias."""
        conv_weight = np.random.randn(32, 16, 3, 3).astype(np.float32)
        bn_gamma = np.ones(32, dtype=np.float32)
        bn_beta = np.zeros(32, dtype=np.float32)
        bn_mean = np.zeros(32, dtype=np.float32)
        bn_var = np.ones(32, dtype=np.float32)

        fused_w, fused_b = compute_conv_bn_weights(
            conv_weight, None, bn_gamma, bn_beta, bn_mean, bn_var
        )

        assert fused_w is not None
        assert fused_b is not None

    def test_compute_conv_bn_weights_epsilon(self):
        """Test weight folding with custom epsilon."""
        conv_weight = np.random.randn(16, 8, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(16).astype(np.float32)
        bn_gamma = np.ones(16, dtype=np.float32)
        bn_beta = np.zeros(16, dtype=np.float32)
        bn_mean = np.zeros(16, dtype=np.float32)
        bn_var = np.ones(16, dtype=np.float32) * 0.1

        fused_w, fused_b = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var, epsilon=1e-3
        )

        assert not np.isnan(fused_w).any()
        assert not np.isnan(fused_b).any()


class TestMixedPrecisionManager:
    """Unit tests for MixedPrecisionManager."""

    def test_mixed_precision_manager_creation(self):
        """Test MixedPrecisionManager creation."""
        policy = PrecisionPolicy.fp16_with_loss_scale()
        mp = MixedPrecisionManager(policy)
        assert mp is not None

    def test_precision_policy_fp16(self):
        """Test FP16 precision policy."""
        policy = PrecisionPolicy.fp16_with_loss_scale()
        assert policy is not None
        assert policy.use_dynamic_loss_scaling

    def test_precision_policy_bf16(self):
        """Test BF16 precision policy."""
        policy = PrecisionPolicy.bf16()
        assert policy is not None
        assert not policy.use_dynamic_loss_scaling

    def test_precision_policy_fp32(self):
        """Test FP32 precision policy."""
        policy = PrecisionPolicy.fp32()
        assert policy is not None
        assert policy.loss_scale == 1.0


class TestProfiler:
    """Unit tests for Profiler."""

    def test_profiler_creation(self):
        """Test Profiler creation."""
        profiler = Profiler()
        assert profiler is not None

    def test_profiler_session(self):
        """Test profiler session."""
        profiler = Profiler()
        with profiler.session("test_session"):
            pass  # No-op

    def test_profiler_measure(self):
        """Test profiler measure context."""
        profiler = Profiler()
        with profiler.session("test"):
            with profiler.measure("op1", "Add"):
                pass  # No-op

    def test_profiler_summary(self):
        """Test getting profiler summary."""
        profiler = Profiler()
        with profiler.session("test"):
            with profiler.measure("op1", "Add"):
                pass

        summary = profiler.summary()
        assert isinstance(summary, str)


class TestOptimizationIntegration:
    """Integration tests for optimization pipeline."""

    def test_full_pipeline(self):
        """Test full optimization pipeline."""
        from zenith.optimization import ConstantFoldingPass, OperatorFusionPass

        # Create graph
        graph = GraphIR(name="integration_test")
        graph.add_node(
            op_type="Conv",
            name="conv1",
            inputs=[],
            outputs=[],
        )
        graph.add_node(
            op_type="Relu",
            name="relu1",
            inputs=[],
            outputs=[],
        )
        graph.add_input(
            TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)
        )
        graph.add_output(
            TensorDescriptor("y", Shape([1, 64, 112, 112]), DataType.Float32)
        )

        # Apply passes
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(OperatorFusionPass())

        result, stats = pm.run(graph)
        assert result is not None

    def test_graph_clone_independence(self):
        """Test that cloned graphs are independent."""
        graph = GraphIR(name="original")
        graph.add_node("Add", "add1", [], [])

        cloned = graph.clone()
        cloned.add_node("Mul", "mul1", [], [])

        assert graph.num_nodes() == 1
        assert cloned.num_nodes() == 2
