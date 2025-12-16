# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for Zenith Optimization Passes

Tests the PoC implementations of:
- Constant Folding
- Dead Code Elimination
- Operator Fusion
- Pass Manager
"""

import pytest
import copy

from zenith.optimization import (
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    PassManager,
    create_default_pass_manager,
    optimize_graph,
)
from zenith.core.graph_ir import GraphIR
from zenith.core.tensor import TensorDescriptor
from zenith.core.node import Node
from zenith.core.types import DataType, Shape


class TestConstantFoldingPass:
    """Tests for Constant Folding optimization pass."""

    @pytest.fixture
    def folding_pass(self):
        """Create a ConstantFoldingPass instance."""
        return ConstantFoldingPass()

    def test_pass_name(self, folding_pass):
        """Test pass has correct name."""
        assert folding_pass.name == "constant_folding"

    def test_no_modification_without_constants(self, folding_pass):
        """Test that graph without foldable ops is unchanged."""
        graph = GraphIR(name="test")
        tensor = TensorDescriptor(
            name="x", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(tensor)
        graph.add_output(tensor)
        graph.add_node("Relu", "relu_0", [tensor], [tensor])

        new_graph, modified = folding_pass.run(graph)

        assert not modified
        assert new_graph.num_nodes() == 1

    def test_fold_constant_operation(self, folding_pass):
        """Test folding of operation with constant inputs."""
        graph = GraphIR(name="test")

        # Add constants
        graph.add_constant("const_a", b"\x01\x02\x03\x04")
        graph.add_constant("const_b", b"\x05\x06\x07\x08")

        tensor_a = TensorDescriptor(
            name="const_a", shape=Shape([1]), dtype=DataType.Float32
        )
        tensor_b = TensorDescriptor(
            name="const_b", shape=Shape([1]), dtype=DataType.Float32
        )
        tensor_out = TensorDescriptor(
            name="result", shape=Shape([1]), dtype=DataType.Float32
        )

        # Add node: result = Add(const_a, const_b)
        graph.add_input(tensor_a)
        graph.add_output(tensor_out)
        graph.add_node("Add", "add_0", [tensor_a, tensor_b], [tensor_out])

        new_graph, modified = folding_pass.run(graph)

        assert modified
        # The Add node should be removed (folded)
        assert new_graph.num_nodes() == 0
        # Result should be added as constant
        assert "result" in new_graph.constants

    def test_non_foldable_op_unchanged(self, folding_pass):
        """Test that non-foldable operations are preserved."""
        graph = GraphIR(name="test")

        # Add constant
        graph.add_constant("weight", b"\x00\x00\x00\x00")

        input_tensor = TensorDescriptor(
            name="input", shape=Shape([1, 3, 224, 224]), dtype=DataType.Float32
        )
        weight_tensor = TensorDescriptor(
            name="weight", shape=Shape([64, 3, 7, 7]), dtype=DataType.Float32
        )
        output_tensor = TensorDescriptor(
            name="output", shape=Shape([1, 64, 112, 112]), dtype=DataType.Float32
        )

        graph.add_input(input_tensor)
        graph.add_output(output_tensor)
        # Conv is foldable but input is not constant
        graph.add_node("Conv", "conv_0", [input_tensor, weight_tensor], [output_tensor])

        new_graph, modified = folding_pass.run(graph)

        # Conv should not be folded because input is not constant
        assert not modified
        assert new_graph.num_nodes() == 1


class TestDeadCodeEliminationPass:
    """Tests for Dead Code Elimination optimization pass."""

    @pytest.fixture
    def dce_pass(self):
        """Create a DeadCodeEliminationPass instance."""
        return DeadCodeEliminationPass()

    def test_pass_name(self, dce_pass):
        """Test pass has correct name."""
        assert dce_pass.name == "dead_code_elimination"

    def test_no_dead_code(self, dce_pass):
        """Test graph with no dead code is unchanged."""
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        output_t = TensorDescriptor(
            name="output", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(output_t)
        graph.add_node("Relu", "relu_0", [input_t], [output_t])

        new_graph, modified = dce_pass.run(graph)

        assert not modified
        assert new_graph.num_nodes() == 1

    def test_remove_dead_node(self, dce_pass):
        """Test removal of unused node."""
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        live_out = TensorDescriptor(
            name="live_out", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        dead_out = TensorDescriptor(
            name="dead_out", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(live_out)  # Only live_out is used

        # This node is live (its output is used)
        graph.add_node("Relu", "relu_live", [input_t], [live_out])
        # This node is dead (its output is never used)
        graph.add_node("Sigmoid", "sigmoid_dead", [input_t], [dead_out])

        new_graph, modified = dce_pass.run(graph)

        assert modified
        assert new_graph.num_nodes() == 1
        assert new_graph.get_node("relu_live") is not None
        assert new_graph.get_node("sigmoid_dead") is None

    def test_chain_dead_code_removal(self, dce_pass):
        """Test removal of chain of unused nodes."""
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        live_out = TensorDescriptor(
            name="live_out", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        dead_mid = TensorDescriptor(
            name="dead_mid", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        dead_out = TensorDescriptor(
            name="dead_out", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(live_out)

        # Live node
        graph.add_node("Relu", "relu_live", [input_t], [live_out])
        # Chain of dead nodes: input -> dead_mid -> dead_out
        graph.add_node("Sigmoid", "sigmoid_dead_1", [input_t], [dead_mid])
        graph.add_node("Tanh", "tanh_dead_2", [dead_mid], [dead_out])

        new_graph, modified = dce_pass.run(graph)

        assert modified
        assert new_graph.num_nodes() == 1
        assert new_graph.get_node("relu_live") is not None


class TestOperatorFusionPass:
    """Tests for Operator Fusion optimization pass."""

    @pytest.fixture
    def fusion_pass(self):
        """Create an OperatorFusionPass instance."""
        return OperatorFusionPass()

    def test_pass_name(self, fusion_pass):
        """Test pass has correct name."""
        assert fusion_pass.name == "operator_fusion"

    def test_no_fusion_possible(self, fusion_pass):
        """Test graph with no fusable patterns is unchanged."""
        graph = GraphIR(name="test")

        tensor = TensorDescriptor(
            name="x", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(tensor)
        graph.add_output(tensor)
        graph.add_node("Softmax", "softmax_0", [tensor], [tensor])

        new_graph, modified = fusion_pass.run(graph)

        assert not modified
        assert new_graph.num_nodes() == 1

    def test_conv_relu_fusion(self, fusion_pass):
        """Test fusion of Conv + Relu pattern."""
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 3, 224, 224]), dtype=DataType.Float32
        )
        conv_out = TensorDescriptor(
            name="conv_out", shape=Shape([1, 64, 112, 112]), dtype=DataType.Float32
        )
        relu_out = TensorDescriptor(
            name="relu_out", shape=Shape([1, 64, 112, 112]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(relu_out)

        graph.add_node("Conv", "conv_0", [input_t], [conv_out])
        graph.add_node("Relu", "relu_0", [conv_out], [relu_out])

        new_graph, modified = fusion_pass.run(graph)

        assert modified
        assert new_graph.num_nodes() == 1

        # Check the fused node
        fused_node = new_graph.nodes[0]
        assert fused_node.op_type == "ConvRelu"
        assert "fused_from" in fused_node.attrs
        assert fused_node.attrs["pattern"] == "conv_relu"

    def test_no_fusion_with_multiple_consumers(self, fusion_pass):
        """Test that fusion is prevented when intermediate has multiple consumers."""
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )
        conv_out = TensorDescriptor(
            name="conv_out", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )
        relu_out = TensorDescriptor(
            name="relu_out", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )
        other_out = TensorDescriptor(
            name="other_out", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(relu_out)
        graph.add_output(other_out)

        # Conv output is used by both Relu and Sigmoid
        graph.add_node("Conv", "conv_0", [input_t], [conv_out])
        graph.add_node("Relu", "relu_0", [conv_out], [relu_out])
        graph.add_node("Sigmoid", "sigmoid_0", [conv_out], [other_out])

        new_graph, modified = fusion_pass.run(graph)

        # Should not fuse because conv_out has multiple consumers
        assert not modified
        assert new_graph.num_nodes() == 3


class TestPassManager:
    """Tests for PassManager orchestration."""

    def test_add_pass(self):
        """Test adding passes to manager."""
        manager = PassManager()
        manager.add_pass(ConstantFoldingPass())
        manager.add_pass(DeadCodeEliminationPass())

        assert len(manager._passes) == 2

    def test_run_all_passes(self):
        """Test running all passes on a graph."""
        manager = PassManager()
        manager.add_pass(DeadCodeEliminationPass())
        manager.add_pass(OperatorFusionPass())

        # Create graph with dead code and fusable ops
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )
        conv_out = TensorDescriptor(
            name="conv_out", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )
        relu_out = TensorDescriptor(
            name="relu_out", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )
        dead_out = TensorDescriptor(
            name="dead_out", shape=Shape([1, 64, 28, 28]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(relu_out)

        # Fusable: Conv + Relu
        graph.add_node("Conv", "conv_0", [input_t], [conv_out])
        graph.add_node("Relu", "relu_0", [conv_out], [relu_out])
        # Dead node
        graph.add_node("Sigmoid", "dead_sigmoid", [input_t], [dead_out])

        optimized, stats = manager.run(graph)

        # Dead code should be eliminated
        assert optimized.get_node("dead_sigmoid") is None
        # Conv+Relu should be fused
        assert optimized.num_nodes() == 1
        # Stats should be recorded
        assert stats["dead_code_elimination"] >= 1
        assert stats["operator_fusion"] >= 1


class TestDefaultPassManager:
    """Tests for default pass manager and optimize_graph function."""

    def test_create_default_manager(self):
        """Test creating default pass manager."""
        manager = create_default_pass_manager()
        assert len(manager._passes) == 3

    def test_optimize_graph_level_0(self):
        """Test optimize_graph with level 0 (no optimization)."""
        graph = GraphIR(name="test")
        tensor = TensorDescriptor(
            name="x", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        graph.add_input(tensor)
        graph.add_output(tensor)
        graph.add_node("Relu", "relu", [tensor], [tensor])

        optimized, stats = optimize_graph(graph, opt_level=0)

        assert optimized.num_nodes() == 1
        assert stats == {}

    def test_optimize_graph_level_2(self):
        """Test optimize_graph with level 2 (standard)."""
        graph = GraphIR(name="test")

        input_t = TensorDescriptor(
            name="input", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        live_out = TensorDescriptor(
            name="live", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        dead_out = TensorDescriptor(
            name="dead", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(input_t)
        graph.add_output(live_out)

        graph.add_node("Relu", "relu_live", [input_t], [live_out])
        graph.add_node("Sigmoid", "sigmoid_dead", [input_t], [dead_out])

        optimized, stats = optimize_graph(graph, opt_level=2)

        # Level 2 includes DCE
        assert optimized.num_nodes() == 1
        assert "dead_code_elimination" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
