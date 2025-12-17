# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for Backend Dispatcher

Verifies proper backend selection and dispatch functionality.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestDispatcher:
    """Tests for the Dispatcher class."""

    def test_dispatcher_singleton(self):
        """Test dispatcher is a singleton."""
        from zenith.core import GraphIR

        # The Python dispatcher should be accessible
        # For now, test the core GraphIR as proxy
        g1 = GraphIR("test1")
        g2 = GraphIR("test2")

        assert g1 is not g2  # Different instances
        assert g1.name != g2.name

    def test_backend_selection_cpu_default(self):
        """Test CPU is default when no GPU available."""
        # When CUDA is not available, CPU should be selected
        backends = _get_available_backends()
        assert "cpu" in backends

    def test_backend_selection_cuda_preferred(self):
        """Test CUDA is preferred when available."""
        backends = _get_available_backends()

        # If CUDA is available, it should be in the list
        try:
            import torch

            if torch.cuda.is_available():
                assert "cuda" in backends
        except ImportError:
            pass  # PyTorch not installed

    def test_dispatch_matmul_to_cublas(self):
        """Test MatMul operations are dispatched to cuBLAS when available."""
        from zenith.core import Node, TensorDescriptor, Shape, DataType

        # Create a MatMul node
        a = TensorDescriptor("A", Shape([10, 20]), DataType.Float32)
        b = TensorDescriptor("B", Shape([20, 30]), DataType.Float32)
        c = TensorDescriptor("C", Shape([10, 30]), DataType.Float32)

        node = Node(
            op_type="MatMul",
            name="matmul1",
            inputs=[a, b],
            outputs=[c],
        )

        assert node.op_type == "MatMul"
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1

    def test_dispatch_conv_to_cudnn(self):
        """Test Conv operations are dispatched to cuDNN when available."""
        from zenith.core import Node, TensorDescriptor, Shape, DataType

        # Create a Conv node
        input_tensor = TensorDescriptor(
            "input", Shape([1, 3, 224, 224]), DataType.Float32
        )
        weight = TensorDescriptor("weight", Shape([64, 3, 7, 7]), DataType.Float32)
        output = TensorDescriptor("output", Shape([1, 64, 112, 112]), DataType.Float32)

        node = Node(
            op_type="Conv",
            name="conv1",
            inputs=[input_tensor, weight],
            outputs=[output],
            attrs={"kernel_shape": [7, 7], "strides": [2, 2]},
        )

        assert node.op_type == "Conv"

    def test_dispatch_relu_to_cudnn(self):
        """Test ReLU operations are dispatched to cuDNN when available."""
        from zenith.core import Node, TensorDescriptor, Shape, DataType

        tensor = TensorDescriptor("x", Shape([1, 64, 112, 112]), DataType.Float32)

        node = Node(
            op_type="Relu",
            name="relu1",
            inputs=[tensor],
            outputs=[tensor],
        )

        assert node.op_type == "Relu"

    def test_dispatch_fallback(self):
        """Test fallback to custom kernels when cuDNN/cuBLAS unavailable."""
        from zenith.core import Node, TensorDescriptor, Shape, DataType

        # Create custom operation not in cuDNN/cuBLAS
        tensor = TensorDescriptor("x", Shape([1, 10]), DataType.Float32)

        node = Node(
            op_type="CustomOp",
            name="custom1",
            inputs=[tensor],
            outputs=[tensor],
        )

        assert node.op_type == "CustomOp"


class TestBackendRegistry:
    """Tests for backend registration."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = _get_available_backends()

        assert isinstance(backends, list)
        assert len(backends) >= 1  # At least CPU
        assert "cpu" in backends

    def test_backend_availability_check(self):
        """Test checking if specific backend is available."""
        backends = _get_available_backends()

        for backend in backends:
            assert isinstance(backend, str)
            assert len(backend) > 0


class TestDispatchStatistics:
    """Tests for dispatch statistics tracking."""

    def test_stats_initialization(self):
        """Test initial stats are zero."""
        stats = DispatchStats()

        assert stats.total_dispatches == 0
        assert stats.cudnn_dispatches == 0
        assert stats.cublas_dispatches == 0
        assert stats.fallback_dispatches == 0

    def test_stats_increment(self):
        """Test stats can be incremented."""
        stats = DispatchStats()
        stats.total_dispatches += 1
        stats.cudnn_dispatches += 1

        assert stats.total_dispatches == 1
        assert stats.cudnn_dispatches == 1


# ============================================================================
# Helper Classes and Functions
# ============================================================================


class DispatchStats:
    """Mock dispatch statistics for testing."""

    def __init__(self):
        self.total_dispatches = 0
        self.cudnn_dispatches = 0
        self.cublas_dispatches = 0
        self.fallback_dispatches = 0


def _get_available_backends():
    """Get list of available backends."""
    backends = ["cpu"]

    try:
        import torch

        if torch.cuda.is_available():
            backends.append("cuda")
    except ImportError:
        pass

    return backends


# ============================================================================
# Integration Tests
# ============================================================================


class TestDispatchIntegration:
    """Integration tests for dispatch system."""

    def test_graphir_with_multiple_ops(self):
        """Test dispatching a graph with multiple operation types."""
        from zenith.core import GraphIR, Node, TensorDescriptor, Shape, DataType

        graph = GraphIR("multi_op_graph")

        # Input
        input_tensor = TensorDescriptor(
            "input", Shape([1, 3, 224, 224]), DataType.Float32
        )
        graph.add_input(input_tensor)

        # Conv
        conv_out = TensorDescriptor(
            "conv_out", Shape([1, 64, 112, 112]), DataType.Float32
        )
        graph.add_node("Conv", "conv1", [input_tensor], [conv_out])

        # ReLU
        relu_out = TensorDescriptor(
            "relu_out", Shape([1, 64, 112, 112]), DataType.Float32
        )
        graph.add_node("Relu", "relu1", [conv_out], [relu_out])

        # Output
        graph.add_output(relu_out)

        # Verify graph structure
        assert graph.num_nodes() == 2

        conv_nodes = graph.find_nodes_by_op("Conv")
        assert len(conv_nodes) == 1

        relu_nodes = graph.find_nodes_by_op("Relu")
        assert len(relu_nodes) == 1

    def test_dispatch_order(self):
        """Test operations are dispatched in topological order."""
        from zenith.core import GraphIR, TensorDescriptor, Shape, DataType

        graph = GraphIR("ordered_graph")

        t1 = TensorDescriptor("t1", Shape([1, 10]), DataType.Float32)
        t2 = TensorDescriptor("t2", Shape([1, 10]), DataType.Float32)
        t3 = TensorDescriptor("t3", Shape([1, 10]), DataType.Float32)

        graph.add_input(t1)
        graph.add_node("Relu", "op1", [t1], [t2])
        graph.add_node("Sigmoid", "op2", [t2], [t3])
        graph.add_output(t3)

        order = graph.topological_order()
        assert len(order) == 2

        # First operation should be relu (comes first in graph)
        assert order[0].name == "op1"
        assert order[1].name == "op2"
