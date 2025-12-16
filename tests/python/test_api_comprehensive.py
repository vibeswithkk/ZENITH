"""
Comprehensive tests for zenith.api module.

Tests the main API entry points.
"""

import pytest

import zenith
from zenith.core import GraphIR, TensorDescriptor, Shape, DataType


class TestZenithModuleInterface:
    """Test zenith module-level interface."""

    def test_zenith_version(self):
        """Test version attribute."""
        assert hasattr(zenith, "__version__")

    def test_zenith_exports(self):
        """Test all expected exports exist."""
        expected = ["compile", "optimize", "GraphIR", "DataType"]
        for name in expected:
            assert hasattr(zenith, name), f"Missing export: {name}"

    def test_zenith_adapters_accessible(self):
        """Test adapters are accessible from zenith."""
        from zenith.adapters import ONNXAdapter

        assert ONNXAdapter is not None

    def test_zenith_optimization_accessible(self):
        """Test optimization module accessible."""
        from zenith.optimization import PassManager

        assert PassManager is not None

    def test_graphir_creation(self):
        """Test GraphIR can be created via zenith."""
        graph = GraphIR(name="api_test")
        assert graph.name == "api_test"

    def test_datatype_enum(self):
        """Test DataType enum values."""
        assert DataType.Float32 is not None
        assert DataType.Float16 is not None
        assert DataType.Int8 is not None

    def test_shape_creation(self):
        """Test Shape creation."""
        shape = Shape([1, 3, 224, 224])
        assert len(shape) == 4

    def test_tensor_descriptor(self):
        """Test TensorDescriptor creation."""
        td = TensorDescriptor("test", Shape([1, 10]), DataType.Float32)
        assert td.name == "test"


class TestGraphIRFromZenith:
    """Test GraphIR functionality accessed via zenith."""

    def test_add_input(self):
        """Test adding input to graph."""
        graph = GraphIR(name="input_test")
        td = TensorDescriptor("x", Shape([1, 64]), DataType.Float32)
        graph.add_input(td)
        assert len(graph.inputs) == 1

    def test_add_output(self):
        """Test adding output to graph."""
        graph = GraphIR(name="output_test")
        td = TensorDescriptor("y", Shape([1, 10]), DataType.Float32)
        graph.add_output(td)
        assert len(graph.outputs) == 1

    def test_graph_summary(self):
        """Test graph summary."""
        graph = GraphIR(name="summary_test")
        summary = graph.summary()
        assert "summary_test" in summary

    def test_graph_num_nodes(self):
        """Test num_nodes method."""
        graph = GraphIR(name="nodes_test")
        assert graph.num_nodes() == 0


class TestAPIIntegration:
    """Integration tests for API functions."""

    def test_passmanager_creation(self):
        """Test PassManager can be created."""
        from zenith.optimization import PassManager

        pm = PassManager()
        assert pm is not None

    def test_constant_folding_pass(self):
        """Test ConstantFoldingPass can be created."""
        from zenith.optimization import ConstantFoldingPass

        cfp = ConstantFoldingPass()
        assert cfp.name == "constant_folding"

    def test_dead_code_elimination_pass(self):
        """Test DeadCodeEliminationPass can be created."""
        from zenith.optimization import DeadCodeEliminationPass

        dce = DeadCodeEliminationPass()
        assert dce.name == "dead_code_elimination"

    def test_quantizer_creation(self):
        """Test Quantizer can be created."""
        from zenith.optimization import Quantizer, QuantizationMode

        q = Quantizer(mode=QuantizationMode.STATIC)
        assert q is not None

    def test_mixed_precision_manager(self):
        """Test MixedPrecisionManager can be created."""
        from zenith.optimization import MixedPrecisionManager, PrecisionPolicy

        policy = PrecisionPolicy.fp16_with_loss_scale()
        mp = MixedPrecisionManager(policy)
        assert mp is not None
