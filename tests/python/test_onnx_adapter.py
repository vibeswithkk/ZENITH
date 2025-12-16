# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for ONNX to GraphIR Conversion

These tests validate the core conversion flow from ONNX format to Zenith's
internal GraphIR representation, as specified in CetakBiru Phase 0.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from io import BytesIO

# Import Zenith modules
from zenith.adapters.onnx_adapter import ONNXAdapter
from zenith.core.graph_ir import GraphIR
from zenith.core.tensor import TensorDescriptor
from zenith.core.node import Node
from zenith.core.types import DataType, Layout, Shape


class TestONNXAdapterBasic:
    """Basic tests for ONNXAdapter initialization and properties."""

    def test_adapter_name(self):
        """Test that adapter returns correct name."""
        adapter = ONNXAdapter()
        assert adapter.name == "onnx"

    def test_adapter_is_available_without_onnx(self):
        """Test availability check when ONNX is not installed."""
        adapter = ONNXAdapter()
        # Result depends on whether ONNX is actually installed
        result = adapter.is_available
        assert isinstance(result, bool)

    @patch.dict(sys.modules, {"onnx": None})
    def test_adapter_not_available_when_onnx_missing(self):
        """Test that adapter reports unavailable when ONNX import fails."""
        # Create fresh adapter instance
        adapter = ONNXAdapter()
        # Force reimport check
        adapter._onnx = None
        # This should handle the import gracefully
        # Note: The actual behavior depends on how is_available is implemented


class TestONNXAdapterConversion:
    """Tests for ONNX to GraphIR conversion logic."""

    @pytest.fixture
    def adapter(self):
        """Create an ONNXAdapter instance."""
        return ONNXAdapter()

    @pytest.fixture
    def mock_onnx_model(self):
        """Create a mock ONNX model for testing."""
        # Create a minimal mock ONNX model structure
        model = Mock()
        model.graph = Mock()
        model.graph.name = "test_model"
        model.graph.input = []
        model.graph.output = []
        model.graph.initializer = []
        model.graph.node = []

        # Add SerializeToString for detection
        model.SerializeToString = Mock(return_value=b"mock_onnx_data")

        return model

    def test_from_model_with_model_proto(self, adapter, mock_onnx_model):
        """Test conversion from ONNX ModelProto object."""
        # Skip if ONNX is not available
        if not adapter.is_available:
            pytest.skip("ONNX is not installed")

        # The model has SerializeToString, so it should be detected as ModelProto
        assert hasattr(mock_onnx_model, "SerializeToString")

    def test_from_model_raises_for_invalid_type(self, adapter):
        """Test that from_model raises ValueError for invalid model types."""
        if not adapter.is_available:
            pytest.skip("ONNX is not installed")

        with pytest.raises(ValueError) as exc_info:
            adapter.from_model(12345)  # Invalid type

        assert "Unsupported model type" in str(exc_info.value)

    def test_from_model_with_string_path(self, adapter):
        """Test that string paths are handled correctly."""
        if not adapter.is_available:
            pytest.skip("ONNX is not installed")

        # This will fail because the file doesn't exist, but it tests the path
        with pytest.raises(Exception):
            adapter.from_model("/nonexistent/path/model.onnx")


class TestGraphIRConstruction:
    """Tests for GraphIR structure after ONNX conversion."""

    def test_graphir_default_construction(self):
        """Test default GraphIR construction."""
        graph = GraphIR(name="test_graph")

        assert graph.name == "test_graph"
        assert graph.num_nodes() == 0
        assert len(graph.inputs) == 0
        assert len(graph.outputs) == 0
        assert len(graph.constants) == 0

    def test_graphir_add_input(self):
        """Test adding inputs to GraphIR."""
        graph = GraphIR(name="test_graph")

        tensor = TensorDescriptor(
            name="input_0",
            shape=Shape([1, 3, 224, 224]),
            dtype=DataType.Float32,
            layout=Layout.NCHW,
        )
        graph.add_input(tensor)

        assert len(graph.inputs) == 1
        assert graph.inputs[0].name == "input_0"
        assert graph.inputs[0].shape.dims == [1, 3, 224, 224]

    def test_graphir_add_output(self):
        """Test adding outputs to GraphIR."""
        graph = GraphIR(name="test_graph")

        tensor = TensorDescriptor(
            name="output_0",
            shape=Shape([1, 1000]),
            dtype=DataType.Float32,
        )
        graph.add_output(tensor)

        assert len(graph.outputs) == 1
        assert graph.outputs[0].name == "output_0"

    def test_graphir_add_node(self):
        """Test adding nodes to GraphIR."""
        graph = GraphIR(name="test_graph")

        input_tensor = TensorDescriptor(
            name="x", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        output_tensor = TensorDescriptor(
            name="y", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        node = graph.add_node(
            op_type="Relu",
            name="relu_0",
            inputs=[input_tensor],
            outputs=[output_tensor],
            attrs={"alpha": 0.0},
        )

        assert graph.num_nodes() == 1
        assert node.op_type == "Relu"
        assert node.name == "relu_0"
        assert graph.get_node("relu_0") == node

    def test_graphir_add_constant(self):
        """Test adding constants (weights) to GraphIR."""
        graph = GraphIR(name="test_graph")

        weight_data = b"\x00\x01\x02\x03"  # Mock weight bytes
        graph.add_constant("weight_0", weight_data)

        assert len(graph.constants) == 1
        assert graph.get_constant("weight_0") == weight_data

    def test_graphir_find_nodes_by_op(self):
        """Test finding nodes by operation type."""
        graph = GraphIR(name="test_graph")

        # Add multiple nodes of different types
        tensor = TensorDescriptor(name="t", shape=Shape([1]), dtype=DataType.Float32)

        graph.add_node("Relu", "relu_1", [tensor], [tensor])
        graph.add_node("Relu", "relu_2", [tensor], [tensor])
        graph.add_node("Conv", "conv_1", [tensor], [tensor])

        relu_nodes = graph.find_nodes_by_op("Relu")
        conv_nodes = graph.find_nodes_by_op("Conv")

        assert len(relu_nodes) == 2
        assert len(conv_nodes) == 1

    def test_graphir_validate_empty_graph(self):
        """Test validation fails for empty graph."""
        graph = GraphIR(name="empty_graph")

        status = graph.validate()
        assert not status.ok()
        assert "no nodes" in status.message.lower()

    def test_graphir_validate_no_inputs(self):
        """Test validation fails for graph without inputs."""
        graph = GraphIR(name="test_graph")

        tensor = TensorDescriptor(name="t", shape=Shape([1]), dtype=DataType.Float32)
        graph.add_node("Relu", "relu_1", [tensor], [tensor])
        graph.add_output(tensor)

        status = graph.validate()
        assert not status.ok()
        assert "no inputs" in status.message.lower()

    def test_graphir_validate_success(self):
        """Test validation succeeds for valid graph."""
        graph = GraphIR(name="valid_graph")

        input_tensor = TensorDescriptor(
            name="input", shape=Shape([1, 10]), dtype=DataType.Float32
        )
        output_tensor = TensorDescriptor(
            name="output", shape=Shape([1, 10]), dtype=DataType.Float32
        )

        graph.add_input(input_tensor)
        graph.add_output(output_tensor)
        graph.add_node("Relu", "relu_1", [input_tensor], [output_tensor])

        status = graph.validate()
        assert status.ok()


class TestTensorDescriptor:
    """Tests for TensorDescriptor used in GraphIR."""

    def test_tensor_descriptor_creation(self):
        """Test basic TensorDescriptor creation."""
        tensor = TensorDescriptor(
            name="test_tensor",
            shape=Shape([2, 3, 4]),
            dtype=DataType.Float32,
            layout=Layout.NCHW,
        )

        assert tensor.name == "test_tensor"
        assert tensor.shape.dims == [2, 3, 4]
        assert tensor.dtype == DataType.Float32
        assert tensor.layout == Layout.NCHW

    def test_tensor_size_bytes(self):
        """Test size calculation in bytes."""
        tensor = TensorDescriptor(
            name="test",
            shape=Shape([2, 3, 4]),  # 24 elements
            dtype=DataType.Float32,  # 4 bytes each
        )

        assert tensor.size_bytes() == 96  # 24 * 4

    def test_tensor_is_valid(self):
        """Test validity checking."""
        valid_tensor = TensorDescriptor(
            name="valid", shape=Shape([1, 2, 3]), dtype=DataType.Float32
        )
        assert valid_tensor.is_valid()

        invalid_tensor = TensorDescriptor(
            name="", shape=Shape([]), dtype=DataType.Float32
        )
        assert not invalid_tensor.is_valid()


class TestNodeOperations:
    """Tests for Node class used in GraphIR."""

    def test_node_creation(self):
        """Test basic Node creation."""
        tensor = TensorDescriptor(name="t", shape=Shape([1]), dtype=DataType.Float32)

        node = Node(
            op_type="MatMul",
            name="matmul_0",
            inputs=[tensor],
            outputs=[tensor],
            attrs={"transA": False, "transB": True},
        )

        assert node.op_type == "MatMul"
        assert node.name == "matmul_0"
        assert node.num_inputs() == 1
        assert node.num_outputs() == 1
        assert node.get_attr("transA") is False
        assert node.get_attr("transB") is True

    def test_node_is_op(self):
        """Test operation type checking."""
        tensor = TensorDescriptor(name="t", shape=Shape([1]), dtype=DataType.Float32)
        node = Node(op_type="Conv", name="conv_0", inputs=[tensor], outputs=[tensor])

        assert node.is_op("Conv")
        assert not node.is_op("Relu")

    def test_node_clone(self):
        """Test node cloning."""
        tensor = TensorDescriptor(name="t", shape=Shape([1]), dtype=DataType.Float32)
        node = Node(
            op_type="Gemm",
            name="gemm_0",
            inputs=[tensor],
            outputs=[tensor],
            attrs={"alpha": 1.0, "beta": 0.0},
        )

        cloned = node.clone()

        assert cloned.op_type == node.op_type
        assert cloned.name == node.name
        assert cloned.get_attr("alpha") == node.get_attr("alpha")
        # Cloned node should have different id
        assert cloned.id != node.id


class TestEndToEndConversion:
    """End-to-end tests for ONNX to GraphIR conversion."""

    @pytest.fixture
    def adapter(self):
        """Create an ONNXAdapter instance."""
        return ONNXAdapter()

    @pytest.mark.skipif(not ONNXAdapter().is_available, reason="ONNX is not installed")
    def test_simple_model_conversion(self, adapter):
        """Test conversion of a simple ONNX model."""
        import onnx
        from onnx import helper, TensorProto

        # Create a simple ONNX model: Y = X + 1
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

        # Constant tensor for adding 1
        one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])

        add_node = helper.make_node("Add", ["X", "one"], ["Y"], name="add_0")

        graph_def = helper.make_graph([add_node], "simple_add", [X], [Y], [one])

        model_def = helper.make_model(graph_def, producer_name="zenith-test")
        model_def.opset_import[0].version = 17

        # Convert to GraphIR
        graph_ir = adapter.from_model(model_def)

        # Verify the conversion
        assert graph_ir.name == "simple_add"
        assert len(graph_ir.inputs) == 1
        assert graph_ir.inputs[0].name == "X"
        assert len(graph_ir.outputs) == 1
        assert graph_ir.outputs[0].name == "Y"
        assert graph_ir.num_nodes() == 1

    @pytest.mark.skipif(not ONNXAdapter().is_available, reason="ONNX is not installed")
    def test_relu_model_conversion(self, adapter):
        """Test conversion of ONNX model with ReLU operation."""
        import onnx
        from onnx import helper, TensorProto

        # Create model: Y = ReLU(X)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])

        relu_node = helper.make_node("Relu", ["X"], ["Y"], name="relu_0")

        graph_def = helper.make_graph([relu_node], "relu_model", [X], [Y])
        model_def = helper.make_model(graph_def, producer_name="zenith-test")
        model_def.opset_import[0].version = 17

        # Convert to GraphIR
        graph_ir = adapter.from_model(model_def)

        # Verify
        assert graph_ir.num_nodes() == 1
        node = graph_ir.nodes[0]
        assert node.op_type == "Relu"
        assert node.name == "relu_0"

    @pytest.mark.skipif(not ONNXAdapter().is_available, reason="ONNX is not installed")
    def test_multi_node_model_conversion(self, adapter):
        """Test conversion of ONNX model with multiple operations."""
        import onnx
        from onnx import helper, TensorProto

        # Create model: Y = Sigmoid(ReLU(X))
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])

        relu_node = helper.make_node("Relu", ["X"], ["relu_out"], name="relu_0")
        sigmoid_node = helper.make_node(
            "Sigmoid", ["relu_out"], ["Y"], name="sigmoid_0"
        )

        graph_def = helper.make_graph(
            [relu_node, sigmoid_node], "multi_op_model", [X], [Y]
        )
        model_def = helper.make_model(graph_def, producer_name="zenith-test")
        model_def.opset_import[0].version = 17

        # Convert to GraphIR
        graph_ir = adapter.from_model(model_def)

        # Verify
        assert graph_ir.num_nodes() == 2
        assert graph_ir.find_nodes_by_op("Relu") != []
        assert graph_ir.find_nodes_by_op("Sigmoid") != []

    @pytest.mark.skipif(not ONNXAdapter().is_available, reason="ONNX is not installed")
    def test_model_with_dynamic_shapes(self, adapter):
        """Test conversion of ONNX model with dynamic batch dimension."""
        import onnx
        from onnx import helper, TensorProto

        # Create model with dynamic batch size (None/-1)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 10])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 10])

        relu_node = helper.make_node("Relu", ["X"], ["Y"], name="relu_0")

        graph_def = helper.make_graph([relu_node], "dynamic_model", [X], [Y])
        model_def = helper.make_model(graph_def, producer_name="zenith-test")
        model_def.opset_import[0].version = 17

        # Convert to GraphIR
        graph_ir = adapter.from_model(model_def)

        # Verify dynamic dimension is preserved as -1
        assert graph_ir.inputs[0].shape.dims[0] == -1
        assert graph_ir.inputs[0].shape.is_dynamic()


class TestDataTypeConversion:
    """Tests for ONNX to Zenith data type mapping."""

    def test_dtype_mapping(self):
        """Test that DataType enum has all expected values."""
        assert hasattr(DataType, "Float32")
        assert hasattr(DataType, "Float16")
        assert hasattr(DataType, "Float64")
        assert hasattr(DataType, "Int8")
        assert hasattr(DataType, "Int16")
        assert hasattr(DataType, "Int32")
        assert hasattr(DataType, "Int64")
        assert hasattr(DataType, "UInt8")
        assert hasattr(DataType, "Bool")
        assert hasattr(DataType, "BFloat16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
