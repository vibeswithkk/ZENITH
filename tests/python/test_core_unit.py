"""
Core Unit Tests

Comprehensive unit testing for all core components as specified in CetakBiru 5.3:
- GraphIR: All methods including add_node, get_node, remove_node, validate, clone
- Node: Creation, attributes, clone
- TensorDescriptor: Creation, size_bytes, is_valid
- Shape: Creation, indexing, numel, rank
- DataType: All types, conversions

Target: >95% coverage for core module (CetakBiru Section 5.3)
"""

import pytest
import numpy as np

from zenith.core import (
    GraphIR,
    Node,
    TensorDescriptor,
    Shape,
    DataType,
)
from zenith.core.types import (
    Status,
    StatusCode,
    Layout,
    dtype_size,
    dtype_to_string,
)
from zenith.optimization.mixed_precision import Precision


class TestShape:
    """Unit tests for Shape class."""

    def test_shape_empty(self):
        """Test empty shape creation."""
        shape = Shape()
        assert shape.rank() == 0

    def test_shape_from_list(self):
        """Test shape from list."""
        shape = Shape([1, 3, 224, 224])
        assert shape.rank() == 4
        assert len(shape) == 4

    def test_shape_indexing(self):
        """Test shape indexing."""
        shape = Shape([1, 3, 224, 224])
        assert shape[0] == 1
        assert shape[1] == 3
        assert shape[2] == 224
        assert shape[3] == 224

    def test_shape_negative_indexing(self):
        """Test negative indexing."""
        shape = Shape([1, 3, 224, 224])
        assert shape[-1] == 224
        assert shape[-2] == 224

    def test_shape_numel(self):
        """Test number of elements calculation."""
        shape = Shape([1, 3, 224, 224])
        expected = 1 * 3 * 224 * 224
        assert shape.numel() == expected

    def test_shape_numel_single_dim(self):
        """Test numel with single dimension."""
        shape = Shape([10])
        assert shape.numel() == 10

    def test_shape_numel_large(self):
        """Test numel with large shape."""
        shape = Shape([64, 512, 7, 7])
        expected = 64 * 512 * 7 * 7
        assert shape.numel() == expected

    def test_shape_iteration(self):
        """Test shape iteration."""
        shape = Shape([1, 3, 224, 224])
        dims = list(shape)
        assert dims == [1, 3, 224, 224]

    def test_shape_repr(self):
        """Test shape string representation."""
        shape = Shape([1, 3, 224, 224])
        repr_str = repr(shape)
        assert "1" in repr_str
        assert "224" in repr_str

    def test_shape_equality(self):
        """Test shape equality."""
        shape1 = Shape([1, 3, 224])
        shape2 = Shape([1, 3, 224])
        assert shape1.dims == shape2.dims


class TestDataType:
    """Unit tests for DataType enum."""

    def test_all_data_types_exist(self):
        """Test all data types are defined."""
        assert DataType.Float32 is not None
        assert DataType.Float16 is not None
        assert DataType.BFloat16 is not None
        assert DataType.Int8 is not None
        assert DataType.Int16 is not None
        assert DataType.Int32 is not None
        assert DataType.Int64 is not None
        assert DataType.UInt8 is not None
        assert DataType.Bool is not None

    def test_dtype_size_float32(self):
        """Test float32 size."""
        assert dtype_size(DataType.Float32) == 4

    def test_dtype_size_float16(self):
        """Test float16 size."""
        assert dtype_size(DataType.Float16) == 2

    def test_dtype_size_int8(self):
        """Test int8 size."""
        assert dtype_size(DataType.Int8) == 1

    def test_dtype_size_int64(self):
        """Test int64 size."""
        assert dtype_size(DataType.Int64) == 8

    def test_dtype_to_string(self):
        """Test dtype to string conversion."""
        assert dtype_to_string(DataType.Float32) == "float32"
        assert dtype_to_string(DataType.Float16) == "float16"
        assert dtype_to_string(DataType.Int8) == "int8"


class TestTensorDescriptor:
    """Unit tests for TensorDescriptor class."""

    def test_tensor_descriptor_creation(self):
        """Test basic tensor descriptor creation."""
        td = TensorDescriptor("input", Shape([1, 3, 224, 224]), DataType.Float32)
        assert td.name == "input"
        assert td.dtype == DataType.Float32

    def test_tensor_descriptor_size_bytes(self):
        """Test size_bytes calculation."""
        td = TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)
        expected = 1 * 3 * 224 * 224 * 4  # 4 bytes for float32
        assert td.size_bytes() == expected

    def test_tensor_descriptor_size_bytes_fp16(self):
        """Test size_bytes for FP16."""
        td = TensorDescriptor("x", Shape([1, 10]), DataType.Float16)
        expected = 10 * 2  # 2 bytes for float16
        assert td.size_bytes() == expected

    def test_tensor_descriptor_is_valid(self):
        """Test is_valid method."""
        td = TensorDescriptor("x", Shape([1, 10]), DataType.Float32)
        assert td.is_valid() is True

    def test_tensor_descriptor_is_invalid_no_name(self):
        """Test is_valid with no name."""
        td = TensorDescriptor("", Shape([1, 10]), DataType.Float32)
        assert td.is_valid() is False

    def test_tensor_descriptor_is_invalid_no_shape(self):
        """Test is_valid with empty shape."""
        td = TensorDescriptor("x", Shape(), DataType.Float32)
        assert td.is_valid() is False

    def test_tensor_descriptor_repr(self):
        """Test string representation."""
        td = TensorDescriptor("input", Shape([1, 3]), DataType.Float32)
        repr_str = repr(td)
        assert "input" in repr_str
        assert "float32" in repr_str

    def test_tensor_descriptor_layout(self):
        """Test tensor layout."""
        td = TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)
        assert td.layout == Layout.NCHW


class TestNode:
    """Unit tests for Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(
            op_type="Conv",
            name="conv1",
            inputs=[TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 64, 112, 112]), DataType.Float32)],
        )
        assert node.op_type == "Conv"
        assert node.name == "conv1"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1

    def test_node_with_attributes(self):
        """Test node with custom attributes."""
        node = Node(
            op_type="Conv",
            name="conv1",
            inputs=[],
            outputs=[],
            attrs={"kernel_size": [3, 3], "stride": [1, 1]},
        )
        assert node.attrs["kernel_size"] == [3, 3]
        assert node.attrs["stride"] == [1, 1]

    def test_node_clone(self):
        """Test node cloning."""
        node = Node(
            op_type="Relu",
            name="relu1",
            inputs=[TensorDescriptor("x", Shape([1, 10]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 10]), DataType.Float32)],
        )
        cloned = node.clone()
        assert cloned.name == node.name
        assert cloned.op_type == node.op_type
        assert cloned is not node

    def test_node_repr(self):
        """Test node string representation."""
        node = Node(op_type="Add", name="add1", inputs=[], outputs=[])
        repr_str = repr(node)
        assert "Add" in repr_str
        assert "add1" in repr_str


class TestGraphIR:
    """Unit tests for GraphIR class."""

    def test_graphir_creation(self):
        """Test basic GraphIR creation."""
        graph = GraphIR(name="test_model")
        assert graph.name == "test_model"
        assert len(graph) == 0

    def test_graphir_add_node(self):
        """Test adding a node."""
        graph = GraphIR(name="test")
        node = graph.add_node(
            op_type="Conv",
            name="conv1",
            inputs=[TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 64, 112, 112]), DataType.Float32)],
        )
        assert node is not None
        assert graph.num_nodes() == 1
        assert len(graph) == 1

    def test_graphir_get_node(self):
        """Test getting a node by name."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        node = graph.get_node("add1")
        assert node is not None
        assert node.name == "add1"

    def test_graphir_get_node_not_found(self):
        """Test getting non-existent node."""
        graph = GraphIR(name="test")
        node = graph.get_node("nonexistent")
        assert node is None

    def test_graphir_remove_node(self):
        """Test removing a node."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        result = graph.remove_node("add1")
        assert result is True
        assert graph.num_nodes() == 0

    def test_graphir_remove_node_not_found(self):
        """Test removing non-existent node."""
        graph = GraphIR(name="test")
        result = graph.remove_node("nonexistent")
        assert result is False

    def test_graphir_add_input(self):
        """Test adding graph inputs."""
        graph = GraphIR(name="test")
        td = TensorDescriptor("x", Shape([1, 10]), DataType.Float32)
        graph.add_input(td)
        assert len(graph.inputs) == 1
        assert graph.inputs[0].name == "x"

    def test_graphir_add_output(self):
        """Test adding graph outputs."""
        graph = GraphIR(name="test")
        td = TensorDescriptor("y", Shape([1, 5]), DataType.Float32)
        graph.add_output(td)
        assert len(graph.outputs) == 1
        assert graph.outputs[0].name == "y"

    def test_graphir_set_inputs_outputs(self):
        """Test setting inputs and outputs at once."""
        graph = GraphIR(name="test")
        inputs = [TensorDescriptor("x", Shape([1, 10]), DataType.Float32)]
        outputs = [TensorDescriptor("y", Shape([1, 5]), DataType.Float32)]
        graph.set_inputs(inputs)
        graph.set_outputs(outputs)
        assert len(graph.inputs) == 1
        assert len(graph.outputs) == 1

    def test_graphir_add_constant(self):
        """Test adding constants."""
        graph = GraphIR(name="test")
        data = b"test_constant_data"
        graph.add_constant("weight", data)
        assert graph.get_constant("weight") == data

    def test_graphir_get_constant_not_found(self):
        """Test getting non-existent constant."""
        graph = GraphIR(name="test")
        assert graph.get_constant("nonexistent") is None

    def test_graphir_find_nodes_by_op(self):
        """Test finding nodes by operation type."""
        graph = GraphIR(name="test")
        graph.add_node("Conv", "conv1", [], [])
        graph.add_node("Relu", "relu1", [], [])
        graph.add_node("Conv", "conv2", [], [])

        conv_nodes = graph.find_nodes_by_op("Conv")
        assert len(conv_nodes) == 2

        relu_nodes = graph.find_nodes_by_op("Relu")
        assert len(relu_nodes) == 1

    def test_graphir_topological_order(self):
        """Test topological ordering."""
        graph = GraphIR(name="test")
        graph.add_node("Conv", "conv1", [], [])
        graph.add_node("Relu", "relu1", [], [])
        order = graph.topological_order()
        assert len(order) == 2

    def test_graphir_validate_empty(self):
        """Test validation of empty graph."""
        graph = GraphIR(name="test")
        status = graph.validate()
        assert status.code != StatusCode.Ok

    def test_graphir_validate_no_inputs(self):
        """Test validation with no inputs."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        status = graph.validate()
        assert status.code == StatusCode.InvalidGraph

    def test_graphir_validate_no_outputs(self):
        """Test validation with no outputs."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        graph.add_input(TensorDescriptor("x", Shape([1]), DataType.Float32))
        status = graph.validate()
        assert status.code == StatusCode.InvalidGraph

    def test_graphir_validate_success(self):
        """Test successful validation."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        graph.add_input(TensorDescriptor("x", Shape([1]), DataType.Float32))
        graph.add_output(TensorDescriptor("y", Shape([1]), DataType.Float32))
        status = graph.validate()
        assert status.code == StatusCode.Ok

    def test_graphir_validate_duplicate_names(self):
        """Test validation with duplicate node names."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        # Manually add duplicate (bypassing normal checks)
        graph._nodes.append(Node(op_type="Add", name="add1", inputs=[], outputs=[]))
        graph.add_input(TensorDescriptor("x", Shape([1]), DataType.Float32))
        graph.add_output(TensorDescriptor("y", Shape([1]), DataType.Float32))
        status = graph.validate()
        assert status.code == StatusCode.InvalidGraph

    def test_graphir_count_ops(self):
        """Test operation counting."""
        graph = GraphIR(name="test")
        graph.add_node("Conv", "conv1", [], [])
        graph.add_node("Relu", "relu1", [], [])
        graph.add_node("Conv", "conv2", [], [])
        graph.add_node("Add", "add1", [], [])

        counts = graph.count_ops()
        assert counts["Conv"] == 2
        assert counts["Relu"] == 1
        assert counts["Add"] == 1

    def test_graphir_summary(self):
        """Test graph summary."""
        graph = GraphIR(name="test_model")
        graph.add_node("Conv", "conv1", [], [])
        graph.add_input(
            TensorDescriptor("x", Shape([1, 3, 224, 224]), DataType.Float32)
        )
        summary = graph.summary()
        assert "test_model" in summary
        assert "Conv" in summary

    def test_graphir_clone(self):
        """Test graph cloning."""
        graph = GraphIR(name="original")
        graph.add_node("Add", "add1", [], [])
        graph.add_input(TensorDescriptor("x", Shape([1]), DataType.Float32))
        graph.add_constant("w", b"data")

        cloned = graph.clone()
        assert cloned.name == graph.name
        assert cloned.num_nodes() == graph.num_nodes()
        assert len(cloned.inputs) == len(graph.inputs)
        assert cloned is not graph

    def test_graphir_repr(self):
        """Test graph string representation."""
        graph = GraphIR(name="test")
        repr_str = repr(graph)
        assert "test" in repr_str

    def test_graphir_nodes_property(self):
        """Test nodes property access."""
        graph = GraphIR(name="test")
        graph.add_node("Relu", "relu1", [], [])
        nodes = graph.nodes
        assert len(nodes) == 1
        assert nodes[0].name == "relu1"


class TestStatus:
    """Unit tests for Status class."""

    def test_status_ok(self):
        """Test OK status creation."""
        status = Status.Ok()
        assert status.code == StatusCode.Ok
        assert status.ok()

    def test_status_error(self):
        """Test error status creation."""
        status = Status.Error(StatusCode.InvalidGraph, "Graph is invalid")
        assert status.code == StatusCode.InvalidGraph
        assert not status.ok()
        assert "invalid" in status.message.lower()


class TestLayout:
    """Unit tests for Layout enum."""

    def test_layouts_exist(self):
        """Test all layouts are defined."""
        assert Layout.NCHW is not None
        assert Layout.NHWC is not None


class TestPrecision:
    """Unit tests for Precision enum."""

    def test_precisions_exist(self):
        """Test all precisions are defined."""
        assert Precision.FP32 is not None
        assert Precision.FP16 is not None
        assert Precision.BF16 is not None
        assert Precision.INT8 is not None
