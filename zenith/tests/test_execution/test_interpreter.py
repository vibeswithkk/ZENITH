# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Integration tests for the ONNX Interpreter.

Tests full graph execution with realistic model structures:
- Simple MLP
- CNN block (Conv-BN-Relu)
"""

import pytest
import numpy as np

from zenith.core import GraphIR, Node, TensorDescriptor, Shape, DataType
from zenith.execution import ONNXInterpreter


class TestSimpleMLP:
    """Test interpreter with a simple MLP graph."""

    def test_mlp_matmul_add_relu(self):
        """Test MLP: MatMul -> Add -> Relu."""
        # Create GraphIR manually
        graph = GraphIR(name="simple_mlp")

        # Input: [batch, in_features]
        graph.add_input(
            TensorDescriptor(name="input", shape=Shape([2, 8]), dtype=DataType.Float32)
        )

        # Output: [batch, out_features]
        graph.add_output(
            TensorDescriptor(name="output", shape=Shape([2, 4]), dtype=DataType.Float32)
        )

        # Add weight constant - pass numpy arrays directly for shape preservation
        weight = np.random.randn(8, 4).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)
        graph.add_constant("weight", weight)
        graph.add_constant("bias", bias)

        # Add nodes
        # Node 1: MatMul
        node1 = Node(
            op_type="MatMul",
            name="matmul0",
            inputs=[
                TensorDescriptor(name="input", shape=Shape([2, 8])),
                TensorDescriptor(name="weight", shape=Shape([8, 4])),
            ],
            outputs=[
                TensorDescriptor(name="matmul_out", shape=Shape([2, 4])),
            ],
        )
        graph._nodes.append(node1)
        graph._name_to_node[node1.name] = node1

        # Node 2: Add bias
        node2 = Node(
            op_type="Add",
            name="add0",
            inputs=[
                TensorDescriptor(name="matmul_out", shape=Shape([2, 4])),
                TensorDescriptor(name="bias", shape=Shape([4])),
            ],
            outputs=[
                TensorDescriptor(name="add_out", shape=Shape([2, 4])),
            ],
        )
        graph._nodes.append(node2)
        graph._name_to_node[node2.name] = node2

        # Node 3: Relu
        node3 = Node(
            op_type="Relu",
            name="relu0",
            inputs=[
                TensorDescriptor(name="add_out", shape=Shape([2, 4])),
            ],
            outputs=[
                TensorDescriptor(name="output", shape=Shape([2, 4])),
            ],
        )
        graph._nodes.append(node3)
        graph._name_to_node[node3.name] = node3

        # Create interpreter
        interpreter = ONNXInterpreter(graph, device="cpu")

        # Verify interpreter state
        assert interpreter.is_fully_supported
        assert len(interpreter.unsupported_operators) == 0
        assert len(interpreter._execution_order) == 3

        # Execute
        input_data = np.random.randn(2, 8).astype(np.float32)
        outputs = interpreter(input=input_data)

        # Verify we got outputs
        assert "output" in outputs
        result = outputs["output"]

        # Compute expected
        expected = np.maximum(0, np.matmul(input_data, weight) + bias)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mlp_two_layers(self):
        """Test 2-layer MLP: Gemm -> Relu -> Gemm -> Softmax."""
        graph = GraphIR(name="two_layer_mlp")

        # Shapes
        batch = 4
        in_features = 16
        hidden = 8
        out_features = 4

        # Input/Output
        graph.add_input(
            TensorDescriptor(
                name="input",
                shape=Shape([batch, in_features]),
                dtype=DataType.Float32,
            )
        )
        graph.add_output(
            TensorDescriptor(
                name="output",
                shape=Shape([batch, out_features]),
                dtype=DataType.Float32,
            )
        )

        # Weights
        W1 = np.random.randn(in_features, hidden).astype(np.float32) * 0.1
        B1 = np.random.randn(hidden).astype(np.float32) * 0.1
        W2 = np.random.randn(hidden, out_features).astype(np.float32) * 0.1
        B2 = np.random.randn(out_features).astype(np.float32) * 0.1

        graph.add_constant("W1", W1)
        graph.add_constant("B1", B1)
        graph.add_constant("W2", W2)
        graph.add_constant("B2", B2)

        # Build nodes
        nodes = [
            Node(
                op_type="Gemm",
                name="gemm1",
                inputs=[
                    TensorDescriptor(name="input", shape=Shape([])),
                    TensorDescriptor(name="W1", shape=Shape([])),
                    TensorDescriptor(name="B1", shape=Shape([])),
                ],
                outputs=[TensorDescriptor(name="h1", shape=Shape([]))],
                attrs={"alpha": 1.0, "beta": 1.0},
            ),
            Node(
                op_type="Relu",
                name="relu1",
                inputs=[TensorDescriptor(name="h1", shape=Shape([]))],
                outputs=[TensorDescriptor(name="h1_relu", shape=Shape([]))],
            ),
            Node(
                op_type="Gemm",
                name="gemm2",
                inputs=[
                    TensorDescriptor(name="h1_relu", shape=Shape([])),
                    TensorDescriptor(name="W2", shape=Shape([])),
                    TensorDescriptor(name="B2", shape=Shape([])),
                ],
                outputs=[TensorDescriptor(name="h2", shape=Shape([]))],
                attrs={"alpha": 1.0, "beta": 1.0},
            ),
            Node(
                op_type="Softmax",
                name="softmax",
                inputs=[TensorDescriptor(name="h2", shape=Shape([]))],
                outputs=[TensorDescriptor(name="output", shape=Shape([]))],
                attrs={"axis": -1},
            ),
        ]

        for node in nodes:
            graph._nodes.append(node)
            graph._name_to_node[node.name] = node

        # Create interpreter and execute
        interpreter = ONNXInterpreter(graph, device="cpu")
        assert interpreter.is_fully_supported

        input_data = np.random.randn(batch, in_features).astype(np.float32)
        outputs = interpreter(input=input_data)

        result = outputs["output"]

        # Check softmax properties
        assert result.shape == (batch, out_features)
        np.testing.assert_allclose(np.sum(result, axis=-1), np.ones(batch), rtol=1e-5)
        assert np.all(result >= 0)


class TestInterpreterFeatures:
    """Test interpreter features and edge cases."""

    def test_summary(self):
        """Test interpreter summary output."""
        graph = GraphIR(name="test_graph")
        graph.add_input(
            TensorDescriptor(name="input", shape=Shape([1, 4]), dtype=DataType.Float32)
        )
        graph.add_output(
            TensorDescriptor(name="output", shape=Shape([1, 4]), dtype=DataType.Float32)
        )

        node = Node(
            op_type="Identity",
            name="id",
            inputs=[TensorDescriptor(name="input", shape=Shape([]))],
            outputs=[TensorDescriptor(name="output", shape=Shape([]))],
        )
        graph._nodes.append(node)
        graph._name_to_node[node.name] = node

        interpreter = ONNXInterpreter(graph, device="cpu")

        summary = interpreter.summary()
        assert "test_graph" in summary
        assert "Device: cpu" in summary

    def test_unsupported_op_detection(self):
        """Test that unsupported operators are detected."""
        graph = GraphIR(name="unsupported_test")
        graph.add_input(
            TensorDescriptor(name="input", shape=Shape([1, 4]), dtype=DataType.Float32)
        )
        graph.add_output(
            TensorDescriptor(name="output", shape=Shape([1, 4]), dtype=DataType.Float32)
        )

        # Add unsupported op
        node = Node(
            op_type="FakeCustomOp",
            name="fake",
            inputs=[TensorDescriptor(name="input", shape=Shape([]))],
            outputs=[TensorDescriptor(name="output", shape=Shape([]))],
        )
        graph._nodes.append(node)
        graph._name_to_node[node.name] = node

        interpreter = ONNXInterpreter(graph, device="cpu", strict=False)

        assert not interpreter.is_fully_supported
        assert "FakeCustomOp" in interpreter.unsupported_operators

    def test_strict_mode_raises(self):
        """Test that strict mode raises for unsupported ops."""
        graph = GraphIR(name="strict_test")
        graph.add_input(
            TensorDescriptor(name="input", shape=Shape([1, 4]), dtype=DataType.Float32)
        )
        graph.add_output(
            TensorDescriptor(name="output", shape=Shape([1, 4]), dtype=DataType.Float32)
        )

        node = Node(
            op_type="NonExistentOp",
            name="bad",
            inputs=[TensorDescriptor(name="input", shape=Shape([]))],
            outputs=[TensorDescriptor(name="output", shape=Shape([]))],
        )
        graph._nodes.append(node)
        graph._name_to_node[node.name] = node

        with pytest.raises(NotImplementedError):
            ONNXInterpreter(graph, device="cpu", strict=True)

    def test_timing(self):
        """Test execution timing."""
        graph = GraphIR(name="timing_test")
        graph.add_input(
            TensorDescriptor(name="input", shape=Shape([4, 8]), dtype=DataType.Float32)
        )
        graph.add_output(
            TensorDescriptor(name="output", shape=Shape([4, 8]), dtype=DataType.Float32)
        )

        weight = np.random.randn(8, 8).astype(np.float32)
        graph.add_constant("weight", weight)

        nodes = [
            Node(
                op_type="MatMul",
                name="matmul",
                inputs=[
                    TensorDescriptor(name="input", shape=Shape([])),
                    TensorDescriptor(name="weight", shape=Shape([])),
                ],
                outputs=[TensorDescriptor(name="mm_out", shape=Shape([]))],
            ),
            Node(
                op_type="Relu",
                name="relu",
                inputs=[TensorDescriptor(name="mm_out", shape=Shape([]))],
                outputs=[TensorDescriptor(name="output", shape=Shape([]))],
            ),
        ]

        for node in nodes:
            graph._nodes.append(node)
            graph._name_to_node[node.name] = node

        interpreter = ONNXInterpreter(graph, device="cpu")

        input_data = np.random.randn(4, 8).astype(np.float32)
        outputs, timings = interpreter.execute_with_timing(input=input_data)

        assert "output" in outputs
        assert "matmul" in timings
        assert "relu" in timings
        assert all(t >= 0 for t in timings.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
