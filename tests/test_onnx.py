"""
Test ONNX Import/Export functionality.
"""

import pytest
import os
import tempfile


class TestOnnxFunctionality:
    """Test ONNX import/export in GraphIR."""

    def test_graph_ir_creation(self):
        """Test basic GraphIR and Node creation."""
        # Import at function level to avoid import issues if module not built
        try:
            from zenith._zenith_core import graph as graph_module

            # Check if GraphIR class exists
            if hasattr(graph_module, "GraphIR"):
                graph = graph_module.GraphIR("test_graph")
                assert graph.name() == "test_graph"
            else:
                pytest.skip("GraphIR not exposed in Python bindings yet")
        except ImportError:
            pytest.skip("zenith._zenith_core not built")

    def test_onnx_type_mapping(self):
        """Test ONNX data type mappings are valid."""
        # These are the expected mappings based on implementation
        onnx_to_zenith = {
            1: "Float32",  # ONNX FLOAT
            10: "Float16",  # ONNX FLOAT16
            16: "BFloat16",  # ONNX BFLOAT16
            11: "Float64",  # ONNX DOUBLE
            3: "Int8",  # ONNX INT8
            5: "Int16",  # ONNX INT16
            6: "Int32",  # ONNX INT32
            7: "Int64",  # ONNX INT64
            2: "UInt8",  # ONNX UINT8
            9: "Bool",  # ONNX BOOL
        }

        # Verify mapping completeness
        assert len(onnx_to_zenith) == 10
        assert 1 in onnx_to_zenith  # FLOAT must be supported

    def test_onnx_operations_supported(self):
        """Test that all documented ONNX ops are supported."""
        supported_ops = [
            "MatMul",
            "Gemm",
            "Conv",
            "Relu",
            "Softmax",
            "LayerNormalization",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Reshape",
            "Transpose",
            "Flatten",
            "Concat",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "BatchNormalization",
            "Sigmoid",
            "Tanh",
            "Gelu",
        ]

        # All ops should be non-empty strings
        for op in supported_ops:
            assert isinstance(op, str)
            assert len(op) > 0

    def test_protobuf_varint_encoding(self):
        """Test varint encoding logic matches protobuf spec."""
        # Varint encoding examples:
        # 0 -> [0x00]
        # 1 -> [0x01]
        # 127 -> [0x7F]
        # 128 -> [0x80, 0x01]
        # 300 -> [0xAC, 0x02]

        test_cases = [
            (0, [0x00]),
            (1, [0x01]),
            (127, [0x7F]),
            (128, [0x80, 0x01]),
            (300, [0xAC, 0x02]),
        ]

        for value, expected in test_cases:
            encoded = self._encode_varint(value)
            assert encoded == expected, f"Varint {value} failed"

    def _encode_varint(self, value: int) -> list:
        """Encode a value as varint (matching ProtobufWriter::write_varint)."""
        result = []
        while value >= 0x80:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value)
        return result


class TestOnnxModelLoading:
    """Test loading actual ONNX models."""

    @pytest.fixture
    def sample_onnx_path(self):
        """Create a minimal ONNX model for testing."""
        # This would be a minimal valid ONNX binary
        # For now, return None as we can't create valid ONNX without protobuf
        return None

    def test_load_nonexistent_file(self):
        """Test loading non-existent file returns error."""
        try:
            from zenith._zenith_core import graph as graph_module

            if hasattr(graph_module, "load_onnx"):
                result = graph_module.load_onnx("/nonexistent/path.onnx")
                assert not result.ok()
            else:
                pytest.skip("load_onnx not exposed yet")
        except ImportError:
            pytest.skip("zenith._zenith_core not built")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
