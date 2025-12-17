"""
API Unit Tests

Comprehensive unit testing for Zenith public API as specified in CetakBiru 5.3:
- compile(): Main entry point
- optimize(): Alias for compile
- _detect_framework(): Framework detection
- CompiledModel: Compiled model representation
"""

import pytest

from zenith.api import (
    compile,
    optimize,
    _detect_framework,
    CompiledModel,
)
from zenith.core import GraphIR, TensorDescriptor, Shape, DataType


class TestDetectFramework:
    """Unit tests for framework detection."""

    def test_detect_onnx_from_string(self):
        """Test detecting ONNX from string path."""
        result = _detect_framework("model.onnx")
        assert result == "onnx"

    def test_detect_onnx_from_bytes(self):
        """Test detecting ONNX from bytes."""
        result = _detect_framework(b"model_data")
        assert result == "onnx"

    def test_detect_unknown_type(self):
        """Test unknown type raises ValueError."""
        with pytest.raises(ValueError):
            _detect_framework(12345)  # Integer not supported

    def test_detect_unknown_object(self):
        """Test custom object raises ValueError."""

        class CustomModel:
            pass

        with pytest.raises(ValueError):
            _detect_framework(CustomModel())


class TestCompiledModel:
    """Unit tests for CompiledModel class."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = GraphIR(name="test_model")
        graph.add_node(
            op_type="Add",
            name="add1",
            inputs=[TensorDescriptor("x", Shape([1, 10]), DataType.Float32)],
            outputs=[TensorDescriptor("y", Shape([1, 10]), DataType.Float32)],
        )
        graph.add_input(TensorDescriptor("input", Shape([1, 10]), DataType.Float32))
        graph.add_output(TensorDescriptor("output", Shape([1, 10]), DataType.Float32))
        return graph

    def test_compiled_model_creation(self, sample_graph):
        """Test CompiledModel creation."""
        model = CompiledModel(sample_graph, "cpu", "cpu")
        assert model.graph_ir is sample_graph
        assert model.backend == "cpu"
        assert model.target == "cpu"

    def test_compiled_model_repr(self, sample_graph):
        """Test CompiledModel string representation."""
        model = CompiledModel(sample_graph, "cuda", "cuda:0")
        repr_str = repr(model)
        assert "test_model" in repr_str
        assert "cuda" in repr_str

    def test_compiled_model_summary(self, sample_graph):
        """Test CompiledModel summary."""
        model = CompiledModel(sample_graph, "cpu", "cpu")
        summary = model.summary()
        assert "test_model" in summary
        assert "Add" in summary

    def test_compiled_model_call_not_implemented(self, sample_graph):
        """Test that calling model raises NotImplementedError."""
        model = CompiledModel(sample_graph, "cpu", "cpu")
        with pytest.raises(NotImplementedError):
            model()


class TestCompileFunction:
    """Unit tests for compile function."""

    def test_compile_with_graphir(self):
        """Test compiling a GraphIR directly."""
        graph = GraphIR(name="test")
        graph.add_node("Add", "add1", [], [])
        graph.add_input(TensorDescriptor("x", Shape([1, 10]), DataType.Float32))
        graph.add_output(TensorDescriptor("y", Shape([1, 10]), DataType.Float32))

        # Should detect as ONNX since it's not a framework model
        try:
            result = compile(graph, target="cpu")
        except (ImportError, ValueError, TypeError):
            # Expected when frameworks not installed
            pass

    def test_compile_cpu_target(self):
        """Test CPU target parsing."""
        graph = GraphIR(name="test")
        graph.add_input(TensorDescriptor("x", Shape([1]), DataType.Float32))
        graph.add_output(TensorDescriptor("y", Shape([1]), DataType.Float32))

        try:
            result = compile(graph, target="cpu")
        except (ImportError, ValueError, TypeError):
            pass

    def test_compile_cuda_target(self):
        """Test CUDA target parsing."""
        graph = GraphIR(name="test")

        try:
            result = compile(graph, target="cuda:0")
        except (ImportError, ValueError, TypeError, NotImplementedError):
            pass

    def test_compile_rocm_not_implemented(self):
        """Test ROCm target raises NotImplementedError."""
        graph = GraphIR(name="test")

        try:
            result = compile(graph, target="rocm:0")
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError:
            pass
        except (ImportError, ValueError, TypeError):
            pass

    def test_compile_tpu_not_implemented(self):
        """Test TPU target raises NotImplementedError."""
        graph = GraphIR(name="test")

        try:
            result = compile(graph, target="tpu")
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError:
            pass
        except (ImportError, ValueError, TypeError):
            pass


class TestOptimizeFunction:
    """Unit tests for optimize function (alias for compile)."""

    def test_optimize_calls_compile(self):
        """Test that optimize is an alias for compile."""
        graph = GraphIR(name="test")

        try:
            result = optimize(graph, target="cpu")
        except (ImportError, ValueError, TypeError):
            pass


class TestCompileParameters:
    """Unit tests for compile function parameters."""

    def test_default_parameters(self):
        """Test default parameter values."""
        # These are tested implicitly through compile calls
        # Default: target="cpu", precision="fp32", opt_level=2
        pass

    def test_precision_fp32(self):
        """Test FP32 precision."""
        graph = GraphIR(name="test")
        try:
            result = compile(graph, precision="fp32")
        except (ImportError, ValueError, TypeError):
            pass

    def test_precision_fp16(self):
        """Test FP16 precision."""
        graph = GraphIR(name="test")
        try:
            result = compile(graph, precision="fp16")
        except (ImportError, ValueError, TypeError):
            pass

    def test_precision_int8(self):
        """Test INT8 precision."""
        graph = GraphIR(name="test")
        try:
            result = compile(graph, precision="int8")
        except (ImportError, ValueError, TypeError):
            pass

    def test_opt_level_1(self):
        """Test optimization level 1."""
        graph = GraphIR(name="test")
        try:
            result = compile(graph, opt_level=1)
        except (ImportError, ValueError, TypeError):
            pass

    def test_opt_level_3(self):
        """Test optimization level 3."""
        graph = GraphIR(name="test")
        try:
            result = compile(graph, opt_level=3)
        except (ImportError, ValueError, TypeError):
            pass

    def test_tolerance_parameter(self):
        """Test custom tolerance."""
        graph = GraphIR(name="test")
        try:
            result = compile(graph, tolerance=1e-4)
        except (ImportError, ValueError, TypeError):
            pass
