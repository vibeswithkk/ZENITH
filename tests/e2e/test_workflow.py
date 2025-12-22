# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
End-to-End Workflow Tests

Tests complete Zenith workflow:
- Model loading
- Compilation
- Execution
- Result verification
"""

import pytest
import numpy as np


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow_graph_ir(self):
        """Test full workflow with GraphIR."""
        from zenith._zenith_core import GraphIR, TensorDescriptor, Shape, DataType
        from zenith.runtime import ZenithEngine

        # 1. Create graph
        graph = GraphIR("test_model")
        td = TensorDescriptor("input", Shape([1, 64]), DataType.Float32)
        graph.add_input(td)
        graph.add_node("Relu", "relu1", [], [])
        td_out = TensorDescriptor("output", Shape([1, 64]), DataType.Float32)
        graph.add_output(td_out)

        # 2. Create engine
        engine = ZenithEngine(backend="cpu")

        # 3. Compile
        compiled = engine.compile(graph)

        # 4. Verify compilation
        assert compiled is not None
        assert hasattr(compiled, "run")

    def test_workflow_with_config(self):
        """Test workflow with custom config."""
        from zenith._zenith_core import GraphIR, TensorDescriptor, Shape, DataType
        from zenith.runtime import ZenithEngine, CompileConfig

        graph = GraphIR("config_test")
        td = TensorDescriptor("x", Shape([1, 128]), DataType.Float32)
        graph.add_input(td)
        graph.add_node("Relu", "relu1", [], [])
        td_out = TensorDescriptor("y", Shape([1, 128]), DataType.Float32)
        graph.add_output(td_out)

        config = CompileConfig(
            precision="fp32",
            verbose=False,
        )

        engine = ZenithEngine(backend="cpu")
        compiled = engine.compile(graph, config=config)

        assert compiled is not None

    def test_pytorch_adapter_workflow(self):
        """Test PyTorch adapter workflow."""
        pytest.importorskip("torch")
        import torch
        from zenith.torch import PyTorchAdapter

        # 1. Create model
        model = torch.nn.Linear(64, 32)

        # 2. Create adapter
        adapter = PyTorchAdapter()

        # 3. Convert to GraphIR
        sample_input = torch.randn(1, 64)
        graph = adapter.from_pytorch(model, sample_input)

        # 4. Verify conversion
        assert graph is not None
        assert graph.num_nodes() > 0

    def test_onnx_import_workflow(self):
        """Test ONNX import workflow."""
        pytest.importorskip("onnx")
        import tempfile
        import torch
        import onnx
        from zenith.adapters import ONNXAdapter

        # 1. Create and export model to ONNX
        model = torch.nn.Linear(64, 32)
        sample_input = torch.randn(1, 64)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, sample_input, f.name)

            # 2. Import ONNX
            adapter = ONNXAdapter()
            graph = adapter.from_onnx(f.name)

            # 3. Verify
            assert graph is not None

    def test_compile_api_workflow(self):
        """Test zenith.compile API."""
        pytest.importorskip("torch")
        import torch
        import zenith

        model = torch.nn.Linear(64, 32)
        compiled = zenith.compile(model, target="cpu")

        assert compiled is not None


class TestPyTorchIntegration:
    """PyTorch integration tests."""

    @pytest.fixture
    def torch_model(self):
        """Create a simple PyTorch model."""
        pytest.importorskip("torch")
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )

    def test_pytorch_model_compile(self, torch_model):
        """Test compiling PyTorch model."""
        import zenith

        compiled = zenith.compile(torch_model, target="cpu")
        assert compiled is not None

    def test_pytorch_sequential_model(self):
        """Test sequential model compilation."""
        pytest.importorskip("torch")
        import torch
        import zenith

        model = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
        )

        compiled = zenith.compile(model, target="cpu")
        assert compiled is not None


class TestErrorHandling:
    """Test error handling in E2E scenarios."""

    def test_invalid_graph_raises_error(self):
        """Test that invalid graph raises appropriate error."""
        from zenith.runtime import ZenithEngine
        from zenith.errors import ValidationError

        engine = ZenithEngine()

        with pytest.raises(ValidationError):
            engine.compile(None)

    def test_empty_graph_raises_error(self):
        """Test that empty graph raises error."""
        from zenith import GraphIR
        from zenith.runtime import ZenithEngine
        from zenith.errors import CompilationError

        graph = GraphIR("empty")
        engine = ZenithEngine()

        with pytest.raises(CompilationError):
            engine.compile(graph)


class TestObservabilityIntegration:
    """Test observability in E2E scenarios."""

    def test_logging_during_compile(self):
        """Test that logging works during compilation."""
        from zenith._zenith_core import GraphIR, TensorDescriptor, Shape, DataType
        from zenith.runtime import ZenithEngine
        from zenith.observability import ZenithLogger, Verbosity
        import io

        # Capture logs
        output = io.StringIO()
        logger = ZenithLogger.get()
        logger.set_output(output)
        logger.set_verbosity(Verbosity.INFO)

        # Compile
        graph = GraphIR("log_test")
        td = TensorDescriptor("x", Shape([1, 64]), DataType.Float32)
        graph.add_input(td)
        graph.add_node("Relu", "relu1", [], [])
        td_out = TensorDescriptor("y", Shape([1, 64]), DataType.Float32)
        graph.add_output(td_out)

        engine = ZenithEngine()
        engine.compile(graph)

        # Verify logs were generated
        logs = output.getvalue()
        assert len(logs) > 0

    def test_metrics_collection(self):
        """Test metrics collection works."""
        from zenith.observability import (
            MetricsCollector,
            InferenceMetrics,
        )

        collector = MetricsCollector()
        collector.record_inference(InferenceMetrics(latency_ms=10.0))
        collector.record_inference(InferenceMetrics(latency_ms=15.0))

        summary = collector.get_summary()
        assert summary["total_inferences"] == 2
        assert summary["latency_mean_ms"] == 12.5


class TestAPIConsistency:
    """Test API consistency across different entry points."""

    def test_zenith_compile_exists(self):
        """Test zenith.compile function exists."""
        import zenith

        assert hasattr(zenith, "compile")
        assert callable(zenith.compile)

    def test_zenith_set_verbosity_exists(self):
        """Test zenith.set_verbosity function exists."""
        import zenith

        assert hasattr(zenith, "set_verbosity")
        zenith.set_verbosity(0)  # Should not raise

    def test_error_classes_exported(self):
        """Test error classes are exported."""
        import zenith

        assert hasattr(zenith, "ZenithError")
        assert hasattr(zenith, "CompilationError")
        assert hasattr(zenith, "ValidationError")

    def test_adapters_exported(self):
        """Test adapters are exported."""
        import zenith

        assert hasattr(zenith, "PyTorchAdapter")
        assert hasattr(zenith, "ONNXAdapter")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
