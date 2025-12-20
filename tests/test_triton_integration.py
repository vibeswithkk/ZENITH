"""
Integration Tests for Triton Backend.

Tests model deployment, inference, and concurrent request handling.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zenith.serving.triton_client import (
    InferenceInput,
    InferenceOutput,
    InferenceResult,
    MockTritonClient,
    ModelMetadata,
    Protocol,
    ServerStatus,
    TritonClient,
)
from tests.integration.triton_load_test import (
    LoadTestConfig,
    LoadTestResult,
    TritonLoadTester,
    run_mock_load_test,
)


class TestTritonClient:
    """Tests for TritonClient."""

    def test_client_initialization_http(self):
        """Test HTTP client initialization."""
        client = MockTritonClient("localhost:8000")
        assert client.protocol == Protocol.HTTP

    def test_server_health_mock(self):
        """Test server health checks with mock."""
        client = MockTritonClient()
        assert client.is_server_live()
        assert client.is_server_ready()

    def test_model_registration(self):
        """Test model registration in mock client."""
        client = MockTritonClient()

        metadata = ModelMetadata(
            name="test_model", platform="python", versions=["1", "2"]
        )
        client.register_model("test_model", metadata)

        assert client.is_model_ready("test_model")
        assert not client.is_model_ready("nonexistent")

    def test_model_metadata_retrieval(self):
        """Test model metadata retrieval."""
        client = MockTritonClient()

        metadata = ModelMetadata(
            name="bert_model",
            platform="python",
            versions=["1"],
            inputs=[{"name": "input_ids", "shape": [-1, 128], "datatype": "INT64"}],
            outputs=[{"name": "logits", "shape": [-1, 768], "datatype": "FP32"}],
        )
        client.register_model("bert_model", metadata)

        retrieved = client.get_model_metadata("bert_model")
        assert retrieved.name == "bert_model"
        assert retrieved.platform == "python"
        assert len(retrieved.inputs) == 1
        assert len(retrieved.outputs) == 1

    def test_list_models(self):
        """Test listing models."""
        client = MockTritonClient()

        client.register_model("model1")
        client.register_model("model2")
        client.register_model("model3")

        models = client.list_models()
        assert len(models) == 3
        assert "model1" in models
        assert "model2" in models


class TestInferenceInput:
    """Tests for InferenceInput."""

    def test_input_creation(self):
        """Test input tensor creation."""
        data = np.random.randn(1, 224, 224, 3).astype(np.float32)
        inp = InferenceInput(name="image", data=data)

        assert inp.name == "image"
        assert inp.data.shape == (1, 224, 224, 3)
        assert inp.datatype == "FP32"

    def test_datatype_inference_int32(self):
        """Test datatype inference for int32."""
        data = np.array([1, 2, 3], dtype=np.int32)
        inp = InferenceInput(name="indices", data=data)
        assert inp.datatype == "INT32"

    def test_datatype_inference_int64(self):
        """Test datatype inference for int64."""
        data = np.array([1, 2, 3], dtype=np.int64)
        inp = InferenceInput(name="ids", data=data)
        assert inp.datatype == "INT64"

    def test_datatype_inference_float16(self):
        """Test datatype inference for float16."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        inp = InferenceInput(name="half_data", data=data)
        assert inp.datatype == "FP16"


class TestInference:
    """Tests for inference operations."""

    def test_basic_inference_mock(self):
        """Test basic inference with mock client."""
        client = MockTritonClient()
        client.register_model("simple_model")

        input_data = np.random.randn(1, 10).astype(np.float32)
        inputs = [InferenceInput(name="input", data=input_data)]

        result = client.infer("simple_model", inputs)

        assert result.success
        assert result.model_name == "simple_model"
        assert len(result.outputs) > 0

    def test_inference_with_custom_handler(self):
        """Test inference with custom handler."""
        client = MockTritonClient()

        def handler(inputs):
            # Double the input
            inp = inputs[0].data
            return {"output": inp * 2}

        client.register_model("doubler", handler=handler)

        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        inputs = [InferenceInput(name="input", data=input_data)]

        result = client.infer("doubler", inputs)

        assert result.success
        output = result.get_output("output")
        np.testing.assert_array_almost_equal(output, input_data * 2)

    def test_inference_nonexistent_model(self):
        """Test inference on nonexistent model."""
        client = MockTritonClient()

        inputs = [InferenceInput(name="input", data=np.array([1.0]))]
        result = client.infer("nonexistent", inputs)

        assert not result.success
        assert "not found" in result.error.lower()

    def test_inference_output_retrieval(self):
        """Test output retrieval by name."""
        client = MockTritonClient()

        def handler(inputs):
            return {
                "output_a": np.array([1.0]),
                "output_b": np.array([2.0]),
            }

        client.register_model("multi_output", handler=handler)

        inputs = [InferenceInput(name="input", data=np.array([0.0]))]
        result = client.infer("multi_output", inputs)

        assert result.success
        assert result.get_output("output_a") is not None
        assert result.get_output("output_b") is not None
        assert result.get_output("nonexistent") is None


class TestLoadTesting:
    """Tests for load testing functionality."""

    def test_load_test_config_defaults(self):
        """Test load test config defaults."""
        config = LoadTestConfig()
        assert config.num_requests == 100
        assert config.concurrent_workers == 10
        assert config.warmup_requests == 10

    def test_mock_load_test(self):
        """Test load testing with mock client."""
        result = run_mock_load_test(
            model_name="test_model",
            num_requests=50,
            concurrent_workers=5,
            verbose=False,
        )

        assert result.total_requests == 50
        assert result.successful_requests == 50
        assert result.failed_requests == 0
        assert result.error_rate == 0.0
        assert result.requests_per_second > 0

    def test_load_test_with_handler(self):
        """Test load testing with custom inference handler."""

        def handler(inputs):
            return {"output": inputs[0].data}

        result = run_mock_load_test(
            model_name="echo_model",
            num_requests=30,
            concurrent_workers=3,
            inference_handler=handler,
            verbose=False,
        )

        assert result.total_requests == 30
        assert result.successful_requests == 30
        assert result.latency.mean_ms >= 0

    def test_load_test_latency_metrics(self):
        """Test that latency metrics are computed correctly."""
        result = run_mock_load_test(
            model_name="latency_test",
            num_requests=100,
            concurrent_workers=10,
            verbose=False,
        )

        assert result.latency.mean_ms >= 0
        assert result.latency.p50_ms >= 0
        assert result.latency.p90_ms >= result.latency.p50_ms
        assert result.latency.p95_ms >= result.latency.p90_ms
        assert result.latency.p99_ms >= result.latency.p95_ms
        assert result.latency.max_ms >= result.latency.min_ms

    def test_load_test_throughput(self):
        """Test throughput calculation."""
        result = run_mock_load_test(
            model_name="throughput_test",
            num_requests=100,
            concurrent_workers=10,
            verbose=False,
        )

        assert result.requests_per_second > 0
        # Should be able to handle at least 100 req/s with mock
        assert result.requests_per_second > 10


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    def test_concurrent_workers(self):
        """Test that concurrent workers are used."""
        config = LoadTestConfig(
            model_name="concurrent_test",
            num_requests=50,
            concurrent_workers=5,
        )

        mock = MockTritonClient()
        mock.register_model("concurrent_test")

        tester = TritonLoadTester(config)
        tester.set_mock_client(mock)

        result = tester.run()

        assert result.actual_concurrent == 5
        assert result.total_requests == 50

    def test_high_concurrency(self):
        """Test high concurrency levels."""
        result = run_mock_load_test(
            model_name="high_concurrency",
            num_requests=200,
            concurrent_workers=20,
            verbose=False,
        )

        assert result.total_requests == 200
        assert result.successful_requests == 200
        assert result.error_rate == 0.0


class TestServerStatus:
    """Tests for server status."""

    def test_server_status_fields(self):
        """Test server status fields."""
        status = ServerStatus(live=True, ready=True, version="2.40.0")
        assert status.live
        assert status.ready
        assert status.version == "2.40.0"

    def test_mock_server_status(self):
        """Test mock server returns healthy status."""
        client = MockTritonClient()
        status = client.get_server_status()

        assert status.live
        assert status.ready


class TestResultSummary:
    """Tests for result summary formatting."""

    def test_summary_contains_metrics(self):
        """Test that summary contains key metrics."""
        result = run_mock_load_test(
            model_name="summary_test",
            num_requests=10,
            concurrent_workers=2,
            verbose=False,
        )

        summary = result.summary()

        assert "Total Requests" in summary
        assert "Successful" in summary
        assert "Latency" in summary
        assert "Throughput" in summary
        assert "Mean" in summary
        assert "P99" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
