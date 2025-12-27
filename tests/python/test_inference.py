# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for Zenith Complete E2E Inference Module.
"""

import pytest
import numpy as np


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from zenith.inference import InferenceConfig

        config = InferenceConfig()

        assert config.backend == "auto"
        assert config.precision == "fp32"
        assert config.batch_size == 1
        assert config.enable_optimization is True
        assert config.warmup_iterations == 3
        assert config.tolerance == 1e-5

    def test_custom_config(self):
        """Test custom configuration."""
        from zenith.inference import InferenceConfig

        config = InferenceConfig(
            backend="cuda",
            precision="fp16",
            batch_size=32,
            enable_cuda_graphs=True,
        )

        assert config.backend == "cuda"
        assert config.precision == "fp16"
        assert config.batch_size == 32
        assert config.enable_cuda_graphs is True


class TestInferenceStats:
    """Tests for InferenceStats."""

    def test_empty_stats(self):
        """Test empty stats."""
        from zenith.inference import InferenceStats

        stats = InferenceStats()

        assert stats.total_runs == 0
        assert stats.mean_latency_ms == 0.0
        # min_latency is 0.0 when empty (for JSON serialization safety)
        assert stats.min_latency_ms == 0.0
        assert stats.max_latency_ms == 0.0

    def test_record_latency(self):
        """Test recording latencies."""
        from zenith.inference import InferenceStats

        stats = InferenceStats()
        stats.record(10.0)
        stats.record(20.0)
        stats.record(15.0)

        assert stats.total_runs == 3
        assert stats.mean_latency_ms == 15.0
        assert stats.min_latency_ms == 10.0
        assert stats.max_latency_ms == 20.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from zenith.inference import InferenceStats

        stats = InferenceStats()
        stats.record(10.0)

        result = stats.to_dict()

        assert "total_runs" in result
        assert "mean_latency_ms" in result
        assert "min_latency_ms" in result
        assert "max_latency_ms" in result
        assert result["total_runs"] == 1


class TestInferenceSession:
    """Tests for InferenceSession."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple PyTorch model."""
        pytest.importorskip("torch")
        import torch

        return torch.nn.Linear(64, 32)

    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        pytest.importorskip("torch")
        import torch

        return torch.randn(1, 64)

    def test_session_creation(self, simple_model, sample_input):
        """Test session creation."""
        from zenith.inference import InferenceSession, InferenceConfig

        config = InferenceConfig(backend="cpu", verbose=0)
        session = InferenceSession(
            simple_model,
            config=config,
            sample_input=sample_input,
        )

        assert session.is_initialized
        assert session.framework == "pytorch"

    def test_session_run(self, simple_model, sample_input):
        """Test running inference."""
        from zenith.inference import InferenceSession, InferenceConfig

        config = InferenceConfig(backend="cpu", verbose=0, warmup_iterations=1)
        session = InferenceSession(
            simple_model,
            config=config,
            sample_input=sample_input,
        )

        result = session.run({"input": sample_input})

        assert result is not None
        assert len(result) > 0

    def test_session_run_with_latency(self, simple_model, sample_input):
        """Test running inference with latency."""
        from zenith.inference import (
            InferenceSession,
            InferenceConfig,
            InferenceResult,
        )

        config = InferenceConfig(backend="cpu", verbose=0, warmup_iterations=1)
        session = InferenceSession(
            simple_model,
            config=config,
            sample_input=sample_input,
        )

        result = session.run({"input": sample_input}, return_latency=True)

        assert isinstance(result, InferenceResult)
        assert result.latency_ms > 0
        assert result.backend_used == "cpu"

    def test_session_stats(self, simple_model, sample_input):
        """Test session statistics."""
        from zenith.inference import InferenceSession, InferenceConfig

        config = InferenceConfig(backend="cpu", verbose=0, warmup_iterations=1)
        session = InferenceSession(
            simple_model,
            config=config,
            sample_input=sample_input,
        )

        # Run multiple inferences
        for _ in range(5):
            session.run({"input": sample_input})

        stats = session.get_stats()

        assert stats["total_runs"] == 5
        assert stats["mean_latency_ms"] > 0

    def test_session_benchmark(self, simple_model, sample_input):
        """Test benchmarking."""
        from zenith.inference import InferenceSession, InferenceConfig

        config = InferenceConfig(backend="cpu", verbose=0, warmup_iterations=0)
        session = InferenceSession(
            simple_model,
            config=config,
            sample_input=sample_input,
        )

        result = session.benchmark(
            {"input": sample_input},
            num_runs=10,
            num_warmup=2,
        )

        assert result["num_runs"] == 10
        assert result["mean_ms"] > 0
        assert result["p50_ms"] > 0
        assert result["p99_ms"] > 0
        assert "throughput_per_sec" in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_session(self):
        """Test create_session function."""
        pytest.importorskip("torch")
        import torch
        from zenith.inference import create_session, InferenceConfig

        model = torch.nn.Linear(32, 16)
        sample_input = torch.randn(1, 32)

        config = InferenceConfig(backend="cpu", verbose=0)
        session = create_session(model, config=config, sample_input=sample_input)

        assert session.is_initialized
        assert session.framework == "pytorch"

    def test_infer_function(self):
        """Test one-shot infer function."""
        pytest.importorskip("torch")
        import torch
        from zenith.inference import infer, InferenceConfig

        model = torch.nn.Linear(32, 16)
        inputs = {"input": torch.randn(1, 32)}

        config = InferenceConfig(backend="cpu", verbose=0)
        result = infer(model, inputs, config=config)

        assert result is not None
        assert len(result) > 0


class TestSequentialModel:
    """Tests for Sequential models."""

    def test_sequential_model_inference(self):
        """Test inference with Sequential model."""
        pytest.importorskip("torch")
        import torch
        from zenith.inference import InferenceSession, InferenceConfig

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )
        sample_input = torch.randn(2, 128)

        config = InferenceConfig(backend="cpu", verbose=0)
        session = InferenceSession(
            model,
            config=config,
            sample_input=sample_input,
        )

        result = session.run({"input": sample_input})
        assert result is not None


class TestModuleExports:
    """Test module exports."""

    def test_exports(self):
        """Test that all expected items are exported."""
        from zenith import inference

        # Check classes
        assert hasattr(inference, "InferenceBackend")
        assert hasattr(inference, "InferencePrecision")
        assert hasattr(inference, "InferenceConfig")
        assert hasattr(inference, "InferenceResult")
        assert hasattr(inference, "InferenceStats")
        assert hasattr(inference, "InferenceSession")

        # Check functions
        assert hasattr(inference, "create_session")
        assert hasattr(inference, "infer")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
