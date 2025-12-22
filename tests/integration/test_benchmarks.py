# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for MLPerf-style Benchmark Suite

Validates:
- BenchmarkConfig validation
- BenchmarkResult generation
- ZenithBenchmark scenarios
- Results table generation
"""

import pytest
import numpy as np

from benchmarks.mlperf_suite import (
    BenchmarkConfig,
    BenchmarkResult,
    ZenithBenchmark,
    generate_results_table,
    compare_results,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.batch_sizes == [1, 4, 8, 16]
        assert config.num_warmup == 10
        assert config.num_runs == 100
        assert config.quality_target == 0.99
        assert config.scenario == "single-stream"
        assert config.precision == "fp32"

    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            model_name="custom-model",
            batch_sizes=[8, 16],
            sequence_lengths=[64, 128],
            num_warmup=5,
            num_runs=50,
            quality_target=0.95,
            scenario="offline",
            precision="fp16",
        )
        assert config.batch_sizes == [8, 16]
        assert config.scenario == "offline"
        assert config.precision == "fp16"

    def test_validate_empty_model_name(self):
        """Test validation rejects empty model name."""
        config = BenchmarkConfig(model_name="")
        with pytest.raises(ValueError, match="model_name is required"):
            config.validate()

    def test_validate_invalid_scenario(self):
        """Test validation rejects invalid scenario."""
        config = BenchmarkConfig(model_name="test", scenario="invalid")
        with pytest.raises(ValueError, match="Invalid scenario"):
            config.validate()

    def test_validate_invalid_precision(self):
        """Test validation rejects invalid precision."""
        config = BenchmarkConfig(model_name="test", precision="fp64")
        with pytest.raises(ValueError, match="Invalid precision"):
            config.validate()

    def test_validate_negative_warmup(self):
        """Test validation rejects negative warmup."""
        config = BenchmarkConfig(model_name="test", num_warmup=-1)
        with pytest.raises(ValueError, match="num_warmup must be non-negative"):
            config.validate()

    def test_validate_zero_runs(self):
        """Test validation rejects zero runs."""
        config = BenchmarkConfig(model_name="test", num_runs=0)
        with pytest.raises(ValueError, match="num_runs must be at least 1"):
            config.validate()


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = BenchmarkResult(
            model_name="test-model",
            scenario="single-stream",
            batch_size=1,
        )
        assert result.latency_mean_ms == 0.0
        assert result.throughput_qps == 0.0
        assert result.quality_passed is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            model_name="test-model",
            scenario="single-stream",
            batch_size=8,
            latency_mean_ms=10.5,
            latency_p50_ms=10.0,
            latency_p90_ms=15.0,
            latency_p99_ms=20.0,
            throughput_qps=100.0,
            quality_score=0.98,
            quality_passed=True,
        )

        d = result.to_dict()
        assert d["model_name"] == "test-model"
        assert d["batch_size"] == 8
        assert d["latency"]["p50_ms"] == 10.0
        assert d["throughput"]["qps"] == 100.0
        assert d["quality"]["passed"] is True

    def test_summary(self):
        """Test summary string generation."""
        result = BenchmarkResult(
            model_name="test-model",
            scenario="single-stream",
            batch_size=8,
            latency_p50_ms=10.0,
            latency_p90_ms=15.0,
            latency_p99_ms=20.0,
            throughput_qps=100.0,
            quality_score=0.98,
            quality_passed=True,
        )

        summary = result.summary()
        assert "test-model" in summary
        assert "single-stream" in summary
        assert "PASS" in summary


class TestZenithBenchmark:
    """Tests for ZenithBenchmark class."""

    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = ZenithBenchmark(device="cpu")
        assert benchmark._device == "cpu"

    def test_single_stream_scenario(self):
        """Test single-stream benchmark scenario."""
        benchmark = ZenithBenchmark(device="cpu")

        def simple_model(x):
            return x * 2

        def input_generator(batch_size, seq_len):
            return np.random.randn(batch_size, seq_len, 64).astype(np.float32)

        config = BenchmarkConfig(
            model_name="simple-test",
            batch_sizes=[1],
            sequence_lengths=[32],
            num_warmup=2,
            num_runs=10,
            scenario="single-stream",
        )

        results = benchmark.run(config, simple_model, input_generator)

        assert len(results) == 1
        result = results[0]
        assert result.model_name == "simple-test"
        assert result.scenario == "single-stream"
        assert result.batch_size == 1
        assert result.latency_p50_ms > 0
        assert result.throughput_qps > 0

    def test_offline_scenario(self):
        """Test offline benchmark scenario."""
        benchmark = ZenithBenchmark(device="cpu")

        def simple_model(x):
            return x + 1

        def input_generator(batch_size, seq_len):
            return np.ones((batch_size, seq_len), dtype=np.float32)

        config = BenchmarkConfig(
            model_name="offline-test",
            batch_sizes=[4],
            sequence_lengths=[16],
            num_warmup=2,
            num_runs=10,
            scenario="offline",
        )

        results = benchmark.run(config, simple_model, input_generator)

        assert len(results) == 1
        result = results[0]
        assert result.scenario == "offline"
        assert result.throughput_samples_per_sec > 0

    def test_server_scenario(self):
        """Test server benchmark scenario."""
        benchmark = ZenithBenchmark(device="cpu")

        def simple_model(x):
            return x

        def input_generator(batch_size, seq_len):
            return np.zeros((batch_size, seq_len), dtype=np.float32)

        config = BenchmarkConfig(
            model_name="server-test",
            batch_sizes=[1],
            sequence_lengths=[8],
            num_warmup=1,
            num_runs=5,
            scenario="server",
            target_qps=100.0,
            target_latency_ms=50.0,
        )

        results = benchmark.run(config, simple_model, input_generator)

        assert len(results) == 1
        result = results[0]
        assert result.scenario == "server"

    def test_quality_verification(self):
        """Test quality verification with reference function."""
        benchmark = ZenithBenchmark(device="cpu")

        def model_fn(x):
            return x * 2

        def reference_fn(x):
            return x * 2

        def input_generator(batch_size, seq_len):
            return np.ones((batch_size, seq_len), dtype=np.float32)

        config = BenchmarkConfig(
            model_name="quality-test",
            batch_sizes=[1],
            sequence_lengths=[4],
            num_warmup=1,
            num_runs=5,
            quality_target=0.9,
        )

        results = benchmark.run(config, model_fn, input_generator, reference_fn)

        assert len(results) == 1
        assert results[0].quality_score == 1.0
        assert results[0].quality_passed is True


class TestResultsTable:
    """Tests for results table generation."""

    def test_generate_results_table(self):
        """Test markdown table generation."""
        results = [
            BenchmarkResult(
                model_name="model-a",
                scenario="single-stream",
                batch_size=1,
                sequence_length=128,
                precision="fp32",
                latency_p50_ms=10.0,
                latency_p90_ms=15.0,
                latency_p99_ms=20.0,
                throughput_qps=100.0,
                quality_passed=True,
            ),
            BenchmarkResult(
                model_name="model-a",
                scenario="single-stream",
                batch_size=8,
                sequence_length=128,
                precision="fp32",
                latency_p50_ms=50.0,
                latency_p90_ms=60.0,
                latency_p99_ms=80.0,
                throughput_qps=160.0,
                quality_passed=True,
            ),
        ]

        table = generate_results_table(results)

        assert "| Model |" in table
        assert "model-a" in table
        assert "100.0" in table
        assert "PASS" in table

    def test_compare_results(self):
        """Test comparison table generation."""
        zenith_results = [
            BenchmarkResult(
                model_name="model-a",
                scenario="single-stream",
                batch_size=1,
                latency_p50_ms=8.0,
                throughput_qps=125.0,
            ),
        ]
        baseline_results = [
            BenchmarkResult(
                model_name="model-a",
                scenario="single-stream",
                batch_size=1,
                latency_p50_ms=10.0,
                throughput_qps=100.0,
            ),
        ]

        comparison = compare_results(zenith_results, baseline_results)

        assert "Speedup" in comparison
        assert "1.25x" in comparison


class TestReportGenerator:
    """Tests for BenchmarkReportGenerator."""

    def test_report_config_defaults(self):
        """Test ReportConfig default values."""
        from benchmarks.report_generator import ReportConfig

        config = ReportConfig()
        assert config.title == "Zenith Benchmark Report"
        assert config.output_dir == "docs/benchmarks"
        assert config.include_charts is True

    def test_get_system_info(self):
        """Test system info collection."""
        from benchmarks.report_generator import get_system_info

        info = get_system_info()
        assert "python_version" in info
        assert "platform" in info
        assert "numpy_version" in info

    def test_markdown_generator_header(self):
        """Test markdown header generation."""
        from benchmarks.report_generator import MarkdownGenerator, ReportConfig

        config = ReportConfig(title="Test Report")
        gen = MarkdownGenerator(config)
        header = gen.generate_header()

        assert "Test Report" in header
        assert "Generated:" in header

    def test_markdown_generator_methodology(self):
        """Test methodology section generation."""
        from benchmarks.report_generator import MarkdownGenerator, ReportConfig

        config = ReportConfig()
        gen = MarkdownGenerator(config)
        methodology = gen.generate_methodology_section()

        assert "Benchmark Methodology" in methodology
        assert "MLPerf" in methodology
        assert "Single-Stream" in methodology

    def test_markdown_generator_results(self):
        """Test results section generation."""
        from benchmarks.report_generator import MarkdownGenerator, ReportConfig

        config = ReportConfig()
        gen = MarkdownGenerator(config)

        results = [
            BenchmarkResult(
                model_name="test",
                scenario="single-stream",
                batch_size=1,
                latency_p50_ms=10.0,
                latency_p90_ms=12.0,
                latency_p99_ms=15.0,
                throughput_qps=100.0,
            ),
        ]

        section = gen.generate_results_section(results)
        assert "Benchmark Results" in section
        assert "test" in section

    def test_chart_generator_initialization(self):
        """Test ChartGenerator initialization."""
        from benchmarks.report_generator import ChartGenerator

        gen = ChartGenerator("/tmp", dpi=100)
        assert gen._output_dir == "/tmp"
        assert gen._dpi == 100

    def test_report_generator_full(self):
        """Test full report generation."""
        import tempfile
        import os
        from benchmarks.report_generator import (
            BenchmarkReportGenerator,
            ReportConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                output_dir=tmpdir,
                include_charts=False,
            )
            generator = BenchmarkReportGenerator(config)

            results = [
                BenchmarkResult(
                    model_name="test-model",
                    scenario="single-stream",
                    batch_size=1,
                    latency_p50_ms=10.0,
                    latency_p90_ms=12.0,
                    latency_p99_ms=15.0,
                    throughput_qps=100.0,
                    quality_passed=True,
                ),
            ]

            report_path = generator.generate(results)
            assert os.path.exists(report_path)

            with open(report_path, "r") as f:
                content = f.read()
            assert "test-model" in content
            assert "Benchmark Report" in content

    def test_json_export(self):
        """Test JSON export functionality."""
        import tempfile
        import os
        import json
        from benchmarks.report_generator import (
            BenchmarkReportGenerator,
            ReportConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(output_dir=tmpdir)
            generator = BenchmarkReportGenerator(config)

            results = [
                BenchmarkResult(
                    model_name="test",
                    scenario="single-stream",
                    batch_size=1,
                    latency_p50_ms=10.0,
                ),
            ]

            json_path = generator.export_json(results)
            assert os.path.exists(json_path)

            with open(json_path, "r") as f:
                data = json.load(f)
            assert "zenith_results" in data
            assert len(data["zenith_results"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
