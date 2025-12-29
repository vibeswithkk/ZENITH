"""
Unit Tests for Advanced Optimization Passes

Tests for:
- Conv-BN-ReLU fusion
- Layout transformation
- Profiler system
- Benchmark system

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from zenith.core import GraphIR, Node, TensorDescriptor, Shape, DataType


class TestConvBNFusion:
    """Tests for Conv-BN-ReLU fusion pass."""

    def test_compute_fused_weights_basic(self):
        """Test basic Conv-BN weight fusion computation."""
        from zenith.optimization.fusion_pass import compute_conv_bn_weights

        # Create sample weights
        conv_weight = np.random.randn(16, 3, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(16).astype(np.float32)

        # BN parameters
        bn_gamma = np.ones(16, dtype=np.float32)
        bn_beta = np.zeros(16, dtype=np.float32)
        bn_mean = np.random.randn(16).astype(np.float32)
        bn_var = np.abs(np.random.randn(16)).astype(np.float32) + 0.1

        # Compute fused weights
        fused_weight, fused_bias = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var
        )

        # Check shapes
        assert fused_weight.shape == conv_weight.shape
        assert fused_bias.shape == conv_bias.shape

        # Check dtypes
        assert fused_weight.dtype == np.float32
        assert fused_bias.dtype == np.float32

    def test_compute_fused_weights_no_conv_bias(self):
        """Test fusion when conv has no bias."""
        from zenith.optimization.fusion_pass import compute_conv_bn_weights

        conv_weight = np.random.randn(8, 3, 3, 3).astype(np.float32)
        conv_bias = None

        bn_gamma = np.ones(8, dtype=np.float32) * 2.0
        bn_beta = np.ones(8, dtype=np.float32) * 0.5
        bn_mean = np.zeros(8, dtype=np.float32)
        bn_var = np.ones(8, dtype=np.float32)

        fused_weight, fused_bias = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var
        )

        assert fused_weight.shape == conv_weight.shape
        assert fused_bias.shape == (8,)

    def test_fusion_numerical_correctness(self):
        """Test that fused weights produce identical output."""
        from zenith.optimization.fusion_pass import compute_conv_bn_weights

        # Simple 1x1 conv for easy verification
        C_out, C_in = 4, 3
        conv_weight = np.random.randn(C_out, C_in, 1, 1).astype(np.float32)
        conv_bias = np.random.randn(C_out).astype(np.float32)

        bn_gamma = np.random.randn(C_out).astype(np.float32)
        bn_beta = np.random.randn(C_out).astype(np.float32)
        bn_mean = np.random.randn(C_out).astype(np.float32)
        bn_var = np.abs(np.random.randn(C_out)).astype(np.float32) + 0.1

        # Input
        x = np.random.randn(1, C_in, 4, 4).astype(np.float32)

        # Reference: Conv then BN
        # Simplified conv (1x1)
        conv_out = np.einsum("nihw,oihw->nohw", x, conv_weight)
        conv_out = conv_out + conv_bias.reshape(1, -1, 1, 1)

        # BN
        std = np.sqrt(bn_var + 1e-5)
        bn_out = bn_gamma.reshape(1, -1, 1, 1) * (
            conv_out - bn_mean.reshape(1, -1, 1, 1)
        ) / std.reshape(1, -1, 1, 1) + bn_beta.reshape(1, -1, 1, 1)

        # Fused: single conv
        fused_weight, fused_bias = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var
        )

        fused_out = np.einsum("nihw,oihw->nohw", x, fused_weight)
        fused_out = fused_out + fused_bias.reshape(1, -1, 1, 1)

        # Should be numerically equal
        np.testing.assert_allclose(fused_out, bn_out, rtol=1e-5, atol=1e-6)

    def test_fusion_pass_pattern_matching(self):
        """Test that fusion pass correctly identifies patterns."""
        from zenith.optimization.fusion_pass import FusionPass

        fusion_pass = FusionPass()
        assert len(fusion_pass.patterns) == 6  # Updated: was 3, now 6 patterns

        # Check pattern names
        pattern_names = [p.name for p in fusion_pass.patterns]
        assert "conv_bn_relu" in pattern_names
        assert "conv_bn" in pattern_names
        assert "gemm_add" in pattern_names
        assert "linear_gelu" in pattern_names  # BERT-specific
        assert "layernorm_add" in pattern_names
        assert "add_layernorm" in pattern_names


class TestLayoutTransform:
    """Tests for layout transformation pass."""

    def test_nhwc_to_nchw_transform(self):
        """Test NHWC to NCHW transformation."""
        from zenith.optimization.layout_pass import transpose_nhwc_to_nchw

        # NHWC tensor [N, H, W, C]
        nhwc = np.random.randn(2, 8, 8, 3).astype(np.float32)

        # Transform to NCHW
        nchw = transpose_nhwc_to_nchw(nhwc)

        # Check shape: [N, C, H, W]
        assert nchw.shape == (2, 3, 8, 8)

        # Check values are correct
        for n in range(2):
            for h in range(8):
                for w in range(8):
                    for c in range(3):
                        assert nhwc[n, h, w, c] == nchw[n, c, h, w]

    def test_nchw_to_nhwc_transform(self):
        """Test NCHW to NHWC transformation."""
        from zenith.optimization.layout_pass import transpose_nchw_to_nhwc

        # NCHW tensor [N, C, H, W]
        nchw = np.random.randn(2, 3, 8, 8).astype(np.float32)

        # Transform to NHWC
        nhwc = transpose_nchw_to_nhwc(nchw)

        # Check shape: [N, H, W, C]
        assert nhwc.shape == (2, 8, 8, 3)

    def test_roundtrip_transform(self):
        """Test that roundtrip transformation preserves data."""
        from zenith.optimization.layout_pass import (
            transpose_nhwc_to_nchw,
            transpose_nchw_to_nhwc,
        )

        original = np.random.randn(1, 16, 16, 64).astype(np.float32)

        # NHWC -> NCHW -> NHWC
        nchw = transpose_nhwc_to_nchw(original)
        recovered = transpose_nchw_to_nhwc(nchw)

        np.testing.assert_array_equal(original, recovered)

    def test_layout_inference(self):
        """Test layout format inference from shape."""
        from zenith.optimization.layout_pass import get_layout_from_shape, LayoutFormat

        # Clear NCHW: small dim[1], large dim[2,3]
        assert get_layout_from_shape([1, 3, 224, 224]) == LayoutFormat.NCHW

        # Clear NHWC: large dim[1,2], small dim[3]
        assert get_layout_from_shape([1, 224, 224, 3]) == LayoutFormat.NHWC

        # 2D tensor
        assert get_layout_from_shape([32, 10]) == LayoutFormat.NC

        # Unknown (ambiguous)
        assert get_layout_from_shape([1, 64, 64, 64]) == LayoutFormat.UNKNOWN

    def test_backend_preferences(self):
        """Test backend layout preferences."""
        from zenith.optimization.layout_pass import (
            BACKEND_LAYOUT_PREFERENCES,
            LayoutFormat,
        )

        # Check CPU AVX2 prefers NCHW
        assert BACKEND_LAYOUT_PREFERENCES["cpu_avx2"].preferred == LayoutFormat.NCHW

        # Check CUDA prefers NCHW
        assert BACKEND_LAYOUT_PREFERENCES["cuda"].preferred == LayoutFormat.NCHW


class TestProfiler:
    """Tests for profiler system."""

    def test_profiler_basic(self):
        """Test basic profiler functionality."""
        from zenith.optimization.profiler import Profiler

        profiler = Profiler()

        with profiler.session("test_session") as session:
            with profiler.measure("op1", "TestOp"):
                time.sleep(0.01)  # 10ms

            with profiler.measure("op2", "TestOp"):
                time.sleep(0.005)  # 5ms

        assert len(profiler.sessions) == 1
        assert session.operation_count == 2
        assert session.total_duration_ms > 15  # At least 15ms

    def test_profiler_export_json(self):
        """Test JSON export."""
        from zenith.optimization.profiler import Profiler

        profiler = Profiler()

        with profiler.session("export_test"):
            with profiler.measure("test_op", "MatMul", input_shapes=[[4, 4]]):
                pass

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        profiler.export_json(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert "sessions" in data
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["name"] == "export_test"

        Path(filepath).unlink()

    def test_profiler_export_csv(self):
        """Test CSV export."""
        from zenith.optimization.profiler import Profiler

        profiler = Profiler()

        with profiler.session("csv_test"):
            with profiler.measure("test_op", "ReLU"):
                pass

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = f.name

        profiler.export_csv(filepath)

        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) == 2  # Header + 1 operation
        assert "session" in lines[0]
        assert "csv_test" in lines[1]

        Path(filepath).unlink()

    def test_profiler_summary(self):
        """Test profiler summary generation."""
        from zenith.optimization.profiler import Profiler

        profiler = Profiler()

        with profiler.session("summary_test"):
            with profiler.measure("matmul", "MatMul"):
                pass
            with profiler.measure("relu", "ReLU"):
                pass
            with profiler.measure("matmul2", "MatMul"):
                pass

        summary = profiler.get_summary()

        assert summary["session_count"] == 1
        session_summary = summary["sessions"][0]
        assert session_summary["operation_count"] == 3
        assert "by_op_type" in session_summary
        assert session_summary["by_op_type"]["MatMul"]["count"] == 2
        assert session_summary["by_op_type"]["ReLU"]["count"] == 1


class TestBenchmark:
    """Tests for benchmark system."""

    def test_benchmark_basic(self):
        """Test basic benchmark functionality."""
        from zenith.optimization.benchmark import Benchmark

        benchmark = Benchmark()

        def dummy_op(x):
            return x * 2

        x = np.ones((10, 10), dtype=np.float32)

        benchmark.add(
            name="test_benchmark",
            fn=dummy_op,
            inputs={"x": x},
            iterations=10,
            warmup=2,
            operation="Mul",
            backend="test",
        )

        results = benchmark.run()

        assert len(results) == 1
        assert results[0].name == "test_benchmark"
        assert results[0].iterations == 10
        assert results[0].mean_time_ms > 0

    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        from zenith.optimization.benchmark import Benchmark

        benchmark = Benchmark()

        def slow_op():
            time.sleep(0.005)

        def fast_op():
            time.sleep(0.001)

        benchmark.add("slow", slow_op, {}, iterations=5, warmup=1)
        benchmark.add("fast", fast_op, {}, iterations=5, warmup=1)

        benchmark.run()

        comparison = benchmark.compare("slow", "fast")

        assert comparison is not None
        assert comparison.speedup > 1.0  # Fast is faster

    def test_benchmark_export(self):
        """Test benchmark export."""
        from zenith.optimization.benchmark import Benchmark

        benchmark = Benchmark()

        benchmark.add("test", lambda: None, {}, iterations=5)
        benchmark.run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        benchmark.export_json(json_path)
        benchmark.export_csv(csv_path)

        # Verify JSON
        with open(json_path) as f:
            data = json.load(f)
        assert "results" in data
        assert len(data["results"]) == 1

        # Verify CSV
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        Path(json_path).unlink()
        Path(csv_path).unlink()


# Required for time module in tests
import time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
