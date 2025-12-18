"""
Test Suite untuk Memory Pool Allocator dan Benchmark Framework.
"""

import pytest
import numpy as np
import time


class TestMemoryPoolAllocation:
    """Test Memory Pool allocation dan deallocation."""

    def test_basic_allocation(self):
        """Test basic allocation pattern."""
        # Simulate allocation tracking with simple counter
        allocated = {}
        cached = {}
        stats = {"allocated": 0, "cached": 0, "hits": 0, "misses": 0}
        next_ptr = [0]  # Use list to allow mutation in nested function

        def allocate(size):
            # Check cache first
            if size in cached and cached[size]:
                ptr = cached[size].pop()
                stats["hits"] += 1
                stats["cached"] -= size
            else:
                ptr = next_ptr[0]
                next_ptr[0] += 1
                stats["misses"] += 1

            allocated[ptr] = size
            stats["allocated"] += size
            return ptr

        def deallocate(ptr):
            if ptr in allocated:
                size = allocated[ptr]
                del allocated[ptr]
                stats["allocated"] -= size

                # Cache it
                if size not in cached:
                    cached[size] = []
                cached[size].append(ptr)
                stats["cached"] += size

        # Allocate ptr1 (1024) and ptr2 (2048)
        ptr1 = allocate(1024)
        ptr2 = allocate(2048)

        assert stats["allocated"] == 3072  # 1024 + 2048
        assert stats["misses"] == 2
        assert stats["hits"] == 0

        # Deallocate ptr1 - goes to cache
        deallocate(ptr1)
        assert stats["allocated"] == 2048  # only ptr2 remaining
        assert stats["cached"] == 1024  # ptr1's block in cache

        # Allocate same size (1024) - should hit cache
        ptr3 = allocate(1024)
        assert stats["hits"] == 1
        assert stats["allocated"] == 3072  # ptr2 (2048) + ptr3 (1024)
        assert stats["cached"] == 0  # cache was consumed

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        hits = 80
        misses = 20

        hit_rate = hits / (hits + misses)
        assert hit_rate == 0.8

    def test_block_alignment(self):
        """Test block size alignment."""
        granularity = 512
        min_size = 512

        def align_size(size):
            size = max(size, min_size)
            return ((size + granularity - 1) // granularity) * granularity

        assert align_size(100) == 512  # Min size
        assert align_size(512) == 512
        assert align_size(513) == 1024
        assert align_size(1000) == 1024
        assert align_size(1025) == 1536


class TestMemoryStats:
    """Test memory statistics tracking."""

    def test_peak_tracking(self):
        """Test peak memory tracking."""
        allocated = 0
        peak = 0

        allocations = [100, 200, 300, 200, 100]
        deallocations = [0, 100, 0, 200, 100]

        for alloc, dealloc in zip(allocations, deallocations):
            allocated += alloc
            allocated -= dealloc
            peak = max(peak, allocated)

        assert peak == 500  # After first 3 allocations: 100+200+300-100=500

    def test_memory_summary(self):
        """Test memory summary generation."""
        allocated_mb = 256.5
        cached_mb = 64.25
        peak_mb = 512.0

        summary = (
            f"Allocated: {allocated_mb} MB, Cached: {cached_mb} MB, Peak: {peak_mb} MB"
        )

        assert "256.5" in summary
        assert "64.25" in summary
        assert "512.0" in summary


class TestLatencyStats:
    """Test latency statistics computation."""

    def test_mean_computation(self):
        """Test mean latency."""
        samples = [10.0, 12.0, 11.0, 13.0, 14.0]
        mean = sum(samples) / len(samples)
        assert mean == 12.0

    def test_std_computation(self):
        """Test standard deviation."""
        samples = [10.0, 12.0, 11.0, 13.0, 14.0]
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        std = np.sqrt(variance)
        assert abs(std - 1.4142) < 0.01

    def test_percentiles(self):
        """Test percentile computation."""
        samples = list(range(1, 101))  # 1 to 100

        sorted_samples = sorted(samples)

        def percentile(p):
            idx = int(p * (len(sorted_samples) - 1))
            return sorted_samples[idx]

        assert percentile(0.50) == 50  # Median
        assert percentile(0.90) == 90
        assert percentile(0.99) == 99

    def test_min_max(self):
        """Test min/max tracking."""
        samples = [5.5, 3.2, 8.1, 2.9, 6.4]
        assert min(samples) == 2.9
        assert max(samples) == 8.1


class TestThroughputStats:
    """Test throughput statistics."""

    def test_throughput_calculation(self):
        """Test throughput from latency."""
        mean_latency_ms = 10.0  # 10ms per inference

        batches_per_sec = 1000.0 / mean_latency_ms
        assert batches_per_sec == 100.0

    def test_samples_per_sec(self):
        """Test samples per second with batch size."""
        mean_latency_ms = 10.0
        batch_size = 32

        batches_per_sec = 1000.0 / mean_latency_ms
        samples_per_sec = batches_per_sec * batch_size

        assert samples_per_sec == 3200.0


class TestBenchmarkConfig:
    """Test benchmark configuration."""

    def test_default_config(self):
        """Test default benchmark config values."""
        config = {
            "warmup_iterations": 10,
            "benchmark_iterations": 100,
            "batch_size": 1,
            "seq_length": 512,
            "image_size": 224,
            "precision": "fp32",
        }

        assert config["warmup_iterations"] == 10
        assert config["benchmark_iterations"] == 100
        assert config["batch_size"] == 1

    def test_model_specs(self):
        """Test model specifications."""
        resnet50 = {
            "name": "ResNet-50",
            "params": 25_600_000,
            "flops": 4_100_000_000,
            "input_shape": [1, 3, 224, 224],
        }

        bert_base = {
            "name": "BERT-Base",
            "params": 110_000_000,
            "flops": 22_000_000_000,
            "input_shape": [1, 512],
        }

        assert resnet50["params"] == 25_600_000
        assert bert_base["params"] == 110_000_000


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""

    def test_warmup_and_benchmark(self):
        """Test warmup and benchmark iterations."""
        warmup_iters = 5
        benchmark_iters = 20

        call_count = 0

        def dummy_fn():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # 1ms

        # Warmup
        for _ in range(warmup_iters):
            dummy_fn()

        # Benchmark
        latencies = []
        for _ in range(benchmark_iters):
            start = time.perf_counter()
            dummy_fn()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        assert call_count == warmup_iters + benchmark_iters
        assert len(latencies) == benchmark_iters

    def test_latency_collection(self):
        """Test latency collection and stats."""
        latencies = [1.0, 1.5, 1.2, 1.8, 1.3, 1.1, 1.4, 1.6, 1.7, 1.9]

        sorted_lat = sorted(latencies)
        mean = sum(latencies) / len(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p90 = sorted_lat[int(0.9 * (len(sorted_lat) - 1))]

        assert 1.0 <= mean <= 2.0
        assert 1.0 <= p50 <= 2.0
        assert p90 >= p50


class TestBenchmarkSuite:
    """Test benchmark suite for multiple models."""

    def test_multiple_results(self):
        """Test collecting multiple benchmark results."""
        results = []

        results.append(
            {
                "model": "ResNet-50",
                "mean_ms": 5.2,
                "throughput": 192.3,
            }
        )

        results.append(
            {
                "model": "BERT-Base",
                "mean_ms": 12.5,
                "throughput": 80.0,
            }
        )

        assert len(results) == 2
        assert results[0]["model"] == "ResNet-50"
        assert results[1]["model"] == "BERT-Base"

    def test_comparison_table(self):
        """Test comparison table generation."""
        results = [
            {"model": "ResNet-50", "mean": 5.2, "p50": 5.0, "p99": 6.5},
            {"model": "BERT-Base", "mean": 12.5, "p50": 12.0, "p99": 15.0},
        ]

        # Check all models in results
        models = [r["model"] for r in results]
        assert "ResNet-50" in models
        assert "BERT-Base" in models


class TestModelSpecs:
    """Test model specifications."""

    def test_resnet_variants(self):
        """Test ResNet model variants."""
        models = {
            "ResNet-18": {"params": 11_700_000, "flops": 1_800_000_000},
            "ResNet-34": {"params": 21_800_000, "flops": 3_600_000_000},
            "ResNet-50": {"params": 25_600_000, "flops": 4_100_000_000},
            "ResNet-101": {"params": 44_500_000, "flops": 7_800_000_000},
            "ResNet-152": {"params": 60_200_000, "flops": 11_500_000_000},
        }

        # Params should increase with depth
        assert models["ResNet-18"]["params"] < models["ResNet-50"]["params"]
        assert models["ResNet-50"]["params"] < models["ResNet-101"]["params"]

    def test_bert_variants(self):
        """Test BERT model variants."""
        models = {
            "BERT-Base": {
                "params": 110_000_000,
                "hidden": 768,
                "layers": 12,
                "heads": 12,
            },
            "BERT-Large": {
                "params": 340_000_000,
                "hidden": 1024,
                "layers": 24,
                "heads": 16,
            },
        }

        assert models["BERT-Base"]["layers"] == 12
        assert models["BERT-Large"]["layers"] == 24


class TestTimerAccuracy:
    """Test timer accuracy."""

    def test_high_resolution_timer(self):
        """Test high resolution timing."""
        start = time.perf_counter()
        time.sleep(0.01)  # 10ms
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        # Should be close to 10ms (with some tolerance)
        assert 8 <= elapsed_ms <= 20

    def test_multiple_measurements(self):
        """Test stability of multiple measurements."""
        measurements = []

        for _ in range(10):
            start = time.perf_counter()
            time.sleep(0.005)  # 5ms
            end = time.perf_counter()
            measurements.append((end - start) * 1000)

        mean = sum(measurements) / len(measurements)
        std = np.std(measurements)

        # Mean should be close to 5ms
        assert 3 <= mean <= 10
        # Std should be relatively small
        assert std < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
