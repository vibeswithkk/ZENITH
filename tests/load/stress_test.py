#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Stress Test - Pure Python

Stress testing for Zenith inference pipeline without external dependencies.
Tests concurrent access, memory stability, and throughput limits.

Usage:
    python tests/load/stress_test.py
    python tests/load/stress_test.py --workers 100 --requests 1000
    python tests/load/stress_test.py --soak --duration 3600

Test Scenarios:
    1. Sequential Baseline: Single-threaded throughput measurement
    2. Concurrent Stress: Multi-threaded inference simulation
    3. Memory Pressure: Rapid allocation/deallocation cycles
    4. Soak Test: Long-running stability test
"""

import argparse
import gc
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional


@dataclass
class StressTestResult:
    """Result of a stress test run."""

    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    latencies_ms: list
    errors: list
    memory_start_mb: float
    memory_end_mb: float

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def throughput(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return self.successful_requests / self.total_time_seconds

    @property
    def p50_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p99_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def memory_delta_mb(self) -> float:
        return self.memory_end_mb - self.memory_start_mb

    def print_report(self) -> None:
        """Print formatted test report."""
        print("\n" + "=" * 60)
        print(f"  STRESS TEST REPORT: {self.name}")
        print("=" * 60)
        print(f"  Total Requests:     {self.total_requests:,}")
        print(f"  Successful:         {self.successful_requests:,}")
        print(f"  Failed:             {self.failed_requests:,}")
        print(f"  Success Rate:       {self.success_rate:.2f}%")
        print("-" * 60)
        print(f"  Total Time:         {self.total_time_seconds:.2f}s")
        print(f"  Throughput:         {self.throughput:.2f} req/s")
        print("-" * 60)
        if self.latencies_ms:
            print(f"  Latency P50:        {self.p50_latency:.3f}ms")
            print(f"  Latency P99:        {self.p99_latency:.3f}ms")
            print(f"  Latency Mean:       {statistics.mean(self.latencies_ms):.3f}ms")
        print("-" * 60)
        print(f"  Memory Start:       {self.memory_start_mb:.2f}MB")
        print(f"  Memory End:         {self.memory_end_mb:.2f}MB")
        print(f"  Memory Delta:       {self.memory_delta_mb:+.2f}MB")
        print("=" * 60)

        if self.errors:
            print(f"\n  ERRORS ({len(self.errors)}):")
            for i, err in enumerate(self.errors[:5]):
                print(f"    {i + 1}. {err}")
            if len(self.errors) > 5:
                print(f"    ... and {len(self.errors) - 5} more")


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # KB to MB on Linux
    except ImportError:
        return 0.0


class ZenithStressTester:
    """
    Stress tester for Zenith components.

    Tests MetricsCollector, inference simulation, and memory stability.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._errors: list = []
        self._latencies: list = []
        self._success_count: int = 0
        self._fail_count: int = 0

    def _reset_counters(self) -> None:
        """Reset all counters for new test."""
        with self._lock:
            self._errors.clear()
            self._latencies.clear()
            self._success_count = 0
            self._fail_count = 0

    def _record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        with self._lock:
            self._success_count += 1
            self._latencies.append(latency_ms)

    def _record_failure(self, error: str) -> None:
        """Record a failed request."""
        with self._lock:
            self._fail_count += 1
            self._errors.append(error)

    def _simulate_inference(self) -> None:
        """
        Simulate an inference operation.

        Records metrics using MetricsCollector.
        """
        start = time.perf_counter()
        try:
            from zenith.observability import get_metrics_collector, InferenceMetrics

            collector = get_metrics_collector()

            # Simulate some computation
            latency = 5.0 + (hash(threading.current_thread().name) % 10)

            collector.record_inference(
                InferenceMetrics(
                    latency_ms=latency,
                    memory_mb=100.0,
                    batch_size=1,
                )
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_success(elapsed_ms)

        except Exception as e:
            self._record_failure(f"{type(e).__name__}: {e}")

    def test_sequential_baseline(self, num_requests: int = 1000) -> StressTestResult:
        """
        Sequential baseline test.

        Measures single-threaded throughput without concurrency overhead.
        """
        self._reset_counters()
        gc.collect()
        memory_start = get_memory_usage_mb()

        start_time = time.perf_counter()

        for _ in range(num_requests):
            self._simulate_inference()

        total_time = time.perf_counter() - start_time
        gc.collect()
        memory_end = get_memory_usage_mb()

        return StressTestResult(
            name="Sequential Baseline",
            total_requests=num_requests,
            successful_requests=self._success_count,
            failed_requests=self._fail_count,
            total_time_seconds=total_time,
            latencies_ms=self._latencies.copy(),
            errors=self._errors.copy(),
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
        )

    def test_concurrent_stress(
        self,
        num_workers: int = 100,
        requests_per_worker: int = 10,
    ) -> StressTestResult:
        """
        Concurrent stress test.

        Args:
            num_workers: Number of concurrent threads
            requests_per_worker: Requests per thread

        Tests thread-safety and concurrent access patterns.
        """
        self._reset_counters()
        gc.collect()
        memory_start = get_memory_usage_mb()

        total_requests = num_workers * requests_per_worker

        def worker_task():
            for _ in range(requests_per_worker):
                self._simulate_inference()

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task) for _ in range(num_workers)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self._record_failure(f"Worker error: {e}")

        total_time = time.perf_counter() - start_time
        gc.collect()
        memory_end = get_memory_usage_mb()

        return StressTestResult(
            name=f"Concurrent Stress ({num_workers} workers)",
            total_requests=total_requests,
            successful_requests=self._success_count,
            failed_requests=self._fail_count,
            total_time_seconds=total_time,
            latencies_ms=self._latencies.copy(),
            errors=self._errors.copy(),
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
        )

    def test_memory_pressure(self, cycles: int = 100) -> StressTestResult:
        """
        Memory pressure test.

        Rapid allocation and deallocation cycles to detect memory leaks.
        """
        self._reset_counters()
        gc.collect()
        memory_start = get_memory_usage_mb()

        start_time = time.perf_counter()

        for cycle in range(cycles):
            try:
                from zenith.observability import (
                    MetricsCollector,
                    InferenceMetrics,
                )

                # Create and populate collector (local instance, not global)
                collector = MetricsCollector()
                for i in range(100):
                    collector.record_inference(
                        InferenceMetrics(
                            latency_ms=float(i),
                            memory_mb=float(i * 10),
                        )
                    )

                # Get summary (forces computation)
                summary = collector.get_summary()

                # Explicitly delete to trigger deallocation
                del collector
                del summary

                elapsed = (time.perf_counter() - start_time) * 1000 / (cycle + 1)
                self._record_success(elapsed)

            except Exception as e:
                self._record_failure(f"Cycle {cycle}: {e}")

            # Force garbage collection every 10 cycles
            if cycle % 10 == 0:
                gc.collect()

        total_time = time.perf_counter() - start_time
        gc.collect()
        memory_end = get_memory_usage_mb()

        return StressTestResult(
            name=f"Memory Pressure ({cycles} cycles)",
            total_requests=cycles,
            successful_requests=self._success_count,
            failed_requests=self._fail_count,
            total_time_seconds=total_time,
            latencies_ms=self._latencies.copy(),
            errors=self._errors.copy(),
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
        )

    def test_soak(self, duration_seconds: int = 3600) -> StressTestResult:
        """
        Soak test for long-running stability.

        Args:
            duration_seconds: How long to run (default: 1 hour)

        Detects slow memory leaks and degradation over time.
        """
        self._reset_counters()
        gc.collect()
        memory_start = get_memory_usage_mb()

        start_time = time.perf_counter()
        request_count = 0

        print(f"Starting soak test for {duration_seconds}s...")
        print("Press Ctrl+C to stop early.\n")

        try:
            while (time.perf_counter() - start_time) < duration_seconds:
                self._simulate_inference()
                request_count += 1

                # Progress update every 10 seconds
                elapsed = time.perf_counter() - start_time
                if request_count % 1000 == 0:
                    current_mem = get_memory_usage_mb()
                    print(
                        f"  [{elapsed:.0f}s] "
                        f"Requests: {request_count:,} | "
                        f"Memory: {current_mem:.1f}MB | "
                        f"Throughput: {request_count / elapsed:.1f}/s"
                    )

                # Periodic GC
                if request_count % 5000 == 0:
                    gc.collect()

        except KeyboardInterrupt:
            print("\nSoak test interrupted by user.")

        total_time = time.perf_counter() - start_time
        gc.collect()
        memory_end = get_memory_usage_mb()

        return StressTestResult(
            name=f"Soak Test ({total_time:.0f}s)",
            total_requests=request_count,
            successful_requests=self._success_count,
            failed_requests=self._fail_count,
            total_time_seconds=total_time,
            latencies_ms=self._latencies.copy(),
            errors=self._errors.copy(),
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
        )


def run_all_tests(
    workers: int = 100,
    requests: int = 1000,
    soak: bool = False,
    soak_duration: int = 3600,
) -> int:
    """
    Run all stress tests.

    Returns:
        0 if all tests pass, 1 if any failures
    """
    print("=" * 60)
    print("  ZENITH STRESS TEST SUITE")
    print("=" * 60)

    tester = ZenithStressTester()
    results: list[StressTestResult] = []

    # Test 1: Sequential Baseline
    print("\n[1/4] Running Sequential Baseline...")
    result = tester.test_sequential_baseline(num_requests=requests)
    result.print_report()
    results.append(result)

    # Test 2: Concurrent Stress
    print(f"\n[2/4] Running Concurrent Stress ({workers} workers)...")
    result = tester.test_concurrent_stress(
        num_workers=workers,
        requests_per_worker=requests // workers,
    )
    result.print_report()
    results.append(result)

    # Test 3: Memory Pressure
    print("\n[3/4] Running Memory Pressure Test...")
    result = tester.test_memory_pressure(cycles=100)
    result.print_report()
    results.append(result)

    # Test 4: Soak Test (optional)
    if soak:
        print(f"\n[4/4] Running Soak Test ({soak_duration}s)...")
        result = tester.test_soak(duration_seconds=soak_duration)
        result.print_report()
        results.append(result)
    else:
        print("\n[4/4] Soak Test skipped (use --soak to enable)")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    all_passed = True
    for r in results:
        status = "PASS" if r.success_rate >= 99.0 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {r.name}: {status} ({r.success_rate:.1f}% success)")

    if all_passed:
        print("\n  ALL TESTS PASSED")
        return 0
    else:
        print("\n  SOME TESTS FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Zenith Stress Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of concurrent workers (default: 100)",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=1000,
        help="Total number of requests (default: 1000)",
    )

    parser.add_argument(
        "--soak",
        action="store_true",
        help="Enable soak test",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Soak test duration in seconds (default: 3600)",
    )

    args = parser.parse_args()

    return run_all_tests(
        workers=args.workers,
        requests=args.requests,
        soak=args.soak,
        soak_duration=args.duration,
    )


if __name__ == "__main__":
    sys.exit(main())
