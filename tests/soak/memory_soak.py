#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Memory Leak Soak Test

Long-running test with automated memory profiling for leak detection.

Features:
- Configurable duration (1h to 72h)
- tracemalloc-based memory tracking
- Periodic snapshots with trend analysis
- Leak threshold alerting
- JSON report generation

Reference:
    Google SRE Testing for Reliability
    Python tracemalloc documentation

Usage:
    # Quick test (5 minutes)
    python tests/soak/memory_soak.py --duration 5m

    # 1 hour test
    python tests/soak/memory_soak.py --duration 1h

    # 72 hour soak test
    python tests/soak/memory_soak.py --duration 72h --output soak_report.json
"""

import argparse
import gc
import json
import resource
import sys
import time
import tracemalloc
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable


@dataclass
class MemorySnapshot:
    """Single memory snapshot."""

    timestamp: str
    elapsed_seconds: float
    rss_mb: float
    tracemalloc_current_mb: float
    tracemalloc_peak_mb: float
    gc_objects: int
    gc_collections: dict = field(default_factory=dict)


@dataclass
class SoakTestConfig:
    """Configuration for soak test."""

    duration_seconds: float = 3600.0  # 1 hour default
    snapshot_interval_seconds: float = 60.0  # Every minute
    workload_interval_seconds: float = 1.0  # Workload every second
    memory_growth_threshold_mb_per_hour: float = 10.0
    object_growth_threshold_per_hour: int = 1000

    def __post_init__(self):
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be positive")


@dataclass
class SoakTestResult:
    """Complete soak test result."""

    start_time: str
    end_time: str
    duration_seconds: float
    passed: bool
    snapshots: List[dict] = field(default_factory=list)
    memory_growth_mb: float = 0.0
    memory_growth_rate_mb_per_hour: float = 0.0
    object_growth: int = 0
    object_growth_rate_per_hour: float = 0.0
    peak_memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Export result as JSON."""
        return json.dumps(asdict(self), indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("  MEMORY SOAK TEST RESULT")
        print("=" * 60)
        print(f"  Duration: {self.duration_seconds / 3600:.2f} hours")
        print(f"  Peak Memory: {self.peak_memory_mb:.2f} MB")
        print(f"  Memory Growth: {self.memory_growth_mb:.2f} MB")
        print(f"  Growth Rate: {self.memory_growth_rate_mb_per_hour:.2f} MB/hour")
        print(f"  Object Growth: {self.object_growth}")
        print(f"  Object Rate: {self.object_growth_rate_per_hour:.0f}/hour")
        print("-" * 60)

        if self.errors:
            print(f"  Errors: {len(self.errors)}")
            for err in self.errors[:5]:
                print(f"    - {err}")

        if self.passed:
            print("  STATUS: PASSED")
        else:
            print("  STATUS: FAILED")
        print("=" * 60)


class MemoryProfiler:
    """
    Automated memory profiler for soak testing.

    Uses tracemalloc and resource module for comprehensive tracking.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshots: List[MemorySnapshot] = []
        self._started = False
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start memory profiling."""
        with self._lock:
            if not self._started:
                tracemalloc.start()
                self._started = True
                self._start_time = time.time()
                self._snapshots.clear()

    def stop(self) -> None:
        """Stop memory profiling."""
        with self._lock:
            if self._started:
                tracemalloc.stop()
                self._started = False

    def take_snapshot(self) -> MemorySnapshot:
        """Take current memory snapshot."""
        with self._lock:
            elapsed = time.time() - (self._start_time or time.time())

            # RSS from resource module
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_mb = usage.ru_maxrss / 1024  # KB to MB on Linux

            # tracemalloc stats
            if self._started:
                current, peak = tracemalloc.get_traced_memory()
                current_mb = current / (1024 * 1024)
                peak_mb = peak / (1024 * 1024)
            else:
                current_mb = 0.0
                peak_mb = 0.0

            # GC stats
            gc.collect()
            gc_objects = len(gc.get_objects())
            gc_stats = {
                f"gen{i}": gc.get_count()[i] if i < len(gc.get_count()) else 0
                for i in range(3)
            }

            snapshot = MemorySnapshot(
                timestamp=datetime.now().isoformat(),
                elapsed_seconds=elapsed,
                rss_mb=rss_mb,
                tracemalloc_current_mb=current_mb,
                tracemalloc_peak_mb=peak_mb,
                gc_objects=gc_objects,
                gc_collections=gc_stats,
            )

            self._snapshots.append(snapshot)
            return snapshot

    @property
    def snapshots(self) -> List[MemorySnapshot]:
        """Get all snapshots."""
        with self._lock:
            return list(self._snapshots)


class SoakTestRunner:
    """
    Runs long-duration soak tests with memory profiling.

    Thread-safe implementation for concurrent workloads.
    """

    def __init__(self, config: Optional[SoakTestConfig] = None):
        self.config = config or SoakTestConfig()
        self.profiler = MemoryProfiler()
        self._stop_flag = threading.Event()
        self._workload: Optional[Callable] = None

    def set_workload(self, workload: Callable) -> None:
        """
        Set the workload function to run during soak test.

        Args:
            workload: Callable that simulates production load
        """
        self._workload = workload

    def _default_workload(self) -> None:
        """Default workload using Zenith components."""
        try:
            from zenith.observability import MetricsCollector, InferenceMetrics

            collector = MetricsCollector()

            for i in range(10):
                collector.record_inference(
                    InferenceMetrics(
                        latency_ms=float(i),
                        memory_mb=float(i * 10),
                    )
                )

            summary = collector.get_summary()

            # Cleanup
            del collector
            del summary

        except ImportError:
            # Fallback if Zenith not available
            data = [i**2 for i in range(1000)]
            del data

    def run(self) -> SoakTestResult:
        """
        Run the soak test.

        Returns:
            SoakTestResult with all metrics
        """
        start_time = datetime.now()
        workload = self._workload or self._default_workload

        print("=" * 60)
        print("  MEMORY SOAK TEST")
        print("=" * 60)
        print(f"  Duration: {self.config.duration_seconds / 3600:.2f} hours")
        print(f"  Snapshot interval: {self.config.snapshot_interval_seconds}s")
        print(f"  Started: {start_time.isoformat()}")
        print("=" * 60)

        # Start profiling
        self.profiler.start()

        # Take initial snapshot
        initial_snapshot = self.profiler.take_snapshot()

        errors: List[str] = []
        elapsed = 0.0
        last_snapshot_time = 0.0
        iteration = 0

        try:
            while (
                elapsed < self.config.duration_seconds and not self._stop_flag.is_set()
            ):
                # Run workload
                try:
                    workload()
                except Exception as e:
                    errors.append(f"Workload error at {elapsed:.0f}s: {e}")

                # Take periodic snapshot
                if (
                    elapsed - last_snapshot_time
                    >= self.config.snapshot_interval_seconds
                ):
                    snapshot = self.profiler.take_snapshot()
                    last_snapshot_time = elapsed

                    # Print progress
                    progress = (elapsed / self.config.duration_seconds) * 100
                    print(
                        f"  [{progress:5.1f}%] "
                        f"RSS: {snapshot.rss_mb:.1f}MB, "
                        f"Objects: {snapshot.gc_objects}"
                    )

                # Wait for next iteration
                time.sleep(self.config.workload_interval_seconds)
                elapsed = time.time() - start_time.timestamp()
                iteration += 1

        except KeyboardInterrupt:
            print("\n  [INTERRUPTED] Stopping soak test...")
            errors.append("Test interrupted by user")

        # Take final snapshot
        final_snapshot = self.profiler.take_snapshot()

        # Stop profiling
        self.profiler.stop()

        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()

        # Calculate metrics - use second snapshot as baseline to skip warmup
        # First snapshot includes import overhead, second is after stabilization
        snapshots = self.profiler.snapshots
        baseline_snapshot = snapshots[1] if len(snapshots) > 1 else initial_snapshot

        memory_growth = final_snapshot.rss_mb - baseline_snapshot.rss_mb
        object_growth = final_snapshot.gc_objects - baseline_snapshot.gc_objects

        hours = actual_duration / 3600
        memory_rate = memory_growth / hours if hours > 0 else 0
        object_rate = object_growth / hours if hours > 0 else 0

        # Find peak
        peak_memory = max(s.rss_mb for s in self.profiler.snapshots)

        # Determine pass/fail
        passed = (
            memory_rate <= self.config.memory_growth_threshold_mb_per_hour
            and object_rate <= self.config.object_growth_threshold_per_hour
            and len(errors) == 0
        )

        result = SoakTestResult(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=actual_duration,
            passed=passed,
            snapshots=[asdict(s) for s in self.profiler.snapshots],
            memory_growth_mb=memory_growth,
            memory_growth_rate_mb_per_hour=memory_rate,
            object_growth=object_growth,
            object_growth_rate_per_hour=object_rate,
            peak_memory_mb=peak_memory,
            errors=errors,
        )

        return result

    def stop(self) -> None:
        """Signal the test to stop."""
        self._stop_flag.set()


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to seconds.

    Supports: 5m, 1h, 72h, 300s
    """
    duration_str = duration_str.strip().lower()

    if duration_str.endswith("s"):
        return float(duration_str[:-1])
    elif duration_str.endswith("m"):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith("h"):
        return float(duration_str[:-1]) * 3600
    else:
        return float(duration_str)


def main():
    parser = argparse.ArgumentParser(
        description="Memory Leak Soak Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--duration",
        type=str,
        default="5m",
        help="Test duration (e.g., 5m, 1h, 72h)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="30s",
        help="Snapshot interval (e.g., 30s, 1m)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON report path",
    )

    parser.add_argument(
        "--threshold-memory",
        type=float,
        default=10.0,
        help="Memory growth threshold (MB/hour)",
    )

    parser.add_argument(
        "--threshold-objects",
        type=int,
        default=1000,
        help="Object growth threshold (per hour)",
    )

    args = parser.parse_args()

    # Parse durations
    duration_seconds = parse_duration(args.duration)
    interval_seconds = parse_duration(args.interval)

    # Create config
    config = SoakTestConfig(
        duration_seconds=duration_seconds,
        snapshot_interval_seconds=interval_seconds,
        memory_growth_threshold_mb_per_hour=args.threshold_memory,
        object_growth_threshold_per_hour=args.threshold_objects,
    )

    # Run test
    runner = SoakTestRunner(config)
    result = runner.run()

    # Print summary
    result.print_summary()

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result.to_json())
        print(f"\nReport saved: {output_path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
