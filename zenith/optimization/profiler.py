"""
Profiler System - Performance Measurement and Analysis

Implements comprehensive profiling for Zenith:
- Per-operation timing
- Memory usage tracking
- Export to JSON and CSV formats

Based on CetakBiru Section 5.1 Phase 2 requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import json
import time
import csv
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from typing import Generator
from pathlib import Path


@dataclass
class OperationMetrics:
    """Metrics for a single operation execution."""

    op_name: str
    op_type: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_bytes: int = 0
    input_shapes: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ProfileSession:
    """A profiling session containing multiple operation measurements."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    operations: list[OperationMetrics] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        """Total session duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def operation_count(self) -> int:
        """Number of operations recorded."""
        return len(self.operations)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory used across all operations."""
        return sum(op.memory_bytes for op in self.operations)

    def summary(self) -> dict:
        """Generate summary statistics."""
        if not self.operations:
            return {"name": self.name, "operations": 0}

        durations = [op.duration_ms for op in self.operations]
        op_types = {}
        for op in self.operations:
            if op.op_type not in op_types:
                op_types[op.op_type] = {"count": 0, "total_ms": 0.0}
            op_types[op.op_type]["count"] += 1
            op_types[op.op_type]["total_ms"] += op.duration_ms

        return {
            "name": self.name,
            "total_duration_ms": self.total_duration_ms,
            "operation_count": self.operation_count,
            "total_memory_bytes": self.total_memory_bytes,
            "avg_op_duration_ms": sum(durations) / len(durations),
            "max_op_duration_ms": max(durations),
            "min_op_duration_ms": min(durations),
            "by_op_type": op_types,
        }


class Profiler:
    """
    Performance profiler for Zenith operations.

    Usage:
        profiler = Profiler()
        with profiler.session("inference") as session:
            with profiler.measure("conv1", "Conv2D"):
                # ... convolution operation
            with profiler.measure("relu1", "ReLU"):
                # ... relu operation

        profiler.export_json("profile.json")
        profiler.export_csv("profile.csv")
    """

    def __init__(self):
        self.sessions: list[ProfileSession] = []
        self._current_session: ProfileSession | None = None
        self._start_time: float = 0.0

    @contextmanager
    def session(self, name: str) -> Generator[ProfileSession, None, None]:
        """
        Start a profiling session.

        Args:
            name: Session name for identification

        Yields:
            ProfileSession object
        """
        session = ProfileSession(name=name, start_time=time.perf_counter())
        self._current_session = session
        try:
            yield session
        finally:
            session.end_time = time.perf_counter()
            self.sessions.append(session)
            self._current_session = None

    @contextmanager
    def measure(
        self,
        op_name: str,
        op_type: str,
        input_shapes: list[list[int]] | None = None,
        output_shapes: list[list[int]] | None = None,
        device: str = "cpu",
    ) -> Generator[OperationMetrics, None, None]:
        """
        Measure a single operation.

        Args:
            op_name: Operation name
            op_type: Operation type (e.g., "Conv2D", "ReLU")
            input_shapes: Input tensor shapes
            output_shapes: Output tensor shapes
            device: Execution device

        Yields:
            OperationMetrics object
        """
        start = time.perf_counter()
        metrics = OperationMetrics(
            op_name=op_name,
            op_type=op_type,
            start_time=start,
            end_time=0.0,
            duration_ms=0.0,
            input_shapes=input_shapes or [],
            output_shapes=output_shapes or [],
            device=device,
        )

        try:
            yield metrics
        finally:
            end = time.perf_counter()
            metrics.end_time = end
            metrics.duration_ms = (end - start) * 1000

            if self._current_session:
                self._current_session.operations.append(metrics)

    def measure_memory(
        self,
        op_name: str,
        memory_bytes: int,
    ) -> None:
        """
        Record memory usage for an operation.

        Args:
            op_name: Operation name
            memory_bytes: Memory used in bytes
        """
        if self._current_session:
            for op in reversed(self._current_session.operations):
                if op.op_name == op_name:
                    op.memory_bytes = memory_bytes
                    break

    def export_json(self, filepath: str | Path) -> None:
        """
        Export profiling data to JSON file.

        Args:
            filepath: Output file path
        """
        data = {
            "sessions": [
                {
                    "name": s.name,
                    "total_duration_ms": s.total_duration_ms,
                    "operation_count": s.operation_count,
                    "operations": [op.to_dict() for op in s.operations],
                    "summary": s.summary(),
                }
                for s in self.sessions
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str | Path) -> None:
        """
        Export profiling data to CSV file.

        Args:
            filepath: Output file path
        """
        fieldnames = [
            "session",
            "op_name",
            "op_type",
            "duration_ms",
            "memory_bytes",
            "device",
            "input_shapes",
            "output_shapes",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for session in self.sessions:
                for op in session.operations:
                    writer.writerow(
                        {
                            "session": session.name,
                            "op_name": op.op_name,
                            "op_type": op.op_type,
                            "duration_ms": f"{op.duration_ms:.4f}",
                            "memory_bytes": op.memory_bytes,
                            "device": op.device,
                            "input_shapes": str(op.input_shapes),
                            "output_shapes": str(op.output_shapes),
                        }
                    )

    def get_summary(self) -> dict:
        """Get summary of all profiling sessions."""
        return {
            "session_count": len(self.sessions),
            "sessions": [s.summary() for s in self.sessions],
        }

    def print_summary(self) -> None:
        """Print profiling summary to console."""
        for session in self.sessions:
            summary = session.summary()
            print(f"\n=== Session: {summary['name']} ===")
            print(f"Total Duration: {summary['total_duration_ms']:.2f} ms")
            print(f"Operations: {summary['operation_count']}")
            print(f"Memory: {summary['total_memory_bytes'] / 1024:.2f} KB")

            if "by_op_type" in summary:
                print("\nBy Operation Type:")
                for op_type, stats in summary["by_op_type"].items():
                    print(
                        f"  {op_type}: {stats['count']} ops, "
                        f"{stats['total_ms']:.2f} ms total"
                    )

    def clear(self) -> None:
        """Clear all profiling data."""
        self.sessions.clear()
        self._current_session = None


# Global profiler instance
_global_profiler: Profiler | None = None


def get_profiler() -> Profiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def reset_profiler() -> None:
    """Reset the global profiler."""
    global _global_profiler
    _global_profiler = Profiler()
