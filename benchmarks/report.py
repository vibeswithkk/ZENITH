# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Benchmark Report Generator

Generates performance reports from benchmark results.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json


def generate_report(
    results: List[Any],
    output_path: str = "BENCHMARK_REPORT.md",
    title: str = "Zenith Performance Benchmark Report",
) -> str:
    """
    Generate a markdown benchmark report.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path to save the report
        title: Report title

    Returns:
        Generated markdown content
    """
    lines = [
        f"# {title}",
        "",
        f"**Generated:** {_get_timestamp()}",
        f"**Total Configurations:** {len(results)}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Group by model
    by_model = _group_by(results, lambda r: r.model_name)

    for model_name, model_results in by_model.items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(
            "| Backend | Precision | Batch | Latency (ms) | Throughput (s/s) | Memory (MB) |"
        )
        lines.append(
            "|---------|-----------|-------|--------------|------------------|-------------|"
        )

        for r in model_results:
            lines.append(
                f"| {r.backend} | {r.precision} | {r.batch_size} | "
                f"{r.latency_mean:.2f} | {r.throughput:.1f} | {r.peak_memory:.1f} |"
            )

        lines.append("")

    # Add detailed statistics
    lines.extend(
        [
            "---",
            "",
            "## Detailed Latency Statistics",
            "",
            "| Model | Backend | P50 (ms) | P95 (ms) | P99 (ms) |",
            "|-------|---------|----------|----------|----------|",
        ]
    )

    for r in results:
        lines.append(
            f"| {r.model_name} | {r.backend} | "
            f"{r.latency_p50:.2f} | {r.latency_p95:.2f} | {r.latency_p99:.2f} |"
        )

    lines.append("")

    content = "\n".join(lines)

    # Write to file
    Path(output_path).write_text(content)

    return content


def _group_by(items: List[Any], key_fn) -> Dict[str, List[Any]]:
    """Group items by key function."""
    result = {}
    for item in items:
        key = key_fn(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def _get_timestamp() -> str:
    """Get current timestamp."""
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S")
