#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Benchmark Report Generator for Zenith

Generates comprehensive benchmark reports including:
- Markdown documentation with methodology and results
- Performance charts (latency, throughput, comparison)
- JSON data export for further analysis

Dependencies:
- matplotlib (optional, for chart generation)
- numpy (required, for statistics)
"""

import json
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger("zenith.benchmarks.report")


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Zenith Benchmark Report"
    output_dir: str = "docs/benchmarks"
    report_filename: str = "BENCHMARK_REPORT.md"
    chart_format: str = "png"
    chart_dpi: int = 150
    include_charts: bool = True
    include_methodology: bool = True
    include_system_info: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def check_matplotlib_available() -> bool:
    """Check if matplotlib is available for chart generation."""
    try:
        import matplotlib

        return True
    except ImportError:
        return False


def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["pytorch_version"] = "not installed"
        info["cuda_available"] = False

    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = "not installed"

    return info


class ChartGenerator:
    """
    Generates benchmark visualization charts.

    Creates latency distribution, throughput comparison, and speedup charts.
    Gracefully handles missing matplotlib dependency.
    """

    def __init__(self, output_dir: str, dpi: int = 150, chart_format: str = "png"):
        self._output_dir = output_dir
        self._dpi = dpi
        self._format = chart_format
        self._matplotlib_available = check_matplotlib_available()

        if not self._matplotlib_available:
            logger.warning("matplotlib not available. Charts will not be generated.")

    def generate_latency_chart(
        self,
        results: list,
        filename: str = "latency_comparison.png",
    ) -> Optional[str]:
        """
        Generate latency comparison bar chart.

        Args:
            results: List of BenchmarkResult objects.
            filename: Output filename.

        Returns:
            Path to generated chart, or None if matplotlib unavailable.
        """
        if not self._matplotlib_available:
            return None

        import matplotlib.pyplot as plt

        models = []
        p50_values = []
        p90_values = []
        p99_values = []

        for r in results:
            label = f"{r.model_name}\nbs={r.batch_size}"
            models.append(label)
            p50_values.append(r.latency_p50_ms)
            p90_values.append(r.latency_p90_ms)
            p99_values.append(r.latency_p99_ms)

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width, p50_values, width, label="P50", color="#2ecc71")
        bars2 = ax.bar(x, p90_values, width, label="P90", color="#f39c12")
        bars3 = ax.bar(x + width, p99_values, width, label="P99", color="#e74c3c")

        ax.set_xlabel("Model Configuration")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Distribution by Model")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        output_path = os.path.join(self._output_dir, filename)
        plt.savefig(output_path, dpi=self._dpi, format=self._format)
        plt.close()

        logger.info(f"Generated latency chart: {output_path}")
        return output_path

    def generate_throughput_chart(
        self,
        results: list,
        filename: str = "throughput_comparison.png",
    ) -> Optional[str]:
        """
        Generate throughput comparison bar chart.

        Args:
            results: List of BenchmarkResult objects.
            filename: Output filename.

        Returns:
            Path to generated chart, or None if matplotlib unavailable.
        """
        if not self._matplotlib_available:
            return None

        import matplotlib.pyplot as plt

        models = []
        qps_values = []
        samples_per_sec = []

        for r in results:
            label = f"{r.model_name}\nbs={r.batch_size}"
            models.append(label)
            qps_values.append(r.throughput_qps)
            samples_per_sec.append(r.throughput_samples_per_sec)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(models))

        bars1 = ax1.bar(x, qps_values, color="#3498db")
        ax1.set_xlabel("Model Configuration")
        ax1.set_ylabel("Queries per Second")
        ax1.set_title("Throughput (QPS)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        bars2 = ax2.bar(x, samples_per_sec, color="#9b59b6")
        ax2.set_xlabel("Model Configuration")
        ax2.set_ylabel("Samples per Second")
        ax2.set_title("Throughput (Samples/sec)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        output_path = os.path.join(self._output_dir, filename)
        plt.savefig(output_path, dpi=self._dpi, format=self._format)
        plt.close()

        logger.info(f"Generated throughput chart: {output_path}")
        return output_path

    def generate_speedup_chart(
        self,
        zenith_results: list,
        baseline_results: list,
        filename: str = "speedup_comparison.png",
    ) -> Optional[str]:
        """
        Generate speedup comparison chart (Zenith vs baseline).

        Args:
            zenith_results: Results from Zenith benchmark.
            baseline_results: Results from baseline (e.g., PyTorch).
            filename: Output filename.

        Returns:
            Path to generated chart, or None if matplotlib unavailable.
        """
        if not self._matplotlib_available:
            return None

        if len(zenith_results) != len(baseline_results):
            logger.warning("Result list lengths do not match. Skipping speedup chart.")
            return None

        import matplotlib.pyplot as plt

        models = []
        speedups = []

        for z, b in zip(zenith_results, baseline_results):
            label = f"{z.model_name}\nbs={z.batch_size}"
            models.append(label)
            if z.latency_p50_ms > 0:
                speedup = b.latency_p50_ms / z.latency_p50_ms
            else:
                speedup = 0.0
            speedups.append(speedup)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        colors = ["#27ae60" if s >= 1.0 else "#e74c3c" for s in speedups]

        bars = ax.bar(x, speedups, color=colors)

        ax.axhline(y=1.0, color="#7f8c8d", linestyle="--", label="Baseline (1.0x)")
        ax.set_xlabel("Model Configuration")
        ax.set_ylabel("Speedup (Higher is Better)")
        ax.set_title("Zenith vs Baseline Speedup")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.annotate(
                f"{speedup:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        output_path = os.path.join(self._output_dir, filename)
        plt.savefig(output_path, dpi=self._dpi, format=self._format)
        plt.close()

        logger.info(f"Generated speedup chart: {output_path}")
        return output_path


class MarkdownGenerator:
    """
    Generates markdown documentation for benchmark reports.

    Creates structured documentation with methodology, results, and analysis.
    """

    def __init__(self, config: ReportConfig):
        self._config = config

    def generate_header(self) -> str:
        """Generate report header with metadata."""
        lines = [
            f"# {self._config.title}",
            "",
            f"**Generated:** {self._config.timestamp}",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    def generate_methodology_section(self) -> str:
        """Generate benchmark methodology documentation."""
        lines = [
            "## Benchmark Methodology",
            "",
            "This benchmark suite follows MLPerf Inference methodology for "
            "consistent and reproducible performance measurements.",
            "",
            "### Scenarios",
            "",
            "| Scenario | Description | Primary Metric |",
            "|----------|-------------|----------------|",
            "| Single-Stream | Per-query latency measurement | P90 Latency |",
            "| Offline | Maximum throughput | Samples/sec |",
            "| Server | Latency under load | P99 Latency at target QPS |",
            "",
            "### Metrics",
            "",
            "- **P50/P90/P99 Latency**: Percentile latencies in milliseconds",
            "- **Throughput (QPS)**: Queries processed per second",
            "- **Throughput (Samples/sec)**: Total samples processed per second",
            "- **Quality Score**: Accuracy relative to reference implementation",
            "",
            "### Measurement Protocol",
            "",
            "1. **Warmup Phase**: Initial runs excluded from timing",
            "2. **Timed Phase**: Multiple iterations with precise timing",
            "3. **Synchronization**: GPU sync before/after each measurement",
            "4. **Quality Verification**: Output comparison with reference",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    def generate_system_info_section(self, system_info: dict) -> str:
        """Generate system information section."""
        lines = [
            "## System Information",
            "",
            "| Property | Value |",
            "|----------|-------|",
        ]

        for key, value in system_info.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"| {formatted_key} | {value} |")

        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def generate_results_section(
        self,
        results: list,
        section_title: str = "Benchmark Results",
    ) -> str:
        """Generate results table section."""
        from benchmarks.mlperf_suite import generate_results_table

        lines = [
            f"## {section_title}",
            "",
            generate_results_table(results),
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    def generate_comparison_section(
        self,
        zenith_results: list,
        baseline_results: list,
        baseline_name: str = "PyTorch",
    ) -> str:
        """Generate comparison section with speedup analysis."""
        from benchmarks.mlperf_suite import compare_results

        lines = [
            f"## Performance Comparison: Zenith vs {baseline_name}",
            "",
            compare_results(zenith_results, baseline_results),
            "",
        ]

        if zenith_results and baseline_results:
            speedups = []
            for z, b in zip(zenith_results, baseline_results):
                if z.latency_p50_ms > 0:
                    speedups.append(b.latency_p50_ms / z.latency_p50_ms)

            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                min_speedup = np.min(speedups)

                lines.extend(
                    [
                        "",
                        "### Speedup Summary",
                        "",
                        f"- **Average Speedup**: {avg_speedup:.2f}x",
                        f"- **Maximum Speedup**: {max_speedup:.2f}x",
                        f"- **Minimum Speedup**: {min_speedup:.2f}x",
                        "",
                    ]
                )

        lines.extend(["---", ""])
        return "\n".join(lines)

    def generate_charts_section(self, chart_paths: list) -> str:
        """Generate section with embedded chart references."""
        if not chart_paths:
            return ""

        lines = [
            "## Performance Charts",
            "",
        ]

        for path in chart_paths:
            if path:
                filename = os.path.basename(path)
                chart_name = filename.replace("_", " ").replace(".png", "").title()
                lines.append(f"### {chart_name}")
                lines.append("")
                lines.append(f"![{chart_name}]({filename})")
                lines.append("")

        lines.extend(["---", ""])
        return "\n".join(lines)

    def generate_summary_section(self, results: list) -> str:
        """Generate executive summary section."""
        if not results:
            return ""

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.quality_passed)
        avg_latency = np.mean([r.latency_p50_ms for r in results])
        total_qps = sum([r.throughput_qps for r in results])

        lines = [
            "## Executive Summary",
            "",
            f"- **Total Benchmark Configurations**: {total_tests}",
            f"- **Quality Verification Passed**: {passed_tests}/{total_tests}",
            f"- **Average P50 Latency**: {avg_latency:.2f} ms",
            f"- **Combined Throughput**: {total_qps:.1f} QPS",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)


class BenchmarkReportGenerator:
    """
    Main report generator that orchestrates markdown and chart generation.

    Example:
        generator = BenchmarkReportGenerator()
        generator.generate(
            zenith_results=my_zenith_results,
            baseline_results=my_baseline_results,
        )
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self._config = config or ReportConfig()
        self._ensure_output_dir()
        self._chart_generator = ChartGenerator(
            output_dir=self._config.output_dir,
            dpi=self._config.chart_dpi,
            chart_format=self._config.chart_format,
        )
        self._markdown_generator = MarkdownGenerator(self._config)

    def _ensure_output_dir(self) -> None:
        """Create output directory if it does not exist."""
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        zenith_results: list,
        baseline_results: Optional[list] = None,
        baseline_name: str = "PyTorch",
    ) -> str:
        """
        Generate complete benchmark report.

        Args:
            zenith_results: Results from Zenith benchmark.
            baseline_results: Optional results from baseline for comparison.
            baseline_name: Name of baseline framework.

        Returns:
            Path to generated report file.
        """
        sections = []

        sections.append(self._markdown_generator.generate_header())

        if self._config.include_methodology:
            sections.append(self._markdown_generator.generate_methodology_section())

        if self._config.include_system_info:
            system_info = get_system_info()
            sections.append(
                self._markdown_generator.generate_system_info_section(system_info)
            )

        sections.append(
            self._markdown_generator.generate_summary_section(zenith_results)
        )

        sections.append(
            self._markdown_generator.generate_results_section(
                zenith_results, "Zenith Benchmark Results"
            )
        )

        if baseline_results:
            sections.append(
                self._markdown_generator.generate_results_section(
                    baseline_results, f"{baseline_name} Baseline Results"
                )
            )
            sections.append(
                self._markdown_generator.generate_comparison_section(
                    zenith_results, baseline_results, baseline_name
                )
            )

        chart_paths = []
        if self._config.include_charts:
            chart_paths.append(
                self._chart_generator.generate_latency_chart(zenith_results)
            )
            chart_paths.append(
                self._chart_generator.generate_throughput_chart(zenith_results)
            )
            if baseline_results:
                chart_paths.append(
                    self._chart_generator.generate_speedup_chart(
                        zenith_results, baseline_results
                    )
                )

            valid_charts = [p for p in chart_paths if p is not None]
            if valid_charts:
                sections.append(
                    self._markdown_generator.generate_charts_section(valid_charts)
                )

        report_content = "\n".join(sections)
        report_path = os.path.join(
            self._config.output_dir, self._config.report_filename
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Generated benchmark report: {report_path}")
        return report_path

    def export_json(
        self,
        zenith_results: list,
        baseline_results: Optional[list] = None,
        filename: str = "benchmark_results.json",
    ) -> str:
        """
        Export results to JSON for further analysis.

        Args:
            zenith_results: Results from Zenith benchmark.
            baseline_results: Optional baseline results.
            filename: Output JSON filename.

        Returns:
            Path to generated JSON file.
        """
        data = {
            "timestamp": self._config.timestamp,
            "system_info": get_system_info(),
            "zenith_results": [r.to_dict() for r in zenith_results],
        }

        if baseline_results:
            data["baseline_results"] = [r.to_dict() for r in baseline_results]

        output_path = os.path.join(self._config.output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported results to JSON: {output_path}")
        return output_path


def generate_report(
    zenith_results: list,
    baseline_results: Optional[list] = None,
    output_dir: str = "docs/benchmarks",
    include_charts: bool = True,
) -> str:
    """
    Convenience function to generate a benchmark report.

    Args:
        zenith_results: List of BenchmarkResult from Zenith.
        baseline_results: Optional list of BenchmarkResult from baseline.
        output_dir: Directory to save report files.
        include_charts: Whether to generate charts.

    Returns:
        Path to the generated report.
    """
    config = ReportConfig(
        output_dir=output_dir,
        include_charts=include_charts,
    )
    generator = BenchmarkReportGenerator(config)
    report_path = generator.generate(zenith_results, baseline_results)
    generator.export_json(zenith_results, baseline_results)
    return report_path


if __name__ == "__main__":
    from benchmarks.mlperf_suite import BenchmarkResult

    sample_results = [
        BenchmarkResult(
            model_name="bert-base",
            scenario="single-stream",
            batch_size=1,
            sequence_length=128,
            precision="fp32",
            latency_p50_ms=10.5,
            latency_p90_ms=12.3,
            latency_p99_ms=15.8,
            throughput_qps=95.2,
            throughput_samples_per_sec=95.2,
            quality_passed=True,
        ),
        BenchmarkResult(
            model_name="bert-base",
            scenario="single-stream",
            batch_size=8,
            sequence_length=128,
            precision="fp32",
            latency_p50_ms=45.2,
            latency_p90_ms=52.1,
            latency_p99_ms=68.4,
            throughput_qps=22.1,
            throughput_samples_per_sec=176.8,
            quality_passed=True,
        ),
    ]

    baseline = [
        BenchmarkResult(
            model_name="bert-base",
            scenario="single-stream",
            batch_size=1,
            sequence_length=128,
            precision="fp32",
            latency_p50_ms=12.8,
            latency_p90_ms=15.2,
            latency_p99_ms=19.1,
            throughput_qps=78.1,
            throughput_samples_per_sec=78.1,
            quality_passed=True,
        ),
        BenchmarkResult(
            model_name="bert-base",
            scenario="single-stream",
            batch_size=8,
            sequence_length=128,
            precision="fp32",
            latency_p50_ms=55.6,
            latency_p90_ms=62.3,
            latency_p99_ms=78.2,
            throughput_qps=18.0,
            throughput_samples_per_sec=144.0,
            quality_passed=True,
        ),
    ]

    report_path = generate_report(sample_results, baseline)
    print(f"Report generated: {report_path}")
