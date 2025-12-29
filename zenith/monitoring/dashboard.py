#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Terminal Dashboard

Real-time TUI monitoring dashboard using Rich library.
Displays latency, memory, throughput metrics with live updates.

Usage:
    zenith dashboard [--host localhost] [--port 8080]

    Or programmatically:
    from zenith.monitoring.dashboard import run_dashboard
    run_dashboard()
"""

import time
import sys
from typing import Optional

# Check for Rich library
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def check_rich_available() -> bool:
    """Check if Rich library is available."""
    return HAS_RICH


class TerminalDashboard:
    """
    Real-time terminal dashboard for Zenith monitoring.

    Features:
    - Latency metrics (P50, P90, P99)
    - Memory usage tracking
    - Throughput display
    - Keyboard controls
    """

    def __init__(self, refresh_rate: float = 1.0):
        """
        Initialize dashboard.

        Args:
            refresh_rate: Update interval in seconds
        """
        if not HAS_RICH:
            raise ImportError(
                "Rich library not installed. Install with: pip install rich"
            )

        self.console = Console()
        self.refresh_rate = refresh_rate
        self._running = False

    def _make_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="latency"),
            Layout(name="memory"),
            Layout(name="throughput"),
        )

        return layout

    def _get_header(self) -> Panel:
        """Create header panel."""
        header_text = Text("ZENITH MONITORING DASHBOARD", style="bold white")
        header_text.stylize("bold cyan")
        return Panel(
            Align.center(header_text),
            style="blue",
        )

    def _get_footer(self) -> Panel:
        """Create footer panel with controls."""
        footer = Text()
        footer.append("[q]", style="bold yellow")
        footer.append(" Quit  ", style="white")
        footer.append("[r]", style="bold yellow")
        footer.append(" Reset  ", style="white")
        footer.append("[h]", style="bold yellow")
        footer.append(" Help", style="white")
        return Panel(Align.center(footer), style="dim")

    def _get_latency_panel(self, metrics: dict) -> Panel:
        """Create latency metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        p50 = metrics.get("latency_p50_ms", 0)
        p90 = metrics.get("latency_p90_ms", 0)
        p99 = metrics.get("latency_p99_ms", 0)
        mean = metrics.get("latency_mean_ms", 0)

        table.add_row("P50", f"{p50:.2f} ms")
        table.add_row("P90", f"{p90:.2f} ms")
        table.add_row("P99", f"{p99:.2f} ms")
        table.add_row("Mean", f"{mean:.2f} ms")

        # Simple sparkline
        sparkline = self._make_sparkline([p50, p90, p99])

        content = Table.grid(padding=1)
        content.add_row(table)
        content.add_row(Text(sparkline, style="green"))

        return Panel(content, title="Latency", border_style="cyan")

    def _get_memory_panel(self, metrics: dict) -> Panel:
        """Create memory metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        mean_mb = metrics.get("memory_mean_mb", 0)
        max_mb = metrics.get("memory_max_mb", 0)

        table.add_row("Current", f"{mean_mb:.1f} MB")
        table.add_row("Peak", f"{max_mb:.1f} MB")

        # Memory bar
        if max_mb > 0:
            pct = min(mean_mb / max_mb * 100, 100)
            bar = self._make_bar(pct)
            table.add_row("Usage", bar)

        return Panel(table, title="Memory", border_style="magenta")

    def _get_throughput_panel(self, metrics: dict) -> Panel:
        """Create throughput metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        total = metrics.get("total_inferences", 0)
        errors = metrics.get("error_count", 0)
        qps = metrics.get("throughput_qps", 0)

        error_rate = (errors / total * 100) if total > 0 else 0
        error_style = "red" if error_rate > 1 else "green"

        table.add_row("Total", f"{total:,}")
        table.add_row(
            "Errors", Text(f"{errors} ({error_rate:.2f}%)", style=error_style)
        )
        table.add_row("QPS", f"{qps:.1f}")

        return Panel(table, title="Throughput", border_style="yellow")

    def _make_sparkline(self, values: list, width: int = 20) -> str:
        """Create a simple sparkline from values."""
        if not values or all(v == 0 for v in values):
            return " " * width

        chars = " ▁▂▃▄▅▆▇█"
        max_val = max(values) if max(values) > 0 else 1

        result = []
        for v in values:
            idx = int((v / max_val) * (len(chars) - 1))
            result.append(chars[idx])

        return "".join(result)

    def _make_bar(self, percentage: float, width: int = 15) -> str:
        """Create a progress bar."""
        filled = int(width * percentage / 100)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"{bar} {percentage:.0f}%"

    def _generate_display(self, metrics: dict) -> Layout:
        """Generate the full dashboard display."""
        layout = self._make_layout()

        layout["header"].update(self._get_header())
        layout["footer"].update(self._get_footer())
        layout["latency"].update(self._get_latency_panel(metrics))
        layout["memory"].update(self._get_memory_panel(metrics))
        layout["throughput"].update(self._get_throughput_panel(metrics))

        return layout

    def run(self, collector=None) -> None:
        """
        Run the dashboard.

        Args:
            collector: Optional MetricsCollector instance
        """
        if collector is None:
            from zenith.observability import get_metrics_collector

            collector = get_metrics_collector()

        self._running = True

        try:
            with Live(
                self._generate_display(collector.get_summary()),
                console=self.console,
                refresh_per_second=1 / self.refresh_rate,
                screen=True,
            ) as live:
                while self._running:
                    metrics = collector.get_summary()
                    live.update(self._generate_display(metrics))
                    time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False


def run_dashboard(
    host: Optional[str] = None,
    port: Optional[int] = None,
    refresh_rate: float = 1.0,
) -> None:
    """
    Run the terminal dashboard.

    Args:
        host: Remote host to connect to (optional)
        port: Remote port (optional)
        refresh_rate: Update interval in seconds
    """
    if not HAS_RICH:
        print("Error: Rich library not installed.")
        print("Install with: pip install rich")
        sys.exit(1)

    dashboard = TerminalDashboard(refresh_rate=refresh_rate)

    print("Starting Zenith Dashboard...")
    print("Press Ctrl+C to exit")

    try:
        dashboard.run()
    except Exception as e:
        print(f"Dashboard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_dashboard()
