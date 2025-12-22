# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Observability Module

Provides structured logging and metrics collection for the Zenith framework.

Components:
- ZenithLogger: Structured logging with JSON output
- MetricsCollector: Inference metrics and statistics
- Verbosity: Logging verbosity levels
"""

from .logger import (
    Verbosity,
    LogEntry,
    ZenithLogger,
    get_logger,
    set_verbosity,
)

from .metrics import (
    InferenceMetrics,
    MetricsCollector,
    get_metrics_collector,
)

__all__ = [
    # Logger
    "Verbosity",
    "LogEntry",
    "ZenithLogger",
    "get_logger",
    "set_verbosity",
    # Metrics
    "InferenceMetrics",
    "MetricsCollector",
    "get_metrics_collector",
]
