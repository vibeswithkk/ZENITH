# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Observability Module

Provides structured logging, metrics collection, GPU monitoring,
event tracking, and request context for the Zenith framework.

Components:
- ZenithLogger: Structured logging with JSON output
- MetricsCollector: Inference metrics and statistics
- GPUMetricsCollector: Real-time GPU metrics (requires pynvml)
- EventEmitter: Custom event system for tracking
- ContextManager: Correlation IDs and request context
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
    reset_metrics_collector,
)

from .gpu_metrics import (
    GPUStats,
    GPUMetricsCollector,
    is_available as gpu_is_available,
    get_current as gpu_get_current,
    get_memory_info as gpu_get_memory_info,
    get_utilization as gpu_get_utilization,
)

from .events import (
    Event,
    EventEmitter,
    EventNames,
    on as event_on,
    off as event_off,
    emit as event_emit,
    enable_history as event_enable_history,
    get_history as event_get_history,
)

from .context import (
    RequestContext,
    ContextManager,
    get_correlation_id,
    set_correlation_id,
    get_context,
    new_context,
    span,
)

# Convenience imports for submodules
from . import gpu_metrics
from . import events
from . import context

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
    "reset_metrics_collector",
    # GPU Metrics
    "GPUStats",
    "GPUMetricsCollector",
    "gpu_is_available",
    "gpu_get_current",
    "gpu_get_memory_info",
    "gpu_get_utilization",
    "gpu_metrics",
    # Events
    "Event",
    "EventEmitter",
    "EventNames",
    "event_on",
    "event_off",
    "event_emit",
    "event_enable_history",
    "event_get_history",
    "events",
    # Context
    "RequestContext",
    "ContextManager",
    "get_correlation_id",
    "set_correlation_id",
    "get_context",
    "new_context",
    "span",
    "context",
]
