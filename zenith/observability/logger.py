# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Structured Logger for Zenith

Provides structured logging with JSON output format for production environments.

Inspired by:
- OpenTelemetry structured logging
- Google Cloud Logging best practices
- Python logging module patterns

Example:
    from zenith.observability import ZenithLogger, Verbosity

    logger = ZenithLogger.get()
    logger.set_verbosity(Verbosity.INFO)
    logger.info("Model compiled", component="compiler", model_name="bert")
"""

import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import IntEnum
from typing import Optional, TextIO


class Verbosity(IntEnum):
    """
    Logging verbosity levels.

    Uses IntEnum for numeric comparison (e.g., if verbosity >= INFO).
    """

    SILENT = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


@dataclass
class LogEntry:
    """
    Structured log entry.

    Attributes:
        level: Log level (ERROR, WARNING, INFO, DEBUG)
        message: Log message
        timestamp: ISO format timestamp
        component: Source component (compiler, runtime, kernel)
        model_name: Optional model name
        operation: Optional operation name
        duration_ms: Optional duration in milliseconds
        memory_mb: Optional memory usage in megabytes
        extra: Additional context fields
    """

    level: str
    message: str
    timestamp: str
    component: str = "zenith"
    model_name: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if not data.get("extra"):
            data.pop("extra", None)
        return json.dumps(data, default=str)

    def to_text(self) -> str:
        """Convert to human-readable text format."""
        parts = [
            f"[{self.level}]",
            f"[{self.component}]",
            self.message,
        ]
        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.2f}ms)")
        if self.memory_mb is not None:
            parts.append(f"[{self.memory_mb:.1f}MB]")
        return " ".join(parts)


class ZenithLogger:
    """
    Structured logger for Zenith.

    Singleton pattern ensures consistent logging configuration across the framework.

    Example:
        logger = ZenithLogger.get()
        logger.set_verbosity(Verbosity.DEBUG)
        logger.info("Processing started", component="runtime")
    """

    _instance: Optional["ZenithLogger"] = None

    def __init__(self):
        """Initialize logger with default settings."""
        self._verbosity = Verbosity.INFO
        self._output: TextIO = sys.stderr
        self._json_format = False
        self._handlers: list = []

        env_verbosity = os.environ.get("ZENITH_VERBOSITY")
        if env_verbosity is not None:
            try:
                self._verbosity = Verbosity(int(env_verbosity))
            except (ValueError, KeyError):
                pass

    @classmethod
    def get(cls) -> "ZenithLogger":
        """Get the singleton logger instance."""
        if cls._instance is None:
            cls._instance = ZenithLogger()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def set_verbosity(self, level: int) -> None:
        """
        Set verbosity level.

        Args:
            level: Verbosity level (0-4 or Verbosity enum)
        """
        if isinstance(level, Verbosity):
            self._verbosity = level
        else:
            self._verbosity = Verbosity(max(0, min(4, level)))

    def get_verbosity(self) -> Verbosity:
        """Get current verbosity level."""
        return self._verbosity

    def set_json_format(self, enabled: bool) -> None:
        """Enable or disable JSON output format."""
        self._json_format = enabled

    def set_output(self, output: TextIO) -> None:
        """Set output stream."""
        self._output = output

    def _emit(self, entry: LogEntry) -> None:
        """Emit a log entry."""
        if self._json_format:
            line = entry.to_json()
        else:
            line = entry.to_text()

        self._output.write(line + "\n")
        self._output.flush()

        for handler in self._handlers:
            handler(entry)

    def add_handler(self, handler) -> None:
        """Add a custom log handler."""
        self._handlers.append(handler)

    def debug(self, message: str, **context) -> None:
        """Log debug message."""
        if self._verbosity >= Verbosity.DEBUG:
            self._emit(
                LogEntry(
                    level="DEBUG",
                    message=message,
                    timestamp=datetime.now().isoformat(),
                    component=context.pop("component", "zenith"),
                    model_name=context.pop("model_name", None),
                    operation=context.pop("operation", None),
                    duration_ms=context.pop("duration_ms", None),
                    memory_mb=context.pop("memory_mb", None),
                    extra=context if context else {},
                )
            )

    def info(self, message: str, **context) -> None:
        """Log info message."""
        if self._verbosity >= Verbosity.INFO:
            self._emit(
                LogEntry(
                    level="INFO",
                    message=message,
                    timestamp=datetime.now().isoformat(),
                    component=context.pop("component", "zenith"),
                    model_name=context.pop("model_name", None),
                    operation=context.pop("operation", None),
                    duration_ms=context.pop("duration_ms", None),
                    memory_mb=context.pop("memory_mb", None),
                    extra=context if context else {},
                )
            )

    def warning(self, message: str, **context) -> None:
        """Log warning message."""
        if self._verbosity >= Verbosity.WARNING:
            self._emit(
                LogEntry(
                    level="WARNING",
                    message=message,
                    timestamp=datetime.now().isoformat(),
                    component=context.pop("component", "zenith"),
                    model_name=context.pop("model_name", None),
                    operation=context.pop("operation", None),
                    duration_ms=context.pop("duration_ms", None),
                    memory_mb=context.pop("memory_mb", None),
                    extra=context if context else {},
                )
            )

    def error(self, message: str, **context) -> None:
        """Log error message."""
        if self._verbosity >= Verbosity.ERROR:
            self._emit(
                LogEntry(
                    level="ERROR",
                    message=message,
                    timestamp=datetime.now().isoformat(),
                    component=context.pop("component", "zenith"),
                    model_name=context.pop("model_name", None),
                    operation=context.pop("operation", None),
                    duration_ms=context.pop("duration_ms", None),
                    memory_mb=context.pop("memory_mb", None),
                    extra=context if context else {},
                )
            )

    def compile_summary(self, stats: dict) -> None:
        """
        Log compilation summary (formatted box output).

        Always shown at INFO level.
        """
        if self._verbosity >= Verbosity.INFO:
            model_name = stats.get("model_name", "N/A")
            target = stats.get("target", "N/A")
            precision = stats.get("precision", "N/A")
            compile_time = stats.get("compile_time", 0)
            fused_ops = stats.get("fused_ops", 0)
            dce_removed = stats.get("dce_removed", 0)
            estimated_speedup = stats.get("estimated_speedup", 1.0)

            summary = f"""
+-----------------------------------------------------------+
| Zenith Compilation Complete                               |
+-----------------------------------------------------------+
| Model:      {model_name:<44} |
| Target:     {target:<44} |
| Precision:  {precision:<44} |
| Time:       {compile_time:.2f}s                                         |
|                                                           |
| Optimizations Applied:                                    |
|   - Fused ops: {fused_ops:<40} |
|   - DCE removed: {dce_removed:<38} |
|   - Est. speedup: {estimated_speedup:.1f}x                                   |
+-----------------------------------------------------------+
"""
            self._output.write(summary)
            self._output.flush()


def get_logger() -> ZenithLogger:
    """Get the global Zenith logger."""
    return ZenithLogger.get()


def set_verbosity(level: int) -> None:
    """
    Set global verbosity level.

    Args:
        level: Verbosity level (0=SILENT, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG)
    """
    ZenithLogger.get().set_verbosity(level)
