# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Event System for Zenith

Provides a lightweight event emitter for tracking and reacting to
system events such as model compilation, inference completion, and errors.

Example:
    from zenith.observability import events

    # Subscribe to events
    events.on("model.compiled", lambda e: print(f"Compiled: {e.data}"))

    # Emit events
    events.emit("model.compiled", model_name="bert", duration_ms=1500)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional
import threading
import fnmatch


@dataclass
class Event:
    """
    Represents a single event occurrence.

    Attributes:
        name: Event name (e.g., "model.compiled", "inference.completed")
        timestamp: ISO format timestamp when event occurred
        data: Event payload as dictionary
        correlation_id: Optional correlation ID for request tracking
    """

    name: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }


EventHandler = Callable[[Event], None]


class EventEmitter:
    """
    Lightweight event emitter with pattern-based subscriptions.

    Supports wildcard patterns for flexible event handling.
    Thread-safe for concurrent usage.

    Example:
        emitter = EventEmitter()

        # Subscribe to specific event
        emitter.on("model.compiled", handler_func)

        # Subscribe to all model events
        emitter.on("model.*", handler_func)

        # Subscribe to all events
        emitter.on("*", handler_func)

        # Emit event
        emitter.emit("model.compiled", model_name="bert")
    """

    def __init__(self):
        """Initialize event emitter."""
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.RLock()
        self._history: List[Event] = []
        self._history_limit = 1000
        self._record_history = False

    def on(self, pattern: str, handler: EventHandler) -> None:
        """
        Subscribe to events matching pattern.

        Args:
            pattern: Event name or glob pattern (e.g., "model.*")
            handler: Callback function receiving Event object
        """
        with self._lock:
            if pattern not in self._handlers:
                self._handlers[pattern] = []
            if handler not in self._handlers[pattern]:
                self._handlers[pattern].append(handler)

    def off(self, pattern: str, handler: Optional[EventHandler] = None) -> None:
        """
        Unsubscribe from events.

        Args:
            pattern: Event pattern to unsubscribe from
            handler: Specific handler to remove, or None to remove all
        """
        with self._lock:
            if pattern not in self._handlers:
                return

            if handler is None:
                del self._handlers[pattern]
            else:
                self._handlers[pattern] = [
                    h for h in self._handlers[pattern] if h != handler
                ]
                if not self._handlers[pattern]:
                    del self._handlers[pattern]

    def emit(self, name: str, correlation_id: Optional[str] = None, **data) -> Event:
        """
        Emit an event.

        Args:
            name: Event name (e.g., "model.compiled")
            correlation_id: Optional correlation ID for tracking
            **data: Event payload as keyword arguments

        Returns:
            The emitted Event object
        """
        event = Event(
            name=name,
            timestamp=datetime.now().isoformat(),
            data=data,
            correlation_id=correlation_id,
        )

        with self._lock:
            # Record to history if enabled
            if self._record_history:
                self._history.append(event)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]

            # Find matching handlers
            handlers_to_call = []
            for pattern, handlers in self._handlers.items():
                if self._matches(name, pattern):
                    handlers_to_call.extend(handlers)

        # Call handlers outside lock to prevent deadlocks
        for handler in handlers_to_call:
            try:
                handler(event)
            except Exception:
                # Silently ignore handler errors to prevent event cascade failures
                pass

        return event

    def _matches(self, name: str, pattern: str) -> bool:
        """Check if event name matches pattern."""
        if pattern == "*":
            return True
        return fnmatch.fnmatch(name, pattern)

    def enable_history(self, limit: int = 1000) -> None:
        """Enable event history recording."""
        with self._lock:
            self._record_history = True
            self._history_limit = limit

    def disable_history(self) -> None:
        """Disable event history recording."""
        with self._lock:
            self._record_history = False

    def get_history(
        self, pattern: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get recorded event history.

        Args:
            pattern: Optional filter pattern
            limit: Maximum number of events to return

        Returns:
            List of Event objects (newest last)
        """
        with self._lock:
            events = self._history.copy()

        if pattern:
            events = [e for e in events if self._matches(e.name, pattern)]

        if limit:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()

    def clear_all(self) -> None:
        """Clear all handlers and history."""
        with self._lock:
            self._handlers.clear()
            self._history.clear()


# Global event emitter instance
_emitter: Optional[EventEmitter] = None


def _get_emitter() -> EventEmitter:
    """Get or create global event emitter."""
    global _emitter
    if _emitter is None:
        _emitter = EventEmitter()
    return _emitter


def on(pattern: str, handler: EventHandler) -> None:
    """Subscribe to events matching pattern."""
    _get_emitter().on(pattern, handler)


def off(pattern: str, handler: Optional[EventHandler] = None) -> None:
    """Unsubscribe from events."""
    _get_emitter().off(pattern, handler)


def emit(name: str, correlation_id: Optional[str] = None, **data) -> Event:
    """Emit an event."""
    return _get_emitter().emit(name, correlation_id=correlation_id, **data)


def enable_history(limit: int = 1000) -> None:
    """Enable event history recording."""
    _get_emitter().enable_history(limit)


def disable_history() -> None:
    """Disable event history recording."""
    _get_emitter().disable_history()


def get_history(
    pattern: Optional[str] = None, limit: Optional[int] = None
) -> List[Event]:
    """Get recorded event history."""
    return _get_emitter().get_history(pattern, limit)


def clear() -> None:
    """Clear all handlers and history."""
    _get_emitter().clear_all()


# Pre-defined event names for consistency
class EventNames:
    """Standard event names for Zenith."""

    # Compilation events
    MODEL_COMPILED = "model.compiled"
    COMPILATION_STARTED = "compilation.started"
    COMPILATION_FAILED = "compilation.failed"

    # Inference events
    INFERENCE_STARTED = "inference.started"
    INFERENCE_COMPLETED = "inference.completed"
    INFERENCE_FAILED = "inference.failed"

    # Memory events
    MEMORY_ALLOCATED = "memory.allocated"
    MEMORY_FREED = "memory.freed"
    MEMORY_WARNING = "memory.warning"

    # Error events
    ERROR_OCCURRED = "error.occurred"
    WARNING_RAISED = "warning.raised"
