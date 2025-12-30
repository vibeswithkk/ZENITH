# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Request Context for Zenith

Provides correlation IDs and request-scoped context for tracing
and debugging distributed operations.

Example:
    from zenith.observability import context

    # Get current correlation ID (auto-generated if not set)
    print(f"Correlation ID: {context.get_correlation_id()}")

    # Set custom correlation ID
    context.set_correlation_id("my-request-123")

    # Use as context manager
    with context.new_context(correlation_id="req-456"):
        # All operations in this block share the correlation ID
        pass
"""

import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class RequestContext:
    """
    Context data for a single request/operation.

    Attributes:
        correlation_id: Unique identifier for tracking related operations
        trace_id: Optional trace ID for distributed tracing
        span_id: Optional span ID for current operation
        parent_span_id: Optional parent span ID
        start_time: When this context was created
        attributes: Custom key-value attributes
    """

    correlation_id: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "attributes": self.attributes,
        }


class ContextManager:
    """
    Thread-local context manager for request tracking.

    Provides correlation IDs that persist across function calls
    within the same thread/async context.

    Example:
        ctx = ContextManager()

        # Auto-generate correlation ID
        cid = ctx.get_correlation_id()

        # Set custom ID
        ctx.set_correlation_id("my-request-123")

        # Add attributes
        ctx.set_attribute("model_name", "bert")
    """

    _local = threading.local()

    def __init__(self):
        """Initialize context manager."""
        pass

    def _get_context(self) -> RequestContext:
        """Get or create thread-local context."""
        if not hasattr(self._local, "context") or self._local.context is None:
            self._local.context = RequestContext(correlation_id=self._generate_id())
        return self._local.context

    def _generate_id(self, length: int = 8) -> str:
        """Generate a short unique ID."""
        return uuid.uuid4().hex[:length]

    def get_correlation_id(self) -> str:
        """
        Get the current correlation ID.

        Returns:
            Correlation ID string (auto-generated if not set)
        """
        return self._get_context().correlation_id

    def set_correlation_id(self, correlation_id: str) -> None:
        """
        Set the correlation ID for current context.

        Args:
            correlation_id: Custom correlation ID
        """
        ctx = self._get_context()
        ctx.correlation_id = correlation_id

    def get_trace_id(self) -> Optional[str]:
        """Get the current trace ID."""
        return self._get_context().trace_id

    def set_trace_id(self, trace_id: str) -> None:
        """Set the trace ID."""
        ctx = self._get_context()
        ctx.trace_id = trace_id

    def get_span_id(self) -> Optional[str]:
        """Get the current span ID."""
        return self._get_context().span_id

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a context attribute."""
        return self._get_context().attributes.get(key, default)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a context attribute."""
        self._get_context().attributes[key] = value

    def get_all_attributes(self) -> dict[str, Any]:
        """Get all context attributes."""
        return self._get_context().attributes.copy()

    def get_context(self) -> RequestContext:
        """Get the full context object."""
        return self._get_context()

    def clear(self) -> None:
        """Clear the current context."""
        self._local.context = None

    @contextmanager
    def new_context(
        self,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        inherit: bool = True,
    ) -> Generator[RequestContext, None, None]:
        """
        Create a new context scope.

        Args:
            correlation_id: Custom correlation ID (auto-generated if None)
            trace_id: Custom trace ID
            inherit: Whether to inherit parent span info

        Yields:
            The new RequestContext
        """
        # Save current context
        old_context = getattr(self._local, "context", None)

        # Create new context
        parent_span = old_context.span_id if old_context and inherit else None

        new_ctx = RequestContext(
            correlation_id=correlation_id or self._generate_id(),
            trace_id=trace_id or (old_context.trace_id if old_context else None),
            span_id=self._generate_id(),
            parent_span_id=parent_span,
        )

        self._local.context = new_ctx

        try:
            yield new_ctx
        finally:
            # Restore old context
            self._local.context = old_context

    @contextmanager
    def span(self, name: str, **attributes) -> Generator[RequestContext, None, None]:
        """
        Create a new span within current context.

        Args:
            name: Span name for tracing
            **attributes: Span attributes

        Yields:
            The context with new span
        """
        ctx = self._get_context()
        old_span_id = ctx.span_id
        old_parent_span_id = ctx.parent_span_id

        # Create new span
        ctx.parent_span_id = ctx.span_id
        ctx.span_id = self._generate_id()
        ctx.attributes["span_name"] = name
        ctx.attributes.update(attributes)

        try:
            yield ctx
        finally:
            # Restore span info
            ctx.span_id = old_span_id
            ctx.parent_span_id = old_parent_span_id
            ctx.attributes.pop("span_name", None)


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def _get_manager() -> ContextManager:
    """Get or create global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _get_manager().get_correlation_id()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID."""
    _get_manager().set_correlation_id(correlation_id)


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return _get_manager().get_trace_id()


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID."""
    _get_manager().set_trace_id(trace_id)


def get_attribute(key: str, default: Any = None) -> Any:
    """Get a context attribute."""
    return _get_manager().get_attribute(key, default)


def set_attribute(key: str, value: Any) -> None:
    """Set a context attribute."""
    _get_manager().set_attribute(key, value)


def get_context() -> RequestContext:
    """Get the full context object."""
    return _get_manager().get_context()


def clear() -> None:
    """Clear the current context."""
    _get_manager().clear()


def new_context(
    correlation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    inherit: bool = True,
):
    """Create a new context scope."""
    return _get_manager().new_context(correlation_id, trace_id, inherit)


def span(name: str, **attributes):
    """Create a new span within current context."""
    return _get_manager().span(name, **attributes)


# Convenience: correlation_id property for direct access
correlation_id: str = property(lambda self: get_correlation_id())
