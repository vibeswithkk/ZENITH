# Zenith Observability Tutorial

A guide to logging, monitoring, and debugging with Zenith.

## Quick Start

```python
import zenith
from zenith.observability import (
    set_verbosity, Verbosity, get_logger,
    events, context, gpu_metrics
)

# Enable verbose logging
set_verbosity(Verbosity.DEBUG)
```

---

## 1. Logging & Verbosity

### Verbosity Levels

| Level | Value | Description |
|-------|-------|-------------|
| `SILENT` | 0 | No output |
| `ERROR` | 1 | Only errors |
| `WARNING` | 2 | Errors + warnings |
| `INFO` | 3 | Standard output |
| `DEBUG` | 4 | Maximum detail |

### Basic Usage

```python
from zenith.observability import ZenithLogger, Verbosity, set_verbosity

# Set global verbosity
set_verbosity(Verbosity.DEBUG)

# Get logger instance
logger = ZenithLogger.get()

# Log messages with context
logger.info("Model loaded", model_name="bert-base")
logger.debug("Memory allocated", memory_mb=1024.5)
logger.warning("High memory usage", threshold_mb=8000)
logger.error("Compilation failed", error="OutOfMemory")
```

### Structured Logging (JSON)

```python
logger = ZenithLogger.get()
logger.set_format("json")  # Output as JSON

# Output:
# {"level":"INFO","message":"Model loaded","timestamp":"...","model_name":"bert-base"}
```

### Environment Variable

```bash
export ZENITH_VERBOSITY=DEBUG
```

---

## 2. Event System

Track and react to system events in real-time.

### Subscribe to Events

```python
from zenith.observability import events, EventNames

# Subscribe to specific event
def on_compiled(event):
    print(f"Model compiled in {event.data.get('duration_ms')}ms")

events.on(EventNames.MODEL_COMPILED, on_compiled)

# Subscribe to all model events (wildcard)
events.on("model.*", lambda e: print(f"Event: {e.name}"))

# Subscribe to everything
events.on("*", lambda e: print(e.to_dict()))
```

### Emit Events

```python
from zenith.observability import events

# Emit with data
events.emit("model.compiled", model_name="bert", duration_ms=1500)
events.emit("inference.completed", latency_ms=45, batch_size=32)
```

### Event History

```python
# Enable history recording
events.enable_history(limit=1000)

# ... operations ...

# Query history
all_events = events.get_history()
model_events = events.get_history(pattern="model.*")
recent = events.get_history(limit=10)
```

### Pre-defined Event Names

```python
from zenith.observability import EventNames

EventNames.MODEL_COMPILED       # "model.compiled"
EventNames.INFERENCE_COMPLETED  # "inference.completed"
EventNames.MEMORY_WARNING       # "memory.warning"
EventNames.ERROR_OCCURRED       # "error.occurred"
```

---

## 3. Correlation IDs & Request Context

Track operations across your application.

### Basic Usage

```python
from zenith.observability import context

# Auto-generated correlation ID
cid = context.get_correlation_id()
print(f"Request ID: {cid}")  # e.g., "a1b2c3d4"

# Set custom ID (e.g., from HTTP header)
context.set_correlation_id("req-12345")
```

### Context Scopes

```python
from zenith.observability import context

# Create isolated context
with context.new_context(correlation_id="batch-job-1"):
    # All operations here share this correlation ID
    process_batch()
    
# After block, reverts to previous context
```

### Span Tracking

```python
from zenith.observability import context

with context.span("compile", model="bert"):
    # Compilation logic
    with context.span("optimize"):
        # Nested operation
        pass
```

### Thread Safety

Each thread gets its own correlation ID automatically:

```python
import threading
from zenith.observability import context

def worker(task_id):
    context.set_correlation_id(f"worker-{task_id}")
    # Thread-isolated context
    
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
```

---

## 4. GPU Metrics

Monitor GPU utilization, memory, and power in real-time.

### Installation

```bash
pip install pyzenith[monitoring]  # Includes pynvml
```

### Basic Usage

```python
from zenith.observability import gpu_metrics

# Check availability
if gpu_metrics.is_available():
    stats = gpu_metrics.get_current()
    
    print(f"GPU: {stats['name']}")
    print(f"Utilization: {stats['utilization_percent']}%")
    print(f"Memory: {stats['memory_used_mb']}/{stats['memory_total_mb']} MB")
    print(f"Temperature: {stats['temperature_celsius']}C")
    print(f"Power: {stats['power_draw_watts']}W")
```

### Memory Monitoring

```python
from zenith.observability import gpu_metrics

memory = gpu_metrics.get_memory_info()
if memory:
    print(f"Used: {memory['used_mb']} MB")
    print(f"Free: {memory['free_mb']} MB")
    print(f"Utilization: {memory['utilization_percent']}%")
```

### Multi-GPU

```python
from zenith.observability.gpu_metrics import GPUMetricsCollector

collector = GPUMetricsCollector.get()
print(f"GPUs found: {collector.get_device_count()}")

for i in range(collector.get_device_count()):
    stats = collector.get_stats(device_index=i)
    print(f"GPU {i}: {stats.name}")
```

---

## 5. Complete Example

```python
import zenith
from zenith.observability import (
    set_verbosity, Verbosity, get_logger,
    events, context, gpu_metrics
)

# Setup
set_verbosity(Verbosity.INFO)
logger = get_logger()
events.enable_history()

# Start request
with context.new_context(correlation_id="inference-001"):
    
    # Log GPU status
    if gpu_metrics.is_available():
        mem = gpu_metrics.get_memory_info()
        logger.info("GPU ready", memory_free_mb=mem['free_mb'])
    
    # Track compilation
    with context.span("compile"):
        events.emit("compilation.started")
        # ... compile model ...
        events.emit("model.compiled", duration_ms=1200)
    
    # Track inference
    with context.span("inference"):
        events.emit("inference.started", batch_size=32)
        # ... run inference ...
        events.emit("inference.completed", latency_ms=45)
    
    logger.info("Request complete", 
                correlation_id=context.get_correlation_id())
```

---

## Summary

| Feature | Import | Purpose |
|---------|--------|---------|
| Verbosity | `set_verbosity(Verbosity.DEBUG)` | Control log detail |
| Logger | `get_logger()` | Structured logging |
| Events | `events.emit()`, `events.on()` | Event tracking |
| Context | `context.get_correlation_id()` | Request tracing |
| GPU | `gpu_metrics.get_current()` | Hardware monitoring |
