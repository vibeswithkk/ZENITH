# Zenith Monitoring Guide

Zenith provides two monitoring options to suit different needs.

## Option 1: Terminal Dashboard

Real-time TUI dashboard that runs in your terminal.

**Requirements:** `pip install rich`

**Usage:**
```bash
zenith dashboard
```

**Features:**
- Latency metrics (P50, P90, P99)
- Memory usage tracking
- Throughput display
- Keyboard controls: `q` quit, `r` reset

---

## Option 2: Grafana Dashboard

Production-grade visualization with Prometheus integration.

### Setup

1. **Start Zenith metrics server:**
```bash
zenith serve --port 8080
```

2. **Configure Prometheus** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'zenith'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

3. **Import Grafana dashboard:**
   - Go to Grafana > Dashboards > Import
   - Upload `grafana/zenith-dashboard.json`
   - Select your Prometheus data source

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| Inference Latency | P50, P90, P99 latency stats |
| Memory Usage | GPU/CPU memory gauge |
| Total Inferences | Counter of all inferences |
| Latency Over Time | Time series chart |
| Inference Rate | Inferences per second |
| Error Rate | Percentage of failed inferences |

---

## Metrics Endpoints

When running `zenith serve`:

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/metrics` | Prometheus | Prometheus scrape endpoint |
| `/summary` | JSON | Metrics summary |
| `/health` | JSON | Health check |
| `/ws` | WebSocket | Real-time streaming |

---

## Programmatic Usage

```python
from zenith.observability import MetricsCollector, InferenceMetrics

collector = MetricsCollector()

# Record inference
collector.record_inference(InferenceMetrics(
    latency_ms=12.5,
    memory_mb=256.0,
    batch_size=32
))

# Get summary
summary = collector.get_summary()
print(f"P99 Latency: {summary['latency_p99_ms']:.2f}ms")
```
