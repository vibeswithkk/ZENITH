# Zenith Load Testing

Load testing infrastructure for Zenith performance validation.

## Components

### 1. Locust HTTP Load Test

Tests the monitoring HTTP server under load.

**Prerequisites:**
```bash
pip install locust
```

**Usage:**
```bash
# Start Zenith server
zenith serve --port 8080

# Run Locust with web UI
locust -f tests/load/locustfile.py --host=http://localhost:8080

# Headless mode for CI (100 users, 10 spawn/s, 60s)
locust -f tests/load/locustfile.py --host=http://localhost:8080 \
       --headless -u 100 -r 10 --run-time 60s
```

**Scenarios:**
- `HealthCheckUser`: Stress `/health` endpoint
- `MetricsScraper`: Simulate Prometheus scraping
- `SummaryPoller`: Poll JSON metrics
- `MixedWorkloadUser`: Realistic usage pattern

---

### 2. Python Stress Test

Pure Python stress test for inference pipeline (no external deps).

**Usage:**
```bash
# Basic test (100 workers, 1000 requests)
python tests/load/stress_test.py

# Custom configuration
python tests/load/stress_test.py --workers 200 --requests 5000

# Soak test (1 hour)
python tests/load/stress_test.py --soak --duration 3600
```

**Scenarios:**
- Sequential Baseline: Single-threaded throughput
- Concurrent Stress: Multi-threaded (100+ workers)
- Memory Pressure: Allocation/deallocation cycles
- Soak Test: Long-running stability (optional)

---

## Success Criteria

| Metric | Target |
|--------|--------|
| `/health` p99 latency | < 10ms |
| Success rate | >= 99.9% |
| Memory leak | 0 MB growth after 1h |
| Throughput | > 500 req/s |

---

## CI Integration

```yaml
# .github/workflows/load-test.yml
- name: Run Stress Test
  run: python tests/load/stress_test.py --workers 50 --requests 500
```
