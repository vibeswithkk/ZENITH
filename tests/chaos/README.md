# Zenith Chaos Testing

Chaos engineering tests for fault tolerance validation.

## Components

### 1. Network Chaos (`network_chaos.py`)

Simulates network failures:
- Connection timeouts
- Connection refused errors
- Artificial latency

**Usage:**
```bash
pytest tests/chaos/network_chaos.py -v
```

---

### 2. Memory Chaos (`memory_chaos.py`)

Simulates memory pressure:
- Allocation stress
- Memory limit testing
- Leak detection

**Usage:**
```bash
pytest tests/chaos/memory_chaos.py -v
```

---

### 3. GPU Chaos (`gpu_chaos.py`)

Simulates GPU OOM (requires CUDA):
- OOM triggering
- Recovery verification
- CPU fallback testing

**Usage:**
```bash
pytest tests/chaos/gpu_chaos.py -v
```

---

## Run All Chaos Tests

```bash
pytest tests/chaos/ -v
```

---

## Success Criteria

| Test | Expected Result |
|------|-----------------|
| Network timeout | Graceful error handling |
| Memory pressure | No leak after recovery |
| GPU OOM | Recovery or CPU fallback |
