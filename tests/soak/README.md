# Zenith Memory Soak Testing

Long-running memory leak detection tests.

## Quick Start

```bash
# 5 minute quick test
python tests/soak/memory_soak.py --duration 5m

# 1 hour test
python tests/soak/memory_soak.py --duration 1h

# 72 hour production soak
python tests/soak/memory_soak.py --duration 72h --output soak_report.json
```

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--duration` | 5m | Test duration (5m, 1h, 72h) |
| `--interval` | 30s | Snapshot interval |
| `--output` | None | JSON report path |
| `--threshold-memory` | 10.0 | MB/hour growth limit |
| `--threshold-objects` | 1000 | Objects/hour limit |

---

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| Memory growth | < 10 MB/hour |
| Object growth | < 1000/hour |
| Errors | 0 |

---

## Output Example

```
============================================================
  MEMORY SOAK TEST RESULT
============================================================
  Duration: 1.00 hours
  Peak Memory: 245.50 MB
  Memory Growth: 2.30 MB
  Growth Rate: 2.30 MB/hour
  Object Growth: 150
  Object Rate: 150/hour
------------------------------------------------------------
  STATUS: PASSED
============================================================
```
