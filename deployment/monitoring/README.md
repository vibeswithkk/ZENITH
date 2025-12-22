# Zenith Monitoring Deployment Guide

## Transparansi

Demo ini menggunakan VPS per-jam untuk testing. Semua kode tersedia untuk deployment mandiri.

## Live Demo (Sementara)

| Service | URL |
|---------|-----|
| Metrics | http://202.155.157.122:8080 |
| Prometheus | http://202.155.157.122:9090 |
| Grafana | http://202.155.157.122:3000 (admin/zenith123) |

## Bukti Berjalan (22 Dec 2025)

```
Health: {"status":"healthy","timestamp":1766349244}
Total Inferences: 883
Models Tracked: bert (291), gpt2 (280), resnet (312)
Latency Mean: 26.8ms
Latency P99: 49.4ms
```

## Quick Start

```bash
cd deployment/monitoring
docker compose up -d
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| GET / | API info |
| GET /health | Health check |
| GET /metrics | Prometheus metrics |
| GET /summary | JSON summary |
| WS /ws | Live WebSocket stream |

## Verifikasi

```bash
curl http://localhost:8080/health
curl http://localhost:8080/summary
curl http://localhost:8080/metrics
```

## Access

- Grafana: http://localhost:3000 (admin / zenith123)
- Prometheus: http://localhost:9090
- Metrics: http://localhost:8080

## Stop

```bash
docker compose down
```

