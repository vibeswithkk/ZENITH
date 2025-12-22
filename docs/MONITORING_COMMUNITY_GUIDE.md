# Zenith Monitoring - Dokumentasi Komunitas

**Tanggal:** 22 December 2025  
**Status:** Demo Sementara (Temporary Demo)

---

## Transparansi

Demi transparansi kepada komunitas, kami jelaskan:

1. **Demo ini bersifat SEMENTARA** - Kami menggunakan VPS per-jam untuk menguji sistem monitoring sebelum rilis resmi.

2. **Tujuan** - Membuktikan bahwa fitur monitoring Zenith benar-benar berfungsi, bukan sekadar klaim yang tidak bisa dibuktikan.

3. **Data yang ditampilkan** - Data yang terlihat di dashboard adalah data demo (simulated) untuk menunjukkan fungsionalitas sistem.

4. **Biaya** - VPS ini dibiayai secara pribadi oleh tim pengembang untuk keperluan testing.

---

## Bukti Sistem Berjalan

### Live URLs (Sementara masih aktif)

| Service | URL | Akses |
|---------|-----|-------|
| Metrics API | http://202.155.157.122:8080 | Public |
| Prometheus | http://202.155.157.122:9090 | Public |
| Grafana | http://202.155.157.122:3000 | admin / zenith123 |

### Verifikasi Langsung

Anda dapat memverifikasi sendiri bahwa sistem berjalan:

```bash
# Health check
curl http://202.155.157.122:8080/health

# Metrics summary
curl http://202.155.157.122:8080/summary

# Prometheus metrics
curl http://202.155.157.122:8080/metrics
```

---

## Cara Deploy Sendiri

Setelah demo ini berakhir, komunitas atau sponsor dapat deploy sendiri dengan langkah berikut:

### Prerequisites

- Docker dan Docker Compose terinstall
- Port 3000, 8080, 9090 tersedia

### Langkah Deployment

```bash
# 1. Clone repository
git clone https://github.com/vibeswithkk/zenith.git
cd zenith

# 2. Masuk ke direktori monitoring
cd deployment/monitoring

# 3. Jalankan stack
docker compose up -d

# 4. Akses dashboard
# Grafana: http://localhost:3000 (admin/zenith123)
# Metrics: http://localhost:8080
# Prometheus: http://localhost:9090
```

### Struktur File

```
deployment/monitoring/
├── docker-compose.yml      # Stack configuration
├── Dockerfile              # Metrics server image
├── prometheus.yml          # Prometheus config
├── standalone_server.py    # Demo server dengan data generator
├── README.md               # Dokumentasi deployment
└── grafana-provisioning/   
    ├── datasources/        # Prometheus datasource
    └── dashboards/         # Pre-built dashboard
```

---

## Integrasi dengan Aplikasi Anda

Untuk mengintegrasikan monitoring dengan aplikasi Zenith Anda:

```python
from zenith.observability import get_metrics_collector, InferenceMetrics

# Record inference metrics
collector = get_metrics_collector()
collector.record_inference(InferenceMetrics(
    latency_ms=10.5,
    memory_mb=256.0,
    model_name="bert"
))

# Metrics akan otomatis tersedia di /metrics endpoint
```

---

## Catatan Penting

- Demo ini akan dimatikan setelah testing selesai
- Semua kode tersedia di repository untuk deployment mandiri
- Tidak ada data sensitif yang dikumpulkan
- Ini adalah infrastruktur monitoring standar industri (Prometheus + Grafana)

---

## Kontribusi

Jika ada sponsor atau anggota komunitas yang ingin menyediakan infrastruktur permanen untuk monitoring, silakan hubungi tim pengembang.
