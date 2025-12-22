# Catatan Diskusi Arsitektur Zenith
**Tanggal:** 21 Desember 2024  
**Topik:** Analisis Inkonsistensi dan Performa Zenith

---

## 1. Temuan Utama: Benchmark Inkonsisten

### Hasil Demo Colab (MNIST CNN):
- PyTorch Native: 8.019 ms/batch
- Zenith Optimized: 8.093 ms/batch  
- **Speedup: 0.99x (LEBIH LAMBAT)**

### Hasil Benchmark Existing (BERT FP16):
- **Speedup: 2-7x LEBIH CEPAT**

### Mengapa Berbeda?
Dua jalur penggunaan yang berbeda dengan hasil berbeda.

---

## 2. Audit: 7 Cara Penggunaan Zenith

| # | Cara | Import | Status | Catatan |
|---|------|--------|--------|---------|
| 1 | Direct CUDA Kernels | `from zenith._zenith_core import cuda` | CEPAT | Low-level, sulit digunakan |
| 2 | PyTorch Adapter | `import zenith.torch as ztorch` | LAMBAT | Tidak pakai kernels |
| 3 | TensorFlow Adapter | `import zenith.tensorflow as ztf` | ⚠️ Tidak diuji | |
| 4 | JAX Adapter | `import zenith.jax as zjax` | ⚠️ Tidak diuji | |
| 5 | ONNX Interpreter | `from zenith.execution import ONNXInterpreter` | ⚠️ Partial | |
| 6 | High-Level API | `zenith.compile(model)` | Tidak ada | Sesuai CetakBiru tapi tidak ada |
| 7 | Triton Serving | `from zenith.serving import TritonBackend` | Ada | Untuk deployment |

---

## 3. Akar Masalah: Missing Link

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   GraphIR    │ ──► │   RUNTIME    │ ──► │   Kernels    │
│ (sudah ada)  │     │ (TIDAK ADA!) │     │ (sudah ada)  │
└──────────────┘     └──────────────┘     └──────────────┘
        ✓                   ✗                    ✓
```

**Zenith Runtime** yang menghubungkan GraphIR ke Kernel **TIDAK ADA**.

Akibatnya:
- `torch.compile(backend="zenith")` → kembali ke PyTorch kernels → LAMBAT
- `cuda.linear_fp16_gpu()` langsung → pakai Zenith kernels → CEPAT

---

## 4. Kondisi Saat Ini

### Yang Sudah Ada dan Bagus:
- CUDA Kernels performant (linear, attention, layernorm, gelu, dll)
- FP16 Tensor Core support
- Kernel Fusion (fused_add_layernorm, dll)
- GraphIR representation
- Framework Adapters (struktur)
- Optimization Passes (fusion, DCE, constant folding)

### Yang Hilang:
- **Zenith Runtime/Executor** - dispatch GraphIR ops ke kernels
- **Unified High-Level API** - `zenith.compile(model)`
- **Konsistensi** - 7 cara berbeda, hanya 1 yang bekerja

---

## 5. Pertanyaan Untuk Dibahas Selanjutnya

1. Apakah ada logging dan real-time monitoring?
2. Apakah arsitektur perlu direstrukturisasi?
3. Prioritas: perbaiki yang ada atau bangun ulang?
4. Timeline dan resource yang dibutuhkan?

---

## 6. Rekomendasi (Pending Diskusi)

### Opsi A: Perbaiki yang Ada
- Buat Zenith Runtime untuk menghubungkan GraphIR → Kernels
- Update adapters untuk menggunakan runtime
- Effort: ~1-2 minggu

### Opsi B: Arsitektur Ulang
- Redesign dengan 1 unified flow
- Lebih clean tapi lebih lama
- Effort: ~1-2 bulan

---

*Catatan ini akan diupdate seiring diskusi berlanjut.*

---

## Update: Blueprint dan Task List Selesai

**Status:** COMPLETE

Berdasarkan diskusi ini, telah dibuat:

1. **[BLUEPRINT_ZENITH_3X.md](./BLUEPRINT_ZENITH_3X.md)** - Blueprint lengkap untuk memperbaiki 7 masalah kritis
   - Riset dari TensorRT, ONNX Runtime, TVM, PyTorch 2.0, MLPerf, OpenTelemetry
   - Arsitektur solusi dengan diagram lengkap
   - Spesifikasi teknis untuk setiap komponen
   - Timeline 6 minggu

2. **[TASK_LIST_ZENITH_3X.md](./TASK_LIST_ZENITH_3X.md)** - Daftar tugas lengkap
   - 33 tugas terstruktur
   - 7 fase implementasi
   - Dependencies dan deliverables jelas
   - Tracking progress

**Next Step:** Mulai implementasi Phase 1 (Runtime Core)
