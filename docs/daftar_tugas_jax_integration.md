# Daftar Tugas: Full JAX Integration

**Project:** Zenith Full JAX Integration  
**Created:** 2025-12-28  
**Target Version:** v0.3.0  
**Estimated Duration:** 8 Weeks

---

## Overview

Daftar tugas lengkap untuk mengimplementasikan Full JAX Integration berdasarkan blueprint yang telah disusun.

**Total Tasks:** 45  
**Priority Distribution:**
- P0 (Critical): 18 tasks
- P1 (High): 15 tasks
- P2 (Medium): 12 tasks

---

## Phase 1: Core JAX Integration (Week 1-2)

### 1.1 JAX Gradient Checkpointing

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-001 | Create `zenith/jax/checkpointing.py` file structure | `zenith/jax/checkpointing.py` | P0 | DONE | 2 |
| JAX-002 | Implement `ZenithJAXCheckpoint` class | `zenith/jax/checkpointing.py` | P0 | DONE | 8 |
| JAX-003 | Implement optimal checkpoint selection (DP algorithm) | `zenith/jax/checkpointing.py` | P0 | DONE | 8 |
| JAX-004 | Implement `checkpoint()` function wrapper around jax.checkpoint | `zenith/jax/checkpointing.py` | P0 | DONE | 4 |
| JAX-005 | Implement `checkpoint_sequential()` for sequential layers | `zenith/jax/checkpointing.py` | P0 | DONE | 6 |
| JAX-006 | Add activation offloading to CPU support | `zenith/jax/checkpointing.py` | P1 | DONE | 4 |
| JAX-007 | Write unit tests for gradient checkpointing | `tests/python/test_jax_integration.py` | P0 | DONE | 6 |

**Deliverable:** Working gradient checkpointing that integrates with jax.grad and reduces memory by 50%+

---

### 1.2 JAX Memory Management

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-008 | Create `zenith/jax/memory_manager.py` file structure | `zenith/jax/memory_manager.py` | P0 | DONE | 2 |
| JAX-009 | Implement `JAXMemoryConfig` dataclass | `zenith/jax/memory_manager.py` | P0 | DONE | 2 |
| JAX-010 | Implement `JAXMemoryManager` class with allocation tracking | `zenith/jax/memory_manager.py` | P0 | DONE | 8 |
| JAX-011 | Implement `JAXMemoryPool` for activation reuse | `zenith/jax/memory_manager.py` | P1 | DONE | 6 |
| JAX-012 | Implement `JAXActivationStore` (analogous to PyTorch version) | `zenith/jax/memory_manager.py` | P0 | DONE | 8 |
| JAX-013 | Implement offload and prefetch functions | `zenith/jax/memory_manager.py` | P1 | DONE | 4 |
| JAX-014 | Implement memory profiler for JAX | `zenith/jax/memory_manager.py` | P2 | DONE | 4 |
| JAX-015 | Write unit tests for memory management | `tests/python/test_jax_integration.py` | P0 | DONE | 6 |

**Deliverable:** Memory manager that prevents OOM and provides profiling

---

### 1.3 Mixed Precision Training

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-016 | Create `zenith/jax/mixed_precision.py` file structure | `zenith/jax/mixed_precision.py` | P0 | DONE | 2 |
| JAX-017 | Implement `MixedPrecisionPolicy` dataclass | `zenith/jax/mixed_precision.py` | P0 | DONE | 3 |
| JAX-018 | Implement `DynamicLossScaler` for FP16 stability | `zenith/jax/mixed_precision.py` | P0 | DONE | 6 |
| JAX-019 | Implement `ZenithMixedPrecision` high-level API | `zenith/jax/mixed_precision.py` | P0 | DONE | 6 |
| JAX-020 | Add BF16 policy (recommended for TPU/Ampere+) | `zenith/jax/mixed_precision.py` | P0 | DONE | 2 |
| JAX-021 | Add FP16 policy with loss scaling | `zenith/jax/mixed_precision.py` | P1 | DONE | 4 |
| JAX-022 | Write unit tests for mixed precision | `tests/python/test_jax_integration.py` | P0 | DONE | 6 |

**Deliverable:** Mixed precision training support with 1.5-2.5x speedup

---

## Phase 2: Backend and Execution (Week 3-4)

### 2.1 XLA Backend

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-023 | Create `zenith/backends/xla_backend.py` file | `zenith/backends/xla_backend.py` | P1 | TODO | 2 |
| JAX-024 | Implement `XLABackend` class extending BaseBackend | `zenith/backends/xla_backend.py` | P1 | TODO | 8 |
| JAX-025 | Implement GraphIR → HLO conversion | `zenith/core/hlo_lowering.py` | P1 | TODO | 16 |
| JAX-026 | Implement XLA compilation and caching | `zenith/backends/xla_backend.py` | P1 | TODO | 8 |
| JAX-027 | Implement XLA execution with device placement | `zenith/backends/xla_backend.py` | P1 | TODO | 6 |
| JAX-028 | Add TPU support detection and handling | `zenith/backends/xla_backend.py` | P2 | TODO | 4 |
| JAX-029 | Register XLA backend in backend registry | `zenith/backends/registry.py` | P1 | TODO | 2 |
| JAX-030 | Write integration tests for XLA backend | `tests/e2e/test_xla_backend.py` | P1 | TODO | 8 |

**Deliverable:** Direct XLA execution from GraphIR

---

### 2.2 ONNX Export Enhancement

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-031 | Create `zenith/jax/onnx_export.py` file | `zenith/jax/onnx_export.py` | P1 | TODO | 2 |
| JAX-032 | Implement `export_to_onnx()` main function | `zenith/jax/onnx_export.py` | P1 | TODO | 8 |
| JAX-033 | Implement `validate_onnx_model()` validation | `zenith/jax/onnx_export.py` | P1 | TODO | 6 |
| JAX-034 | Add StableHLO → ONNX conversion path | `zenith/jax/onnx_export.py` | P1 | TODO | 8 |
| JAX-035 | Implement ONNX optimization passes | `zenith/jax/onnx_export.py` | P2 | TODO | 4 |
| JAX-036 | Write tests for ONNX export | `tests/python/test_jax_onnx_export.py` | P1 | TODO | 6 |

**Deliverable:** Robust ONNX export with validation

---

## Phase 3: Custom Primitives and Kernels (Week 5-6)

### 3.1 JAX Primitives Framework

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-037 | Create `zenith/jax/primitives.py` file | `zenith/jax/primitives.py` | P1 | TODO | 2 |
| JAX-038 | Implement primitive registration framework | `zenith/jax/primitives.py` | P1 | TODO | 8 |
| JAX-039 | Implement fused attention primitive with abstract_eval | `zenith/jax/primitives.py` | P1 | TODO | 12 |
| JAX-040 | Implement JVP (forward-mode diff) for attention | `zenith/jax/primitives.py` | P1 | TODO | 8 |
| JAX-041 | Implement VJP (reverse-mode diff) for attention | `zenith/jax/primitives.py` | P1 | TODO | 8 |
| JAX-042 | Add fused LayerNorm primitive | `zenith/jax/primitives.py` | P2 | TODO | 8 |
| JAX-043 | Write tests for JAX primitives | `tests/python/test_jax_primitives.py` | P1 | TODO | 8 |

**Deliverable:** Custom JAX primitives with full JVP/VJP support

---

### 3.2 XLA Custom Kernels

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-044 | Create `zenith/runtime/xla_kernels.py` file | `zenith/runtime/xla_kernels.py` | P2 | TODO | 2 |
| JAX-045 | Implement XLA FFI Python bindings | `zenith/runtime/xla_kernels.py` | P2 | TODO | 8 |
| JAX-046 | Create C++ XLA kernel registry (if needed) | `zenith/_native/xla_kernels.cpp` | P2 | TODO | 16 |
| JAX-047 | Implement fused attention XLA kernel | `zenith/runtime/xla_kernels.py` | P2 | TODO | 12 |
| JAX-048 | Write performance tests for XLA kernels | `tests/perf/test_xla_kernels.py` | P2 | TODO | 6 |

**Deliverable:** XLA-optimized kernels via CustomCall

---

## Phase 4: Validation and Polish (Week 7-8)

### 4.1 Integration and Testing

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-049 | Create comprehensive E2E test suite | `tests/e2e/test_jax_workflow.py` | P0 | TODO | 8 |
| JAX-050 | Test Flax model full workflow | `tests/e2e/test_jax_workflow.py` | P0 | TODO | 4 |
| JAX-051 | Test Haiku model full workflow | `tests/e2e/test_jax_workflow.py` | P1 | TODO | 4 |
| JAX-052 | Test HuggingFace Flax BERT optimization | `tests/e2e/test_jax_workflow.py` | P1 | TODO | 4 |
| JAX-053 | Create Colab validation notebook | `notebooks/zenith_jax_validation.ipynb` | P0 | TODO | 8 |
| JAX-054 | Run full validation on Colab with GPU | Manual | P0 | TODO | 4 |

**Deliverable:** Fully validated JAX integration

---

### 4.2 Documentation

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-055 | Write JAX integration main documentation | `docs/jax_integration.md` | P0 | TODO | 8 |
| JAX-056 | Write API reference for zenith.jax module | `docs/api/zenith_jax.md` | P1 | TODO | 6 |
| JAX-057 | Write tutorial: JAX gradient checkpointing | `docs/tutorials/jax_checkpointing.md` | P1 | TODO | 4 |
| JAX-058 | Write tutorial: JAX mixed precision | `docs/tutorials/jax_mixed_precision.md` | P1 | TODO | 4 |
| JAX-059 | Update README with JAX examples | `README.md` | P0 | TODO | 2 |

**Deliverable:** Comprehensive documentation

---

### 4.3 Performance Benchmarks

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-060 | Create JAX benchmark suite | `benchmarks/jax_benchmarks.py` | P1 | TODO | 8 |
| JAX-061 | Benchmark: Gradient checkpointing memory reduction | `benchmarks/jax_benchmarks.py` | P1 | TODO | 4 |
| JAX-062 | Benchmark: Mixed precision speedup | `benchmarks/jax_benchmarks.py` | P1 | TODO | 4 |
| JAX-063 | Benchmark: Custom primitives vs baseline | `benchmarks/jax_benchmarks.py` | P1 | TODO | 4 |
| JAX-064 | Generate benchmark report | `BENCHMARK_REPORT_JAX.md` | P1 | TODO | 4 |

**Deliverable:** Performance benchmark report

---

### 4.4 Release Preparation

| ID | Task | File | Priority | Status | Est. Hours |
|----|------|------|----------|--------|------------|
| JAX-065 | Update version to 0.3.0 | `pyproject.toml` | P0 | TODO | 1 |
| JAX-066 | Update CHANGELOG with JAX features | `CHANGELOG.md` | P0 | TODO | 2 |
| JAX-067 | Update zenith/jax/__init__.py exports | `zenith/jax/__init__.py` | P0 | TODO | 2 |
| JAX-068 | Run full test suite before release | Manual | P0 | TODO | 4 |
| JAX-069 | Build and test PyPI package | Manual | P0 | TODO | 2 |
| JAX-070 | Publish pyzenith 0.3.0 to PyPI | Manual | P0 | TODO | 1 |

**Deliverable:** Released version 0.3.0 with Full JAX Integration

---

## Summary by Priority

### P0 (Critical) - Must Complete
- JAX-001 to JAX-007 (Gradient Checkpointing)
- JAX-008, JAX-009, JAX-010, JAX-012, JAX-015 (Memory Management core)
- JAX-016 to JAX-020, JAX-022 (Mixed Precision core)
- JAX-049, JAX-050, JAX-053, JAX-054 (E2E Testing)
- JAX-055, JAX-059 (Documentation core)
- JAX-065 to JAX-070 (Release)

**Total P0 Hours:** ~150 hours

### P1 (High) - Should Complete
- JAX-006 (Offloading)
- JAX-011, JAX-013 (Memory Pool)
- JAX-021 (FP16 policy)
- JAX-023 to JAX-036 (XLA Backend + ONNX)
- JAX-037 to JAX-043 (Primitives)
- JAX-051, JAX-052, JAX-056-058, JAX-060-064

**Total P1 Hours:** ~160 hours

### P2 (Medium) - Nice to Have
- JAX-014 (Memory profiler)
- JAX-028 (TPU support)
- JAX-035 (ONNX optimization)
- JAX-042 (LayerNorm primitive)
- JAX-044 to JAX-048 (XLA kernels)

**Total P2 Hours:** ~60 hours

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| 1-2 | Core JAX Integration | Checkpointing, Memory Manager, Mixed Precision |
| 3-4 | Backend & Execution | XLA Backend, ONNX Export |
| 5-6 | Primitives & Kernels | JAX Primitives, XLA Kernels |
| 7-8 | Validation & Polish | E2E Tests, Docs, Benchmarks, Release |

---

## Dependencies

### External Libraries Required
1. `jax>=0.4.14` - For jax.export and modern features
2. `flax>=0.8.0` - For Flax model support
3. `optax>=0.1.7` - For Flax training
4. `jax2onnx>=0.1.0` - For ONNX export
5. `onnx>=1.14.0` - ONNX model handling

### Internal Dependencies
- `zenith/core/graph_ir.py` - GraphIR must support HLO lowering
- `zenith/backends/base.py` - XLABackend extends BaseBackend
- `zenith/memory/native_checkpointing.py` - Reference for JAX implementation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Memory reduction (checkpointing) | ≥ 50% |
| Speedup (mixed precision) | ≥ 1.5x |
| Test coverage | ≥ 90% |
| Colab validation | All tests PASS |
| Documentation coverage | 100% |

---

## Notes

1. **Prioritas utama:** Fokus pada P0 tasks terlebih dahulu
2. **Testing:** Setiap fitur harus memiliki unit test sebelum merge
3. **Validation:** Colab notebook validation wajib PASS sebelum release
4. **Documentation:** Update docs bersamaan dengan implementasi

---

*End of Task List*
