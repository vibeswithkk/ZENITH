# Zenith Benchmark Report

**Generated:** 2024-12-22 00:45:00

---

## Benchmark Methodology

This benchmark suite follows MLPerf Inference methodology for consistent and reproducible performance measurements.

### Scenarios

| Scenario | Description | Primary Metric |
|----------|-------------|----------------|
| Single-Stream | Per-query latency measurement | P90 Latency |
| Offline | Maximum throughput | Samples/sec |
| Server | Latency under load | P99 Latency at target QPS |

### Metrics

- **P50/P90/P99 Latency**: Percentile latencies in milliseconds
- **Throughput (QPS)**: Queries processed per second
- **Throughput (Samples/sec)**: Total samples processed per second
- **Quality Score**: Accuracy relative to reference implementation

### Measurement Protocol

1. **Warmup Phase**: Initial runs excluded from timing (default: 10 iterations)
2. **Timed Phase**: Multiple iterations with precise timing (default: 100 iterations)
3. **Synchronization**: GPU sync before/after each measurement (CUDA)
4. **Quality Verification**: Output comparison with reference implementation

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_warmup | 10 | Warmup iterations |
| num_runs | 100 | Timed benchmark iterations |
| quality_target | 0.99 | Minimum accuracy threshold |
| target_latency_ms | 100.0 | Server scenario latency target |
| target_qps | 10.0 | Server scenario QPS target |

---

## System Requirements

### Minimum Requirements

- Python 3.9 or higher
- NumPy 1.20 or higher
- 4 GB RAM

### Recommended Requirements

- Python 3.11 or higher
- NVIDIA GPU with CUDA 11.0+
- PyTorch 2.0+ (for baseline comparison)
- 16 GB RAM
- matplotlib (for chart generation)

---

## Available Benchmarks

### BERT Benchmark

File: `benchmarks/bert_mlperf.py`

Benchmarks BERT-base encoder with various configurations:
- Batch sizes: 1, 4, 8, 16
- Sequence lengths: 32, 64, 128, 256
- Precision modes: FP32, FP16

Usage:
```bash
python -m benchmarks.bert_mlperf --batch-sizes 1 4 8 --seq-lengths 128 --scenario single-stream
```

### ResNet-50 Benchmark

File: `benchmarks/resnet_mlperf.py`

Benchmarks ResNet-50 image classification:
- Batch sizes: 1, 4, 8, 16, 32
- Image size: 224x224
- Precision modes: FP32, FP16

Usage:
```bash
python -m benchmarks.resnet_mlperf --batch-sizes 1 8 16 --scenario offline
```

---

## Running Benchmarks

### Quick Start

```bash
# Run BERT single-stream benchmark
cd /path/to/zenith
python benchmarks/bert_mlperf.py --batch-sizes 1 4 8 --num-runs 50

# Run ResNet offline benchmark
python benchmarks/resnet_mlperf.py --scenario offline --batch-sizes 16 32
```

### Full Benchmark Suite

```bash
# Run all benchmarks with default settings
python -c "
from benchmarks.mlperf_suite import BenchmarkConfig, ZenithBenchmark
import numpy as np

benchmark = ZenithBenchmark(device='cpu')

def model(x):
    return x * 2

def gen(batch, seq):
    return np.random.randn(batch, seq, 768).astype(np.float32)

config = BenchmarkConfig(
    model_name='test-model',
    batch_sizes=[1, 4, 8],
    sequence_lengths=[128],
    num_runs=100,
)

results = benchmark.run(config, model, gen)
for r in results:
    print(r.summary())
"
```

### Generating Reports

```bash
# Generate full benchmark report with charts
python -c "
from benchmarks.report_generator import generate_report, BenchmarkResult

results = [...]  # Your benchmark results
report_path = generate_report(results, output_dir='docs/benchmarks')
print(f'Report saved to: {report_path}')
"
```

---

## Expected Performance

### BERT-base (Single-Stream, FP32, CPU)

| Batch Size | Seq Len | P50 (ms) | P90 (ms) | P99 (ms) | QPS |
|------------|---------|----------|----------|----------|-----|
| 1 | 128 | 10-15 | 12-18 | 15-25 | 65-100 |
| 4 | 128 | 35-50 | 40-60 | 50-75 | 20-30 |
| 8 | 128 | 65-90 | 75-110 | 90-130 | 10-15 |
| 16 | 128 | 120-170 | 140-200 | 170-250 | 5-8 |

### ResNet-50 (Single-Stream, FP32, CPU)

| Batch Size | P50 (ms) | P90 (ms) | P99 (ms) | Images/sec |
|------------|----------|----------|----------|------------|
| 1 | 20-35 | 25-45 | 30-55 | 28-50 |
| 4 | 70-110 | 80-130 | 100-160 | 36-57 |
| 8 | 130-200 | 150-240 | 180-300 | 40-62 |
| 16 | 250-380 | 290-450 | 350-550 | 42-64 |

Note: Actual performance varies based on hardware configuration.

---

## Comparison with Baseline

### Zenith Optimizations

Zenith provides performance improvements through:

1. **Kernel Fusion**: Combining multiple operations into single kernels
2. **Memory Optimization**: Reduced memory allocation overhead
3. **Precision Flexibility**: FP16/INT8 support where applicable
4. **Graph Optimization**: Operator reordering and constant folding

### Expected Speedup

| Model | Precision | Expected Speedup |
|-------|-----------|------------------|
| BERT-base | FP32 | 1.0-1.3x |
| BERT-base | FP16 | 1.5-2.5x |
| ResNet-50 | FP32 | 1.0-1.2x |
| ResNet-50 | FP16 | 1.5-2.2x |

Note: Speedups are relative to native PyTorch execution. Actual results depend on hardware and workload.

---

## Troubleshooting

### Common Issues

**Issue**: `matplotlib not available`
**Solution**: Install matplotlib with `pip install matplotlib`

**Issue**: Low throughput on GPU
**Solution**: Ensure CUDA is available with `torch.cuda.is_available()`

**Issue**: Quality verification fails
**Solution**: Check precision settings and numerical tolerances

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## API Reference

### BenchmarkConfig

```python
@dataclass
class BenchmarkConfig:
    model_name: str              # Required: Name of the model
    batch_sizes: list            # Default: [1, 4, 8, 16]
    sequence_lengths: list       # Default: [128]
    num_warmup: int              # Default: 10
    num_runs: int                # Default: 100
    quality_target: float        # Default: 0.99
    scenario: str                # Default: "single-stream"
    precision: str               # Default: "fp32"
    target_latency_ms: float     # Default: 100.0
    target_qps: float            # Default: 10.0
```

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    model_name: str
    scenario: str
    batch_size: int
    sequence_length: int
    precision: str
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float
    throughput_qps: float
    throughput_samples_per_sec: float
    quality_score: float
    quality_passed: bool
    total_samples: int
    total_time_sec: float
    warmup_time_sec: float
```

### ZenithBenchmark

```python
class ZenithBenchmark:
    def __init__(self, device: str = "cpu"):
        ...

    def run(
        self,
        config: BenchmarkConfig,
        model_fn: Callable,
        input_generator: Callable,
        reference_fn: Optional[Callable] = None,
    ) -> list[BenchmarkResult]:
        ...
```

---

## References

- MLPerf Inference Rules v3.1: https://github.com/mlcommons/inference_policies
- MLPerf Inference Paper (MLSys 2020)
- Zenith Framework Documentation: https://github.com/vibeswithkk/ZENITH

---

## Changelog

### Version 3.0 (December 2024)

- Added MLPerf-style benchmark suite
- Implemented single-stream, offline, and server scenarios
- Added quality verification against reference
- Added BERT and ResNet benchmark scripts
- Added report generator with chart support
- Added P50/P90/P99 latency metrics
