# Zenith Benchmark - Google Colab Setup
# =====================================
# Copy-paste this entire cell into a Colab notebook with GPU runtime
# Make sure to enable GPU: Runtime -> Change runtime type -> GPU

# =============================================================================
# CELL 1: Install Dependencies & Clone Repository
# =============================================================================

print("=" * 60)
print("ZENITH BENCHMARK SETUP")
print("=" * 60)

# Clone repository
import os
if not os.path.exists('/content/ZENITH'):
    print("[1/4] Cloning ZENITH repository...")
    !git clone --depth 1 https://github.com/vibeswithkk/ZENITH.git /content/ZENITH
else:
    print("[1/4] Repository already exists")

# Change directory
%cd /content/ZENITH

# Install required packages
print("[2/4] Installing dependencies...")
!pip install -q torch torchvision numpy psutil

# Verify GPU
print("[3/4] Checking GPU...")
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"    GPU: {gpu_name}")
    print(f"    Memory: {gpu_mem:.1f} GB")
    print(f"    CUDA: {torch.version.cuda}")
else:
    print("    WARNING: No GPU detected! Enable GPU in Runtime settings.")

print("[4/4] Setup complete!")
print("=" * 60)

# =============================================================================
# CELL 2: Run Single Model Benchmark
# =============================================================================

print("\n" + "=" * 60)
print("RUNNING BENCHMARK: ResNet-50 on CUDA")
print("=" * 60 + "\n")

!python benchmarks/run_benchmarks.py --model resnet50 --backend cuda --iterations 50

# =============================================================================
# CELL 3: Run All Benchmarks (Optional)
# =============================================================================

# Uncomment to run full benchmark suite:
# !python benchmarks/run_benchmarks.py --all --output /content/benchmark_results.json

# =============================================================================
# CELL 4: Quick Manual Benchmark (Alternative)
# =============================================================================

import torch
import torchvision.models as models
import time
import numpy as np

def quick_benchmark(model_name="resnet50", batch_size=1, warmup=10, iterations=100):
    """Quick inline benchmark without external dependencies."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading {model_name}...")
    model = getattr(models, model_name)(pretrained=False)
    model.eval()
    model.to(device)
    
    # Create input
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    # Results
    latencies = np.array(latencies)
    print(f"\nResults for {model_name} (batch_size={batch_size}):")
    print(f"  Mean latency:  {latencies.mean():.2f} ms")
    print(f"  P50 latency:   {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95 latency:   {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency:   {np.percentile(latencies, 99):.2f} ms")
    print(f"  Throughput:    {batch_size * 1000 / latencies.mean():.1f} samples/sec")
    print(f"  GPU Memory:    {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")
    
    return latencies

# Run quick benchmark
print("\n" + "=" * 60)
print("QUICK BENCHMARK (inline)")
print("=" * 60 + "\n")

# Test different models
for model_name in ["resnet18", "resnet50", "mobilenet_v2"]:
    quick_benchmark(model_name, batch_size=1)
    print()

# Test different batch sizes
print("=" * 60)
print("BATCH SIZE COMPARISON (ResNet-50)")
print("=" * 60 + "\n")

for bs in [1, 4, 8, 16]:
    try:
        quick_benchmark("resnet50", batch_size=bs, iterations=50)
        print()
    except RuntimeError as e:
        print(f"Batch size {bs}: OOM - {e}")
        break
