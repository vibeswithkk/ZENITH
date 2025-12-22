# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
ResNet Deployment Example

Demonstrates deploying a ResNet model with Zenith.
"""

import torch
import time


def main():
    """Run ResNet deployment example."""
    try:
        import torchvision.models as models
    except ImportError:
        print("Please install torchvision: pip install torchvision")
        return

    import zenith

    # Load model
    print("Loading ResNet-50 model...")
    model = models.resnet50(pretrained=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Prepare input
    batch_size = 16
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # Baseline
    print("\nBaseline inference...")
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            output = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) * 50

    # Compile with Zenith
    print("Compiling with Zenith...")
    optimized = zenith.compile(
        model,
        target=device,
        precision="fp16" if device == "cuda" else "fp32",
    )

    # Optimized
    print("\nOptimized inference...")
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            output = optimized(x)
        if device == "cuda":
            torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) * 50

    # Results
    print(f"\nResults (batch_size={batch_size}):")
    print(f"  Baseline:   {baseline_time:.2f}ms")
    print(f"  Optimized:  {optimized_time:.2f}ms")
    print(f"  Speedup:    {baseline_time / optimized_time:.2f}x")
    print(f"  Throughput: {1000 / optimized_time * batch_size:.1f} images/sec")


if __name__ == "__main__":
    main()
