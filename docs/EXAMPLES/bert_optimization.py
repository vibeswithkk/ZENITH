# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
BERT Optimization Example

Demonstrates optimizing a BERT model with Zenith.
"""

import torch
import time


def main():
    """Run BERT optimization example."""
    try:
        from transformers import BertModel, BertTokenizer
    except ImportError:
        print("Please install transformers: pip install transformers")
        return

    import zenith

    # Load model
    print("Loading BERT model...")
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Prepare input
    text = "Hello, this is a test sentence for BERT optimization."
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Baseline
    print("\nBaseline inference...")
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(10):
            output = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) * 100

    # Compile with Zenith
    print("Compiling with Zenith...")
    zenith.set_verbosity(2)
    optimized = zenith.compile(
        model,
        target=device,
        precision="fp16" if device == "cuda" else "fp32",
    )

    # Optimized
    print("\nOptimized inference...")
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(10):
            output = optimized(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) * 100

    # Results
    print(f"\nResults:")
    print(f"  Baseline:  {baseline_time:.2f}ms per inference")
    print(f"  Optimized: {optimized_time:.2f}ms per inference")
    print(f"  Speedup:   {baseline_time / optimized_time:.2f}x")


if __name__ == "__main__":
    main()
