# Production-Scale BERT Benchmark Suite
# Tests various batch sizes and sequence lengths

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH PRODUCTION BENCHMARK SUITE")
print("=" * 70)

from zenith._zenith_core import cuda


def benchmark_config(batch_size, seq_len, num_layers=12, num_runs=20):
    """Benchmark a specific configuration."""
    import torch
    from transformers import BertModel, BertConfig

    hidden_size = 768
    num_heads = 12

    config = BertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=3072,
        num_hidden_layers=num_layers,
        hidden_act="gelu_new",
    )
    torch_bert = BertModel(config).cuda()
    torch_bert.eval()

    # Import here to avoid circular issues
    from bert_fp16_full import BertEncoderFp16

    zenith_encoder = BertEncoderFp16(num_layers, hidden_size, num_heads)
    zenith_encoder.load_from_pytorch(torch_bert.encoder)

    # Create input
    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    x_torch = torch.from_numpy(x_np).cuda()
    x_2d = x_np.reshape(batch_size * seq_len, hidden_size)
    x_gpu = cuda.to_gpu(np.ascontiguousarray(x_2d))

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = torch_bert.encoder(x_torch)
        _ = zenith_encoder.forward(x_gpu, batch_size, seq_len)

    # PyTorch timing
    torch.cuda.synchronize()
    pytorch_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch_bert.encoder(x_torch)
        torch.cuda.synchronize()
        pytorch_times.append((time.perf_counter() - t0) * 1000)

    # Zenith timing
    zenith_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = zenith_encoder.forward(x_gpu, batch_size, seq_len)
        zenith_times.append((time.perf_counter() - t0) * 1000)

    # Accuracy check
    with torch.no_grad():
        torch_out = torch_bert.encoder(x_torch)[0].cpu().numpy()
    zenith_out = zenith_encoder.forward(x_gpu, batch_size, seq_len)
    zenith_out_np = zenith_out.to_numpy().reshape(batch_size, seq_len, hidden_size)
    max_diff = np.max(np.abs(zenith_out_np - torch_out))

    return {
        "batch": batch_size,
        "seq": seq_len,
        "pytorch_ms": np.mean(pytorch_times),
        "pytorch_std": np.std(pytorch_times),
        "zenith_ms": np.mean(zenith_times),
        "zenith_std": np.std(zenith_times),
        "speedup": np.mean(pytorch_times) / np.mean(zenith_times),
        "max_diff": max_diff,
    }


if __name__ == "__main__":
    import torch
    from transformers import BertModel, BertConfig

    # Test configurations
    CONFIGS = [
        (1, 32),  # Small
        (1, 64),  # Medium
        (1, 128),  # Standard
        (4, 32),  # Batch
        (8, 32),  # Larger batch
        (1, 256),  # Long sequence
        (4, 128),  # Production-like
    ]

    print("\nRunning benchmarks for multiple configurations...\n")
    print(
        f"{'Batch':>6} {'Seq':>6} {'PyTorch':>12} {'Zenith':>12} "
        f"{'Speedup':>10} {'MaxDiff':>12}"
    )
    print("-" * 70)

    results = []
    for batch, seq in CONFIGS:
        try:
            result = benchmark_config(batch, seq)
            results.append(result)

            speedup_str = f"{result['speedup']:.2f}x"
            if result["speedup"] > 1:
                speedup_str += " ✓"

            print(
                f"{batch:>6} {seq:>6} "
                f"{result['pytorch_ms']:>10.2f}ms "
                f"{result['zenith_ms']:>10.2f}ms "
                f"{speedup_str:>10} "
                f"{result['max_diff']:>10.2e}"
            )
        except Exception as e:
            print(f"{batch:>6} {seq:>6} ERROR: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_speedup = np.mean([r["speedup"] for r in results])
    max_speedup = max([r["speedup"] for r in results])
    min_speedup = min([r["speedup"] for r in results])

    passing = sum(1 for r in results if r["max_diff"] < 0.1)

    print(f"\n  Configurations tested: {len(results)}")
    print(f"  Accuracy passing (diff < 0.1): {passing}/{len(results)}")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Best speedup: {max_speedup:.2f}x")
    print(f"  Worst speedup: {min_speedup:.2f}x")

    if avg_speedup > 1:
        print(f"\n  OVERALL: Zenith FP16 is {avg_speedup:.2f}x FASTER on average!")
    else:
        print(f"\n  OVERALL: Zenith FP16 is {1 / avg_speedup:.2f}x slower on average")

    print("=" * 70)
