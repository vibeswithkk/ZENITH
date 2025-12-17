# ResNet-50 with GPU-Resident Tensors (Zero-Copy Pipeline)
# Uses cuda._gpu operations that keep tensors on GPU
# Expected speedup: 10-50x from baseline

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH GPU-RESIDENT ResNet-50 (Zero-Copy)")
print("=" * 70)

from zenith._zenith_core import cuda


class ZenithResNet50GPU:
    """
    GPU-Resident ResNet-50 using zero-copy operations.
    All tensors stay on GPU throughout forward pass.
    """

    def __init__(self, pytorch_model):
        """Cache all weights on GPU."""
        print("  Uploading weights to GPU...")
        start = time.perf_counter()

        # Stem weights (upload to GPU once)
        self.conv1_w = cuda.to_gpu(
            np.ascontiguousarray(pytorch_model.conv1.weight.detach().cpu().numpy())
        )
        self.bn1 = self._upload_bn(pytorch_model.bn1)

        # Layer weights
        self.layer1 = [self._upload_bottleneck(b) for b in pytorch_model.layer1]
        self.layer2 = [self._upload_bottleneck(b) for b in pytorch_model.layer2]
        self.layer3 = [self._upload_bottleneck(b) for b in pytorch_model.layer3]
        self.layer4 = [self._upload_bottleneck(b) for b in pytorch_model.layer4]

        # FC weights (keep on CPU for final matmul, or use GPU)
        self.fc_w = np.ascontiguousarray(pytorch_model.fc.weight.detach().cpu().numpy())
        self.fc_b = np.ascontiguousarray(pytorch_model.fc.bias.detach().cpu().numpy())

        upload_time = (time.perf_counter() - start) * 1000
        print(f"  Weights uploaded in {upload_time:.1f} ms")

    def _upload_bn(self, bn):
        """Upload BatchNorm params to GPU."""
        return {
            "gamma": cuda.to_gpu(
                np.ascontiguousarray(bn.weight.detach().cpu().numpy())
            ),
            "beta": cuda.to_gpu(np.ascontiguousarray(bn.bias.detach().cpu().numpy())),
            "mean": cuda.to_gpu(
                np.ascontiguousarray(bn.running_mean.detach().cpu().numpy())
            ),
            "var": cuda.to_gpu(
                np.ascontiguousarray(bn.running_var.detach().cpu().numpy())
            ),
            "eps": bn.eps,
        }

    def _upload_bottleneck(self, block):
        """Upload all bottleneck weights to GPU."""
        result = {
            "conv1_w": cuda.to_gpu(
                np.ascontiguousarray(block.conv1.weight.detach().cpu().numpy())
            ),
            "bn1": self._upload_bn(block.bn1),
            "conv2_w": cuda.to_gpu(
                np.ascontiguousarray(block.conv2.weight.detach().cpu().numpy())
            ),
            "conv2_s": block.conv2.stride[0],
            "conv2_p": block.conv2.padding[0],
            "bn2": self._upload_bn(block.bn2),
            "conv3_w": cuda.to_gpu(
                np.ascontiguousarray(block.conv3.weight.detach().cpu().numpy())
            ),
            "bn3": self._upload_bn(block.bn3),
            "downsample": None,
        }

        if block.downsample is not None:
            result["downsample"] = {
                "conv_w": cuda.to_gpu(
                    np.ascontiguousarray(
                        block.downsample[0].weight.detach().cpu().numpy()
                    )
                ),
                "conv_s": block.downsample[0].stride[0],
                "bn": self._upload_bn(block.downsample[1]),
            }

        return result

    def _bn_gpu(self, x, bn):
        """Apply batch norm on GPU tensor."""
        return cuda.batch_norm_gpu(
            x, bn["gamma"], bn["beta"], bn["mean"], bn["var"], bn["eps"]
        )

    def _bottleneck_gpu(self, x, block):
        """Run bottleneck on GPU tensors."""
        identity = x

        # Conv1 -> BN1 -> ReLU
        out = cuda.conv2d_gpu(x, block["conv1_w"], stride=1, padding=0)
        out = self._bn_gpu(out, block["bn1"])
        out = cuda.relu_gpu(out)

        # Conv2 -> BN2 -> ReLU
        out = cuda.conv2d_gpu(
            out, block["conv2_w"], stride=block["conv2_s"], padding=block["conv2_p"]
        )
        out = self._bn_gpu(out, block["bn2"])
        out = cuda.relu_gpu(out)

        # Conv3 -> BN3
        out = cuda.conv2d_gpu(out, block["conv3_w"], stride=1, padding=0)
        out = self._bn_gpu(out, block["bn3"])

        # Downsample
        if block["downsample"] is not None:
            ds = block["downsample"]
            identity = cuda.conv2d_gpu(x, ds["conv_w"], stride=ds["conv_s"], padding=0)
            identity = self._bn_gpu(identity, ds["bn"])

        # Add + ReLU
        out = cuda.add_gpu(out, identity)
        out = cuda.relu_gpu(out)
        return out

    def forward(self, x_np):
        """
        Forward pass with GPU-resident tensors.
        Only one H2D at start, one D2H at end.
        """
        # Single H2D transfer at start
        x = cuda.to_gpu(np.ascontiguousarray(x_np.astype(np.float32)))

        # Stem (all on GPU)
        x = cuda.conv2d_gpu(x, self.conv1_w, stride=2, padding=3)
        x = self._bn_gpu(x, self.bn1)
        x = cuda.relu_gpu(x)
        x = cuda.maxpool2d_gpu(x, kernel_size=3, stride=2, padding=1)

        # Residual layers (all on GPU)
        for block in self.layer1:
            x = self._bottleneck_gpu(x, block)
        for block in self.layer2:
            x = self._bottleneck_gpu(x, block)
        for block in self.layer3:
            x = self._bottleneck_gpu(x, block)
        for block in self.layer4:
            x = self._bottleneck_gpu(x, block)

        # Global average pool
        x = cuda.global_avgpool_gpu(x)

        # Single D2H transfer at end
        x_np = x.to_numpy()
        x_flat = x_np.reshape(x_np.shape[0], -1)

        # FC (CPU for now)
        output = x_flat @ self.fc_w.T + self.fc_b
        return output

    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":
    import torch
    import torchvision.models as models

    # Load PyTorch model
    print("\n[1/5] Loading PyTorch ResNet-50...")
    torch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_model.eval()
    torch_model.cuda()
    print(f"  Loaded: {sum(p.numel() for p in torch_model.parameters()):,} parameters")

    # Create GPU-resident Zenith model
    print("\n[2/5] Creating GPU-resident Zenith model...")
    zenith_model = ZenithResNet50GPU(torch_model)

    # Prepare input
    print("\n[3/5] Preparing input...")
    np.random.seed(42)
    torch.manual_seed(42)
    input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_torch = torch.from_numpy(input_np).cuda()
    print(f"  Input: {input_np.shape}")

    # Run inference
    print("\n[4/5] Running inference...")

    # PyTorch reference
    print("  Running PyTorch...")
    with torch.no_grad():
        torch_output = torch_model(input_torch).cpu().numpy()

    # Zenith GPU-resident
    print("  Running Zenith (GPU-resident)...")
    t0 = time.perf_counter()
    zenith_output = zenith_model(input_np)
    zenith_time = (time.perf_counter() - t0) * 1000
    print(f"  Zenith inference time: {zenith_time:.2f} ms")

    # Verify accuracy
    print("\n[5/5] Verifying accuracy...")
    max_diff = np.max(np.abs(zenith_output - torch_output))
    mean_diff = np.mean(np.abs(zenith_output - torch_output))

    print(f"  Output shape: {zenith_output.shape}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    torch_top5 = np.argsort(torch_output[0])[-5:][::-1]
    zenith_top5 = np.argsort(zenith_output[0])[-5:][::-1]
    top5_match = np.array_equal(torch_top5, zenith_top5)

    print(f"  PyTorch top-5: {torch_top5}")
    print(f"  Zenith top-5:  {zenith_top5}")
    print(f"  Top-5 match: {'PASS' if top5_match else 'FAIL'}")

    # Performance benchmark
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        _ = zenith_model(input_np)

    # Benchmark Zenith GPU-resident
    print("Benchmarking Zenith GPU-RESIDENT (10 iterations)...")
    zenith_times = []
    for i in range(10):
        t0 = time.perf_counter()
        _ = zenith_model(input_np)
        zenith_times.append((time.perf_counter() - t0) * 1000)
        print(f"  Iter {i + 1}: {zenith_times[-1]:.2f} ms")

    # Benchmark PyTorch
    print("\nBenchmarking PyTorch (10 iterations)...")
    torch.cuda.synchronize()
    pytorch_times = []
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch_model(input_torch)
        torch.cuda.synchronize()
        pytorch_times.append((time.perf_counter() - t0) * 1000)
        print(f"  Iter {i + 1}: {pytorch_times[-1]:.2f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"\nModel: ResNet-50 ({sum(p.numel() for p in torch_model.parameters()):,} parameters)"
    )
    print(f"Input: {input_np.shape}")

    baseline = 225.16  # Previous non-GPU-resident time
    pytorch_avg = np.mean(pytorch_times)
    zenith_avg = np.mean(zenith_times)

    print(f"\nLatency:")
    print(f"  PyTorch:            {pytorch_avg:.2f} ± {np.std(pytorch_times):.2f} ms")
    print(f"  Zenith GPU-RESIDENT: {zenith_avg:.2f} ± {np.std(zenith_times):.2f} ms")
    print(f"  Ratio vs PyTorch:   {zenith_avg / pytorch_avg:.2f}x")

    print(f"\nSpeedup from baseline (CPU-transfer):")
    print(f"  Baseline:  {baseline:.2f} ms")
    print(f"  GPU-Resident: {zenith_avg:.2f} ms")
    speedup = (baseline - zenith_avg) / baseline * 100
    print(f"  Speedup:   {speedup:.1f}%")

    print(f"\nAccuracy:")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Top-5 match: {'PASS' if top5_match else 'FAIL'}")
    print("=" * 70)

    # Memory stats
    print("\nGPU Memory Pool Stats:")
    stats = cuda.memory_stats()
    print(f"  Allocations: {stats['allocations']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache returns: {stats['cache_returns']}")
    print(f"  Total allocated: {stats['total_allocated'] / 1024 / 1024:.1f} MB")
