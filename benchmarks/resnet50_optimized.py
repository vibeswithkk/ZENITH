# Optimized ResNet-50 Implementation with Weight Caching
# Based on NVIDIA cuDNN/TensorRT best practices research
# Key optimization: Pre-extract and cache weights at initialization
# Expected speedup: 50%+ from baseline

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH OPTIMIZED ResNet-50 (Weight Caching)")
print("=" * 70)

from zenith._zenith_core import cuda


# ============================================================================
# ZenithResNet50: Optimized Inference Class with Weight Caching
# ============================================================================


class ZenithResNet50:
    """
    Optimized ResNet-50 inference using Zenith operators.

    Key optimization: All weights are extracted and cached at initialization.
    This eliminates the overhead of .detach().cpu().numpy() during inference.

    Based on NVIDIA best practices:
    - Pre-allocate and reuse device buffers
    - Cache frequently accessed data
    - Reduce CPU-GPU transfer overhead
    """

    def __init__(self, pytorch_model):
        """
        Initialize with a PyTorch ResNet-50 model.
        Extracts and caches all weights as contiguous numpy arrays.
        """
        print("  Caching weights...")
        start = time.perf_counter()

        # Stem weights
        self.conv1_w = np.ascontiguousarray(
            pytorch_model.conv1.weight.detach().cpu().numpy()
        )
        self.bn1 = self._extract_bn(pytorch_model.bn1)

        # Layer weights (3, 4, 6, 3 blocks)
        self.layer1 = [self._extract_bottleneck(b) for b in pytorch_model.layer1]
        self.layer2 = [self._extract_bottleneck(b) for b in pytorch_model.layer2]
        self.layer3 = [self._extract_bottleneck(b) for b in pytorch_model.layer3]
        self.layer4 = [self._extract_bottleneck(b) for b in pytorch_model.layer4]

        # FC weights
        self.fc_w = np.ascontiguousarray(pytorch_model.fc.weight.detach().cpu().numpy())
        self.fc_b = np.ascontiguousarray(pytorch_model.fc.bias.detach().cpu().numpy())

        cache_time = (time.perf_counter() - start) * 1000
        print(f"  Weights cached in {cache_time:.1f} ms")

        # Calculate memory usage
        total_bytes = (
            self.conv1_w.nbytes
            + sum(p.nbytes for p in self.bn1[:4])
            + sum(sum(p.nbytes for p in self._flatten_block(b)) for b in self.layer1)
            + sum(sum(p.nbytes for p in self._flatten_block(b)) for b in self.layer2)
            + sum(sum(p.nbytes for p in self._flatten_block(b)) for b in self.layer3)
            + sum(sum(p.nbytes for p in self._flatten_block(b)) for b in self.layer4)
            + self.fc_w.nbytes
            + self.fc_b.nbytes
        )
        print(f"  Memory usage: {total_bytes / 1024 / 1024:.1f} MB")

    def _extract_bn(self, bn):
        """Extract batch norm parameters as contiguous arrays."""
        return (
            np.ascontiguousarray(bn.weight.detach().cpu().numpy()),
            np.ascontiguousarray(bn.bias.detach().cpu().numpy()),
            np.ascontiguousarray(bn.running_mean.detach().cpu().numpy()),
            np.ascontiguousarray(bn.running_var.detach().cpu().numpy()),
            bn.eps,
        )

    def _extract_bottleneck(self, block):
        """Extract all weights for a bottleneck block."""
        # Conv1 (1x1)
        conv1_w = np.ascontiguousarray(block.conv1.weight.detach().cpu().numpy())
        bn1 = self._extract_bn(block.bn1)

        # Conv2 (3x3)
        conv2_w = np.ascontiguousarray(block.conv2.weight.detach().cpu().numpy())
        conv2_s = block.conv2.stride[0]
        conv2_p = block.conv2.padding[0]
        bn2 = self._extract_bn(block.bn2)

        # Conv3 (1x1)
        conv3_w = np.ascontiguousarray(block.conv3.weight.detach().cpu().numpy())
        bn3 = self._extract_bn(block.bn3)

        # Downsample (optional)
        downsample = None
        if block.downsample is not None:
            ds_w = np.ascontiguousarray(
                block.downsample[0].weight.detach().cpu().numpy()
            )
            ds_s = block.downsample[0].stride[0]
            ds_bn = self._extract_bn(block.downsample[1])
            downsample = (ds_w, ds_s, ds_bn)

        return {
            "conv1_w": conv1_w,
            "bn1": bn1,
            "conv2_w": conv2_w,
            "conv2_s": conv2_s,
            "conv2_p": conv2_p,
            "bn2": bn2,
            "conv3_w": conv3_w,
            "bn3": bn3,
            "downsample": downsample,
        }

    def _flatten_block(self, block):
        """Flatten block weights for memory calculation."""
        params = [block["conv1_w"], block["conv2_w"], block["conv3_w"]]
        for bn_key in ["bn1", "bn2", "bn3"]:
            params.extend(block[bn_key][:4])
        if block["downsample"]:
            params.append(block["downsample"][0])
            params.extend(block["downsample"][2][:4])
        return params

    def _run_bottleneck(self, x, block):
        """Run a cached bottleneck block."""
        x = np.ascontiguousarray(x.astype(np.float32))
        identity = x

        # Conv1 -> BN1 -> ReLU
        out = cuda.conv2d(x, block["conv1_w"], stride=1, padding=0)
        bn1 = block["bn1"]
        out = cuda.batch_norm(out, bn1[0], bn1[1], bn1[2], bn1[3], bn1[4])
        out = cuda.relu(out)

        # Conv2 -> BN2 -> ReLU
        out = cuda.conv2d(
            out, block["conv2_w"], stride=block["conv2_s"], padding=block["conv2_p"]
        )
        bn2 = block["bn2"]
        out = cuda.batch_norm(out, bn2[0], bn2[1], bn2[2], bn2[3], bn2[4])
        out = cuda.relu(out)

        # Conv3 -> BN3 (no ReLU before add)
        out = cuda.conv2d(out, block["conv3_w"], stride=1, padding=0)
        bn3 = block["bn3"]
        out = cuda.batch_norm(out, bn3[0], bn3[1], bn3[2], bn3[3], bn3[4])

        # Downsample if needed
        if block["downsample"] is not None:
            ds_w, ds_s, ds_bn = block["downsample"]
            identity = cuda.conv2d(x, ds_w, stride=ds_s, padding=0)
            identity = cuda.batch_norm(
                identity, ds_bn[0], ds_bn[1], ds_bn[2], ds_bn[3], ds_bn[4]
            )

        # Add + ReLU
        out = cuda.add(out, identity)
        out = cuda.relu(out)
        return out

    def forward(self, x):
        """
        Run optimized forward pass.
        No weight extraction during inference - uses cached weights only.
        """
        x = np.ascontiguousarray(x.astype(np.float32))

        # Stem: Conv1 -> BN1 -> ReLU -> MaxPool
        x = cuda.conv2d(x, self.conv1_w, stride=2, padding=3)
        x = cuda.batch_norm(
            x, self.bn1[0], self.bn1[1], self.bn1[2], self.bn1[3], self.bn1[4]
        )
        x = cuda.relu(x)
        x = cuda.maxpool2d(x, kernel_size=3, stride=2, padding=1)

        # Residual layers (using cached weights)
        for block in self.layer1:
            x = self._run_bottleneck(x, block)
        for block in self.layer2:
            x = self._run_bottleneck(x, block)
        for block in self.layer3:
            x = self._run_bottleneck(x, block)
        for block in self.layer4:
            x = self._run_bottleneck(x, block)

        # Global average pool
        x = cuda.global_avgpool(x)
        x = x.reshape(x.shape[0], -1)

        # FC (using cached weights)
        x = x @ self.fc_w.T + self.fc_b

        return x

    def __call__(self, x):
        return self.forward(x)


# ============================================================================
# Main Benchmark
# ============================================================================

if __name__ == "__main__":
    import torch
    import torchvision.models as models

    # Load PyTorch model
    print("\n[1/5] Loading PyTorch ResNet-50...")
    torch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_model.eval()
    torch_model.cuda()
    print(f"  Loaded: {sum(p.numel() for p in torch_model.parameters()):,} parameters")

    # Create optimized Zenith model
    print("\n[2/5] Creating optimized Zenith model...")
    zenith_model = ZenithResNet50(torch_model)

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

    # Zenith optimized
    print("  Running Zenith (optimized)...")
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

    # Benchmark Zenith optimized
    print("Benchmarking Zenith OPTIMIZED (10 iterations)...")
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
    print(f"\nLatency:")
    print(
        f"  PyTorch:          {np.mean(pytorch_times):.2f} ± {np.std(pytorch_times):.2f} ms"
    )
    print(
        f"  Zenith OPTIMIZED: {np.mean(zenith_times):.2f} ± {np.std(zenith_times):.2f} ms"
    )
    print(f"  Ratio:            {np.mean(zenith_times) / np.mean(pytorch_times):.2f}x")
    print(f"\nSpeedup from baseline (225ms):")
    baseline = 225.16
    optimized = np.mean(zenith_times)
    speedup = (baseline - optimized) / baseline * 100
    print(f"  Baseline:  {baseline:.2f} ms")
    print(f"  Optimized: {optimized:.2f} ms")
    print(f"  Speedup:   {speedup:.1f}%")
    print(f"\nAccuracy:")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Top-5 match: {'PASS' if top5_match else 'FAIL'}")
    print("=" * 70)
