# Full ResNet-50 Implementation Using Zenith Operators
# Uses the PROVEN CORRECT implementation from debug_resnet_extended.py
# Loads pretrained weights from PyTorch and runs full inference

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH FULL ResNet-50 IMPLEMENTATION")
print("=" * 70)

from zenith._zenith_core import cuda

# ============================================================================
# Helper Functions (PROVEN CORRECT from debug_resnet_extended.py)
# ============================================================================


def extract_bn(bn):
    """Extract batch norm parameters - all as contiguous arrays"""
    return (
        np.ascontiguousarray(bn.weight.detach().cpu().numpy()),
        np.ascontiguousarray(bn.bias.detach().cpu().numpy()),
        np.ascontiguousarray(bn.running_mean.detach().cpu().numpy()),
        np.ascontiguousarray(bn.running_var.detach().cpu().numpy()),
        bn.eps,
    )


def zenith_bottleneck(x, block):
    """Run a bottleneck block with Zenith - PROVEN CORRECT implementation"""
    x = np.ascontiguousarray(x.astype(np.float32))
    identity = x

    # Conv1 (1x1)
    w1 = np.ascontiguousarray(block.conv1.weight.detach().cpu().numpy())
    bn1 = extract_bn(block.bn1)
    out = cuda.conv2d(x, w1, stride=1, padding=0)
    out = cuda.batch_norm(out, bn1[0], bn1[1], bn1[2], bn1[3], bn1[4])
    out = cuda.relu(out)

    # Conv2 (3x3)
    w2 = np.ascontiguousarray(block.conv2.weight.detach().cpu().numpy())
    s2 = block.conv2.stride[0]
    p2 = block.conv2.padding[0]
    bn2 = extract_bn(block.bn2)
    out = cuda.conv2d(out, w2, stride=s2, padding=p2)
    out = cuda.batch_norm(out, bn2[0], bn2[1], bn2[2], bn2[3], bn2[4])
    out = cuda.relu(out)

    # Conv3 (1x1)
    w3 = np.ascontiguousarray(block.conv3.weight.detach().cpu().numpy())
    bn3 = extract_bn(block.bn3)
    out = cuda.conv2d(out, w3, stride=1, padding=0)
    out = cuda.batch_norm(out, bn3[0], bn3[1], bn3[2], bn3[3], bn3[4])

    # Downsample
    if block.downsample is not None:
        ds_w = np.ascontiguousarray(block.downsample[0].weight.detach().cpu().numpy())
        ds_s = block.downsample[0].stride[0]
        ds_bn = extract_bn(block.downsample[1])
        identity = cuda.conv2d(x, ds_w, stride=ds_s, padding=0)
        identity = cuda.batch_norm(
            identity, ds_bn[0], ds_bn[1], ds_bn[2], ds_bn[3], ds_bn[4]
        )

    # Add + ReLU
    out = cuda.add(out, identity)
    out = cuda.relu(out)
    return out


def zenith_resnet50_forward(x, model):
    """Full ResNet-50 forward pass - PROVEN CORRECT implementation"""
    x = np.ascontiguousarray(x.astype(np.float32))

    # Stem: conv1 -> bn1 -> relu -> maxpool
    conv1_w = np.ascontiguousarray(model.conv1.weight.detach().cpu().numpy())
    bn1 = extract_bn(model.bn1)

    x = cuda.conv2d(x, conv1_w, stride=2, padding=3)
    x = cuda.batch_norm(x, bn1[0], bn1[1], bn1[2], bn1[3], bn1[4])
    x = cuda.relu(x)
    x = cuda.maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # Layer groups
    for block in model.layer1:
        x = zenith_bottleneck(x, block)
    for block in model.layer2:
        x = zenith_bottleneck(x, block)
    for block in model.layer3:
        x = zenith_bottleneck(x, block)
    for block in model.layer4:
        x = zenith_bottleneck(x, block)

    # Global average pool
    x = cuda.global_avgpool(x)
    x = x.reshape(x.shape[0], -1)  # Flatten to [N, C]

    # FC layer (using numpy matmul for now)
    fc_w = np.ascontiguousarray(model.fc.weight.detach().cpu().numpy())
    fc_b = np.ascontiguousarray(model.fc.bias.detach().cpu().numpy())
    x = x @ fc_w.T + fc_b

    return x


# ============================================================================
# Load PyTorch Model
# ============================================================================
print("\n[1/5] Loading PyTorch ResNet-50...")

import torch
import torchvision.models as models

torch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_model.cuda()

print(f"  Loaded: {sum(p.numel() for p in torch_model.parameters()):,} parameters")

# ============================================================================
# Prepare Input
# ============================================================================
print("\n[2/5] Preparing input...")

# Use fixed seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

batch_size = 1
input_np = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
input_torch = torch.from_numpy(input_np).cuda()

print(f"  Input: {input_np.shape}")

# ============================================================================
# Run Inference
# ============================================================================
print("\n[3/5] Running inference...")

# PyTorch reference
print("  Running PyTorch...")
with torch.no_grad():
    torch_output = torch_model(input_torch).cpu().numpy()

# Zenith full inference
print("  Running Zenith...")
t0 = time.perf_counter()
zenith_output = zenith_resnet50_forward(input_np, torch_model)
zenith_time = (time.perf_counter() - t0) * 1000

print(f"  Zenith inference time: {zenith_time:.2f} ms")

# ============================================================================
# Verify Accuracy
# ============================================================================
print("\n[4/5] Verifying accuracy...")

max_diff = np.max(np.abs(zenith_output - torch_output))
mean_diff = np.mean(np.abs(zenith_output - torch_output))

print(f"  Output shape: {zenith_output.shape}")
print(f"  Max diff: {max_diff:.2e}")
print(f"  Mean diff: {mean_diff:.2e}")

# Check top-5 predictions match
torch_top5 = np.argsort(torch_output[0])[-5:][::-1]
zenith_top5 = np.argsort(zenith_output[0])[-5:][::-1]

print(f"  PyTorch top-5: {torch_top5}")
print(f"  Zenith top-5:  {zenith_top5}")
top5_match = np.array_equal(torch_top5, zenith_top5)
print(f"  Top-5 match: {'PASS' if top5_match else 'FAIL'}")

# ============================================================================
# Performance Benchmark
# ============================================================================
print("\n[5/5] Performance benchmark...")
print("=" * 70)

# Warmup
print("\nWarming up...")
for _ in range(3):
    _ = zenith_resnet50_forward(input_np, torch_model)

# Benchmark Zenith
print("Benchmarking Zenith (10 iterations)...")
zenith_times = []
for i in range(10):
    t0 = time.perf_counter()
    _ = zenith_resnet50_forward(input_np, torch_model)
    zenith_times.append((time.perf_counter() - t0) * 1000)
    print(f"  Iter {i + 1}: {zenith_times[-1]:.2f} ms")

# PyTorch benchmark
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

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(
    f"\nModel: ResNet-50 ({sum(p.numel() for p in torch_model.parameters()):,} parameters)"
)
print(f"Input: {input_np.shape}")
print(f"\nLatency:")
print(f"  PyTorch: {np.mean(pytorch_times):.2f} ± {np.std(pytorch_times):.2f} ms")
print(f"  Zenith:  {np.mean(zenith_times):.2f} ± {np.std(zenith_times):.2f} ms")
print(f"  Ratio:   {np.mean(zenith_times) / np.mean(pytorch_times):.2f}x")
print(f"\nAccuracy:")
print(f"  Max diff: {max_diff:.2e}")
print(f"  Top-5 match: {'PASS' if top5_match else 'FAIL'}")
print("=" * 70)
