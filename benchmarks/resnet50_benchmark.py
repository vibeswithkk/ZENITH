# End-to-End ResNet-50 Benchmark with Zenith
# Loads pretrained weights from PyTorch and runs inference using Zenith operators
# Compares performance with native PyTorch

import numpy as np
import time
import sys

sys.path.insert(0, "build/python")

print("=" * 70)
print("ZENITH ResNet-50 END-TO-END BENCHMARK")
print("=" * 70)

# ============================================================================
# Load PyTorch ResNet-50 for reference
# ============================================================================
print("\n[1/4] Loading PyTorch ResNet-50...")

import torch
import torchvision.models as models

# Load pretrained ResNet-50
torch_model = models.resnet50(pretrained=True)
torch_model.eval()
torch_model.cuda()

print(
    f"  Model loaded: {sum(p.numel() for p in torch_model.parameters()):,} parameters"
)

# ============================================================================
# Prepare input
# ============================================================================
print("\n[2/4] Preparing input...")

# Standard ImageNet input: [1, 3, 224, 224]
batch_size = 1
input_np = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
input_torch = torch.from_numpy(input_np).cuda()

print(f"  Input shape: {input_np.shape}")

# ============================================================================
# PyTorch Inference Benchmark
# ============================================================================
print("\n[3/4] PyTorch inference benchmark...")

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = torch_model(input_torch)
torch.cuda.synchronize()

# Benchmark
pytorch_times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_torch = torch_model(input_torch)
    torch.cuda.synchronize()
    pytorch_times.append((time.perf_counter() - t0) * 1000)

pytorch_mean = np.mean(pytorch_times)
pytorch_std = np.std(pytorch_times)
print(f"  PyTorch: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")

# ============================================================================
# Zenith Inference (Layer by Layer)
# ============================================================================
print("\n[4/4] Zenith inference benchmark...")

from zenith._zenith_core import cuda


def extract_conv_params(conv_layer):
    """Extract conv layer parameters"""
    weight = conv_layer.weight.detach().cpu().numpy()
    bias = (
        conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else None
    )
    stride = conv_layer.stride[0]
    padding = conv_layer.padding[0]
    return weight, bias, stride, padding


def extract_bn_params(bn_layer):
    """Extract batch norm parameters"""
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta = bn_layer.bias.detach().cpu().numpy()
    mean = bn_layer.running_mean.detach().cpu().numpy()
    var = bn_layer.running_var.detach().cpu().numpy()
    eps = bn_layer.eps
    return gamma, beta, mean, var, eps


# Extract first conv layer params for quick test
conv1_weight, conv1_bias, conv1_stride, conv1_padding = extract_conv_params(
    torch_model.conv1
)
bn1_gamma, bn1_beta, bn1_mean, bn1_var, bn1_eps = extract_bn_params(torch_model.bn1)

print(f"  Conv1 weight shape: {conv1_weight.shape}")
print(f"  BN1 gamma shape: {bn1_gamma.shape}")


# Run first few layers with Zenith
def zenith_first_block(x):
    """Run conv1 -> bn1 -> relu -> maxpool using Zenith"""
    # Conv1: 7x7, stride 2, padding 3
    x = cuda.conv2d(x, conv1_weight, stride=conv1_stride, padding=conv1_padding)
    # BN1
    x = cuda.batch_norm(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, bn1_eps)
    # ReLU
    x = cuda.relu(x)
    # MaxPool: 3x3, stride 2, padding 1
    x = cuda.maxpool2d(x, kernel_size=3, stride=2, padding=1)
    return x


# Warmup Zenith
for _ in range(10):
    _ = zenith_first_block(input_np)

# Benchmark Zenith first block
zenith_times = []
for _ in range(100):
    t0 = time.perf_counter()
    output_zenith = zenith_first_block(input_np)
    zenith_times.append((time.perf_counter() - t0) * 1000)

zenith_mean = np.mean(zenith_times)
zenith_std = np.std(zenith_times)
print(f"  Zenith first block: {zenith_mean:.3f} ± {zenith_std:.3f} ms")

# Verify correctness against PyTorch first block
torch_first_block = torch.nn.Sequential(
    torch_model.conv1, torch_model.bn1, torch_model.relu, torch_model.maxpool
).cuda()

with torch.no_grad():
    torch_output = torch_first_block(input_torch).cpu().numpy()

max_diff = np.max(np.abs(output_zenith - torch_output))
print(f"  Max diff vs PyTorch: {max_diff:.2e}")
print(f"  Accuracy: {'PASS' if max_diff < 1e-3 else 'FAIL'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print(f"\nFirst Block (Conv->BN->ReLU->MaxPool):")
print(f"  PyTorch:  Layer integrated in full model")
print(f"  Zenith:   {zenith_mean:.3f} ms per inference")
print(f"  Accuracy: Max diff = {max_diff:.2e}")

print(f"\nFull ResNet-50:")
print(f"  PyTorch:  {pytorch_mean:.3f} ms ({1000 / pytorch_mean:.1f} FPS)")

print("\nNote: Full ResNet-50 with Zenith requires implementing all 50+ layers.")
print("Current implementation verifies operator correctness against PyTorch.")
print("=" * 70)
