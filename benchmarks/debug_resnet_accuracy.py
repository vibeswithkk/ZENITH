# Debug ResNet-50 layer-by-layer to find accuracy divergence
# Run in Colab: python benchmarks/debug_resnet_accuracy.py

import sys

sys.path.insert(0, "build/python")

import numpy as np
import torch
import torchvision.models as models

print("=" * 70)
print("ZENITH vs PyTorch LAYER-BY-LAYER COMPARISON")
print("=" * 70)

from zenith._zenith_core import cuda

# Load PyTorch model
print("\n[1/5] Loading PyTorch ResNet-50...")
torch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_model.cuda()

# Fixed input for reproducibility
np.random.seed(42)
torch.manual_seed(42)
input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_torch = torch.from_numpy(input_np).cuda()


def extract_bn(bn):
    return (
        bn.weight.detach().cpu().numpy(),
        bn.bias.detach().cpu().numpy(),
        bn.running_mean.detach().cpu().numpy(),
        bn.running_var.detach().cpu().numpy(),
        bn.eps,
    )


def compare(name, zenith_out, torch_out):
    """Compare Zenith and PyTorch outputs"""
    max_diff = np.max(np.abs(zenith_out - torch_out))
    mean_diff = np.mean(np.abs(zenith_out - torch_out))
    status = "PASS" if max_diff < 1e-3 else ("WARN" if max_diff < 1e-1 else "FAIL")
    print(f"  {name}: max={max_diff:.2e}, mean={mean_diff:.2e} [{status}]")
    return max_diff


# ============================================================================
# Test 1: Conv1 only
# ============================================================================
print("\n[2/5] Testing CONV1 only...")

conv1_w = torch_model.conv1.weight.detach().cpu().numpy()
conv1_w = np.ascontiguousarray(conv1_w)

# Zenith conv1
z_out = cuda.conv2d(np.ascontiguousarray(input_np), conv1_w, stride=2, padding=3)

# PyTorch conv1
with torch.no_grad():
    t_out = torch_model.conv1(input_torch).cpu().numpy()

compare("Conv1", z_out, t_out)

# ============================================================================
# Test 2: Conv1 + BN1
# ============================================================================
print("\n[3/5] Testing CONV1 + BN1...")

bn1_params = extract_bn(torch_model.bn1)

# Zenith
z_out = cuda.conv2d(np.ascontiguousarray(input_np), conv1_w, stride=2, padding=3)
z_out = cuda.batch_norm(
    z_out, bn1_params[0], bn1_params[1], bn1_params[2], bn1_params[3], bn1_params[4]
)

# PyTorch
with torch.no_grad():
    t_out = torch_model.bn1(torch_model.conv1(input_torch)).cpu().numpy()

compare("Conv1+BN1", z_out, t_out)

# ============================================================================
# Test 3: Conv1 + BN1 + ReLU
# ============================================================================
print("\n[4/5] Testing CONV1 + BN1 + ReLU...")

z_out = cuda.relu(z_out)

with torch.no_grad():
    t_out = (
        torch_model.relu(torch_model.bn1(torch_model.conv1(input_torch))).cpu().numpy()
    )

compare("Conv1+BN1+ReLU", z_out, t_out)

# ============================================================================
# Test 4: First Block (Stem)
# ============================================================================
print("\n[5/5] Testing STEM (conv1+bn1+relu+maxpool)...")

# Zenith full stem
x = np.ascontiguousarray(input_np)
x = cuda.conv2d(x, conv1_w, stride=2, padding=3)
x = cuda.batch_norm(
    x, bn1_params[0], bn1_params[1], bn1_params[2], bn1_params[3], bn1_params[4]
)
x = cuda.relu(x)
z_stem = cuda.maxpool2d(x, kernel_size=3, stride=2, padding=1)

# PyTorch stem
with torch.no_grad():
    t_stem = (
        torch_model.maxpool(
            torch_model.relu(torch_model.bn1(torch_model.conv1(input_torch)))
        )
        .cpu()
        .numpy()
    )

compare("Stem", z_stem, t_stem)
print(f"  Shapes: Zenith={z_stem.shape}, PyTorch={t_stem.shape}")

# ============================================================================
# Test 5: First Bottleneck Block
# ============================================================================
print("\n[BONUS] Testing Layer1 Block0...")

block = torch_model.layer1[0]

# Get stem output from PyTorch as input to layer1
with torch.no_grad():
    stem_out = torch_model.maxpool(
        torch_model.relu(torch_model.bn1(torch_model.conv1(input_torch)))
    )

x_np = stem_out.cpu().numpy()
x_np = np.ascontiguousarray(x_np)

# Bottleneck conv1 (1x1)
conv1_w = np.ascontiguousarray(block.conv1.weight.detach().cpu().numpy())
bn1_p = extract_bn(block.bn1)

z = cuda.conv2d(x_np, conv1_w, stride=1, padding=0)
z = cuda.batch_norm(z, bn1_p[0], bn1_p[1], bn1_p[2], bn1_p[3], bn1_p[4])
z = cuda.relu(z)

# PyTorch
with torch.no_grad():
    t = block.relu(block.bn1(block.conv1(stem_out))).cpu().numpy()

compare("Layer1.0.conv1", z, t)

# Bottleneck conv2 (3x3)
conv2_w = np.ascontiguousarray(block.conv2.weight.detach().cpu().numpy())
conv2_s = block.conv2.stride[0]
conv2_p = block.conv2.padding[0]
bn2_p = extract_bn(block.bn2)

z = cuda.conv2d(z, conv2_w, stride=conv2_s, padding=conv2_p)
z = cuda.batch_norm(z, bn2_p[0], bn2_p[1], bn2_p[2], bn2_p[3], bn2_p[4])
z = cuda.relu(z)

with torch.no_grad():
    t_inter = block.relu(block.bn1(block.conv1(stem_out)))
    t = block.relu(block.bn2(block.conv2(t_inter))).cpu().numpy()

compare("Layer1.0.conv2", z, t)

# Bottleneck conv3 (1x1)
conv3_w = np.ascontiguousarray(block.conv3.weight.detach().cpu().numpy())
bn3_p = extract_bn(block.bn3)

z = cuda.conv2d(z, conv3_w, stride=1, padding=0)
z_pre_add = cuda.batch_norm(z, bn3_p[0], bn3_p[1], bn3_p[2], bn3_p[3], bn3_p[4])

with torch.no_grad():
    t_inter2 = block.relu(
        block.bn2(block.conv2(block.relu(block.bn1(block.conv1(stem_out)))))
    )
    t_pre_add = block.bn3(block.conv3(t_inter2)).cpu().numpy()

compare("Layer1.0.conv3 (pre-add)", z_pre_add, t_pre_add)

# Downsample path
if block.downsample is not None:
    ds_conv_w = np.ascontiguousarray(block.downsample[0].weight.detach().cpu().numpy())
    ds_conv_s = block.downsample[0].stride[0]
    ds_bn_p = extract_bn(block.downsample[1])

    z_identity = cuda.conv2d(x_np, ds_conv_w, stride=ds_conv_s, padding=0)
    z_identity = cuda.batch_norm(
        z_identity, ds_bn_p[0], ds_bn_p[1], ds_bn_p[2], ds_bn_p[3], ds_bn_p[4]
    )

    with torch.no_grad():
        t_identity = block.downsample(stem_out).cpu().numpy()

    compare("Layer1.0.downsample", z_identity, t_identity)

    # Add + ReLU
    z_out = cuda.add(z_pre_add, z_identity)
    z_out = cuda.relu(z_out)

    with torch.no_grad():
        t_out = block(stem_out).cpu().numpy()

    compare("Layer1.0 FULL", z_out, t_out)

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
print("\nIf any layer shows FAIL/WARN, that's where the bug is!")
