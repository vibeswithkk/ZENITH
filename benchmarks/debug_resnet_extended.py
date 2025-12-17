# Extended debug - test all layer groups to find divergence
# Run in Colab: python benchmarks/debug_resnet_extended.py

import sys

sys.path.insert(0, "build/python")

import numpy as np
import torch
import torchvision.models as models

print("=" * 70)
print("ZENITH vs PyTorch - EXTENDED LAYER COMPARISON")
print("=" * 70)

from zenith._zenith_core import cuda


def extract_bn(bn):
    return (
        np.ascontiguousarray(bn.weight.detach().cpu().numpy()),
        np.ascontiguousarray(bn.bias.detach().cpu().numpy()),
        np.ascontiguousarray(bn.running_mean.detach().cpu().numpy()),
        np.ascontiguousarray(bn.running_var.detach().cpu().numpy()),
        bn.eps,
    )


def compare(name, zenith_out, torch_out):
    max_diff = np.max(np.abs(zenith_out - torch_out))
    mean_diff = np.mean(np.abs(zenith_out - torch_out))
    status = "PASS" if max_diff < 1e-4 else ("WARN" if max_diff < 1e-1 else "FAIL")
    print(f"  {name}: max={max_diff:.2e}, mean={mean_diff:.2e} [{status}]")
    if status == "FAIL":
        print(f"    Zenith range: [{zenith_out.min():.4f}, {zenith_out.max():.4f}]")
        print(f"    PyTorch range: [{torch_out.min():.4f}, {torch_out.max():.4f}]")
    return max_diff, status


def zenith_bottleneck(x, block):
    """Run a bottleneck block with Zenith"""
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


# Load model
print("\nLoading PyTorch ResNet-50...")
torch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_model.cuda()

# Fixed input
np.random.seed(42)
torch.manual_seed(42)
input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_torch = torch.from_numpy(input_np).cuda()

# Stem
print("\n[STEM] Testing stem...")
conv1_w = np.ascontiguousarray(torch_model.conv1.weight.detach().cpu().numpy())
bn1 = extract_bn(torch_model.bn1)

z = np.ascontiguousarray(input_np)
z = cuda.conv2d(z, conv1_w, stride=2, padding=3)
z = cuda.batch_norm(z, bn1[0], bn1[1], bn1[2], bn1[3], bn1[4])
z = cuda.relu(z)
z = cuda.maxpool2d(z, kernel_size=3, stride=2, padding=1)

with torch.no_grad():
    t = torch_model.maxpool(
        torch_model.relu(torch_model.bn1(torch_model.conv1(input_torch)))
    )

compare("Stem", z, t.cpu().numpy())


# Test each layer group
layer_groups = [
    ("layer1", torch_model.layer1),
    ("layer2", torch_model.layer2),
    ("layer3", torch_model.layer3),
    ("layer4", torch_model.layer4),
]

z_out = z  # Zenith output (starts from stem)
t_out = t  # PyTorch tensor

for layer_name, layer in layer_groups:
    print(f"\n[{layer_name.upper()}] Testing {len(layer)} blocks...")

    for i, block in enumerate(layer):
        # Zenith
        z_out = zenith_bottleneck(z_out, block)

        # PyTorch
        with torch.no_grad():
            t_out = block(t_out)

        max_diff, status = compare(f"{layer_name}.{i}", z_out, t_out.cpu().numpy())

        if status == "FAIL":
            print(f"    --> STOPPING: Found divergence at {layer_name}.{i}")
            break
    else:
        continue
    break  # Break outer loop if inner broke

# Final comparison
print("\n[FINAL] After all layers...")
compare("Full backbone", z_out, t_out.cpu().numpy())

# Global avgpool + FC
print("\n[CLASSIFIER] Testing classifier...")
z_pool = cuda.global_avgpool(z_out)
z_flat = z_pool.reshape(1, -1)

with torch.no_grad():
    t_pool = torch_model.avgpool(t_out)
    t_flat = torch.flatten(t_pool, 1)

compare("Global AvgPool", z_pool.reshape(-1), t_flat.cpu().numpy().reshape(-1))

# FC layer
fc_w = np.ascontiguousarray(torch_model.fc.weight.detach().cpu().numpy())
fc_b = np.ascontiguousarray(torch_model.fc.bias.detach().cpu().numpy())

# Manual FC: output = input @ W^T + bias
z_fc = z_flat @ fc_w.T + fc_b

with torch.no_grad():
    t_fc = torch_model.fc(t_flat).cpu().numpy()

compare("FC output", z_fc, t_fc)

# Top-5 comparison
z_top5 = np.argsort(z_fc[0])[-5:][::-1]
t_top5 = np.argsort(t_fc[0])[-5:][::-1]
print(f"\n  Zenith top-5: {z_top5}")
print(f"  PyTorch top-5: {t_top5}")
print(f"  Match: {'YES' if np.array_equal(z_top5, t_top5) else 'NO'}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
