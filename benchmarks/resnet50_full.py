# Full ResNet-50 Implementation Using Zenith Operators
# Implements all 50+ layers: conv1, bn1, 4 layer groups (16 bottleneck blocks), fc
# Loads pretrained weights from PyTorch and runs full inference

import numpy as np
import time
import sys

sys.path.insert(0, "build/python")

print("=" * 70)
print("ZENITH FULL ResNet-50 IMPLEMENTATION")
print("=" * 70)

from zenith._zenith_core import cuda

# ============================================================================
# Helper Functions
# ============================================================================


def extract_conv(conv_layer):
    """Extract conv layer parameters"""
    w = conv_layer.weight.detach().cpu().numpy()
    b = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else None
    s = conv_layer.stride[0]
    p = conv_layer.padding[0]
    return w, b, s, p


def extract_bn(bn_layer):
    """Extract batch norm parameters"""
    return (
        bn_layer.weight.detach().cpu().numpy(),
        bn_layer.bias.detach().cpu().numpy(),
        bn_layer.running_mean.detach().cpu().numpy(),
        bn_layer.running_var.detach().cpu().numpy(),
        bn_layer.eps,
    )


def zenith_conv_bn_relu(x, conv_params, bn_params, relu=True):
    """Conv -> BatchNorm -> ReLU"""
    w, b, s, p = conv_params
    gamma, beta, mean, var, eps = bn_params

    x = cuda.conv2d(x, w, b, stride=s, padding=p)
    x = cuda.batch_norm(x, gamma, beta, mean, var, eps)
    if relu:
        x = cuda.relu(x)
    return x


def zenith_bottleneck(x, block, downsample=None):
    """ResNet Bottleneck block: 1x1 -> 3x3 -> 1x1 with skip connection"""
    identity = x

    # Extract parameters
    conv1_w = block.conv1.weight.detach().cpu().numpy()
    bn1_params = extract_bn(block.bn1)

    conv2_w = block.conv2.weight.detach().cpu().numpy()
    conv2_s = block.conv2.stride[0]
    conv2_p = block.conv2.padding[0]
    bn2_params = extract_bn(block.bn2)

    conv3_w = block.conv3.weight.detach().cpu().numpy()
    bn3_params = extract_bn(block.bn3)

    # Forward pass
    # Conv1: 1x1, stride=1, padding=0
    out = cuda.conv2d(x, conv1_w, stride=1, padding=0)
    out = cuda.batch_norm(out, *bn1_params[:4], bn1_params[4])
    out = cuda.relu(out)

    # Conv2: 3x3, stride varies (1 or 2), padding=1
    out = cuda.conv2d(out, conv2_w, stride=conv2_s, padding=conv2_p)
    out = cuda.batch_norm(out, *bn2_params[:4], bn2_params[4])
    out = cuda.relu(out)

    # Conv3: 1x1, stride=1, padding=0
    out = cuda.conv2d(out, conv3_w, stride=1, padding=0)
    out = cuda.batch_norm(out, *bn3_params[:4], bn3_params[4])

    # Downsample if needed (for dimension matching)
    if downsample is not None:
        ds_conv_w = downsample[0].weight.detach().cpu().numpy()
        ds_conv_s = downsample[0].stride[0]
        ds_bn_params = extract_bn(downsample[1])

        identity = cuda.conv2d(x, ds_conv_w, stride=ds_conv_s, padding=0)
        identity = cuda.batch_norm(identity, *ds_bn_params[:4], ds_bn_params[4])

    # Residual connection
    out = cuda.add(out, identity)
    out = cuda.relu(out)

    return out


def zenith_layer(x, layer):
    """Process an entire layer (sequence of bottleneck blocks)"""
    for i, block in enumerate(layer):
        downsample = (
            block.downsample
            if hasattr(block, "downsample") and block.downsample
            else None
        )
        x = zenith_bottleneck(x, block, downsample)
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
# Extract All Weights
# ============================================================================
print("\n[2/5] Extracting weights...")

# First layers
conv1_params = extract_conv(torch_model.conv1)
bn1_params = extract_bn(torch_model.bn1)

# FC layer
fc_weight = torch_model.fc.weight.detach().cpu().numpy()
fc_bias = torch_model.fc.bias.detach().cpu().numpy()

print(f"  conv1: {conv1_params[0].shape}")
print(f"  fc: {fc_weight.shape}")


# ============================================================================
# Full ResNet-50 Forward Pass
# ============================================================================
def zenith_resnet50(x, model):
    """Full ResNet-50 forward pass using Zenith"""
    # Stem: conv1 -> bn1 -> relu -> maxpool
    x = zenith_conv_bn_relu(x, conv1_params, bn1_params, relu=True)
    x = cuda.maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # Residual layers
    x = zenith_layer(x, model.layer1)
    x = zenith_layer(x, model.layer2)
    x = zenith_layer(x, model.layer3)
    x = zenith_layer(x, model.layer4)

    # Classifier: global avgpool -> flatten -> fc
    x = cuda.global_avgpool(x)

    # Flatten: [N, C, 1, 1] -> [N, C]
    N, C = x.shape[0], x.shape[1]
    x = x.reshape(N, C)

    # FC layer using matmul
    x_gpu = cuda.to_gpu(x)
    w_gpu = cuda.to_gpu(fc_weight.T)
    out_gpu = cuda.matmul_gpu(x_gpu, w_gpu)
    x = out_gpu.to_numpy()
    x = x + fc_bias

    return x


# ============================================================================
# Prepare Input
# ============================================================================
print("\n[3/5] Preparing input...")

batch_size = 1
input_np = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
input_torch = torch.from_numpy(input_np).cuda()

print(f"  Input: {input_np.shape}")

# ============================================================================
# Run Inference
# ============================================================================
print("\n[4/5] Running inference...")

# PyTorch reference
print("  Running PyTorch...")
with torch.no_grad():
    torch_output = torch_model(input_torch).cpu().numpy()

# Zenith full inference
print("  Running Zenith...")
t0 = time.perf_counter()
zenith_output = zenith_resnet50(input_np, torch_model)
zenith_time = (time.perf_counter() - t0) * 1000

print(f"  Zenith inference time: {zenith_time:.2f} ms")

# ============================================================================
# Verify Accuracy
# ============================================================================
print("\n[5/5] Verifying accuracy...")

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
print(f"  Top-5 match: {'PASS' if np.array_equal(torch_top5, zenith_top5) else 'FAIL'}")

# ============================================================================
# Performance Benchmark
# ============================================================================
print("\n" + "=" * 70)
print("PERFORMANCE BENCHMARK")
print("=" * 70)

# Warmup
print("\nWarming up...")
for _ in range(3):
    _ = zenith_resnet50(input_np, torch_model)

# Benchmark
print("Benchmarking Zenith (10 iterations)...")
zenith_times = []
for i in range(10):
    t0 = time.perf_counter()
    _ = zenith_resnet50(input_np, torch_model)
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
print(f"  Top-5 match: {'PASS' if np.array_equal(torch_top5, zenith_top5) else 'FAIL'}")
print("=" * 70)
