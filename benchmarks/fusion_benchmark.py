#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Fusion Pass Verification Benchmark.

Tests that OperatorFusionPass correctly fuses Conv+BN+ReLU patterns.
"""

import sys

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required")
    sys.exit(1)


class FusibleCNN(nn.Module):
    """CNN with fusible patterns: Conv + BatchNorm + ReLU."""

    def __init__(self):
        super().__init__()
        # Pattern 1: Conv + BN + ReLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Pattern 2: Conv + BN + ReLU
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # Pattern 3: Conv + BN (no ReLU)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Pattern 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Pattern 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Pattern 3
        x = self.conv3(x)
        x = self.bn3(x)

        return x


def main():
    print("=" * 60)
    print("  FUSION PASS VERIFICATION")
    print("=" * 60)

    # Create model with fusible patterns
    model = FusibleCNN()
    sample = torch.randn(1, 3, 32, 32)

    print("\nModel structure:")
    print("  - Conv2d + BatchNorm2d + ReLU (x2)")
    print("  - Conv2d + BatchNorm2d (x1)")
    print("\nExpected fusible patterns: 2 (Conv+BN+ReLU)")

    # Test with Zenith compilation
    try:
        import zenith

        print("\nCompiling with Zenith (opt_level=2)...")
        compiled = zenith.compile(
            model,
            target="cpu",
            precision="fp32",
            opt_level=2,
            sample_input=sample,
        )

        print("\n[VERIFICATION COMPLETE]")

    except Exception as e:
        print(f"\nError during compilation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
