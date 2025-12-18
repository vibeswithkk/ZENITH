# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Command Line Interface

Simple CLI for Zenith operations.
"""

from __future__ import annotations

import argparse
import sys


def main():
    """Main entry point for Zenith CLI."""
    parser = argparse.ArgumentParser(
        prog="zenith",
        description="Zenith - Cross-Platform ML Optimization Framework",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version information",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information",
    )

    args = parser.parse_args()

    if args.version:
        from zenith import __version__

        print(f"Zenith v{__version__}")
        return 0

    if args.info:
        _show_info()
        return 0

    # Default: show help
    parser.print_help()
    return 0


def _show_info():
    """Show system and Zenith information."""
    import platform

    print("=" * 50)
    print("Zenith System Information")
    print("=" * 50)

    # Version
    try:
        from zenith import __version__

        print(f"Zenith Version: {__version__}")
    except ImportError:
        print("Zenith Version: unknown")

    # Python
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    # CUDA
    try:
        from zenith import backends

        cuda_available = backends.is_cuda_available()
        print(f"CUDA Available: {cuda_available}")
    except Exception:
        print("CUDA Available: unknown")

    # Operators
    try:
        from zenith.execution import OperatorRegistry
        from zenith.execution.operators import (
            math_ops,
            activation_ops,
            conv_ops,
            shape_ops,  # noqa: F401
        )

        print(f"ONNX Operators: {OperatorRegistry.count()}")
    except Exception:
        print("ONNX Operators: unknown")

    print("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
