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
        description="Zenith - ML Optimization Toolkit",
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

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start terminal monitoring dashboard",
    )
    dashboard_parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Refresh rate in seconds (default: 1.0)",
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start metrics HTTP server",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )

    args = parser.parse_args()

    if args.version:
        from zenith import __version__

        print(f"Zenith v{__version__}")
        return 0

    if args.info:
        _show_info()
        return 0

    if args.command == "dashboard":
        return _run_dashboard(args)

    if args.command == "serve":
        return _run_serve(args)

    # Default: show help
    parser.print_help()
    return 0


def _run_dashboard(args):
    """Run the terminal dashboard."""
    try:
        from zenith.monitoring.dashboard import run_dashboard

        run_dashboard(refresh_rate=args.refresh)
        return 0
    except ImportError as e:
        print(f"Error: {e}")
        print("Install Rich with: pip install rich")
        return 1
    except KeyboardInterrupt:
        return 0


def _run_serve(args):
    """Run the metrics HTTP server."""
    try:
        from zenith.monitoring import start_server

        start_server(host=args.host, port=args.port)
        return 0
    except ImportError as e:
        print(f"Error: {e}")
        print("Install FastAPI with: pip install fastapi uvicorn")
        return 1
    except KeyboardInterrupt:
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
