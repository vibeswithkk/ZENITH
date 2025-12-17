# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Benchmarks Package

Provides end-to-end benchmarking utilities for measuring Zenith performance
against native framework execution.

Per CetakBiru Section 5.3 - Performance Regression Testing
"""

from .models import get_model, list_models
from .run_benchmarks import benchmark_model, run_all_benchmarks
from .report import generate_report

__all__ = [
    "get_model",
    "list_models",
    "benchmark_model",
    "run_all_benchmarks",
    "generate_report",
]
