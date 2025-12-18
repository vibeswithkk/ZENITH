# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Pytest configuration for Zenith Python tests.
"""

import sys
from pathlib import Path

# Add the project root to sys.path so we can import zenith
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Skip test modules that require optional dependencies not installed
collect_ignore = []

# Check for hypothesis
try:
    import hypothesis
except ImportError:
    collect_ignore.append("test_property_based.py")

# Check for zenith._zenith_core (CUDA build)
try:
    from zenith._zenith_core import cuda
except ImportError:
    collect_ignore.append("test_model_zoo.py")
