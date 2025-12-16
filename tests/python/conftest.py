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
