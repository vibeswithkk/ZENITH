# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Integrations Module

Provides seamless integration with various ML frameworks:
- TorchDynamo: torch.compile backend registration
"""

from .torch_dynamo import (
    register_backend as register_torch_backend,
    is_registered as is_torch_backend_registered,
    get_backend as get_torch_backend,
)

__all__ = [
    "register_torch_backend",
    "is_torch_backend_registered",
    "get_torch_backend",
]
