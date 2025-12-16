# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tensor Descriptor (Pure Python Implementation)
"""

from dataclasses import dataclass, field

from .types import DataType, Layout, Shape, dtype_size


@dataclass
class TensorDescriptor:
    """
    Describes a tensor's metadata without holding actual data.

    This is the primary way tensors are represented in the Graph IR.
    """

    name: str = ""
    shape: Shape = field(default_factory=Shape)
    dtype: DataType = DataType.Float32
    layout: Layout = Layout.NCHW

    def size_bytes(self) -> int:
        """Calculate the size in bytes."""
        elements = self.shape.numel()
        if elements < 0:
            return 0  # Dynamic shape
        return elements * dtype_size(self.dtype)

    def is_valid(self) -> bool:
        """Check if tensor has a valid shape."""
        return bool(self.name) and self.shape.rank() > 0

    def __repr__(self) -> str:
        from .types import dtype_to_string

        return (
            f"TensorDescriptor(name='{self.name}', "
            f"shape={self.shape}, dtype={dtype_to_string(self.dtype)})"
        )
