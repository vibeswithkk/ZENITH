# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Core Types (Pure Python Implementation)

This module provides pure Python implementations of core types
as a fallback when native C++ bindings are not available.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Union


class DataType(Enum):
    """Supported data types for tensors."""

    Float32 = auto()
    Float16 = auto()
    BFloat16 = auto()
    Float64 = auto()
    Int8 = auto()
    Int16 = auto()
    Int32 = auto()
    Int64 = auto()
    UInt8 = auto()
    Bool = auto()


def dtype_size(dtype: DataType) -> int:
    """Get the size in bytes for a data type."""
    sizes = {
        DataType.Float32: 4,
        DataType.Float16: 2,
        DataType.BFloat16: 2,
        DataType.Float64: 8,
        DataType.Int8: 1,
        DataType.Int16: 2,
        DataType.Int32: 4,
        DataType.Int64: 8,
        DataType.UInt8: 1,
        DataType.Bool: 1,
    }
    return sizes.get(dtype, 0)


def dtype_to_string(dtype: DataType) -> str:
    """Get string representation of data type."""
    return dtype.name.lower()


class Layout(Enum):
    """Memory layout for tensors."""

    NCHW = auto()  # Batch, Channels, Height, Width (PyTorch default)
    NHWC = auto()  # Batch, Height, Width, Channels (TensorFlow default)
    NC = auto()  # Batch, Channels (1D)


class StatusCode(Enum):
    """Result status codes for operations."""

    Ok = auto()
    InvalidArgument = auto()
    NotFound = auto()
    AlreadyExists = auto()
    OutOfMemory = auto()
    NotImplemented = auto()
    InternalError = auto()
    InvalidGraph = auto()
    OptimizationFailed = auto()


@dataclass
class Status:
    """Status class for operation results."""

    code: StatusCode = StatusCode.Ok
    message: str = ""

    def ok(self) -> bool:
        return self.code == StatusCode.Ok

    @classmethod
    def Ok(cls) -> "Status":
        return cls()

    @classmethod
    def Error(cls, code: StatusCode, message: str) -> "Status":
        return cls(code=code, message=message)

    def __bool__(self) -> bool:
        return self.ok()


@dataclass
class Shape:
    """Represents tensor dimensions."""

    dims: list[int] = field(default_factory=list)

    def rank(self) -> int:
        """Get number of dimensions."""
        return len(self.dims)

    def numel(self) -> int:
        """Get total number of elements."""
        if not self.dims:
            return 0
        result = 1
        for d in self.dims:
            if d < 0:
                return -1  # Dynamic dimension
            result *= d
        return result

    def is_dynamic(self) -> bool:
        """Check if shape has dynamic dimensions."""
        return any(d < 0 for d in self.dims)

    def __getitem__(self, idx: int) -> int:
        return self.dims[idx]

    def __len__(self) -> int:
        return len(self.dims)

    def __repr__(self) -> str:
        return f"Shape({self.dims})"


# Attribute value types
AttributeValue = Union[int, float, str, list[int], list[float], list[str], bool]
AttributeMap = dict[str, AttributeValue]
