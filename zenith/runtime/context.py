# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Execution Context - Holds state during model execution.

Inspired by:
- TensorRT: Execution context for dynamic network state
- ONNX Runtime: Memory management and tensor lifecycle
- PyTorch: Tensor storage and device management

This module provides:
1. Tensor storage during execution
2. Input/output management
3. Memory tracking
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
import numpy as np


@dataclass
class TensorInfo:
    """Information about a tensor in the execution context."""

    name: str
    data: Any  # numpy array or GPU tensor
    shape: tuple
    dtype: str
    is_gpu: bool = False
    memory_bytes: int = 0


class ExecutionContext:
    """
    Holds execution state during model inference.

    Manages:
    - Input tensors
    - Intermediate tensors (activations)
    - Output tensors
    - Memory allocation tracking

    Example:
        context = ExecutionContext()
        context.set_input("x", input_tensor)

        # During execution
        context.set_tensor("hidden", hidden_tensor)

        # Get outputs
        outputs = context.get_outputs()
    """

    def __init__(
        self,
        input_names: list[str] = None,
        output_names: list[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize execution context.

        Args:
            input_names: Names of input tensors
            output_names: Names of output tensors
            device: Target device ("cuda" or "cpu")
        """
        self.input_names = input_names or []
        self.output_names = output_names or []
        self.device = device

        # Tensor storage
        self._tensors: dict[str, TensorInfo] = {}

        # Memory tracking
        self._total_memory_bytes = 0
        self._peak_memory_bytes = 0

        # Execution metadata
        self._execution_count = 0

    def set_input(self, name: str, data: Any) -> None:
        """
        Set an input tensor.

        Args:
            name: Input tensor name
            data: Input data (numpy array, PyTorch tensor, or GPU tensor)
        """
        tensor_data, is_gpu = self._normalize_tensor(data)
        shape = tuple(tensor_data.shape) if hasattr(tensor_data, "shape") else ()
        dtype = str(tensor_data.dtype) if hasattr(tensor_data, "dtype") else "unknown"

        memory = self._estimate_memory(tensor_data)

        self._tensors[name] = TensorInfo(
            name=name,
            data=tensor_data,
            shape=shape,
            dtype=dtype,
            is_gpu=is_gpu,
            memory_bytes=memory,
        )

        self._update_memory(memory)

    def set_tensor(self, name: str, data: Any) -> None:
        """
        Set an intermediate or output tensor.

        Args:
            name: Tensor name
            data: Tensor data
        """
        tensor_data, is_gpu = self._normalize_tensor(data)
        shape = tuple(tensor_data.shape) if hasattr(tensor_data, "shape") else ()
        dtype = str(tensor_data.dtype) if hasattr(tensor_data, "dtype") else "unknown"

        memory = self._estimate_memory(tensor_data)

        # Free previous tensor memory if replacing
        if name in self._tensors:
            self._total_memory_bytes -= self._tensors[name].memory_bytes

        self._tensors[name] = TensorInfo(
            name=name,
            data=tensor_data,
            shape=shape,
            dtype=dtype,
            is_gpu=is_gpu,
            memory_bytes=memory,
        )

        self._update_memory(memory)

    def get_tensor(self, name: str) -> Any:
        """
        Get a tensor by name.

        Args:
            name: Tensor name

        Returns:
            Tensor data

        Raises:
            KeyError: If tensor not found
        """
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found in context")
        return self._tensors[name].data

    def get_tensor_info(self, name: str) -> Optional[TensorInfo]:
        """Get tensor info by name."""
        return self._tensors.get(name)

    def has_tensor(self, name: str) -> bool:
        """Check if tensor exists."""
        return name in self._tensors

    def get_inputs(self) -> dict[str, Any]:
        """Get all input tensors."""
        return {
            name: self._tensors[name].data
            for name in self.input_names
            if name in self._tensors
        }

    def get_outputs(self) -> dict[str, Any]:
        """Get all output tensors."""
        return {
            name: self._tensors[name].data
            for name in self.output_names
            if name in self._tensors
        }

    def get_output(self, name: str = None) -> Any:
        """
        Get output tensor.

        Args:
            name: Output name (default: first output)

        Returns:
            Output tensor data
        """
        if name is None and self.output_names:
            name = self.output_names[0]
        if name and name in self._tensors:
            return self._tensors[name].data
        return None

    def delete_tensor(self, name: str) -> None:
        """Delete a tensor to free memory."""
        if name in self._tensors:
            self._total_memory_bytes -= self._tensors[name].memory_bytes
            del self._tensors[name]

    def clear(self) -> None:
        """Clear all tensors (except inputs)."""
        to_delete = [name for name in self._tensors if name not in self.input_names]
        for name in to_delete:
            self.delete_tensor(name)

    def _normalize_tensor(self, data: Any) -> tuple[Any, bool]:
        """
        Normalize tensor to internal format.

        Returns:
            Tuple of (normalized_data, is_gpu)
        """
        # Check if it's a Zenith GPU tensor
        if hasattr(data, "to_numpy"):
            return data, True

        # Check if it's a PyTorch tensor
        if hasattr(data, "detach") and hasattr(data, "cpu"):
            # PyTorch tensor
            is_gpu = data.is_cuda if hasattr(data, "is_cuda") else False
            return data, is_gpu

        # Check if it's a numpy array
        if isinstance(data, np.ndarray):
            return data, False

        # Try to convert to numpy
        try:
            return np.asarray(data), False
        except:
            return data, False

    def _estimate_memory(self, data: Any) -> int:
        """Estimate memory usage of a tensor."""
        if hasattr(data, "nbytes"):
            return data.nbytes
        if hasattr(data, "numel") and hasattr(data, "element_size"):
            return data.numel() * data.element_size()
        if hasattr(data, "shape") and hasattr(data, "dtype"):
            shape = data.shape
            try:
                itemsize = np.dtype(data.dtype).itemsize
                return int(np.prod(shape)) * itemsize
            except:
                pass
        return 0

    def _update_memory(self, added_bytes: int) -> None:
        """Update memory tracking."""
        self._total_memory_bytes += added_bytes
        self._peak_memory_bytes = max(self._peak_memory_bytes, self._total_memory_bytes)

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._total_memory_bytes / (1024 * 1024)

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory usage in MB."""
        return self._peak_memory_bytes / (1024 * 1024)

    def summary(self) -> dict:
        """Get execution context summary."""
        return {
            "num_tensors": len(self._tensors),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "device": self.device,
            "memory_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "tensors": {
                name: {
                    "shape": info.shape,
                    "dtype": info.dtype,
                    "is_gpu": info.is_gpu,
                    "memory_kb": info.memory_bytes / 1024,
                }
                for name, info in self._tensors.items()
            },
        }
