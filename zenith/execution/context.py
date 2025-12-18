# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Execution Context

Manages tensor buffers and state during ONNX graph execution.
Handles GPU memory allocation, data transfers, and cleanup.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np


class ExecutionContext:
    """
    Manages tensor buffers during ONNX graph execution.

    The ExecutionContext is responsible for:
    - Storing intermediate tensor values during graph execution
    - Managing GPU tensor lifecycle
    - Handling data transfers between CPU and GPU
    - Providing access to model constants (weights, biases)

    Example:
        ctx = ExecutionContext(device="cuda")
        ctx.set_tensor("input", input_array)
        ctx.set_constant("weight", weight_array)

        # During execution
        input_gpu = ctx.get_tensor("input")
        weight_gpu = ctx.get_tensor("weight")

        # After execution
        output = ctx.get_tensor_numpy("output")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize execution context.

        Args:
            device: Target device ("cuda" or "cpu").
        """
        self.device = device
        self._tensors: Dict[str, Any] = {}  # name → GpuTensor or ndarray
        self._constants: Dict[str, np.ndarray] = {}  # name → ndarray
        self._gpu_tensors: Dict[str, Any] = {}  # name → GpuTensor (cached)
        self._cuda_available = False

        # Try to import CUDA module
        self._init_cuda()

    def _init_cuda(self) -> None:
        """Initialize CUDA backend if available."""
        if self.device == "cuda":
            try:
                from zenith import _zenith_core

                self._cuda = _zenith_core.cuda
                self._cuda_available = self._cuda.is_available()
            except (ImportError, AttributeError):
                self._cuda = None
                self._cuda_available = False

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    def set_tensor(self, name: str, value: Any) -> None:
        """
        Store a tensor value.

        Args:
            name: Tensor name (from ONNX graph).
            value: Tensor value (numpy array, GpuTensor, or other).
        """
        self._tensors[name] = value

    def get_tensor(self, name: str) -> Any:
        """
        Retrieve a tensor value.

        Args:
            name: Tensor name.

        Returns:
            Tensor value (may be numpy array or GpuTensor).

        Raises:
            KeyError: If tensor not found.
        """
        # Check tensors first
        if name in self._tensors:
            return self._tensors[name]

        # Check constants
        if name in self._constants:
            return self._constants[name]

        # Check GPU tensors cache
        if name in self._gpu_tensors:
            return self._gpu_tensors[name]

        raise KeyError(f"Tensor '{name}' not found in execution context")

    def has_tensor(self, name: str) -> bool:
        """Check if tensor exists in context."""
        return (
            name in self._tensors
            or name in self._constants
            or name in self._gpu_tensors
        )

    def set_constant(self, name: str, value: np.ndarray) -> None:
        """
        Store a constant (weight, bias).

        Constants are typically loaded once at initialization
        and remain unchanged during execution.

        Args:
            name: Constant name.
            value: Constant value as numpy array.
        """
        self._constants[name] = value

    def get_constant(self, name: str) -> np.ndarray:
        """
        Retrieve a constant value.

        Args:
            name: Constant name.

        Returns:
            Constant as numpy array.
        """
        return self._constants[name]

    def upload_to_gpu(self, name: str, data: np.ndarray, cache: bool = True) -> Any:
        """
        Upload numpy array to GPU memory.

        Args:
            name: Tensor name for caching.
            data: Numpy array to upload.
            cache: Whether to cache the GPU tensor.

        Returns:
            GpuTensor object.
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA not available")

        # Ensure contiguous array
        data = np.ascontiguousarray(data, dtype=np.float32)

        # Create GPU tensor
        gpu_tensor = self._cuda.to_gpu(data)

        if cache:
            self._gpu_tensors[name] = gpu_tensor

        return gpu_tensor

    def download_from_gpu(self, gpu_tensor: Any) -> np.ndarray:
        """
        Download GPU tensor to numpy array.

        Args:
            gpu_tensor: GpuTensor to download.

        Returns:
            Numpy array with tensor data.
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA not available")

        return self._cuda.to_numpy(gpu_tensor)

    def get_tensor_numpy(self, name: str) -> np.ndarray:
        """
        Get tensor as numpy array (downloading from GPU if needed).

        Args:
            name: Tensor name.

        Returns:
            Tensor as numpy array.
        """
        tensor = self.get_tensor(name)

        # Check if it's a GpuTensor
        if hasattr(tensor, "is_valid") and tensor.is_valid():
            return self.download_from_gpu(tensor)

        # Already numpy array
        return np.asarray(tensor)

    def get_gpu_tensor(self, name: str) -> Any:
        """
        Get tensor as GPU tensor (uploading from CPU if needed).

        Args:
            name: Tensor name.

        Returns:
            GpuTensor object.
        """
        # Check cached GPU tensors first
        if name in self._gpu_tensors:
            return self._gpu_tensors[name]

        # Get the tensor value
        tensor = self.get_tensor(name)

        # Check if already a GpuTensor
        if hasattr(tensor, "is_valid"):
            return tensor

        # Upload to GPU
        return self.upload_to_gpu(name, np.asarray(tensor))

    def clear(self) -> None:
        """Clear all tensor values (but keep constants)."""
        self._tensors.clear()
        self._gpu_tensors.clear()

    def clear_all(self) -> None:
        """Clear all tensors and constants."""
        self._tensors.clear()
        self._constants.clear()
        self._gpu_tensors.clear()

    def get_tensor_names(self) -> list:
        """Get all tensor names in context."""
        names = set(self._tensors.keys())
        names.update(self._constants.keys())
        names.update(self._gpu_tensors.keys())
        return list(names)

    def __repr__(self) -> str:
        return (
            f"ExecutionContext(device='{self.device}', "
            f"tensors={len(self._tensors)}, "
            f"constants={len(self._constants)}, "
            f"gpu_tensors={len(self._gpu_tensors)})"
        )
