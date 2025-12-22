# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Memory Manager - GPU memory allocation and management.

Inspired by:
- TensorRT: Memory optimization and reuse
- PyTorch: CUDA caching allocator
- TVM: Memory planning

This module provides:
1. GPU memory allocation
2. Memory pool management
3. Memory reuse optimization
"""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class MemoryBlock:
    """A block of allocated memory."""

    name: str
    size_bytes: int
    offset: int
    is_free: bool = True
    tensor_name: Optional[str] = None


class MemoryManager:
    """
    Manages GPU memory for efficient execution.

    Features:
    - Memory pool allocation
    - Memory reuse between non-overlapping tensors
    - Memory usage tracking

    Example:
        manager = MemoryManager(device="cuda")
        manager.allocate("activation_0", size_bytes=1024*1024)
        ptr = manager.get_ptr("activation_0")
    """

    def __init__(self, device: str = "cuda", pool_size_mb: int = 512):
        """
        Initialize memory manager.

        Args:
            device: Target device
            pool_size_mb: Initial memory pool size in MB
        """
        self.device = device
        self.pool_size_bytes = pool_size_mb * 1024 * 1024

        # Memory blocks
        self._blocks: dict[str, MemoryBlock] = {}

        # Total allocated
        self._total_allocated = 0
        self._peak_allocated = 0

        # GPU memory pool (lazy init)
        self._pool = None
        self._cuda_available = None

    def allocate(self, name: str, size_bytes: int) -> bool:
        """
        Allocate memory for a tensor.

        Args:
            name: Tensor name
            size_bytes: Required size in bytes

        Returns:
            True if allocation succeeded
        """
        if name in self._blocks:
            # Already allocated
            return True

        # Track allocation
        offset = self._total_allocated
        self._blocks[name] = MemoryBlock(
            name=name,
            size_bytes=size_bytes,
            offset=offset,
            is_free=False,
            tensor_name=name,
        )

        self._total_allocated += size_bytes
        self._peak_allocated = max(self._peak_allocated, self._total_allocated)

        return True

    def _coalesce_adjacent_blocks(self) -> int:
        """
        Merge adjacent free memory blocks to reduce fragmentation.

        Algorithm complexity:
            - Sort by offset: O(n log n)
            - Linear scan: O(n)
            - Overall: O(n log n)

        Returns:
            Number of blocks that were merged (coalesced count).
        """
        if len(self._blocks) < 2:
            return 0

        sorted_names = sorted(self._blocks.keys(), key=lambda n: self._blocks[n].offset)

        merged_count = 0
        i = 0

        while i < len(sorted_names) - 1:
            curr_name = sorted_names[i]
            next_name = sorted_names[i + 1]

            curr_block = self._blocks.get(curr_name)
            next_block = self._blocks.get(next_name)

            if curr_block is None or next_block is None:
                i += 1
                continue

            curr_end = curr_block.offset + curr_block.size_bytes
            is_adjacent = curr_end == next_block.offset

            if curr_block.is_free and next_block.is_free and is_adjacent:
                curr_block.size_bytes += next_block.size_bytes
                del self._blocks[next_name]
                sorted_names.pop(i + 1)
                merged_count += 1
            else:
                i += 1

        return merged_count

    def free(self, name: str) -> None:
        """
        Free a memory allocation with automatic coalescing.

        Marks the block as free and attempts to merge with adjacent
        free blocks to reduce memory fragmentation.

        Args:
            name: Name of the tensor/allocation to free.
        """
        if name not in self._blocks:
            return

        block = self._blocks[name]
        block.is_free = True
        block.tensor_name = None

        self._coalesce_adjacent_blocks()

    def get_size(self, name: str) -> int:
        """Get allocated size for a tensor."""
        if name in self._blocks:
            return self._blocks[name].size_bytes
        return 0

    def plan_memory(
        self, tensor_sizes: dict[str, int], liveness: dict[str, tuple[int, int]]
    ) -> dict:
        """
        Plan memory allocation with reuse optimization.

        Args:
            tensor_sizes: Map of tensor name to size in bytes
            liveness: Map of tensor name to (start_op, end_op) indices

        Returns:
            Memory plan with tensor offsets
        """
        # Sort tensors by size (largest first) for better packing
        sorted_tensors = sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)

        # Simple first-fit allocation
        allocations = {}
        memory_timeline = []  # (offset, size, end_time)
        current_offset = 0

        for name, size in sorted_tensors:
            start_time, end_time = liveness.get(name, (0, float("inf")))

            # Try to reuse freed memory
            best_fit = None
            for i, (offset, block_size, block_end) in enumerate(memory_timeline):
                if block_end <= start_time and block_size >= size:
                    if best_fit is None or block_size < best_fit[1]:
                        best_fit = (offset, block_size, i)

            if best_fit is not None:
                # Reuse memory
                allocations[name] = best_fit[0]
                memory_timeline[best_fit[2]] = (best_fit[0], best_fit[1], end_time)
            else:
                # New allocation
                allocations[name] = current_offset
                memory_timeline.append((current_offset, size, end_time))
                current_offset += size

        return {
            "allocations": allocations,
            "total_size_bytes": current_offset,
            "total_size_mb": current_offset / (1024 * 1024),
        }

    def to_gpu(self, data: np.ndarray) -> Any:
        """
        Transfer numpy array to GPU.

        Args:
            data: Numpy array

        Returns:
            GPU tensor
        """
        try:
            from zenith._zenith_core import cuda

            return cuda.to_gpu(data)
        except ImportError:
            return data  # Fallback to CPU

    def to_cpu(self, gpu_tensor: Any) -> np.ndarray:
        """
        Transfer GPU tensor to CPU.

        Args:
            gpu_tensor: GPU tensor

        Returns:
            Numpy array
        """
        if hasattr(gpu_tensor, "to_numpy"):
            return gpu_tensor.to_numpy()
        if hasattr(gpu_tensor, "cpu"):
            return gpu_tensor.cpu().numpy()
        return np.asarray(gpu_tensor)

    def clear(self) -> None:
        """Clear all allocations."""
        self._blocks.clear()
        self._total_allocated = 0

    @property
    def total_allocated_mb(self) -> float:
        """Total currently allocated memory in MB."""
        return self._total_allocated / (1024 * 1024)

    @property
    def peak_allocated_mb(self) -> float:
        """Peak allocated memory in MB."""
        return self._peak_allocated / (1024 * 1024)

    def summary(self) -> dict:
        """Get memory manager summary."""
        return {
            "device": self.device,
            "num_allocations": len(self._blocks),
            "total_allocated_mb": self.total_allocated_mb,
            "peak_allocated_mb": self.peak_allocated_mb,
            "blocks": {
                name: {
                    "size_mb": block.size_bytes / (1024 * 1024),
                    "is_free": block.is_free,
                }
                for name, block in self._blocks.items()
            },
        }
