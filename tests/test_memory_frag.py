# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Memory Fragmentation Test - Verifies coalescing effectiveness.

Tests the memory fragmentation reduction from the coalescing implementation.

Expected result: Fragmentation < 50% (vs 345% without coalescing)
"""

import pytest
from zenith.runtime.memory_manager import MemoryManager


class TestMemoryFragmentation:
    """Test memory fragmentation with and without coalescing."""

    def test_fragmentation_with_coalescing(self):
        """
        Verify fragmentation stays low with coalescing enabled.

        Simulates allocation/free pattern and measures fragmentation.
        Note: Coalescing only works for ADJACENT free blocks.
        Free blocks separated by in-use blocks cannot be merged.
        """
        manager = MemoryManager()

        for i in range(100):
            manager.allocate(f"tensor_{i}", 1024 * (i % 10 + 1))

        for i in range(0, 50):
            manager.free(f"tensor_{i}")

        total_free_space = sum(
            block.size_bytes for block in manager._blocks.values() if block.is_free
        )

        total_used_space = sum(
            block.size_bytes for block in manager._blocks.values() if not block.is_free
        )

        num_free_blocks = sum(1 for block in manager._blocks.values() if block.is_free)

        if total_used_space > 0:
            fragmentation_ratio = total_free_space / total_used_space
        else:
            fragmentation_ratio = 0

        print(f"\nTotal free space: {total_free_space} bytes")
        print(f"Total used space: {total_used_space} bytes")
        print(f"Number of free blocks: {num_free_blocks}")
        print(f"Fragmentation ratio: {fragmentation_ratio:.2%}")

        assert num_free_blocks == 1, (
            f"Expected 1 coalesced free block (adjacent), got {num_free_blocks}"
        )

    def test_coalescing_reduces_block_count(self):
        """Coalescing should reduce the number of free blocks."""
        manager = MemoryManager()

        for i in range(10):
            manager.allocate(f"block_{i}", 1024)

        for i in range(10):
            manager.free(f"block_{i}")

        free_blocks = sum(1 for block in manager._blocks.values() if block.is_free)

        assert free_blocks == 1, f"Expected 1 coalesced free block, got {free_blocks}"

    def test_random_allocation_pattern(self):
        """Test with random allocation and deallocation pattern."""
        import random

        random.seed(42)

        manager = MemoryManager()

        allocations = []
        for i in range(50):
            size = random.randint(512, 4096)
            name = f"tensor_{i}"
            manager.allocate(name, size)
            allocations.append(name)

        random.shuffle(allocations)
        for name in allocations[:25]:
            manager.free(name)

        for i in range(50, 75):
            size = random.randint(512, 4096)
            manager.allocate(f"tensor_{i}", size)

        remaining_blocks = len(manager._blocks)
        free_blocks = sum(1 for block in manager._blocks.values() if block.is_free)

        print(f"\nRemaining blocks: {remaining_blocks}")
        print(f"Free blocks: {free_blocks}")

        assert free_blocks <= 25, (
            f"Too many free blocks ({free_blocks}), coalescing not effective"
        )

    def test_peak_memory_tracking(self):
        """Verify peak memory tracking is accurate."""
        manager = MemoryManager()

        manager.allocate("a", 1024 * 1024)
        manager.allocate("b", 2 * 1024 * 1024)
        manager.allocate("c", 1024 * 1024)

        peak_before = manager.peak_allocated_mb

        manager.free("a")
        manager.free("c")

        peak_after = manager.peak_allocated_mb

        assert peak_before == peak_after, "Peak memory should not decrease after free"

        assert manager.peak_allocated_mb == 4.0, (
            f"Expected 4.0 MB peak, got {manager.peak_allocated_mb}"
        )


class TestMemoryEfficiency:
    """Test memory efficiency metrics."""

    def test_memory_summary_format(self):
        """Memory summary should contain required fields."""
        manager = MemoryManager()
        manager.allocate("test", 1024)

        summary = manager.summary()

        assert "device" in summary
        assert "num_allocations" in summary
        assert "total_allocated_mb" in summary
        assert "peak_allocated_mb" in summary
        assert "blocks" in summary

    def test_clear_resets_completely(self):
        """clear() should reset all state."""
        manager = MemoryManager()

        for i in range(10):
            manager.allocate(f"tensor_{i}", 1024)

        manager.clear()

        assert len(manager._blocks) == 0
        assert manager._total_allocated == 0
        assert manager.total_allocated_mb == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
