# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for Memory Manager Coalescing.

Tests the _coalesce_adjacent_blocks() method and auto-coalescing
behavior in the free() method.
"""

import pytest
from zenith.runtime.memory_manager import MemoryManager


class TestMemoryCoalescing:
    """Test memory block coalescing functionality."""

    def test_coalesce_empty_manager(self):
        """Empty manager should return 0 merged blocks."""
        manager = MemoryManager()
        merged = manager._coalesce_adjacent_blocks()
        assert merged == 0

    def test_coalesce_single_block(self):
        """Single block cannot be coalesced."""
        manager = MemoryManager()
        manager.allocate("block_a", 1024)
        manager.free("block_a")

        assert len(manager._blocks) == 1

    def test_coalesce_two_adjacent_free_blocks(self):
        """Two adjacent free blocks should merge into one."""
        manager = MemoryManager()

        manager.allocate("block_a", 1024)
        manager.allocate("block_b", 2048)

        manager._blocks["block_a"].is_free = True
        manager._blocks["block_b"].is_free = True

        merged = manager._coalesce_adjacent_blocks()

        assert merged == 1
        assert len(manager._blocks) == 1
        remaining = list(manager._blocks.values())[0]
        assert remaining.size_bytes == 3072

    def test_no_coalesce_non_adjacent_blocks(self):
        """Non-adjacent free blocks should not merge."""
        manager = MemoryManager()

        manager.allocate("block_a", 1024)
        manager.allocate("block_b", 2048)
        manager.allocate("block_c", 512)

        manager._blocks["block_a"].is_free = True
        manager._blocks["block_c"].is_free = True

        merged = manager._coalesce_adjacent_blocks()

        assert merged == 0
        assert len(manager._blocks) == 3

    def test_no_coalesce_in_use_blocks(self):
        """In-use blocks should not be coalesced even if adjacent."""
        manager = MemoryManager()

        manager.allocate("block_a", 1024)
        manager.allocate("block_b", 2048)

        merged = manager._coalesce_adjacent_blocks()

        assert merged == 0
        assert len(manager._blocks) == 2

    def test_chain_coalesce(self):
        """Three adjacent free blocks should merge into one."""
        manager = MemoryManager()

        manager.allocate("block_a", 1024)
        manager.allocate("block_b", 1024)
        manager.allocate("block_c", 1024)

        manager._blocks["block_a"].is_free = True
        manager._blocks["block_b"].is_free = True
        manager._blocks["block_c"].is_free = True

        merged = manager._coalesce_adjacent_blocks()

        assert merged == 2
        assert len(manager._blocks) == 1
        remaining = list(manager._blocks.values())[0]
        assert remaining.size_bytes == 3072

    def test_auto_coalesce_on_free(self):
        """free() should automatically coalesce adjacent blocks."""
        manager = MemoryManager()

        manager.allocate("block_a", 1024)
        manager.allocate("block_b", 2048)
        manager.allocate("block_c", 512)

        manager.free("block_a")
        assert len(manager._blocks) == 3

        manager.free("block_b")

        assert len(manager._blocks) == 2
        assert "block_a" in manager._blocks
        assert manager._blocks["block_a"].size_bytes == 3072

    def test_free_nonexistent_block(self):
        """Freeing a nonexistent block should not raise."""
        manager = MemoryManager()
        manager.free("nonexistent")

    def test_offsets_preserved_correctly(self):
        """Merged block should preserve earliest offset."""
        manager = MemoryManager()

        manager.allocate("block_a", 1024)
        manager.allocate("block_b", 2048)

        original_offset = manager._blocks["block_a"].offset

        manager.free("block_a")
        manager.free("block_b")

        remaining = list(manager._blocks.values())[0]
        assert remaining.offset == original_offset


class TestMemoryManagerBasic:
    """Test basic MemoryManager functionality."""

    def test_allocate_tracks_size(self):
        """Allocation should track size correctly."""
        manager = MemoryManager()
        manager.allocate("tensor_a", 1024 * 1024)

        assert manager.get_size("tensor_a") == 1024 * 1024

    def test_peak_allocation_tracking(self):
        """Peak allocation should be tracked."""
        manager = MemoryManager()
        manager.allocate("tensor_a", 1024)
        manager.allocate("tensor_b", 2048)

        assert manager.peak_allocated_mb == (3072) / (1024 * 1024)

    def test_clear_resets_allocations(self):
        """clear() should reset all allocations."""
        manager = MemoryManager()
        manager.allocate("tensor_a", 1024)
        manager.clear()

        assert len(manager._blocks) == 0
        assert manager._total_allocated == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
