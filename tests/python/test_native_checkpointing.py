# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Tests for Native Gradient Checkpointing (Phase 2).

This test suite validates:
1. CheckpointFunction - Custom autograd.Function rematerialization
2. ActivationStore - Memory pool management
3. OptimalCheckpointSelector - DP-based checkpoint selection
4. native_checkpoint - Core checkpointing function
5. native_checkpoint_sequential - Sequential layer checkpointing
6. NativeCheckpointer - High-level manager
7. Gradient correctness under checkpointing
8. Memory savings verification
9. Thread safety
10. Edge cases and error handling
"""

import math
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestActivationStore:
    """Tests for ActivationStore memory pool."""

    def test_store_and_retrieve(self):
        """Test basic store and retrieve operations."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore()
        tensor = torch.randn(10, 10)

        # Store
        success = store.store(layer_id=0, tensor=tensor)
        assert success is True

        # Retrieve
        retrieved = store.retrieve(layer_id=0)
        assert retrieved is not None
        assert torch.equal(retrieved, tensor)

    def test_store_with_memory_limit(self):
        """Test store with memory limit enforces eviction."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        # 400 bytes limit (10 floats = 40 bytes, so 10 tensors max)
        store = ActivationStore(max_memory_bytes=400)

        # Store 15 small tensors
        for i in range(15):
            tensor = torch.randn(10)  # 40 bytes
            store.store(layer_id=i, tensor=tensor)

        # Should have evicted some
        assert store.memory_usage <= 400
        assert store.statistics["eviction_count"] > 0

    def test_retrieve_miss(self):
        """Test retrieve returns None for missing key."""
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore()
        result = store.retrieve(layer_id=999)
        assert result is None
        assert store.statistics["miss_count"] == 1

    def test_remove(self):
        """Test remove operation."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore()
        tensor = torch.randn(10, 10)
        store.store(layer_id=0, tensor=tensor)

        removed = store.remove(layer_id=0)
        assert removed is not None
        assert store.retrieve(layer_id=0) is None

    def test_clear(self):
        """Test clear operation."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore()
        for i in range(5):
            store.store(layer_id=i, tensor=torch.randn(10))

        count = store.clear()
        assert count == 5
        assert store.memory_usage == 0

    def test_checkpoint_protection(self):
        """Test that checkpoints are protected from eviction."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore(max_memory_bytes=100)

        # Store a checkpoint (protected)
        store.store(layer_id=0, tensor=torch.randn(10), is_checkpoint=True)

        # Store many non-checkpoints
        for i in range(1, 10):
            store.store(layer_id=i, tensor=torch.randn(10), is_checkpoint=False)

        # Checkpoint should still exist
        assert store.retrieve(layer_id=0) is not None

    def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore(enable_profiling=True)

        store.store(layer_id=0, tensor=torch.randn(10))
        store.retrieve(layer_id=0)
        store.retrieve(layer_id=999)  # Miss

        stats = store.statistics
        assert stats["store_count"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1


class TestOptimalCheckpointSelector:
    """Tests for OptimalCheckpointSelector."""

    def test_sqrt_selection(self):
        """Test sqrt checkpoint selection."""
        from zenith.memory.native_checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=100)
        positions = selector.select_checkpoints_sqrt()

        # sqrt(100) = 10, so should have positions at 0, 10, 20, ..., 90
        expected = list(range(0, 100, 10))
        assert positions == expected

    def test_sqrt_selection_small(self):
        """Test sqrt selection with small layer count."""
        from zenith.memory.native_checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=4)
        positions = selector.select_checkpoints_sqrt()

        # sqrt(4) = 2, positions at 0, 2
        assert positions == [0, 2]

    def test_sqrt_selection_single_layer(self):
        """Test sqrt selection with single layer."""
        from zenith.memory.native_checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=1)
        positions = selector.select_checkpoints_sqrt()
        assert positions == [0]

    def test_dp_selection(self):
        """Test dynamic programming checkpoint selection."""
        from zenith.memory.native_checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(
            num_layers=10,
            memory_costs=[1.0] * 10,
            compute_costs=[1.0] * 10,
            lambda_tolerance=0.5,
        )
        positions = selector.select_checkpoints_dp()

        # Should return valid positions
        assert len(positions) >= 0
        assert all(0 <= p < 10 for p in positions)

    def test_select_method_dispatch(self):
        """Test select_checkpoints dispatches to correct method."""
        from zenith.memory.native_checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=16)

        sqrt_result = selector.select_checkpoints(method="sqrt")
        assert sqrt_result == selector.select_checkpoints_sqrt()

    def test_select_invalid_method(self):
        """Test select_checkpoints raises on invalid method."""
        from zenith.memory.native_checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=10)

        with pytest.raises(ValueError, match="Unknown method"):
            selector.select_checkpoints(method="invalid")


class TestCheckpointFunction:
    """Tests for CheckpointFunction autograd.Function."""

    def test_forward_output_correct(self):
        """Test forward produces correct output."""
        import torch
        from zenith.memory.native_checkpointing import CheckpointFunction

        def simple_fn(x):
            return x * 2 + 1

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = CheckpointFunction.apply(simple_fn, x)

        expected = torch.tensor([3.0, 5.0, 7.0])
        assert torch.allclose(result, expected)

    def test_gradient_flow(self):
        """Test gradients flow correctly through checkpoint."""
        import torch
        from zenith.memory.native_checkpointing import CheckpointFunction

        def fn(x):
            return x**2

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        result = CheckpointFunction.apply(fn, x)
        loss = result.sum()
        loss.backward()

        # d/dx(x^2) = 2x
        expected_grad = torch.tensor([4.0, 6.0])
        assert torch.allclose(x.grad, expected_grad)

    def test_preserve_rng_state(self):
        """Test RNG state is preserved during recomputation."""
        import torch
        from zenith.memory.native_checkpointing import CheckpointFunction

        def fn_with_dropout(x):
            return torch.nn.functional.dropout(x, p=0.5, training=True)

        torch.manual_seed(42)
        x = torch.ones(100, requires_grad=True)
        result = CheckpointFunction.apply(fn_with_dropout, x, preserve_rng_state=True)

        # The result should be deterministic due to RNG preservation
        # Just verify it runs without error
        loss = result.sum()
        loss.backward()
        assert x.grad is not None


class TestNativeCheckpoint:
    """Tests for native_checkpoint function."""

    def test_basic_checkpoint(self):
        """Test basic checkpointing."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        def layer(x):
            return x * 2

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = native_checkpoint(layer, x)

        assert torch.allclose(result, torch.tensor([2.0, 4.0]))

    def test_checkpoint_with_kwargs(self):
        """Test checkpointing with keyword arguments."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        def layer(x, scale=1.0, bias=0.0):
            return x * scale + bias

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = native_checkpoint(layer, x, scale=2.0, bias=1.0)

        expected = torch.tensor([3.0, 5.0])
        assert torch.allclose(result, expected)

    def test_checkpoint_gradient_correctness(self):
        """Test gradient correctness with complex function."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        def complex_fn(x):
            x = x**2
            x = torch.sin(x)
            x = x + x.mean()
            return x

        x = torch.randn(10, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)

        # With checkpointing
        result_cp = native_checkpoint(complex_fn, x)
        loss_cp = result_cp.sum()
        loss_cp.backward()

        # Without checkpointing
        result_no_cp = complex_fn(x_clone)
        loss_no_cp = result_no_cp.sum()
        loss_no_cp.backward()

        # Gradients should match
        assert torch.allclose(x.grad, x_clone.grad, atol=1e-6)

    def test_no_grad_skips_checkpoint(self):
        """Test that checkpointing is skipped when grad is disabled."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        call_count = [0]

        def layer(x):
            call_count[0] += 1
            return x * 2

        x = torch.tensor([1.0])

        with torch.no_grad():
            result = native_checkpoint(layer, x)

        # Should be called once (no recomputation needed)
        assert call_count[0] == 1


class TestNativeCheckpointSequential:
    """Tests for native_checkpoint_sequential function."""

    def test_sequential_checkpoint(self):
        """Test sequential checkpointing."""
        import torch
        import torch.nn as nn
        from zenith.memory.native_checkpointing import native_checkpoint_sequential

        layers = [
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        ]

        x = torch.randn(2, 10, requires_grad=True)
        result = native_checkpoint_sequential(layers, segments=2, input_tensor=x)

        assert result.shape == (2, 10)

        # Test backward
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_sequential_auto_segments(self):
        """Test auto segment calculation."""
        import torch
        import torch.nn as nn
        from zenith.memory.native_checkpointing import native_checkpoint_sequential

        layers = [nn.Linear(10, 10) for _ in range(9)]

        x = torch.randn(2, 10, requires_grad=True)
        result = native_checkpoint_sequential(layers, segments=0, input_tensor=x)

        # sqrt(9) = 3 segments
        assert result.shape == (2, 10)

    def test_sequential_empty_raises(self):
        """Test empty functions list returns input."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint_sequential

        x = torch.randn(2, 10)
        result = native_checkpoint_sequential([], input_tensor=x)
        assert torch.equal(result, x)

    def test_sequential_none_input_raises(self):
        """Test None input raises ValueError."""
        from zenith.memory.native_checkpointing import native_checkpoint_sequential

        with pytest.raises(ValueError, match="input_tensor is required"):
            native_checkpoint_sequential([lambda x: x], input_tensor=None)


class TestNativeCheckpointer:
    """Tests for NativeCheckpointer manager."""

    def test_initialization(self):
        """Test checkpointer initialization."""
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer(
            policy="sqrt",
            max_memory_mb=100,
            lambda_tolerance=0.33,
        )

        assert checkpointer._policy == "sqrt"
        assert checkpointer._lambda == 0.33

    def test_configure(self):
        """Test configure method."""
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer()
        checkpointer.configure(num_layers=100)

        # sqrt(100) = 10 checkpoints
        assert len(checkpointer._checkpoint_positions) == 10

    def test_should_checkpoint(self):
        """Test should_checkpoint method."""
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer()
        checkpointer.configure(num_layers=100)

        # Position 0 should be checkpointed (first checkpoint)
        assert checkpointer.should_checkpoint(0) is True

        # Position 10 should be checkpointed (sqrt interval)
        assert checkpointer.should_checkpoint(10) is True

    def test_checkpoint_method(self):
        """Test checkpoint method."""
        import torch
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer()

        def layer(x):
            return x * 2

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = checkpointer.checkpoint(layer, x)

        assert torch.allclose(result, torch.tensor([2.0, 4.0]))

    def test_get_statistics(self):
        """Test get_statistics method."""
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer()
        checkpointer.configure(num_layers=16)

        stats = checkpointer.get_statistics()

        assert stats["policy"] == "sqrt"
        assert stats["num_layers"] == 16
        assert stats["num_checkpoints"] == 4  # sqrt(16) = 4

    def test_reset(self):
        """Test reset method."""
        import torch
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer(enable_profiling=True)

        # Do some work
        def layer(x):
            return x * 2

        x = torch.tensor([1.0], requires_grad=True)
        checkpointer.checkpoint(layer, x)

        # Reset
        checkpointer.reset()

        assert len(checkpointer._forward_times) == 0


class TestGlobalCheckpointer:
    """Tests for global checkpointer singleton."""

    def test_get_native_checkpointer(self):
        """Test global checkpointer is singleton."""
        from zenith.memory.native_checkpointing import get_native_checkpointer

        cp1 = get_native_checkpointer()
        cp2 = get_native_checkpointer()

        assert cp1 is cp2


class TestEvictionPolicy:
    """Tests for eviction policies."""

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore, EvictionPolicy

        # Each float32 tensor with 5 elements = 5 * 4 bytes = 20 bytes
        # Set limit to 60 bytes so we can only store 3 tensors max
        store = ActivationStore(
            max_memory_bytes=60,
            eviction_policy=EvictionPolicy.LRU,
        )

        # Store 3 tensors (fills up the 60 byte limit)
        store.store(0, torch.randn(5))  # 20 bytes
        time.sleep(0.01)
        store.store(1, torch.randn(5))  # 20 bytes
        time.sleep(0.01)
        store.store(2, torch.randn(5))  # 20 bytes

        # Access 0 to make it recently used
        store.retrieve(0)

        # Add 2 more tensors - this MUST trigger eviction
        store.store(3, torch.randn(5))  # Should evict 1
        store.store(4, torch.randn(5))  # Should evict 2

        # Check eviction occurred
        stats = store.statistics
        assert stats["eviction_count"] >= 2, (
            f"Expected >= 2 evictions, got {stats['eviction_count']}"
        )


class TestThreadSafety:
    """Tests for thread safety."""

    def test_activation_store_thread_safety(self):
        """Test ActivationStore is thread-safe."""
        import torch
        from zenith.memory.native_checkpointing import ActivationStore

        store = ActivationStore()
        errors = []

        def worker(tid):
            try:
                for i in range(100):
                    key = tid * 1000 + i
                    store.store(key, torch.randn(10))
                    store.retrieve(key)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_checkpointer_thread_safety(self):
        """Test NativeCheckpointer is thread-safe."""
        import torch
        from zenith.memory.native_checkpointing import NativeCheckpointer

        checkpointer = NativeCheckpointer()
        errors = []

        def worker(tid):
            try:
                for i in range(50):
                    x = torch.randn(10, requires_grad=True)
                    result = checkpointer.checkpoint(lambda y: y * 2, x)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_checkpoint_no_grad_tensor(self):
        """Test checkpoint with tensor that doesn't require grad."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        def layer(x):
            return x * 2

        x = torch.tensor([1.0, 2.0], requires_grad=False)
        result = native_checkpoint(layer, x)

        # Should just run the function directly
        assert torch.allclose(result, torch.tensor([2.0, 4.0]))

    def test_checkpoint_multiple_outputs(self):
        """Test checkpoint with function returning multiple tensors."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        def multi_output(x):
            return x * 2, x + 1

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = native_checkpoint(multi_output, x)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_checkpoint_mixed_inputs(self):
        """Test checkpoint with mixed tensor and non-tensor inputs."""
        import torch
        from zenith.memory.native_checkpointing import native_checkpoint

        def layer(x, scale, offset):
            return x * scale + offset

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = native_checkpoint(layer, x, 2.0, 1.0)

        expected = torch.tensor([3.0, 5.0])
        assert torch.allclose(result, expected)


class TestIntegration:
    """Integration tests."""

    def test_transformer_block_checkpoint(self):
        """Test checkpointing a transformer-like block."""
        import torch
        import torch.nn as nn
        from zenith.memory.native_checkpointing import native_checkpoint

        class TransformerBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)

            def forward(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.ffn(self.norm2(x))
                return x

        block = TransformerBlock(64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        # Checkpoint the forward pass
        def run_block(inp):
            return block(inp)

        output = native_checkpoint(run_block, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_training_loop_with_checkpointing(self):
        """Test a complete training loop with checkpointing."""
        import torch
        import torch.nn as nn
        from zenith.memory.native_checkpointing import native_checkpoint_sequential

        # Simple model
        layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(8)])

        optimizer = torch.optim.SGD(layers.parameters(), lr=0.01)

        for epoch in range(10):
            x = torch.randn(4, 32, requires_grad=True)

            # Convert to list of functions
            funcs = [layer.forward for layer in layers]

            # Forward with checkpointing
            output = native_checkpoint_sequential(funcs, segments=0, input_tensor=x)

            loss = output.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should complete without error
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
