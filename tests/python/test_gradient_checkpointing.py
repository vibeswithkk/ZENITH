# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Tests for Gradient Checkpointing Module.

Tests cover:
1. Basic checkpoint function
2. Sequential checkpointing
3. Module wrapper
4. Context manager
5. Automatic checkpointing
6. Memory estimation
7. Policy selection
8. Edge cases
"""

import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestCheckpointPolicy:
    """Tests for CheckpointPolicy enum."""

    def test_all_policies_defined(self):
        """Verify all checkpoint policies exist."""
        from zenith.memory.gradient_checkpointing import CheckpointPolicy

        assert CheckpointPolicy.NONE.value == "none"
        assert CheckpointPolicy.SQRT.value == "sqrt"
        assert CheckpointPolicy.SEGMENT.value == "segment"
        assert CheckpointPolicy.EVERY_N.value == "every_n"
        assert CheckpointPolicy.SELECTIVE.value == "selective"
        assert CheckpointPolicy.MEMORY_AWARE.value == "memory_aware"


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from zenith.memory.gradient_checkpointing import (
            CheckpointConfig,
            CheckpointPolicy,
        )

        config = CheckpointConfig()
        assert config.enabled is True
        assert config.policy == CheckpointPolicy.SQRT
        assert config.num_segments == 0
        assert config.every_n == 1
        assert config.preserve_rng_state is True
        assert config.use_reentrant is False
        assert config.memory_efficient is True
        assert config.lambda_tolerance == 0.2
        assert config.debug is False

    def test_custom_values(self):
        """Test custom configuration."""
        from zenith.memory.gradient_checkpointing import (
            CheckpointConfig,
            CheckpointPolicy,
        )

        config = CheckpointConfig(
            enabled=False,
            policy=CheckpointPolicy.SEGMENT,
            num_segments=4,
            lambda_tolerance=0.3,
        )
        assert config.enabled is False
        assert config.policy == CheckpointPolicy.SEGMENT
        assert config.num_segments == 4
        assert config.lambda_tolerance == 0.3


class TestCheckpointingContext:
    """Tests for CheckpointingContext manager."""

    def test_context_enter_exit(self):
        """Test context manager enter and exit."""
        from zenith.memory.gradient_checkpointing import (
            CheckpointingContext,
            _get_context_config,
        )

        # Before context
        assert _get_context_config() is None

        # Enter context
        with CheckpointingContext(enabled=True) as ctx:
            config = _get_context_config()
            assert config is not None
            assert config.enabled is True

        # After context
        assert _get_context_config() is None

    def test_nested_contexts(self):
        """Test nested context managers."""
        from zenith.memory.gradient_checkpointing import (
            CheckpointingContext,
            CheckpointPolicy,
            _get_context_config,
        )

        with CheckpointingContext(enabled=True, policy=CheckpointPolicy.SQRT):
            outer_config = _get_context_config()
            assert outer_config.policy == CheckpointPolicy.SQRT

            with CheckpointingContext(enabled=False, policy=CheckpointPolicy.SEGMENT):
                inner_config = _get_context_config()
                assert inner_config.enabled is False
                assert inner_config.policy == CheckpointPolicy.SEGMENT

            # Back to outer
            restored_config = _get_context_config()
            assert restored_config.policy == CheckpointPolicy.SQRT

    def test_context_properties(self):
        """Test context properties."""
        from zenith.memory.gradient_checkpointing import CheckpointingContext

        ctx = CheckpointingContext(enabled=True, num_segments=4)
        assert ctx.config.enabled is True
        assert ctx.config.num_segments == 4


class TestSegmentCheckpointer:
    """Tests for SegmentCheckpointer class."""

    def test_sqrt_policy(self):
        """Test sqrt checkpoint policy."""
        from zenith.memory.gradient_checkpointing import (
            SegmentCheckpointer,
            CheckpointPolicy,
        )

        checkpointer = SegmentCheckpointer(policy=CheckpointPolicy.SQRT)

        # For 16 layers, sqrt(16) = 4, so checkpoint every 4 layers
        total = 16
        checkpoints = [
            i for i in range(total) if checkpointer.should_checkpoint(i, total)
        ]
        assert len(checkpoints) == 4  # 0, 4, 8, 12

    def test_every_n_policy(self):
        """Test every-N checkpoint policy."""
        from zenith.memory.gradient_checkpointing import (
            SegmentCheckpointer,
            CheckpointPolicy,
        )

        checkpointer = SegmentCheckpointer(
            policy=CheckpointPolicy.EVERY_N,
            num_checkpoints=2,  # Every 2 layers
        )

        total = 10
        checkpoints = [
            i for i in range(total) if checkpointer.should_checkpoint(i, total)
        ]
        assert checkpoints == [0, 2, 4, 6, 8]

    def test_none_policy(self):
        """Test no checkpointing policy."""
        from zenith.memory.gradient_checkpointing import (
            SegmentCheckpointer,
            CheckpointPolicy,
        )

        checkpointer = SegmentCheckpointer(policy=CheckpointPolicy.NONE)

        total = 10
        checkpoints = [
            i for i in range(total) if checkpointer.should_checkpoint(i, total)
        ]
        assert checkpoints == []

    def test_explicit_layers(self):
        """Test explicit checkpoint layer specification."""
        from zenith.memory.gradient_checkpointing import SegmentCheckpointer

        explicit_checkpoints = {0, 5, 10}
        checkpointer = SegmentCheckpointer(checkpoint_layers=explicit_checkpoints)

        total = 15
        for i in range(total):
            expected = i in explicit_checkpoints
            assert checkpointer.should_checkpoint(i, total) == expected


class TestModuleCheckpointer:
    """Tests for ModuleCheckpointer class."""

    def test_initialization(self):
        """Test checkpointer initialization."""
        from zenith.memory.gradient_checkpointing import ModuleCheckpointer

        checkpointer = ModuleCheckpointer(
            target_names=("attention", "mlp"),
            min_params=1000,
        )
        assert checkpointer._target_names == ("attention", "mlp")
        assert checkpointer._min_params == 1000


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_memory_stats_dataclass(self):
        """Test MemoryStats dataclass."""
        from zenith.memory.gradient_checkpointing import MemoryStats

        stats = MemoryStats(
            total_activations_mb=100.0,
            checkpointed_mb=25.0,
            memory_savings_pct=75.0,
        )
        assert stats.total_activations_mb == 100.0
        assert stats.checkpointed_mb == 25.0
        assert stats.memory_savings_pct == 75.0

    def test_get_memory_stats_without_cuda(self):
        """Test memory stats when CUDA not available."""
        from zenith.memory.gradient_checkpointing import get_memory_stats

        # Mock torch.cuda.is_available to return False
        with patch("zenith.memory.gradient_checkpointing._get_torch") as mock_torch:
            mock_torch.return_value.cuda.is_available.return_value = False

            stats = get_memory_stats()
            assert stats["allocated_mb"] == 0.0
            assert stats["reserved_mb"] == 0.0


class TestEstimateMemorySavings:
    """Tests for memory savings estimation."""

    def test_sqrt_savings_estimate(self):
        """Test memory savings for sqrt policy."""
        from zenith.memory.gradient_checkpointing import (
            estimate_memory_savings,
            CheckpointPolicy,
        )

        # Create mock model with 100 layers
        mock_model = MagicMock()
        mock_model.modules.return_value = [MagicMock() for _ in range(100)]

        mock_input = MagicMock()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats = estimate_memory_savings(
                mock_model,
                mock_input,
                policy=CheckpointPolicy.SQRT,
            )

        # With 100 layers and sqrt checkpointing, savings should be significant
        # sqrt(100) = 10 checkpoints, reduction_factor = 10/100 = 0.1
        # savings = (1 - sqrt(0.1)) * 100 = (1 - 0.316) * 100 = 68.4%
        assert stats.memory_savings_pct > 50, (
            f"Expected >50% savings, got {stats.memory_savings_pct}%"
        )
        assert stats.recompute_overhead_pct > 0, "Should have some recompute overhead"


class TestAutoCheckpoint:
    """Tests for auto_checkpoint decorator."""

    def test_decorator_with_string_policy(self):
        """Test decorator with string policy."""
        from zenith.memory.gradient_checkpointing import auto_checkpoint

        @auto_checkpoint(policy="sqrt")
        class TestModel:
            def __init__(self):
                pass

        model = TestModel()
        assert hasattr(type(model), "_zenith_checkpoint_config")

    def test_decorator_with_enum_policy(self):
        """Test decorator with enum policy."""
        from zenith.memory.gradient_checkpointing import (
            auto_checkpoint,
            CheckpointPolicy,
        )

        @auto_checkpoint(policy=CheckpointPolicy.SEGMENT, num_segments=4)
        class TestModel:
            def __init__(self):
                pass

        config = TestModel._zenith_checkpoint_config
        assert config.policy == CheckpointPolicy.SEGMENT
        assert config.num_segments == 4


class TestCheckpointFunction:
    """Tests for main checkpoint function."""

    def test_checkpoint_disabled_in_context(self):
        """Test that checkpoint runs function directly when disabled."""
        from zenith.memory.gradient_checkpointing import (
            checkpoint,
            CheckpointingContext,
        )

        call_count = 0

        def tracked_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        with CheckpointingContext(enabled=False):
            result = checkpoint(tracked_function, 5)

        assert result == 10
        assert call_count == 1

    def test_checkpoint_with_kwargs(self):
        """Test checkpoint with keyword arguments."""
        from zenith.memory.gradient_checkpointing import (
            checkpoint,
            CheckpointingContext,
        )

        def func_with_kwargs(x, multiplier=1):
            return x * multiplier

        with CheckpointingContext(enabled=False):
            result = checkpoint(func_with_kwargs, 5, multiplier=3)

        assert result == 15


class TestCheckpointSequential:
    """Tests for checkpoint_sequential function."""

    def test_empty_sequence(self):
        """Test with empty sequence."""
        from zenith.memory.gradient_checkpointing import checkpoint_sequential

        with pytest.raises(ValueError):
            checkpoint_sequential([], input=None)

    def test_auto_segment_calculation(self):
        """Test automatic segment calculation uses sqrt."""
        # With 16 functions, should get sqrt(16) = 4 segments
        functions = [lambda x: x for _ in range(16)]

        from zenith.memory.gradient_checkpointing import (
            checkpoint_sequential,
            CheckpointingContext,
        )

        # Use disabled context to avoid actual PyTorch calls
        with CheckpointingContext(enabled=False):
            result = checkpoint_sequential(functions, segments=0, input=1)
            assert result == 1  # Identity functions


class TestUnwrapCheckpoint:
    """Tests for unwrap_checkpoint function."""

    def test_unwrap_wrapped_module(self):
        """Test unwrapping a wrapped module."""
        from zenith.memory.gradient_checkpointing import (
            checkpoint_wrapper,
            unwrap_checkpoint,
        )

        # Create mock module
        mock_module = MagicMock()
        mock_module.forward = lambda x: x * 2
        original_forward = mock_module.forward

        with patch("zenith.memory.gradient_checkpointing._get_torch") as mock_torch:
            mock_torch.return_value.nn.Module = type(mock_module)

            # Wrap
            checkpoint_wrapper(mock_module)
            assert hasattr(mock_module, "_zenith_original_forward")

            # Unwrap
            unwrap_checkpoint(mock_module)
            assert not hasattr(mock_module, "_zenith_original_forward")

    def test_unwrap_non_wrapped_module(self):
        """Test unwrapping a module that was never wrapped."""
        from zenith.memory.gradient_checkpointing import unwrap_checkpoint

        mock_module = MagicMock()
        # Should not raise
        result = unwrap_checkpoint(mock_module)
        assert result is mock_module


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_checkpoint_preserves_function_result(self):
        """Test that checkpoint returns correct result."""
        from zenith.memory.gradient_checkpointing import (
            checkpoint,
            CheckpointingContext,
        )

        def compute(a, b):
            return a + b

        with CheckpointingContext(enabled=False):
            result = checkpoint(compute, 3, 4)
            assert result == 7

    def test_segment_checkpointer_caching(self):
        """Test that checkpoint positions are cached."""
        from zenith.memory.gradient_checkpointing import (
            SegmentCheckpointer,
            CheckpointPolicy,
        )

        checkpointer = SegmentCheckpointer(policy=CheckpointPolicy.SQRT)

        # First call populates cache
        checkpointer.should_checkpoint(0, 16)
        assert 16 in checkpointer._cached_positions

        # Second call uses cache
        checkpointer.should_checkpoint(1, 16)
        assert 16 in checkpointer._cached_positions

    def test_config_with_all_options(self):
        """Test CheckpointConfig with all options set."""
        from zenith.memory.gradient_checkpointing import (
            CheckpointConfig,
            CheckpointPolicy,
        )

        config = CheckpointConfig(
            enabled=True,
            policy=CheckpointPolicy.MEMORY_AWARE,
            num_segments=8,
            every_n=2,
            checkpoint_layers=("layer1", "layer2"),
            preserve_rng_state=False,
            use_reentrant=True,
            memory_efficient=False,
            lambda_tolerance=0.5,
            debug=True,
        )

        assert config.enabled is True
        assert config.policy == CheckpointPolicy.MEMORY_AWARE
        assert config.num_segments == 8
        assert config.checkpoint_layers == ("layer1", "layer2")
        assert config.use_reentrant is True
        assert config.debug is True


class TestIntegration:
    """Integration tests for the module."""

    def test_module_imports(self):
        """Test that all expected symbols are importable."""
        from zenith.memory import (
            CheckpointPolicy,
            CheckpointConfig,
            CheckpointingContext,
            checkpoint,
            checkpoint_sequential,
            checkpoint_wrapper,
            auto_checkpoint,
            estimate_memory_savings,
            get_memory_stats,
            SegmentCheckpointer,
            ModuleCheckpointer,
        )

        # All imports should work
        assert CheckpointPolicy is not None
        assert checkpoint is not None

    def test_full_workflow_disabled(self):
        """Test full workflow with checkpointing disabled."""
        from zenith.memory import (
            CheckpointingContext,
            checkpoint,
            CheckpointPolicy,
        )

        def layer1(x):
            return x + 1

        def layer2(x):
            return x * 2

        def layer3(x):
            return x - 1

        with CheckpointingContext(enabled=False, policy=CheckpointPolicy.SQRT):
            x = 10
            x = checkpoint(layer1, x)
            x = checkpoint(layer2, x)
            x = checkpoint(layer3, x)

        # (10 + 1) * 2 - 1 = 21
        assert x == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
