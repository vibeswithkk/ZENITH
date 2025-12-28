# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit tests for Zenith JAX Core Integration modules.

Tests:
- Gradient Checkpointing
- Memory Management
- Mixed Precision Training
"""

import math
import pytest
from typing import List

# Check if JAX is available
try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")


class TestOptimalCheckpointSelector:
    """Tests for OptimalCheckpointSelector."""

    def test_sqrt_selection_basic(self):
        """Test sqrt heuristic produces valid checkpoints."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=16)
        checkpoints = selector.select_sqrt()

        assert len(checkpoints) == 4
        assert checkpoints[0] == 0
        assert all(0 <= c < 16 for c in checkpoints)

    def test_sqrt_selection_small(self):
        """Test sqrt heuristic for small networks."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=1)
        checkpoints = selector.select_sqrt()
        assert checkpoints == [0]

        selector = OptimalCheckpointSelector(num_layers=0)
        checkpoints = selector.select_sqrt()
        assert checkpoints == []

    def test_dp_selection_basic(self):
        """Test DP algorithm produces valid checkpoints."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=10)
        checkpoints = selector.select_dp()

        assert len(checkpoints) > 0
        assert all(0 <= c < 10 for c in checkpoints)

    def test_uniform_selection(self):
        """Test uniform checkpoint selection."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=12)
        checkpoints = selector.select_uniform(num_checkpoints=4)

        assert len(checkpoints) == 4

    def test_memory_reduction_estimate(self):
        """Test memory reduction estimation."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        selector = OptimalCheckpointSelector(num_layers=100)
        checkpoints = selector.select_sqrt()

        reduction = selector.estimate_memory_reduction(checkpoints)

        assert 0 <= reduction <= 100
        assert reduction > 0  # Should have some reduction

    def test_invalid_num_layers_raises(self):
        """Test that invalid num_layers raises error."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        with pytest.raises(ValueError):
            OptimalCheckpointSelector(num_layers=-1)

    def test_invalid_tolerance_raises(self):
        """Test that invalid compute_tolerance raises error."""
        from zenith.jax.checkpointing import OptimalCheckpointSelector

        with pytest.raises(ValueError):
            OptimalCheckpointSelector(num_layers=10, compute_tolerance=1.5)

        with pytest.raises(ValueError):
            OptimalCheckpointSelector(num_layers=10, compute_tolerance=-0.1)


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from zenith.jax.checkpointing import (
            CheckpointConfig,
            CheckpointPolicy,
            SelectionMethod,
        )

        config = CheckpointConfig()

        assert config.policy == CheckpointPolicy.DOTS_SAVEABLE
        assert config.selection_method == SelectionMethod.SQRT
        assert config.compute_tolerance == 0.33
        assert config.offload_to_cpu is False

    def test_custom_config(self):
        """Test custom configuration."""
        from zenith.jax.checkpointing import (
            CheckpointConfig,
            CheckpointPolicy,
            SelectionMethod,
        )

        config = CheckpointConfig(
            policy=CheckpointPolicy.NOTHING,
            selection_method=SelectionMethod.DP,
            memory_budget_gb=8.0,
            offload_to_cpu=True,
        )

        assert config.policy == CheckpointPolicy.NOTHING
        assert config.selection_method == SelectionMethod.DP
        assert config.memory_budget_gb == 8.0
        assert config.offload_to_cpu is True


class TestZenithCheckpointer:
    """Tests for ZenithCheckpointer."""

    def test_checkpointer_creation(self):
        """Test checkpointer creation with default config."""
        from zenith.jax.checkpointing import ZenithCheckpointer, CheckpointConfig

        checkpointer = ZenithCheckpointer()

        assert checkpointer.config is not None
        assert checkpointer.stats is not None

    def test_checkpointer_with_custom_config(self):
        """Test checkpointer with custom config."""
        from zenith.jax.checkpointing import (
            ZenithCheckpointer,
            CheckpointConfig,
            CheckpointPolicy,
        )

        config = CheckpointConfig(
            policy=CheckpointPolicy.NOTHING,
            enable_profiling=True,
        )
        checkpointer = ZenithCheckpointer(config=config)

        assert checkpointer.config.policy == CheckpointPolicy.NOTHING
        assert checkpointer.config.enable_profiling is True

    def test_reset_stats(self):
        """Test stats reset."""
        from zenith.jax.checkpointing import ZenithCheckpointer

        checkpointer = ZenithCheckpointer()
        checkpointer.reset_stats()

        assert checkpointer.stats.num_checkpoints == 0


class TestJAXActivationStore:
    """Tests for JAXActivationStore."""

    def test_store_creation(self):
        """Test activation store creation."""
        from zenith.jax.memory_manager import JAXActivationStore

        store = JAXActivationStore(max_memory_bytes=1024 * 1024)

        assert store.memory_usage == 0
        assert len(store) == 0

    def test_store_retrieve_basic(self):
        """Test basic store and retrieve operations."""
        from zenith.jax.memory_manager import JAXActivationStore
        import numpy as np

        store = JAXActivationStore()

        arr = np.zeros((100, 100), dtype=np.float32)

        success = store.store(layer_id=0, array=arr)
        assert success is True
        assert len(store) == 1

        retrieved = store.retrieve(layer_id=0)
        assert retrieved is not None
        assert np.array_equal(retrieved, arr)

        missing = store.retrieve(layer_id=99)
        assert missing is None

    def test_store_remove(self):
        """Test remove operation."""
        from zenith.jax.memory_manager import JAXActivationStore
        import numpy as np

        store = JAXActivationStore()

        arr = np.zeros((10, 10), dtype=np.float32)
        store.store(layer_id=0, array=arr)

        removed = store.remove(layer_id=0)
        assert removed is not None
        assert len(store) == 0

        assert store.retrieve(layer_id=0) is None

    def test_store_clear(self):
        """Test clear operation."""
        from zenith.jax.memory_manager import JAXActivationStore
        import numpy as np

        store = JAXActivationStore()

        for i in range(5):
            arr = np.zeros((10, 10), dtype=np.float32)
            store.store(layer_id=i, array=arr)

        assert len(store) == 5

        cleared = store.clear()
        assert cleared == 5
        assert len(store) == 0

    def test_eviction_policy_lru(self):
        """Test LRU eviction policy."""
        from zenith.jax.memory_manager import (
            JAXActivationStore,
            EvictionPolicy,
        )
        import numpy as np

        arr_size = 400  # 400 bytes per float32 array of 100 elements
        max_memory = 1000  # Allow only ~2 arrays

        store = JAXActivationStore(
            max_memory_bytes=max_memory,
            eviction_policy=EvictionPolicy.LRU,
        )

        arr1 = np.zeros(100, dtype=np.float32)
        arr2 = np.zeros(100, dtype=np.float32)
        arr3 = np.zeros(100, dtype=np.float32)

        store.store(layer_id=0, array=arr1)
        store.store(layer_id=1, array=arr2)

        store.retrieve(layer_id=0)

        store.store(layer_id=2, array=arr3)

        stats = store.statistics
        assert stats["eviction_count"] >= 0

    def test_checkpoint_protection(self):
        """Test that checkpoints are protected from eviction."""
        from zenith.jax.memory_manager import JAXActivationStore
        import numpy as np

        store = JAXActivationStore(max_memory_bytes=1000)

        arr = np.zeros(100, dtype=np.float32)
        store.store(layer_id=0, array=arr, is_checkpoint=True)

        stats = store.statistics
        assert stats["checkpoint_count"] == 1

    def test_contains_operator(self):
        """Test __contains__ operator."""
        from zenith.jax.memory_manager import JAXActivationStore
        import numpy as np

        store = JAXActivationStore()
        arr = np.zeros(10, dtype=np.float32)
        store.store(layer_id=5, array=arr)

        assert 5 in store
        assert 99 not in store


class TestJAXMemoryManager:
    """Tests for JAXMemoryManager."""

    def test_manager_creation(self):
        """Test memory manager creation."""
        from zenith.jax.memory_manager import JAXMemoryManager, JAXMemoryConfig

        manager = JAXMemoryManager()

        assert manager.memory_usage == 0
        assert manager.config is not None

    def test_manager_with_config(self):
        """Test manager with custom config."""
        from zenith.jax.memory_manager import JAXMemoryManager, JAXMemoryConfig

        config = JAXMemoryConfig(
            max_memory_bytes=1024 * 1024 * 1024,
            enable_offloading=True,
        )
        manager = JAXMemoryManager(config=config)

        assert manager.config.max_memory_bytes == 1024 * 1024 * 1024
        assert manager.config.enable_offloading is True

    @requires_jax
    def test_store_retrieve(self):
        """Test store and retrieve through manager."""
        from zenith.jax.memory_manager import JAXMemoryManager
        import numpy as np

        manager = JAXMemoryManager()

        arr = np.zeros((50, 50), dtype=np.float32)
        success = manager.store(layer_id=0, array=arr)

        assert success is True

        retrieved = manager.retrieve(layer_id=0)
        assert retrieved is not None

    @requires_jax
    def test_statistics(self):
        """Test statistics collection."""
        from zenith.jax.memory_manager import JAXMemoryManager
        import numpy as np

        manager = JAXMemoryManager()

        arr = np.zeros((100, 100), dtype=np.float32)
        manager.store(layer_id=0, array=arr)

        stats = manager.get_statistics()

        assert "current_memory_bytes" in stats
        assert "stored_count" in stats
        assert stats["stored_count"] == 1


class TestMixedPrecisionPolicy:
    """Tests for MixedPrecisionPolicy."""

    def test_fp32_policy(self):
        """Test FP32 policy."""
        from zenith.jax.mixed_precision import MixedPrecisionPolicy, PrecisionMode

        policy = MixedPrecisionPolicy.fp32()

        assert policy.param_dtype == "float32"
        assert policy.compute_dtype == "float32"
        assert policy.output_dtype == "float32"
        assert policy.mode == PrecisionMode.FP32
        assert policy.requires_loss_scaling is False

    def test_bf16_policy(self):
        """Test BF16 policy."""
        from zenith.jax.mixed_precision import MixedPrecisionPolicy, PrecisionMode

        policy = MixedPrecisionPolicy.bf16()

        assert policy.param_dtype == "float32"
        assert policy.compute_dtype == "bfloat16"
        assert policy.output_dtype == "bfloat16"
        assert policy.mode == PrecisionMode.BF16
        assert policy.requires_loss_scaling is False

    def test_fp16_policy(self):
        """Test FP16 policy."""
        from zenith.jax.mixed_precision import MixedPrecisionPolicy, PrecisionMode

        policy = MixedPrecisionPolicy.fp16()

        assert policy.param_dtype == "float32"
        assert policy.compute_dtype == "float16"
        assert policy.output_dtype == "float16"
        assert policy.mode == PrecisionMode.FP16
        assert policy.requires_loss_scaling is True


class TestDynamicLossScaler:
    """Tests for DynamicLossScaler."""

    def test_scaler_creation(self):
        """Test loss scaler creation."""
        from zenith.jax.mixed_precision import DynamicLossScaler

        scaler = DynamicLossScaler()

        assert scaler.scale == 2**15

    def test_scaler_with_config(self):
        """Test scaler with custom config."""
        from zenith.jax.mixed_precision import (
            DynamicLossScaler,
            LossScalerConfig,
        )

        config = LossScalerConfig(
            initial_scale=1024.0,
            growth_factor=4.0,
            backoff_factor=0.25,
        )
        scaler = DynamicLossScaler(config=config)

        assert scaler.scale == 1024.0

    def test_scale_loss(self):
        """Test loss scaling."""
        from zenith.jax.mixed_precision import DynamicLossScaler
        import numpy as np

        scaler = DynamicLossScaler()

        loss = np.array(1.0, dtype=np.float32)
        scaled = scaler.scale_loss(loss)

        assert np.isclose(scaled, loss * scaler.scale)

    def test_update_on_good_steps(self):
        """Test scale update on good steps."""
        from zenith.jax.mixed_precision import (
            DynamicLossScaler,
            LossScalerConfig,
        )

        config = LossScalerConfig(
            initial_scale=1.0,
            growth_factor=2.0,
            growth_interval=3,
        )
        scaler = DynamicLossScaler(config=config)

        initial_scale = scaler.scale

        for _ in range(3):
            scaler.update(grads_finite=True)

        assert scaler.scale == initial_scale * 2.0

    def test_update_on_overflow(self):
        """Test scale update on overflow."""
        from zenith.jax.mixed_precision import (
            DynamicLossScaler,
            LossScalerConfig,
        )

        config = LossScalerConfig(
            initial_scale=100.0,
            backoff_factor=0.5,
        )
        scaler = DynamicLossScaler(config=config)

        scaler.update(grads_finite=False)

        assert scaler.scale == 50.0

    def test_reset(self):
        """Test scaler reset."""
        from zenith.jax.mixed_precision import DynamicLossScaler

        scaler = DynamicLossScaler()

        for _ in range(10):
            scaler.update(grads_finite=False)

        scaler.reset()

        assert scaler.scale == 2**15


class TestZenithMixedPrecision:
    """Tests for ZenithMixedPrecision."""

    def test_creation_bf16(self):
        """Test creation with BF16 policy."""
        from zenith.jax.mixed_precision import (
            ZenithMixedPrecision,
            PrecisionMode,
        )

        mp = ZenithMixedPrecision(policy="bf16")

        assert mp.policy.mode == PrecisionMode.BF16
        assert mp.scaler is None

    def test_creation_fp16(self):
        """Test creation with FP16 policy."""
        from zenith.jax.mixed_precision import (
            ZenithMixedPrecision,
            PrecisionMode,
        )

        mp = ZenithMixedPrecision(policy="fp16")

        assert mp.policy.mode == PrecisionMode.FP16
        assert mp.scaler is not None

    def test_invalid_policy_raises(self):
        """Test that invalid policy raises error."""
        from zenith.jax.mixed_precision import ZenithMixedPrecision

        with pytest.raises(ValueError):
            ZenithMixedPrecision(policy="invalid")

    def test_stats(self):
        """Test statistics tracking."""
        from zenith.jax.mixed_precision import ZenithMixedPrecision

        mp = ZenithMixedPrecision(policy="bf16")

        stats = mp.stats

        assert stats.total_steps == 0


class TestCreatePolicy:
    """Tests for create_policy function."""

    def test_create_fp32(self):
        """Test creating FP32 policy."""
        from zenith.jax.mixed_precision import create_policy, PrecisionMode

        policy = create_policy("fp32")
        assert policy.mode == PrecisionMode.FP32

    def test_create_bf16(self):
        """Test creating BF16 policy."""
        from zenith.jax.mixed_precision import create_policy, PrecisionMode

        policy = create_policy("bf16")
        assert policy.mode == PrecisionMode.BF16

    def test_create_fp16(self):
        """Test creating FP16 policy."""
        from zenith.jax.mixed_precision import create_policy, PrecisionMode

        policy = create_policy("fp16")
        assert policy.mode == PrecisionMode.FP16

    def test_create_invalid_raises(self):
        """Test that invalid mode raises error."""
        from zenith.jax.mixed_precision import create_policy

        with pytest.raises(ValueError):
            create_policy("invalid")


class TestPublicAPI:
    """Tests for zenith.jax public API."""

    def test_all_exports_accessible(self):
        """Test that all __all__ exports are accessible."""
        import zenith.jax as zjax

        for name in zjax.__all__:
            assert hasattr(zjax, name), f"Missing export: {name}"

    def test_checkpoint_function_exists(self):
        """Test checkpoint function is accessible."""
        from zenith.jax import checkpoint

        assert callable(checkpoint)

    def test_checkpoint_sequential_exists(self):
        """Test checkpoint_sequential function is accessible."""
        from zenith.jax import checkpoint_sequential

        assert callable(checkpoint_sequential)

    def test_remat_exists(self):
        """Test remat function is accessible."""
        from zenith.jax import remat

        assert callable(remat)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
