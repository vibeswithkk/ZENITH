# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Tests for CUDA Graphs implementation.

This test suite validates:
1. CudaGraphManager functionality
2. GraphCaptureContext context manager
3. CachedGraphModel wrapper
4. Graph caching and replay
5. Statistics tracking
6. Thread safety
7. LRU cache eviction
8. Static buffer management
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch


class TestGraphCaptureMode:
    """Tests for GraphCaptureMode enum."""

    def test_capture_modes_exist(self):
        """Verify all capture modes are defined."""
        from zenith.runtime.cuda_graphs import GraphCaptureMode

        assert GraphCaptureMode.GLOBAL.value == "global"
        assert GraphCaptureMode.THREAD_LOCAL.value == "thread_local"
        assert GraphCaptureMode.RELAXED.value == "relaxed"


class TestGraphStatus:
    """Tests for GraphStatus enum."""

    def test_graph_statuses_exist(self):
        """Verify all graph statuses are defined."""
        from zenith.runtime.cuda_graphs import GraphStatus

        assert GraphStatus.NOT_CAPTURED.value == "not_captured"
        assert GraphStatus.CAPTURING.value == "capturing"
        assert GraphStatus.CAPTURED.value == "captured"
        assert GraphStatus.INSTANTIATED.value == "instantiated"
        assert GraphStatus.INVALID.value == "invalid"


class TestGraphStatistics:
    """Tests for GraphStatistics dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        from zenith.runtime.cuda_graphs import GraphStatistics

        stats = GraphStatistics()
        assert stats.node_count == 0
        assert stats.launch_count == 0
        assert stats.total_time_ms == 0.0
        assert stats.average_time_ms == 0.0

    def test_update_average(self):
        """Test average calculation."""
        from zenith.runtime.cuda_graphs import GraphStatistics

        stats = GraphStatistics()
        stats.launch_count = 10
        stats.total_time_ms = 100.0
        stats.update_average()
        assert stats.average_time_ms == 10.0

    def test_update_average_zero_launches(self):
        """Test average with zero launches."""
        from zenith.runtime.cuda_graphs import GraphStatistics

        stats = GraphStatistics()
        stats.update_average()
        assert stats.average_time_ms == 0.0


class TestCachedGraph:
    """Tests for CachedGraph dataclass."""

    def test_default_values(self):
        """Test default cached graph values."""
        from zenith.runtime.cuda_graphs import CachedGraph, GraphStatus

        cached = CachedGraph(key="test")
        assert cached.key == "test"
        assert cached.status == GraphStatus.NOT_CAPTURED
        assert cached.graph_handle is None
        assert cached.exec_handle is None
        assert cached.static_inputs is None
        assert cached.static_outputs is None


class TestCudaGraphManager:
    """Tests for CudaGraphManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        assert manager.cache_size == 0
        assert manager.total_launches == 0

    def test_initialization_with_params(self):
        """Test manager initialization with custom parameters."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager(
            max_cached_graphs=50,
            warmup_iterations=5,
            enable_stats=False,
        )
        assert manager._max_cached == 50
        assert manager._warmup_iterations == 5
        assert manager._enable_stats is False

    def test_clear(self):
        """Test clearing cached graphs."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager()
        manager._cache["test"] = CachedGraph(key="test")
        manager._cache_order.append("test")
        assert manager.cache_size == 1

        cleared = manager.clear()
        assert cleared == 1
        assert manager.cache_size == 0
        assert len(manager._cache_order) == 0

    def test_invalidate_nonexistent(self):
        """Test invalidating a non-existent graph."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        result = manager.invalidate("nonexistent")
        assert result is False

    def test_invalidate_existing(self):
        """Test invalidating an existing graph."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager()
        manager._cache["test"] = CachedGraph(key="test")
        manager._cache_order.append("test")

        result = manager.invalidate("test")
        assert result is True
        assert "test" not in manager._cache
        assert "test" not in manager._cache_order

    def test_replay_nonexistent(self):
        """Test replaying a non-existent graph."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        result = manager.replay("nonexistent")
        assert result is False

    def test_get_statistics_nonexistent(self):
        """Test getting statistics for non-existent graph."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        stats = manager.get_statistics("nonexistent")
        assert stats is None

    def test_get_all_statistics(self):
        """Test getting all statistics."""
        from zenith.runtime.cuda_graphs import (
            CudaGraphManager,
            CachedGraph,
            GraphStatistics,
        )

        manager = CudaGraphManager()
        manager._cache["test1"] = CachedGraph(
            key="test1",
            statistics=GraphStatistics(launch_count=5),
        )
        manager._cache["test2"] = CachedGraph(
            key="test2",
            statistics=GraphStatistics(launch_count=10),
        )

        all_stats = manager.get_all_statistics()
        assert len(all_stats) == 2
        assert all_stats["test1"].launch_count == 5
        assert all_stats["test2"].launch_count == 10


class TestLRUCacheEviction:
    """Tests for LRU cache eviction."""

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager(max_cached_graphs=3)

        # Add 3 entries
        for i in range(3):
            key = f"test{i}"
            manager._cache[key] = CachedGraph(key=key)
            manager._cache_order.append(key)

        assert manager.cache_size == 3

        # Enforce limit - should evict oldest
        manager._enforce_cache_limit()

        # Should have evicted "test0" (oldest)
        assert manager.cache_size == 2
        assert "test0" not in manager._cache

    def test_lru_update_order(self):
        """Test that LRU order is updated on access."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        manager._cache_order = ["a", "b", "c"]

        manager._update_lru("a")

        # "a" should now be at the end (most recently used)
        assert manager._cache_order == ["b", "c", "a"]


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_cache_access(self):
        """Test concurrent access to cache."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager()
        errors = []

        def worker(tid):
            try:
                for i in range(10):
                    key = f"thread{tid}_item{i}"
                    manager._cache[key] = CachedGraph(key=key)
                    manager._cache_order.append(key)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

    def test_global_manager_thread_safety(self):
        """Test that global manager is thread-safe."""
        from zenith.runtime.cuda_graphs import get_global_manager

        managers = []

        def get_manager():
            managers.append(get_global_manager())

        threads = [threading.Thread(target=get_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same manager
        assert len(set(id(m) for m in managers)) == 1


class TestGraphCaptureContext:
    """Tests for GraphCaptureContext class."""

    def test_context_without_cuda(self):
        """Test capture context when CUDA not available."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        manager._cuda_available = False

        with manager.capture("test") as ctx:
            assert ctx is not None

    def test_context_properties(self):
        """Test context properties."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        ctx = manager.capture("test", stream=None)
        assert ctx._key == "test"

    def test_context_exception_handling(self):
        """Test that context invalidates graph on exception."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager()
        manager._cuda_available = False  # Disable CUDA for testing

        try:
            with manager.capture("test") as ctx:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Graph should have been invalidated
        assert "test" not in manager._cache


class TestCachedGraphModel:
    """Tests for CachedGraphModel class."""

    def test_initialization(self):
        """Test model wrapper initialization."""
        from zenith.runtime.cuda_graphs import CachedGraphModel

        mock_model = MagicMock()
        cached_model = CachedGraphModel(mock_model)
        assert cached_model._model is mock_model

    def test_fallback_without_cuda(self):
        """Test that model falls back to direct execution without CUDA."""
        from zenith.runtime.cuda_graphs import CachedGraphModel

        mock_model = MagicMock()
        mock_model.return_value = "output"

        cached_model = CachedGraphModel(mock_model)
        cached_model._manager._cuda_available = False

        result = cached_model("input")
        mock_model.assert_called_once_with("input")

    def test_invalidate(self):
        """Test cache invalidation."""
        from zenith.runtime.cuda_graphs import CachedGraphModel

        mock_model = MagicMock()
        cached_model = CachedGraphModel(mock_model)

        cached_model._captured_keys.add("test")
        cached_model._warmup_counters["test"] = 0
        cached_model._static_outputs["test"] = "output"

        cached_model.invalidate()
        assert len(cached_model._captured_keys) == 0
        assert len(cached_model._warmup_counters) == 0
        assert len(cached_model._static_outputs) == 0

    def test_shape_key_generation(self):
        """Test shape-based cache key generation."""
        from zenith.runtime.cuda_graphs import CachedGraphModel

        mock_model = MagicMock()
        cached_model = CachedGraphModel(mock_model)

        tensor1 = MagicMock()
        tensor1.shape = (1, 768)
        tensor2 = MagicMock()
        tensor2.shape = (1, 768)
        tensor3 = MagicMock()
        tensor3.shape = (2, 768)

        key1 = cached_model._get_shape_key((tensor1,), {})
        key2 = cached_model._get_shape_key((tensor2,), {})
        key3 = cached_model._get_shape_key((tensor3,), {})

        assert key1 == key2
        assert key1 != key3


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_global_manager(self):
        """Test getting global manager."""
        from zenith.runtime.cuda_graphs import get_global_manager, CudaGraphManager

        manager = get_global_manager()
        assert isinstance(manager, CudaGraphManager)

        manager2 = get_global_manager()
        assert manager is manager2

    def test_capture_function(self):
        """Test module-level capture function."""
        from zenith.runtime.cuda_graphs import capture, GraphCaptureContext

        ctx = capture("test_key")
        assert isinstance(ctx, GraphCaptureContext)

    def test_replay_function(self):
        """Test module-level replay function."""
        from zenith.runtime.cuda_graphs import replay

        result = replay("nonexistent")
        assert result is False


class TestGraphCachedDecorator:
    """Tests for graph_cached decorator."""

    def test_decorator_without_cuda(self):
        """Test decorator when CUDA not available."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        manager._cuda_available = False

        @manager.graph_cached("test_func")
        def test_function(x):
            return x * 2

        result = test_function(5)
        assert result == 10

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()

        @manager.graph_cached()
        def my_function(x):
            """Test docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "Test docstring."

    def test_decorator_warmup_counter(self):
        """Test that decorator respects warmup iterations."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager(warmup_iterations=2)
        manager._cuda_available = False  # Disable CUDA

        call_count = 0

        @manager.graph_cached()
        def counted_function(x):
            nonlocal call_count
            call_count += 1
            return x

        # Call multiple times
        for i in range(5):
            counted_function(i)

        # All calls should have executed (no CUDA)
        assert call_count == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_replay_not_instantiated(self):
        """Test replaying a graph that is not instantiated."""
        from zenith.runtime.cuda_graphs import (
            CudaGraphManager,
            CachedGraph,
            GraphStatus,
        )

        manager = CudaGraphManager()
        manager._cuda_available = True
        manager._torch_available = True
        manager._cache["test"] = CachedGraph(
            key="test",
            status=GraphStatus.CAPTURED,
        )

        with pytest.raises(ValueError, match="not instantiated"):
            manager.replay("test")

    def test_empty_statistics(self):
        """Test statistics with no data."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        all_stats = manager.get_all_statistics()
        assert len(all_stats) == 0

    def test_total_launches_empty(self):
        """Test total launches with empty cache."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        assert manager.total_launches == 0

    def test_total_launches_with_data(self):
        """Test total launches calculation."""
        from zenith.runtime.cuda_graphs import (
            CudaGraphManager,
            CachedGraph,
            GraphStatistics,
        )

        manager = CudaGraphManager()
        manager._cache["test1"] = CachedGraph(
            key="test1",
            statistics=GraphStatistics(launch_count=5),
        )
        manager._cache["test2"] = CachedGraph(
            key="test2",
            statistics=GraphStatistics(launch_count=10),
        )

        assert manager.total_launches == 15


class TestGraphProperties:
    """Tests for manager properties."""

    def test_cuda_available_property(self):
        """Test cuda_available property."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        assert isinstance(manager.cuda_available, bool)

    def test_graphs_enabled_property(self):
        """Test graphs_enabled property."""
        from zenith.runtime.cuda_graphs import CudaGraphManager

        manager = CudaGraphManager()
        assert isinstance(manager.graphs_enabled, bool)

    def test_cache_size_property(self):
        """Test cache_size property."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager()
        assert manager.cache_size == 0

        manager._cache["test"] = CachedGraph(key="test")
        assert manager.cache_size == 1


class TestStaticBufferManagement:
    """Tests for static buffer handling."""

    def test_cached_graph_static_fields(self):
        """Test that CachedGraph has static buffer fields."""
        from zenith.runtime.cuda_graphs import CachedGraph

        cached = CachedGraph(
            key="test",
            static_inputs=[1, 2, 3],
            static_outputs={"result": 42},
        )

        assert cached.static_inputs == [1, 2, 3]
        assert cached.static_outputs == {"result": 42}

    def test_invalidate_clears_static_buffers(self):
        """Test that invalidate clears static buffers."""
        from zenith.runtime.cuda_graphs import CudaGraphManager, CachedGraph

        manager = CudaGraphManager()
        manager._cache["test"] = CachedGraph(
            key="test",
            static_inputs=[1, 2],
            static_outputs={"x": 1},
        )
        manager._cache_order.append("test")

        manager.invalidate("test")

        # After invalidation, key should not exist
        assert "test" not in manager._cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
