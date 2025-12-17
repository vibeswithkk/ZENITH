"""
Autotuner Unit Tests

Comprehensive unit testing for kernel auto-tuning system as specified in CetakBiru 5.3:
- TuningConfig: Configuration hash, equality
- TuningResult: Serialization, deserialization
- SearchSpace: Parameter definition, iteration, sampling
- GridSearch: Exhaustive search
- RandomSearch: Random sampling
- TuningCache: Persistent caching
- KernelAutotuner: Main interface
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from zenith.optimization.autotuner import (
    TuningConfig,
    TuningResult,
    SearchSpace,
    GridSearch,
    RandomSearch,
    TuningCache,
    KernelAutotuner,
    MATMUL_SEARCH_SPACE,
    CONV2D_SEARCH_SPACE,
    RELU_SEARCH_SPACE,
    get_search_space,
    autotune_matmul,
)


class TestTuningConfig:
    """Unit tests for TuningConfig."""

    def test_tuning_config_creation(self):
        """Test TuningConfig creation."""
        config = TuningConfig(
            op_name="matmul",
            input_shapes=[(256, 256), (256, 256)],
        )
        assert config.op_name == "matmul"
        assert config.dtype == "float32"
        assert config.device == "cpu"

    def test_tuning_config_custom_params(self):
        """Test TuningConfig with custom parameters."""
        config = TuningConfig(
            op_name="conv2d",
            input_shapes=[(1, 3, 224, 224)],
            dtype="float16",
            device="cuda",
        )
        assert config.dtype == "float16"
        assert config.device == "cuda"

    def test_tuning_config_hash(self):
        """Test config hash generation."""
        config = TuningConfig("matmul", [(256, 256)])
        hash_val = config.config_hash()
        assert len(hash_val) == 16  # 16 hex chars
        assert isinstance(hash_val, str)

    def test_tuning_config_hash_deterministic(self):
        """Test that same config produces same hash."""
        config1 = TuningConfig("matmul", [(256, 256)])
        config2 = TuningConfig("matmul", [(256, 256)])
        assert config1.config_hash() == config2.config_hash()

    def test_tuning_config_hash_different(self):
        """Test that different configs produce different hashes."""
        config1 = TuningConfig("matmul", [(256, 256)])
        config2 = TuningConfig("matmul", [(512, 512)])
        assert config1.config_hash() != config2.config_hash()


class TestTuningResult:
    """Unit tests for TuningResult."""

    def test_tuning_result_creation(self):
        """Test TuningResult creation."""
        result = TuningResult(
            config_hash="abc123",
            params={"tile_m": 32},
            mean_time_ms=1.5,
            std_time_ms=0.1,
        )
        assert result.config_hash == "abc123"
        assert result.mean_time_ms == 1.5

    def test_tuning_result_to_dict(self):
        """Test converting result to dict."""
        result = TuningResult(
            config_hash="abc123",
            params={"tile_m": 32},
            mean_time_ms=1.5,
            std_time_ms=0.1,
            gflops=100.0,
        )
        d = result.to_dict()
        assert d["config_hash"] == "abc123"
        assert d["gflops"] == 100.0

    def test_tuning_result_from_dict(self):
        """Test creating result from dict."""
        data = {
            "config_hash": "xyz789",
            "params": {"tile_m": 64},
            "mean_time_ms": 2.0,
            "std_time_ms": 0.2,
            "gflops": 0.0,
            "memory_bandwidth_gbps": 0.0,
            "timestamp": 0.0,
        }
        result = TuningResult.from_dict(data)
        assert result.config_hash == "xyz789"
        assert result.params["tile_m"] == 64


class TestSearchSpace:
    """Unit tests for SearchSpace."""

    def test_search_space_creation(self):
        """Test SearchSpace creation."""
        space = SearchSpace("test")
        assert space.name == "test"
        assert space.size() == 0

    def test_search_space_define(self):
        """Test defining parameters."""
        space = SearchSpace("test")
        space.define("tile_m", [16, 32, 64])
        assert "tile_m" in space.params
        assert space.params["tile_m"] == [16, 32, 64]

    def test_search_space_chaining(self):
        """Test method chaining."""
        space = (
            SearchSpace("test").define("tile_m", [16, 32]).define("tile_n", [16, 32])
        )
        assert "tile_m" in space.params
        assert "tile_n" in space.params

    def test_search_space_size(self):
        """Test size calculation."""
        space = (
            SearchSpace("test")
            .define("tile_m", [16, 32, 64])  # 3 options
            .define("tile_n", [16, 32])  # 2 options
        )
        assert space.size() == 6  # 3 * 2 = 6

    def test_search_space_iterate(self):
        """Test iterating over configurations."""
        space = SearchSpace("test").define("a", [1, 2]).define("b", [10, 20])
        configs = list(space.iterate())
        assert len(configs) == 4
        assert {"a": 1, "b": 10} in configs
        assert {"a": 2, "b": 20} in configs

    def test_search_space_iterate_empty(self):
        """Test iterating over empty space."""
        space = SearchSpace("empty")
        configs = list(space.iterate())
        assert len(configs) == 1  # One empty config

    def test_search_space_sample(self):
        """Test random sampling."""
        space = (
            SearchSpace("test")
            .define("a", [1, 2, 3, 4, 5])
            .define("b", [1, 2, 3, 4, 5])
        )
        samples = space.sample(5)
        assert len(samples) == 5
        for sample in samples:
            assert "a" in sample
            assert "b" in sample

    def test_search_space_sample_with_rng(self):
        """Test sampling with custom RNG."""
        space = SearchSpace("test").define("x", [1, 2, 3, 4, 5])
        rng = np.random.default_rng(42)
        samples = space.sample(3, rng)
        assert len(samples) == 3


class TestGridSearch:
    """Unit tests for GridSearch strategy."""

    def test_grid_search_basic(self):
        """Test basic grid search."""
        space = SearchSpace("test").define("x", [1, 2, 3])

        def evaluate(params):
            return params["x"] ** 2  # Minimum at x=1

        searcher = GridSearch()
        best_params, best_time = searcher.search(space, evaluate, max_trials=10)
        assert best_params["x"] == 1
        assert best_time == 1

    def test_grid_search_max_trials(self):
        """Test grid search with max trials limit."""
        space = SearchSpace("test").define("x", list(range(100)))

        eval_count = 0

        def evaluate(params):
            nonlocal eval_count
            eval_count += 1
            return params["x"]

        searcher = GridSearch()
        searcher.search(space, evaluate, max_trials=10)
        assert eval_count == 10


class TestRandomSearch:
    """Unit tests for RandomSearch strategy."""

    def test_random_search_basic(self):
        """Test basic random search."""
        space = SearchSpace("test").define("x", [1, 2, 3, 4, 5])

        def evaluate(params):
            return params["x"]

        searcher = RandomSearch(seed=42)
        best_params, best_time = searcher.search(space, evaluate, max_trials=5)
        assert "x" in best_params
        assert best_params["x"] in [1, 2, 3, 4, 5]

    def test_random_search_with_seed(self):
        """Test reproducibility with seed."""
        space = SearchSpace("test").define("x", list(range(10)))

        def evaluate(params):
            return params["x"]

        searcher1 = RandomSearch(seed=42)
        best1, _ = searcher1.search(space, evaluate, max_trials=5)

        searcher2 = RandomSearch(seed=42)
        best2, _ = searcher2.search(space, evaluate, max_trials=5)

        assert best1 == best2


class TestTuningCache:
    """Unit tests for TuningCache."""

    def test_cache_creation(self):
        """Test cache creation with temp file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache = TuningCache(f.name)
            assert cache.cache_path == Path(f.name)

    def test_cache_put_get(self):
        """Test putting and getting from cache."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache = TuningCache(f.name)
            config = TuningConfig("matmul", [(256, 256)])
            result = TuningResult(
                config_hash=config.config_hash(),
                params={"tile_m": 32},
                mean_time_ms=1.5,
                std_time_ms=0.1,
            )

            cache.put(config, result)
            retrieved = cache.get(config)

            assert retrieved is not None
            assert retrieved.params["tile_m"] == 32

    def test_cache_get_not_found(self):
        """Test getting non-existent entry."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache = TuningCache(f.name)
            config = TuningConfig("nonexistent", [])
            result = cache.get(config)
            assert result is None

    def test_cache_clear(self):
        """Test clearing cache."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache = TuningCache(f.name)
            config = TuningConfig("matmul", [(256, 256)])
            result = TuningResult(
                config_hash=config.config_hash(),
                params={},
                mean_time_ms=1.0,
                std_time_ms=0.1,
            )
            cache.put(config, result)
            cache.clear()
            assert cache.get(config) is None


class TestKernelAutotuner:
    """Unit tests for KernelAutotuner."""

    def test_autotuner_creation(self):
        """Test autotuner creation."""
        tuner = KernelAutotuner()
        assert tuner is not None
        assert tuner.strategy is not None

    def test_autotuner_with_strategy(self):
        """Test autotuner with custom strategy."""
        tuner = KernelAutotuner(strategy=GridSearch())
        assert isinstance(tuner.strategy, GridSearch)

    def test_autotuner_tune_basic(self):
        """Test basic tuning."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tuner = KernelAutotuner(cache_path=f.name)
            config = TuningConfig("test", [(10, 10)])
            space = SearchSpace("test").define("x", [1, 2, 3])

            def evaluate(params):
                return params["x"]  # No-op kernel

            best_params, best_time = tuner.tune(
                config,
                space,
                evaluate,
                max_trials=3,
                warmup=0,
                repetitions=1,
            )

            assert "x" in best_params

    def test_autotuner_uses_cache(self):
        """Test that autotuner uses cache."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tuner = KernelAutotuner(cache_path=f.name)
            config = TuningConfig("cached_test", [(10, 10)])
            space = SearchSpace("test").define("x", [1])

            call_count = 0

            def evaluate(params):
                nonlocal call_count
                call_count += 1
                return 1.0

            # First call should evaluate
            tuner.tune(config, space, evaluate, max_trials=1, warmup=0, repetitions=1)
            first_count = call_count

            # Second call should use cache
            tuner.tune(config, space, evaluate, max_trials=1, warmup=0, repetitions=1)
            assert call_count == first_count  # No new evaluations

    def test_autotuner_force_retune(self):
        """Test force retuning."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tuner = KernelAutotuner(cache_path=f.name)
            config = TuningConfig("retune_test", [(10, 10)])
            space = SearchSpace("test").define("x", [1])

            call_count = 0

            def evaluate(params):
                nonlocal call_count
                call_count += 1
                return 1.0

            # First call
            tuner.tune(config, space, evaluate, max_trials=1, warmup=0, repetitions=1)
            first_count = call_count

            # Force retune should evaluate again
            tuner.tune(
                config,
                space,
                evaluate,
                max_trials=1,
                warmup=0,
                repetitions=1,
                force_retune=True,
            )
            assert call_count > first_count


class TestPredefinedSearchSpaces:
    """Unit tests for predefined search spaces."""

    def test_matmul_search_space(self):
        """Test MATMUL_SEARCH_SPACE is defined."""
        assert MATMUL_SEARCH_SPACE is not None
        assert "tile_m" in MATMUL_SEARCH_SPACE.params
        assert "tile_n" in MATMUL_SEARCH_SPACE.params

    def test_conv2d_search_space(self):
        """Test CONV2D_SEARCH_SPACE is defined."""
        assert CONV2D_SEARCH_SPACE is not None
        assert "tile_h" in CONV2D_SEARCH_SPACE.params
        assert "tile_w" in CONV2D_SEARCH_SPACE.params

    def test_relu_search_space(self):
        """Test RELU_SEARCH_SPACE is defined."""
        assert RELU_SEARCH_SPACE is not None
        assert "vector_width" in RELU_SEARCH_SPACE.params

    def test_get_search_space_matmul(self):
        """Test getting matmul search space."""
        space = get_search_space("matmul")
        assert space == MATMUL_SEARCH_SPACE

    def test_get_search_space_unknown(self):
        """Test getting unknown search space."""
        space = get_search_space("unknown_op")
        assert "unroll_factor" in space.params  # Default space


class TestAutotuneMatmul:
    """Unit tests for autotune_matmul convenience function."""

    def test_autotune_matmul_basic(self):
        """Test autotune_matmul function."""

        def mock_evaluate(params):
            return 1.0  # Mock kernel execution

        best_params = autotune_matmul(
            M=256,
            K=256,
            N=256,
            evaluate=mock_evaluate,
            max_trials=5,
        )

        assert isinstance(best_params, dict)
