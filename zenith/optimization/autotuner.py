"""
Kernel Auto-tuning System

Implements automatic kernel parameter optimization with:
- Search space definition per kernel type
- Grid/random search strategies
- Persistent caching to file
- Performance prediction model

Based on CetakBiru Section 5.1 requirements and Apache TVM approach.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Any
import numpy as np


@dataclass
class TuningConfig:
    """Configuration for a specific kernel operation."""

    op_name: str
    input_shapes: list[tuple]
    dtype: str = "float32"
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def config_hash(self) -> str:
        """Generate unique hash for this configuration."""
        data = f"{self.op_name}_{self.input_shapes}_{self.dtype}_{self.device}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


@dataclass
class TuningResult:
    """Result of a tuning trial."""

    config_hash: str
    params: dict
    mean_time_ms: float
    std_time_ms: float
    gflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TuningResult":
        return cls(**data)


class SearchSpace:
    """Defines a search space for kernel parameters."""

    def __init__(self, name: str):
        self.name = name
        self.params: dict[str, list] = {}

    def define(self, param_name: str, values: list) -> "SearchSpace":
        """Define a parameter with possible values."""
        self.params[param_name] = values
        return self

    def size(self) -> int:
        """Total number of configurations in search space."""
        if not self.params:
            return 0
        total = 1
        for values in self.params.values():
            total *= len(values)
        return total

    def iterate(self):
        """Iterate over all configurations."""
        if not self.params:
            yield {}
            return

        param_names = list(self.params.keys())
        param_values = list(self.params.values())

        def recursive(idx: int, current: dict):
            if idx == len(param_names):
                yield current.copy()
                return
            for val in param_values[idx]:
                current[param_names[idx]] = val
                yield from recursive(idx + 1, current)

        yield from recursive(0, {})

    def sample(self, n: int, rng: np.random.Generator | None = None):
        """Randomly sample n configurations."""
        if rng is None:
            rng = np.random.default_rng()

        all_configs = list(self.iterate())
        n = min(n, len(all_configs))
        indices = rng.choice(len(all_configs), size=n, replace=False)
        return [all_configs[i] for i in indices]


# Predefined search spaces for common operations

MATMUL_SEARCH_SPACE = (
    SearchSpace("matmul")
    .define("tile_m", [16, 32, 64, 128])
    .define("tile_n", [16, 32, 64, 128])
    .define("tile_k", [8, 16, 32, 64])
    .define("unroll_factor", [1, 2, 4, 8])
)

CONV2D_SEARCH_SPACE = (
    SearchSpace("conv2d")
    .define("tile_h", [1, 2, 4, 8])
    .define("tile_w", [1, 2, 4, 8])
    .define("tile_c", [16, 32, 64])
    .define("unroll_factor", [1, 2, 4])
)

RELU_SEARCH_SPACE = (
    SearchSpace("relu")
    .define("vector_width", [4, 8, 16, 32])
    .define("unroll_factor", [1, 2, 4, 8])
)


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def search(
        self,
        space: SearchSpace,
        evaluate: Callable[[dict], float],
        max_trials: int,
    ) -> tuple[dict, float]:
        """
        Search for optimal parameters.

        Args:
            space: Search space to explore
            evaluate: Function that takes params and returns time in ms
            max_trials: Maximum number of trials

        Returns:
            (best_params, best_time)
        """
        pass


class GridSearch(SearchStrategy):
    """Exhaustive grid search strategy."""

    def search(
        self,
        space: SearchSpace,
        evaluate: Callable[[dict], float],
        max_trials: int,
    ) -> tuple[dict, float]:
        best_params = {}
        best_time = float("inf")

        for i, params in enumerate(space.iterate()):
            if i >= max_trials:
                break

            time_ms = evaluate(params)
            if time_ms < best_time:
                best_time = time_ms
                best_params = params.copy()

        return best_params, best_time


class RandomSearch(SearchStrategy):
    """Random search strategy with optional seed."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def search(
        self,
        space: SearchSpace,
        evaluate: Callable[[dict], float],
        max_trials: int,
    ) -> tuple[dict, float]:
        best_params = {}
        best_time = float("inf")

        samples = space.sample(max_trials, self.rng)
        for params in samples:
            time_ms = evaluate(params)
            if time_ms < best_time:
                best_time = time_ms
                best_params = params.copy()

        return best_params, best_time


class TuningCache:
    """Persistent cache for tuning results."""

    def __init__(self, cache_path: str | Path = ".zenith_tuning_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache: dict[str, TuningResult] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    data = json.load(f)
                self.cache = {k: TuningResult.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, KeyError):
                self.cache = {}

    def save(self) -> None:
        """Save cache to file."""
        data = {k: v.to_dict() for k, v in self.cache.items()}
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, config: TuningConfig) -> TuningResult | None:
        """Get cached result for config."""
        return self.cache.get(config.config_hash())

    def put(self, config: TuningConfig, result: TuningResult) -> None:
        """Cache a result."""
        self.cache[config.config_hash()] = result
        self.save()

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()


class KernelAutotuner:
    """
    Main auto-tuning interface for kernel optimization.

    Usage:
        tuner = KernelAutotuner()

        # Define how to run the kernel with given params
        def run_matmul(params):
            # Execute kernel with params and return time
            return execute_kernel(params)

        # Find optimal parameters
        config = TuningConfig("matmul", [(256, 256), (256, 256)])
        best_params, best_time = tuner.tune(
            config,
            MATMUL_SEARCH_SPACE,
            run_matmul,
            max_trials=50,
        )
    """

    def __init__(
        self,
        strategy: SearchStrategy | None = None,
        cache_path: str | Path = ".zenith_tuning_cache.json",
    ):
        self.strategy = strategy or RandomSearch()
        self.cache = TuningCache(cache_path)
        self.history: list[TuningResult] = []

    def tune(
        self,
        config: TuningConfig,
        space: SearchSpace,
        evaluate: Callable[[dict], float],
        max_trials: int = 100,
        warmup: int = 3,
        repetitions: int = 10,
        force_retune: bool = False,
    ) -> tuple[dict, float]:
        """
        Tune kernel parameters for given configuration.

        Args:
            config: Tuning configuration
            space: Search space for parameters
            evaluate: Function that executes kernel with params
            max_trials: Maximum tuning trials
            warmup: Warmup iterations before timing
            repetitions: Repetitions for timing
            force_retune: Force retuning even if cached

        Returns:
            (best_params, best_time_ms)
        """
        # Check cache first
        if not force_retune:
            cached = self.cache.get(config)
            if cached is not None:
                return cached.params, cached.mean_time_ms

        # Create timed evaluator
        def timed_evaluate(params: dict) -> float:
            # Warmup
            for _ in range(warmup):
                evaluate(params)

            # Timed runs
            times = []
            for _ in range(repetitions):
                start = time.perf_counter()
                evaluate(params)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            return float(np.mean(times))

        # Run search
        best_params, best_time = self.strategy.search(space, timed_evaluate, max_trials)

        # Calculate statistics
        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            evaluate(best_params)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        result = TuningResult(
            config_hash=config.config_hash(),
            params=best_params,
            mean_time_ms=float(np.mean(times)),
            std_time_ms=float(np.std(times)),
            timestamp=time.time(),
        )

        # Cache and record
        self.cache.put(config, result)
        self.history.append(result)

        return best_params, result.mean_time_ms

    def get_cached(self, config: TuningConfig) -> TuningResult | None:
        """Get cached tuning result for a configuration."""
        return self.cache.get(config)

    def clear_cache(self) -> None:
        """Clear the tuning cache."""
        self.cache.clear()

    def export_history(self, filepath: str | Path) -> None:
        """Export tuning history to file."""
        data = [r.to_dict() for r in self.history]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def get_search_space(op_name: str) -> SearchSpace:
    """Get predefined search space for an operation."""
    spaces = {
        "matmul": MATMUL_SEARCH_SPACE,
        "conv2d": CONV2D_SEARCH_SPACE,
        "relu": RELU_SEARCH_SPACE,
    }
    if op_name in spaces:
        return spaces[op_name]

    # Default minimal search space
    return SearchSpace(op_name).define("unroll_factor", [1, 2, 4])


def autotune_matmul(
    M: int,
    K: int,
    N: int,
    evaluate: Callable[[dict], Any],
    max_trials: int = 50,
) -> dict:
    """
    Convenience function to autotune matrix multiplication.

    Args:
        M, K, N: Matrix dimensions
        evaluate: Kernel execution function
        max_trials: Maximum trials

    Returns:
        Best parameters dictionary
    """
    tuner = KernelAutotuner()
    config = TuningConfig(
        op_name="matmul",
        input_shapes=[(M, K), (K, N)],
    )
    best_params, _ = tuner.tune(
        config,
        MATMUL_SEARCH_SPACE,
        evaluate,
        max_trials=max_trials,
    )
    return best_params
