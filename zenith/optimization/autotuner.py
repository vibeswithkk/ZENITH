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
from itertools import product
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
        # MD5 used for cache key only, not for security purposes
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()[:16]


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
        """Iterate over all configurations using itertools.product."""
        if not self.params:
            yield {}
            return

        param_names = list(self.params.keys())
        param_values = list(self.params.values())

        for combination in product(*param_values):
            yield dict(zip(param_names, combination))

    def _config_at_index(self, idx: int) -> dict:
        """Get configuration at specific index without full iteration.

        This enables O(1) access to any configuration in the search space.
        """
        config = {}
        for name, values in self.params.items():
            config[name] = values[idx % len(values)]
            idx //= len(values)
        return config

    def sample(self, n: int, rng: np.random.Generator | None = None):
        """Randomly sample n configurations."""
        if rng is None:
            rng = np.random.default_rng()

        space_size = self.size()
        if space_size == 0:
            return []

        n = min(n, space_size)

        # For small spaces or when sampling most of it, materialize
        if space_size <= n * 2:
            all_configs = list(self.iterate())
            indices = rng.choice(len(all_configs), size=n, replace=False)
            return [all_configs[i] for i in indices]

        # For large spaces, use indexed sampling
        indices = rng.choice(space_size, size=n, replace=False)
        return [self._config_at_index(i) for i in indices]


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


class SimulatedAnnealingSearch(SearchStrategy):
    """
    Simulated Annealing search strategy.

    Uses probabilistic acceptance of worse solutions to escape local minima,
    with temperature gradually decreasing over iterations.

    Reference: Kirkpatrick et al. "Optimization by Simulated Annealing"
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.01,
        seed: int | None = None,
    ):
        """
        Initialize Simulated Annealing search.

        Args:
            initial_temp: Starting temperature
            cooling_rate: Multiplicative cooling factor (0 < rate < 1)
            min_temp: Minimum temperature (stopping condition)
            seed: Random seed for reproducibility
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.rng = np.random.default_rng(seed)

    def search(
        self,
        space: SearchSpace,
        evaluate: Callable[[dict], float],
        max_trials: int,
    ) -> tuple[dict, float]:
        if space.size() == 0:
            return {}, float("inf")

        # Initialize with random configuration
        all_configs = list(space.iterate())
        current_params = self.rng.choice(all_configs)
        current_cost = evaluate(current_params)

        best_params = current_params.copy()
        best_cost = current_cost

        temperature = self.initial_temp
        trial = 0

        while trial < max_trials and temperature > self.min_temp:
            # Generate neighbor by modifying one parameter
            neighbor = self._get_neighbor(current_params, space)
            neighbor_cost = evaluate(neighbor)

            # Accept or reject based on Metropolis criterion
            delta = neighbor_cost - current_cost
            if delta < 0:
                # Better solution - always accept
                current_params = neighbor
                current_cost = neighbor_cost
            else:
                # Worse solution - accept with probability
                acceptance_prob = np.exp(-delta / (temperature * best_cost + 1e-10))
                if self.rng.random() < acceptance_prob:
                    current_params = neighbor
                    current_cost = neighbor_cost

            # Update best if improved
            if current_cost < best_cost:
                best_params = current_params.copy()
                best_cost = current_cost

            # Cool down
            temperature *= self.cooling_rate
            trial += 1

        return best_params, best_cost

    def _get_neighbor(self, params: dict, space: SearchSpace) -> dict:
        """Generate a neighboring configuration by modifying one parameter."""
        neighbor = params.copy()
        param_names = list(space.params.keys())

        # Pick a random parameter to modify
        param_to_modify = self.rng.choice(param_names)
        possible_values = space.params[param_to_modify]

        # Pick a new value (different from current if possible)
        current_val = params[param_to_modify]
        if len(possible_values) > 1:
            other_values = [v for v in possible_values if v != current_val]
            neighbor[param_to_modify] = self.rng.choice(other_values)
        else:
            neighbor[param_to_modify] = possible_values[0]

        return neighbor


class GeneticAlgorithmSearch(SearchStrategy):
    """
    Genetic Algorithm search strategy.

    Uses evolutionary principles (selection, crossover, mutation) to
    evolve a population of configurations towards optimal solutions.

    Reference: Mitchell, Melanie. "An Introduction to Genetic Algorithms"
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        seed: int | None = None,
    ):
        """
        Initialize Genetic Algorithm search.

        Args:
            population_size: Number of individuals in each generation
            elite_ratio: Fraction of best individuals to keep unchanged
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of crossover between parents
            seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)

    def search(
        self,
        space: SearchSpace,
        evaluate: Callable[[dict], float],
        max_trials: int,
    ) -> tuple[dict, float]:
        if space.size() == 0:
            return {}, float("inf")

        param_names = list(space.params.keys())

        # Initialize population
        all_configs = list(space.iterate())
        pop_size = min(self.population_size, len(all_configs))
        population = list(self.rng.choice(all_configs, size=pop_size, replace=False))

        # Evaluate initial population
        fitness = [evaluate(ind) for ind in population]
        trials_used = pop_size

        best_idx = int(np.argmin(fitness))
        best_params = population[best_idx].copy()
        best_cost = fitness[best_idx]

        num_elite = max(1, int(pop_size * self.elite_ratio))
        generations = (max_trials - pop_size) // pop_size

        for _ in range(generations):
            if trials_used >= max_trials:
                break

            # Selection: rank-based
            sorted_indices = np.argsort(fitness)
            selected = [population[i] for i in sorted_indices[: pop_size // 2]]

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            for i in range(num_elite):
                new_population.append(population[sorted_indices[i]].copy())

            # Crossover and mutation
            while len(new_population) < pop_size:
                # Select parents
                parent1 = selected[self.rng.integers(len(selected))]
                parent2 = selected[self.rng.integers(len(selected))]

                # Crossover
                if self.rng.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, param_names)
                else:
                    child = parent1.copy()

                # Mutation
                child = self._mutate(child, space)

                new_population.append(child)

            population = new_population[:pop_size]

            # Evaluate new population (skip elites already evaluated)
            fitness = []
            for i, ind in enumerate(population):
                if i < num_elite:
                    # Re-evaluate elites (they may have been modified)
                    pass
                cost = evaluate(ind)
                fitness.append(cost)
                trials_used += 1

                if cost < best_cost:
                    best_params = ind.copy()
                    best_cost = cost

        return best_params, best_cost

    def _crossover(self, parent1: dict, parent2: dict, param_names: list[str]) -> dict:
        """Uniform crossover between two parents."""
        child = {}
        for param in param_names:
            if self.rng.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child

    def _mutate(self, individual: dict, space: SearchSpace) -> dict:
        """Mutate individual by randomly changing parameters."""
        mutated = individual.copy()
        for param_name, values in space.params.items():
            if self.rng.random() < self.mutation_rate:
                mutated[param_name] = self.rng.choice(values)
        return mutated


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
