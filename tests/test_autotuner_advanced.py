"""
Test Suite for Advanced Auto-Tuning Search Strategies.

Tests for:
- Simulated Annealing search
- Genetic Algorithm search
- Hardware profiling
- Search space constraints

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import pytest
import numpy as np


class TestSimulatedAnnealingSearch:
    """Tests for Simulated Annealing search strategy."""

    def test_initialization(self):
        """Test SA search initialization."""
        from zenith.optimization.autotuner import SimulatedAnnealingSearch

        sa = SimulatedAnnealingSearch(
            initial_temp=2.0,
            cooling_rate=0.9,
            min_temp=0.001,
            seed=42,
        )

        assert sa.initial_temp == 2.0
        assert sa.cooling_rate == 0.9
        assert sa.min_temp == 0.001

    def test_search_finds_optimum(self):
        """Test that SA finds near-optimal solution."""
        from zenith.optimization.autotuner import (
            SimulatedAnnealingSearch,
            SearchSpace,
        )

        # Create simple search space
        space = SearchSpace("test")
        space.define("x", [1, 2, 3, 4, 5])
        space.define("y", [1, 2, 3, 4, 5])

        # Objective: minimize (x - 3)^2 + (y - 4)^2
        def evaluate(params):
            return (params["x"] - 3) ** 2 + (params["y"] - 4) ** 2

        sa = SimulatedAnnealingSearch(seed=42)
        best_params, best_cost = sa.search(space, evaluate, max_trials=100)

        # Should find or approach optimum (x=3, y=4, cost=0)
        assert best_cost <= 2  # Allow some tolerance

    def test_search_deterministic_with_seed(self):
        """Test that SA is deterministic with same seed."""
        from zenith.optimization.autotuner import (
            SimulatedAnnealingSearch,
            SearchSpace,
        )

        space = SearchSpace("test")
        space.define("x", [1, 2, 3, 4, 5])

        def evaluate(params):
            return params["x"] ** 2

        # Run twice with same seed
        sa1 = SimulatedAnnealingSearch(seed=123)
        result1 = sa1.search(space, evaluate, max_trials=50)

        sa2 = SimulatedAnnealingSearch(seed=123)
        result2 = sa2.search(space, evaluate, max_trials=50)

        assert result1 == result2

    def test_search_empty_space(self):
        """Test behavior with empty search space."""
        from zenith.optimization.autotuner import (
            SimulatedAnnealingSearch,
            SearchSpace,
        )

        space = SearchSpace("empty")
        sa = SimulatedAnnealingSearch()

        best_params, best_cost = sa.search(space, lambda p: 0, max_trials=10)

        assert best_params == {}
        assert best_cost == float("inf")


class TestGeneticAlgorithmSearch:
    """Tests for Genetic Algorithm search strategy."""

    def test_initialization(self):
        """Test GA search initialization."""
        from zenith.optimization.autotuner import GeneticAlgorithmSearch

        ga = GeneticAlgorithmSearch(
            population_size=30,
            elite_ratio=0.1,
            mutation_rate=0.2,
            crossover_rate=0.7,
            seed=42,
        )

        assert ga.population_size == 30
        assert ga.elite_ratio == 0.1
        assert ga.mutation_rate == 0.2
        assert ga.crossover_rate == 0.7

    def test_search_finds_optimum(self):
        """Test that GA finds near-optimal solution."""
        from zenith.optimization.autotuner import (
            GeneticAlgorithmSearch,
            SearchSpace,
        )

        # Create search space
        space = SearchSpace("test")
        space.define("a", [0, 1, 2, 3, 4])
        space.define("b", [0, 1, 2, 3, 4])
        space.define("c", [0, 1, 2, 3, 4])

        # Objective: minimize sum of squares of deviation from (2, 2, 2)
        def evaluate(params):
            return (
                (params["a"] - 2) ** 2 + (params["b"] - 2) ** 2 + (params["c"] - 2) ** 2
            )

        ga = GeneticAlgorithmSearch(population_size=10, seed=42)
        best_params, best_cost = ga.search(space, evaluate, max_trials=100)

        # Should find optimum (a=2, b=2, c=2, cost=0)
        assert best_cost <= 3  # Allow some tolerance

    def test_search_deterministic_with_seed(self):
        """Test that GA is deterministic with same seed."""
        from zenith.optimization.autotuner import (
            GeneticAlgorithmSearch,
            SearchSpace,
        )

        space = SearchSpace("test")
        space.define("x", [1, 2, 3])
        space.define("y", [1, 2, 3])

        def evaluate(params):
            return params["x"] + params["y"]

        # Run twice with same seed
        ga1 = GeneticAlgorithmSearch(population_size=5, seed=456)
        result1 = ga1.search(space, evaluate, max_trials=50)

        ga2 = GeneticAlgorithmSearch(population_size=5, seed=456)
        result2 = ga2.search(space, evaluate, max_trials=50)

        assert result1 == result2

    def test_crossover(self):
        """Test crossover operation."""
        from zenith.optimization.autotuner import GeneticAlgorithmSearch

        ga = GeneticAlgorithmSearch(seed=42)
        parent1 = {"a": 1, "b": 2, "c": 3}
        parent2 = {"a": 4, "b": 5, "c": 6}

        child = ga._crossover(parent1, parent2, ["a", "b", "c"])

        # Child should have values from either parent
        for key in ["a", "b", "c"]:
            assert child[key] in [parent1[key], parent2[key]]

    def test_mutation(self):
        """Test mutation operation."""
        from zenith.optimization.autotuner import (
            GeneticAlgorithmSearch,
            SearchSpace,
        )

        space = SearchSpace("test")
        space.define("x", [1, 2, 3, 4, 5])

        # High mutation rate to ensure mutation happens
        ga = GeneticAlgorithmSearch(mutation_rate=1.0, seed=42)
        individual = {"x": 3}

        mutated = ga._mutate(individual, space)

        # Mutated value should be from the search space
        assert mutated["x"] in [1, 2, 3, 4, 5]


class TestHardwareProfile:
    """Tests for hardware profiling module."""

    def test_detect_cpu_info(self):
        """Test CPU info detection."""
        from zenith.optimization.hardware_profile import detect_cpu_info

        cpu_info = detect_cpu_info()

        assert cpu_info is not None
        assert cpu_info.num_cores >= 1

    def test_detect_hardware(self):
        """Test full hardware detection."""
        from zenith.optimization.hardware_profile import detect_hardware

        hw_info = detect_hardware()

        assert hw_info is not None
        assert hw_info.os_name is not None
        assert hw_info.cpu is not None

    def test_get_hardware_info_caching(self):
        """Test that hardware info is cached."""
        from zenith.optimization.hardware_profile import (
            get_hardware_info,
            refresh_hardware_info,
        )

        info1 = get_hardware_info()
        info2 = get_hardware_info()

        # Should return same cached instance
        assert info1 is info2

        # Refresh should create new instance
        info3 = refresh_hardware_info()
        assert info3 is not info1

    def test_device_type_enum(self):
        """Test DeviceType enum values."""
        from zenith.optimization.hardware_profile import DeviceType

        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.ROCM.value == "rocm"

    def test_cuda_device_info_properties(self):
        """Test CUDADeviceInfo computed properties."""
        from zenith.optimization.hardware_profile import CUDADeviceInfo

        # Volta-class GPU
        device = CUDADeviceInfo(
            compute_capability=(7, 0),
            multiprocessor_count=80,
            clock_rate_mhz=1500,
        )

        assert device.supports_tensor_cores is True
        assert device.supports_fp16 is True
        assert device.supports_bf16 is False
        assert device.supports_int8_tensor_cores is False

        # Ampere-class GPU
        ampere = CUDADeviceInfo(compute_capability=(8, 0))
        assert ampere.supports_bf16 is True
        assert ampere.supports_tensor_cores is True

    def test_hardware_constraints(self):
        """Test hardware constraints generation."""
        from zenith.optimization.hardware_profile import (
            get_constraints_for_device,
            DeviceType,
        )

        cuda_constraints = get_constraints_for_device(DeviceType.CUDA)
        assert cuda_constraints.max_threads_per_block == 1024

        cpu_constraints = get_constraints_for_device(DeviceType.CPU)
        assert cpu_constraints.max_threads_per_block == 1

    def test_simd_features(self):
        """Test SIMD feature detection."""
        from zenith.optimization.hardware_profile import SIMDFeatures

        # AVX512 capable
        simd = SIMDFeatures(sse=True, avx=True, avx512f=True)
        assert simd.max_vector_width == 512

        # AVX only
        simd_avx = SIMDFeatures(sse=True, avx=True)
        assert simd_avx.max_vector_width == 256

        # SSE only
        simd_sse = SIMDFeatures(sse=True)
        assert simd_sse.max_vector_width == 128


class TestSearchStrategiesComparison:
    """Compare different search strategies."""

    def test_all_strategies_find_solution(self):
        """Test that all strategies can find solutions."""
        from zenith.optimization.autotuner import (
            GridSearch,
            RandomSearch,
            SimulatedAnnealingSearch,
            GeneticAlgorithmSearch,
            SearchSpace,
        )

        space = SearchSpace("test")
        space.define("x", [1, 2, 3, 4, 5])

        def evaluate(params):
            return (params["x"] - 3) ** 2

        strategies = [
            GridSearch(),
            RandomSearch(seed=42),
            SimulatedAnnealingSearch(seed=42),
            GeneticAlgorithmSearch(population_size=5, seed=42),
        ]

        for strategy in strategies:
            best_params, best_cost = strategy.search(space, evaluate, 25)
            # All strategies should find solution with cost <= 1
            assert best_cost <= 1, f"{strategy.__class__.__name__} failed"

    def test_strategy_performance_scaling(self):
        """Test that strategies scale with larger spaces."""
        from zenith.optimization.autotuner import (
            RandomSearch,
            SimulatedAnnealingSearch,
            SearchSpace,
        )

        # Larger search space
        space = SearchSpace("large")
        for i in range(5):
            space.define(f"p{i}", list(range(10)))

        # Multi-modal objective
        def evaluate(params):
            return sum(
                min((params[f"p{i}"] - 5) ** 2, (params[f"p{i}"] - 8) ** 2)
                for i in range(5)
            )

        random = RandomSearch(seed=42)
        sa = SimulatedAnnealingSearch(seed=42)

        _, random_cost = random.search(space, evaluate, 100)
        _, sa_cost = sa.search(space, evaluate, 100)

        # Both should find reasonable solutions
        assert random_cost < 50
        assert sa_cost < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
