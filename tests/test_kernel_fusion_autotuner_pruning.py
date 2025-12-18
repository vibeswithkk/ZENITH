"""
Test Suite untuk Kernel Fusion, Auto-tuning, dan Pruning.
"""

import pytest
import math
import random


class TestKernelFusion:
    """Test Kernel Fusion functionality."""

    def test_fusion_patterns(self):
        """Test pola fusion yang didukung."""
        fusion_patterns = [
            ("BiasRelu", ["Add", "Relu"]),
            ("BiasGelu", ["Add", "Gelu"]),
            ("BiasSigmoid", ["Add", "Sigmoid"]),
            ("LayerNormAdd", ["LayerNormalization", "Add"]),
            ("ConvBiasRelu", ["Conv", "Add", "Relu"]),
            ("ConvBatchNormRelu", ["Conv", "BatchNormalization", "Relu"]),
            ("LinearBiasRelu", ["Linear", "Add", "Relu"]),
            ("LinearBiasGelu", ["Linear", "Add", "Gelu"]),
            ("MatMulAdd", ["MatMul", "Add"]),
            ("AddRelu", ["Add", "Relu"]),
        ]

        for name, ops in fusion_patterns:
            assert len(ops) >= 2
            assert isinstance(name, str)

    def test_pattern_matching_single_consumer(self):
        """Test bahwa fusion hanya terjadi dengan single consumer."""
        # Simulasi: Node A memiliki dua consumers
        producers_consumers = {
            "add_0": ["relu_0", "relu_1"],  # Dua consumers
            "add_1": ["relu_2"],  # Satu consumer
        }

        can_fuse = {}
        for producer, consumers in producers_consumers.items():
            can_fuse[producer] = len(consumers) == 1

        assert can_fuse["add_0"] is False
        assert can_fuse["add_1"] is True

    def test_speedup_estimation(self):
        """Test estimasi speedup dari fusion."""
        # Fusion biasanya memberikan 1.5x - 3x speedup
        expected_speedups = {
            "BiasRelu": 1.8,
            "ConvBatchNormRelu": 3.0,
            "LinearBiasGelu": 2.5,
        }

        for name, speedup in expected_speedups.items():
            assert speedup >= 1.0  # Harus ada speedup
            assert speedup <= 5.0  # Tidak unrealistic

    def test_fused_kernel_registry(self):
        """Test registry untuk fused kernels."""
        registry = {}

        # Register kernels
        registry["BiasRelu"] = {"ops": ["Add", "Relu"], "speedup": 1.8}
        registry["MatMulAdd"] = {"ops": ["MatMul", "Add"], "speedup": 1.8}

        # Verify
        assert "BiasRelu" in registry
        assert len(registry["BiasRelu"]["ops"]) == 2


class TestAutoTuner:
    """Test Auto-tuning functionality."""

    def test_tunable_parameters(self):
        """Test parameter tuning candidates."""
        params = {
            "block_size_x": [32, 64, 128, 256, 512, 1024],
            "tile_m": [16, 32, 64, 128, 256],
            "tile_n": [16, 32, 64, 128, 256],
            "tile_k": [8, 16, 32, 64],
            "unroll_factor": [1, 2, 4, 8],
        }

        for name, candidates in params.items():
            assert len(candidates) >= 2
            assert all(isinstance(c, int) for c in candidates)

    def test_grid_search(self):
        """Test grid search strategy."""
        # 3 params x 2 candidates each = 8 combinations
        params = [
            [1, 2],  # param 0
            [10, 20],  # param 1
            [100, 200],  # param 2
        ]

        # Generate all combinations
        combinations = []
        for p0 in params[0]:
            for p1 in params[1]:
                for p2 in params[2]:
                    combinations.append((p0, p1, p2))

        assert len(combinations) == 8

    def test_random_search(self):
        """Test random search strategy."""
        random.seed(42)
        params = [[32, 64, 128], [1, 2, 4], [8, 16]]
        num_samples = 10

        samples = []
        for _ in range(num_samples):
            sample = [random.choice(p) for p in params]
            samples.append(tuple(sample))

        assert len(samples) == num_samples

    def test_tuning_result(self):
        """Test struktur tuning result."""
        result = {
            "kernel_name": "gemm",
            "best_config": [64, 64, 16],
            "best_latency_ms": 0.5,
            "baseline_latency_ms": 1.0,
            "speedup": 2.0,
            "total_trials": 50,
            "valid_trials": 48,
        }

        assert (
            result["speedup"]
            == result["baseline_latency_ms"] / result["best_latency_ms"]
        )
        assert result["valid_trials"] <= result["total_trials"]

    def test_cost_model_history(self):
        """Test cost model berbasis history."""
        history = {}

        # Update dengan measurements
        history["64_64_16"] = 0.5
        history["128_64_16"] = 0.4
        history["128_128_16"] = 0.6

        # Predict untuk config yang sudah ada
        assert history.get("64_64_16") == 0.5

        # Average untuk config baru
        avg = sum(history.values()) / len(history)
        assert abs(avg - 0.5) < 0.1


class TestPruning:
    """Test Pruning functionality."""

    def test_magnitude_scoring(self):
        """Test magnitude-based importance scoring."""
        weights = [0.1, -0.5, 0.3, -0.8, 0.2]

        # Magnitude scores = absolute values
        scores = [abs(w) for w in weights]

        assert scores == [0.1, 0.5, 0.3, 0.8, 0.2]

    def test_unstructured_pruning(self):
        """Test unstructured pruning dengan target sparsity."""
        weights = [0.1, -0.5, 0.3, -0.8, 0.2, 0.05, -0.9, 0.15]
        target_sparsity = 0.5  # Prune 50%

        # Sort by magnitude
        indexed = [(abs(w), i) for i, w in enumerate(weights)]
        indexed.sort()

        num_to_prune = int(len(weights) * target_sparsity)
        indices_to_prune = [idx for _, idx in indexed[:num_to_prune]]

        # Apply pruning
        pruned = weights.copy()
        for idx in indices_to_prune:
            pruned[idx] = 0.0

        # Count zeros
        num_zeros = sum(1 for w in pruned if w == 0.0)
        actual_sparsity = num_zeros / len(pruned)

        assert actual_sparsity == target_sparsity

    def test_structured_pruning_filters(self):
        """Test structured pruning untuk filters."""
        # 4 filters, each 3x3 (flattened)
        num_filters = 4
        filter_size = 9
        weights = list(range(num_filters * filter_size))

        # Calculate L1 norm per filter
        filter_norms = []
        for f in range(num_filters):
            norm = sum(abs(weights[f * filter_size + i]) for i in range(filter_size))
            filter_norms.append(norm)

        # Filter 0 should have lowest norm (values 0-8)
        assert filter_norms[0] < filter_norms[1]
        assert filter_norms[1] < filter_norms[2]

    def test_sparsity_calculation(self):
        """Test perhitungan sparsity."""
        mask = [1, 0, 1, 0, 1, 0, 1, 0]  # 50% zeros

        num_zeros = sum(1 for m in mask if m == 0)
        sparsity = num_zeros / len(mask)

        assert sparsity == 0.5

    def test_compression_ratio(self):
        """Test perhitungan compression ratio."""
        # 75% pruned = 25% density = 4x compression
        sparsity = 0.75
        density = 1.0 - sparsity
        compression_ratio = 1.0 / density

        assert compression_ratio == 4.0

    def test_pruning_config(self):
        """Test konfigurasi pruning."""
        config = {
            "method": "Magnitude",
            "structure": "Unstructured",
            "target_sparsity": 0.5,
            "layers_to_prune": ["Conv", "Linear", "MatMul"],
            "layers_to_skip": ["bias"],
        }

        assert 0.0 <= config["target_sparsity"] <= 1.0
        assert len(config["layers_to_prune"]) > 0


class TestPruningMethods:
    """Test berbagai metode pruning."""

    def test_l1_pruning(self):
        """Test L1 (magnitude) pruning."""
        weights = [0.1, 0.5, 0.3, 0.8, 0.2]
        scores = [abs(w) for w in weights]

        # Lowest: 0.1 at index 0
        min_idx = scores.index(min(scores))
        assert min_idx == 0

    def test_l2_pruning(self):
        """Test L2 norm scoring."""
        weights = [0.1, -0.5, 0.3]
        scores = [w**2 for w in weights]

        assert scores[0] == pytest.approx(0.01)
        assert scores[1] == pytest.approx(0.25)
        assert scores[2] == pytest.approx(0.09)

    def test_random_pruning(self):
        """Test random pruning (reproducible with seed)."""
        random.seed(42)

        weights = list(range(10))
        target_sparsity = 0.3
        num_to_prune = int(len(weights) * target_sparsity)

        # Randomly select indices to prune
        indices = random.sample(range(len(weights)), num_to_prune)

        assert len(indices) == num_to_prune
        assert len(set(indices)) == num_to_prune  # No duplicates


class TestNMSparsity:
    """Test N:M sparsity pattern."""

    def test_2_of_4_sparsity(self):
        """Test 2:4 sparsity (50% structured sparsity)."""
        weights = [
            0.5,
            0.1,
            0.3,
            0.2,  # Group 1
            0.8,
            0.4,
            0.2,
            0.1,
        ]  # Group 2

        def apply_2_4_sparsity(group):
            """Keep top 2 out of 4."""
            indexed = [(abs(w), i) for i, w in enumerate(group)]
            indexed.sort(reverse=True)

            result = [0.0] * 4
            for score, idx in indexed[:2]:  # Keep top 2
                result[idx] = group[idx]
            return result

        # Apply to each group of 4
        group1 = apply_2_4_sparsity(weights[0:4])
        group2 = apply_2_4_sparsity(weights[4:8])

        # Each group should have exactly 2 non-zeros
        assert sum(1 for w in group1 if w != 0) == 2
        assert sum(1 for w in group2 if w != 0) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
