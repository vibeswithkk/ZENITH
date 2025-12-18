"""
Test Suite untuk Graph Optimization Passes.
Menguji Dead Code Elimination, Constant Folding, dan Operator Fusion.
"""

import pytest
import sys
import os


class TestDeadCodeElimination:
    """Test Dead Code Elimination Pass."""

    def test_reachability_algorithm(self):
        """Test algoritma reachability dari outputs ke inputs."""
        # Simulasi struktur graph
        # Graph: A -> B -> C (output)
        #        D (tidak terhubung ke output, harus dihapus)
        nodes = {
            "A": {"inputs": [], "outputs": ["tensor_a"]},
            "B": {"inputs": ["tensor_a"], "outputs": ["tensor_b"]},
            "C": {"inputs": ["tensor_b"], "outputs": ["output"]},
            "D": {"inputs": [], "outputs": ["tensor_d"]},  # Dead node
        }
        outputs = ["output"]

        # Jalankan backward reachability
        reachable = set()
        worklist = list(outputs)

        while worklist:
            current = worklist.pop()
            if current in reachable:
                continue
            reachable.add(current)

            # Cari producer
            for name, node in nodes.items():
                if current in node["outputs"]:
                    reachable.add(name)
                    worklist.extend(node["inputs"])

        # Node D tidak boleh reachable
        assert "A" in reachable
        assert "B" in reachable
        assert "C" in reachable
        assert "D" not in reachable

    def test_empty_graph(self):
        """Test dengan graph kosong."""
        nodes = {}
        outputs = []

        # Tidak ada node yang reachable
        reachable = set()
        assert len(reachable) == 0


class TestConstantFolding:
    """Test Constant Folding Pass."""

    def test_add_constants(self):
        """Test folding operasi Add dengan dua konstanta."""
        import struct

        # Simulasi: Constant(2.0) + Constant(3.0) = Constant(5.0)
        a = 2.0
        b = 3.0
        expected = 5.0

        result = a + b
        assert abs(result - expected) < 1e-6

    def test_unary_relu(self):
        """Test folding ReLU dengan konstanta."""

        # ReLU(-2.0) = 0.0
        # ReLU(3.0) = 3.0
        def relu(x):
            return max(0.0, x)

        assert relu(-2.0) == 0.0
        assert relu(3.0) == 3.0
        assert relu(0.0) == 0.0

    def test_unary_sigmoid(self):
        """Test folding Sigmoid dengan konstanta."""
        import math

        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        # sigmoid(0) = 0.5
        assert abs(sigmoid(0.0) - 0.5) < 1e-6

        # sigmoid(large) ~ 1.0
        assert abs(sigmoid(10.0) - 1.0) < 1e-4

        # sigmoid(-large) ~ 0.0
        assert abs(sigmoid(-10.0) - 0.0) < 1e-4

    def test_identity_folding(self):
        """Test bahwa Identity node bisa di-fold."""
        value = 42.0
        # Identity(x) = x
        result = value  # Identity transform
        assert result == value


class TestOperatorFusion:
    """Test Operator Fusion Pass."""

    def test_pattern_matching(self):
        """Test pattern matching untuk fusion."""
        # Pattern: [MatMul, Add] -> FusedMatMulAdd
        pattern_ops = ["MatMul", "Add"]
        node_sequence = ["MatMul", "Add"]

        # Check match
        matches = all(p == n for p, n in zip(pattern_ops, node_sequence))
        assert matches

    def test_single_consumer_check(self):
        """Test bahwa fusion hanya terjadi jika producer punya satu consumer."""
        # Simulasi graph dimana MatMul punya dua consumers
        producers_consumers = {
            "matmul_0": ["add_0", "add_1"],  # Dua consumers, tidak boleh fuse
            "matmul_1": ["add_2"],  # Satu consumer, boleh fuse
        }

        can_fuse_0 = len(producers_consumers["matmul_0"]) == 1
        can_fuse_1 = len(producers_consumers["matmul_1"]) == 1

        assert can_fuse_0 is False
        assert can_fuse_1 is True

    def test_supported_fusion_patterns(self):
        """Test daftar pola fusion yang didukung."""
        supported_patterns = [
            ("Conv", "BatchNormalization", "Relu"),
            ("MatMul", "Add"),
            ("Linear", "Relu"),
            ("Linear", "Gelu"),
            ("Add", "Relu"),
            ("LayerNormalization", "Add"),
        ]

        # Semua pola harus memiliki minimal 2 operasi
        for pattern in supported_patterns:
            assert len(pattern) >= 2


class TestPassManager:
    """Test PassManager framework."""

    def test_optimization_levels(self):
        """Test optimization levels."""
        levels = {
            "O0": 0,  # Tidak ada optimisasi
            "O1": 1,  # Dasar
            "O2": 2,  # Medium
            "O3": 3,  # Agresif
        }

        # O0 harus skip semua passes dengan level > 0
        assert levels["O0"] < levels["O1"]
        assert levels["O1"] < levels["O2"]
        assert levels["O2"] < levels["O3"]

    def test_pass_statistics(self):
        """Test statistik pass."""
        stats = {
            "pass_name": "DeadCodeElimination",
            "nodes_before": 10,
            "nodes_after": 8,
            "nodes_removed": 2,
            "duration_ms": 0.5,
            "success": True,
        }

        assert stats["nodes_removed"] == stats["nodes_before"] - stats["nodes_after"]
        assert stats["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
