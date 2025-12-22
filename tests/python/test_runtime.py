# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for Zenith Runtime Module

Tests:
1. KernelRegistry initialization
2. ExecutionContext tensor handling
3. KernelDispatcher operation routing
4. GraphExecutor execution
5. ZenithEngine compilation
"""

import pytest
import numpy as np


class TestKernelRegistry:
    """Tests for KernelRegistry."""

    def test_registry_creation(self):
        """Test registry can be created."""
        from zenith.runtime.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        assert registry is not None

    def test_precision_enum(self):
        """Test Precision enum values."""
        from zenith.runtime.kernel_registry import Precision

        assert Precision.FP32.value == "fp32"
        assert Precision.FP16.value == "fp16"
        assert Precision.INT8.value == "int8"

    def test_kernel_spec_creation(self):
        """Test KernelSpec creation."""
        from zenith.runtime.kernel_registry import KernelSpec, Precision

        spec = KernelSpec(
            name="test_kernel",
            op_types=["MatMul", "Linear"],
            precision=Precision.FP32,
            kernel_fn=lambda x, y: x @ y,
            priority=10,
        )

        assert spec.name == "test_kernel"
        assert spec.supports_op("MatMul")
        assert spec.supports_op("Linear")
        assert not spec.supports_op("Conv")

    def test_registry_register_and_query(self):
        """Test registering and querying kernels."""
        from zenith.runtime.kernel_registry import KernelRegistry, KernelSpec, Precision

        registry = KernelRegistry()

        spec = KernelSpec(
            name="matmul_fp32",
            op_types=["MatMul"],
            precision=Precision.FP32,
            priority=10,
        )

        registry.register(spec)

        assert registry.is_supported("MatMul", Precision.FP32)
        assert not registry.is_supported("Conv2d", Precision.FP32)

        kernel = registry.get_kernel("MatMul", Precision.FP32)
        assert kernel is not None
        assert kernel.name == "matmul_fp32"

    def test_global_registry(self):
        """Test global registry initialization."""
        from zenith.runtime.kernel_registry import get_registry, reset_registry

        reset_registry()
        registry = get_registry()

        assert registry is not None
        # Should have registered kernels
        supported_ops = registry.list_supported_ops()
        assert len(supported_ops) > 0


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_context_creation(self):
        """Test context creation."""
        from zenith.runtime.context import ExecutionContext

        context = ExecutionContext(input_names=["x"], output_names=["y"], device="cuda")

        assert context.input_names == ["x"]
        assert context.output_names == ["y"]

    def test_set_get_tensor(self):
        """Test setting and getting tensors."""
        from zenith.runtime.context import ExecutionContext

        context = ExecutionContext()

        tensor = np.random.randn(2, 3).astype(np.float32)
        context.set_tensor("test", tensor)

        assert context.has_tensor("test")
        retrieved = context.get_tensor("test")
        np.testing.assert_array_equal(retrieved, tensor)

    def test_tensor_info(self):
        """Test tensor info tracking."""
        from zenith.runtime.context import ExecutionContext

        context = ExecutionContext()

        tensor = np.random.randn(4, 5).astype(np.float32)
        context.set_tensor("test", tensor)

        info = context.get_tensor_info("test")
        assert info is not None
        assert info.shape == (4, 5)
        assert "float32" in info.dtype

    def test_memory_tracking(self):
        """Test memory tracking."""
        from zenith.runtime.context import ExecutionContext

        context = ExecutionContext()

        tensor = np.zeros((1000, 1000), dtype=np.float32)  # ~4MB
        context.set_tensor("large", tensor)

        assert context.memory_usage_mb > 0


class TestKernelDispatcher:
    """Tests for KernelDispatcher."""

    def test_dispatcher_creation(self):
        """Test dispatcher creation."""
        from zenith.runtime.dispatcher import KernelDispatcher
        from zenith.runtime.kernel_registry import Precision

        dispatcher = KernelDispatcher(precision=Precision.FP32)
        assert dispatcher is not None

    def test_op_type_normalization(self):
        """Test operation type normalization."""
        from zenith.runtime.dispatcher import KernelDispatcher

        dispatcher = KernelDispatcher()

        assert dispatcher._normalize_op_type("MatMul") == "Linear"
        assert dispatcher._normalize_op_type("Gemm") == "Linear"
        assert dispatcher._normalize_op_type("Relu") == "ReLU"


class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_manager_creation(self):
        """Test memory manager creation."""
        from zenith.runtime.memory_manager import MemoryManager

        manager = MemoryManager(device="cuda")
        assert manager is not None

    def test_allocation(self):
        """Test memory allocation."""
        from zenith.runtime.memory_manager import MemoryManager

        manager = MemoryManager()

        success = manager.allocate("tensor1", size_bytes=1024 * 1024)
        assert success

        assert manager.get_size("tensor1") == 1024 * 1024
        assert manager.total_allocated_mb > 0

    def test_memory_plan(self):
        """Test memory planning with reuse."""
        from zenith.runtime.memory_manager import MemoryManager

        manager = MemoryManager()

        tensor_sizes = {
            "a": 1024,
            "b": 2048,
            "c": 1024,
        }

        liveness = {
            "a": (0, 2),  # Used in ops 0-2
            "b": (1, 3),  # Used in ops 1-3
            "c": (3, 5),  # Used in ops 3-5 (can reuse "a")
        }

        plan = manager.plan_memory(tensor_sizes, liveness)

        assert "allocations" in plan
        assert "total_size_bytes" in plan


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_execution_plan_creation(self):
        """Test execution plan creation."""
        from zenith.runtime.executor import ExecutionPlan

        plan = ExecutionPlan(
            nodes=[],
            input_names=["input"],
            output_names=["output"],
        )

        assert plan.input_names == ["input"]
        assert plan.output_names == ["output"]
        assert plan.total_ops == 0


class TestGraphExecutor:
    """Tests for GraphExecutor."""

    def test_executor_creation(self):
        """Test executor creation."""
        from zenith.runtime.executor import GraphExecutor, ExecutionPlan
        from zenith.runtime.kernel_registry import Precision

        plan = ExecutionPlan(
            nodes=[],
            input_names=["x"],
            output_names=["y"],
        )

        executor = GraphExecutor(
            execution_plan=plan,
            precision=Precision.FP32,
        )

        assert executor is not None


class TestZenithEngine:
    """Tests for ZenithEngine."""

    def test_engine_creation(self):
        """Test engine creation."""
        from zenith.runtime.engine import ZenithEngine

        engine = ZenithEngine(backend="cuda")
        assert engine.backend == "cuda"

    def test_compile_config(self):
        """Test CompileConfig."""
        from zenith.runtime.engine import CompileConfig
        from zenith.runtime.kernel_registry import Precision

        config = CompileConfig(
            precision="fp16",
            mode="default",
        )

        assert config.precision == "fp16"
        assert config.get_precision() == Precision.FP16

    def test_list_supported_ops(self):
        """Test listing supported operations."""
        from zenith.runtime.engine import ZenithEngine

        engine = ZenithEngine()
        ops = engine.list_supported_ops()

        assert isinstance(ops, list)
        # Should have at least some ops registered
        # (may be empty if CUDA not available)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_import_runtime(self):
        """Test importing runtime module."""
        from zenith.runtime import (
            ZenithEngine,
            CompileConfig,
            GraphExecutor,
            KernelDispatcher,
            KernelRegistry,
            ExecutionContext,
            MemoryManager,
        )

        # All should be importable
        assert ZenithEngine is not None
        assert CompileConfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
