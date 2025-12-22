# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Graph Executor - Executes compiled models using Zenith kernels.

Inspired by:
- TensorRT: Engine execution with execution context
- ONNX Runtime: Session.run() with feeds and fetches
- TVM: Module execution with packed function calls

This module provides the actual execution of compiled models.
"""

from typing import Any, Optional, Union
from dataclasses import dataclass, field
import time
import numpy as np

from .context import ExecutionContext
from .dispatcher import KernelDispatcher
from .kernel_registry import Precision


@dataclass
class ExecutionPlan:
    """
    Execution plan for a compiled model.

    Contains:
    - Ordered list of nodes to execute
    - Memory allocation plan
    - Pre-selected kernels
    """

    nodes: list[Any]  # Ordered nodes
    input_names: list[str]
    output_names: list[str]
    node_weights: dict[str, dict[str, Any]] = field(default_factory=dict)
    memory_plan: dict = field(default_factory=dict)
    total_ops: int = 0


@dataclass
class ExecutionStats:
    """Statistics from model execution."""

    total_time_ms: float
    node_times_ms: dict[str, float]
    memory_mb: float
    num_kernel_calls: int


class GraphExecutor:
    """
    Executes a compiled model using Zenith kernels.

    This is the runtime execution engine that processes the execution plan
    and dispatches operations to the appropriate kernels.

    Example:
        executor = GraphExecutor(execution_plan, precision=Precision.FP16)

        inputs = {"input": input_tensor}
        outputs = executor.run(inputs)
    """

    def __init__(
        self,
        execution_plan: ExecutionPlan,
        precision: Precision = Precision.FP32,
        device: str = "cuda",
        enable_profiling: bool = False,
    ):
        """
        Initialize executor.

        Args:
            execution_plan: Plan with ordered nodes and kernel assignments
            precision: Precision for kernel selection
            device: Target device
            enable_profiling: Whether to profile execution
        """
        self.plan = execution_plan
        self.precision = precision
        self.device = device
        self.enable_profiling = enable_profiling

        # Create dispatcher
        self.dispatcher = KernelDispatcher(precision=precision, device=device)

        # Execution stats
        self._last_stats: Optional[ExecutionStats] = None
        self._execution_count = 0

    def run(self, inputs: dict[str, Any], return_numpy: bool = True) -> dict[str, Any]:
        """
        Execute the model with given inputs.

        Args:
            inputs: Dictionary of input name -> tensor
            return_numpy: Whether to convert outputs to numpy arrays

        Returns:
            Dictionary of output name -> tensor
        """
        start_time = time.perf_counter()
        node_times = {} if self.enable_profiling else None

        # Create execution context
        context = ExecutionContext(
            input_names=self.plan.input_names,
            output_names=self.plan.output_names,
            device=self.device,
        )

        # Set inputs
        for name, tensor in inputs.items():
            # Convert to GPU if needed
            gpu_tensor = self._to_device(tensor)
            context.set_input(name, gpu_tensor)

        # Execute nodes in order
        for node in self.plan.nodes:
            node_start = time.perf_counter() if self.enable_profiling else 0

            # Get weights for this node
            node_name = node.name if hasattr(node, "name") else str(node)
            weights = self.plan.node_weights.get(node_name, {})

            # Dispatch to kernel
            self.dispatcher.dispatch(node, context, weights)

            if self.enable_profiling:
                node_times[node_name] = (time.perf_counter() - node_start) * 1000

        # Collect outputs
        outputs = {}
        for name in self.plan.output_names:
            if context.has_tensor(name):
                output = context.get_tensor(name)
                if return_numpy:
                    output = self._to_numpy(output)
                outputs[name] = output

        # If no named outputs, return first available
        if not outputs and context._tensors:
            last_tensor_name = list(context._tensors.keys())[-1]
            output = context.get_tensor(last_tensor_name)
            if return_numpy:
                output = self._to_numpy(output)
            outputs[last_tensor_name] = output

        # Record stats
        total_time = (time.perf_counter() - start_time) * 1000
        self._last_stats = ExecutionStats(
            total_time_ms=total_time,
            node_times_ms=node_times or {},
            memory_mb=context.memory_usage_mb,
            num_kernel_calls=self.dispatcher._dispatch_count,
        )

        self._execution_count += 1

        return outputs

    def run_single(self, inputs: dict[str, Any], output_name: str = None) -> Any:
        """
        Execute and return single output tensor.

        Args:
            inputs: Input dictionary
            output_name: Specific output to return (default: first)

        Returns:
            Single output tensor
        """
        outputs = self.run(inputs)

        if output_name and output_name in outputs:
            return outputs[output_name]

        if outputs:
            return next(iter(outputs.values()))

        return None

    def _to_device(self, tensor: Any) -> Any:
        """Convert tensor to target device."""
        if self.device == "cuda":
            try:
                from zenith._zenith_core import cuda

                # Already GPU tensor
                if hasattr(tensor, "to_numpy"):
                    return tensor

                # PyTorch tensor
                if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
                    np_array = tensor.detach().cpu().numpy()
                    return cuda.to_gpu(np.ascontiguousarray(np_array))

                # Numpy array
                if isinstance(tensor, np.ndarray):
                    return cuda.to_gpu(np.ascontiguousarray(tensor))

                # Try to convert
                return cuda.to_gpu(np.ascontiguousarray(np.asarray(tensor)))

            except ImportError:
                pass

        # CPU fallback - handle PyTorch tensors properly
        if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
            # PyTorch tensor (possibly on CUDA)
            return tensor.detach().cpu().numpy()
        if hasattr(tensor, "numpy"):
            # Try direct numpy, handle CUDA tensor
            try:
                return tensor.numpy()
            except (TypeError, RuntimeError):
                # CUDA tensor - need to move to CPU first
                if hasattr(tensor, "cpu"):
                    return tensor.cpu().numpy()
        return np.asarray(tensor)

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, "to_numpy"):
            return tensor.to_numpy()
        if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
            return tensor.cpu().numpy()
        return np.asarray(tensor)

    def get_stats(self) -> Optional[ExecutionStats]:
        """Get stats from last execution."""
        return self._last_stats

    def profile(self, inputs: dict[str, Any], num_runs: int = 10) -> dict:
        """
        Profile execution over multiple runs.

        Args:
            inputs: Input dictionary
            num_runs: Number of runs for averaging

        Returns:
            Profiling results
        """
        # Warm-up
        self.run(inputs)

        times = []
        self.enable_profiling = True

        for _ in range(num_runs):
            self.run(inputs)
            if self._last_stats:
                times.append(self._last_stats.total_time_ms)

        self.enable_profiling = False

        return {
            "num_runs": num_runs,
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "throughput": 1000 / np.mean(times) if times else 0,
        }


class EagerExecutor:
    """
    Eager execution mode - execute operations immediately.

    Useful for debugging and development.
    """

    def __init__(self, precision: Precision = Precision.FP32, device: str = "cuda"):
        self.precision = precision
        self.device = device
        self.dispatcher = KernelDispatcher(precision=precision, device=device)

    def execute_op(
        self,
        op_type: str,
        inputs: list[Any],
        weights: dict[str, Any] = None,
        attributes: dict = None,
    ) -> Any:
        """
        Execute a single operation.

        Args:
            op_type: Operation type
            inputs: Input tensors
            weights: Weight tensors
            attributes: Operation attributes

        Returns:
            Output tensor
        """

        # Create minimal node-like object
        class OpNode:
            def __init__(self, op_type, inputs, outputs, attrs):
                self.op_type = op_type
                self.inputs = inputs
                self.outputs = outputs
                self.attributes = attrs

        # Create context
        context = ExecutionContext(device=self.device)

        # Set inputs
        input_names = [f"input_{i}" for i in range(len(inputs))]
        for name, tensor in zip(input_names, inputs):
            context.set_tensor(name, tensor)

        # Create node
        node = OpNode(
            op_type=op_type,
            inputs=input_names,
            outputs=["output"],
            attrs=attributes or {},
        )

        # Dispatch
        return self.dispatcher.dispatch(node, context, weights)
