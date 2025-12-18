# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
ONNX Graph Interpreter

Full interpreter that walks ONNX computation graphs and executes
each node using Zenith's CUDA operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np

from .context import ExecutionContext
from .registry import OperatorRegistry


class ONNXInterpreter:
    """
    Executes ONNX graphs using Zenith CUDA operations.

    This is a full interpreter that walks the ONNX graph and
    executes each node using GPU-accelerated kernels.

    The interpreter follows the ONNX Runtime architecture:
    1. Load graph and constants
    2. Topologically sort nodes for execution order
    3. Execute each node in order, using registered kernels
    4. Return output tensors

    Example:
        from zenith.execution import ONNXInterpreter
        from zenith.adapters import PyTorchAdapter

        # Convert model to GraphIR
        adapter = PyTorchAdapter()
        graph_ir = adapter.from_model(model, sample_input)

        # Create interpreter
        interpreter = ONNXInterpreter(graph_ir, device="cuda")

        # Execute
        outputs = interpreter(input=input_data)
    """

    def __init__(
        self,
        graph_ir: Any,
        device: str = "cuda",
        strict: bool = False,
    ):
        """
        Initialize the ONNX interpreter.

        Args:
            graph_ir: GraphIR representation of the model.
            device: Target device ("cuda" or "cpu").
            strict: If True, raise error for unsupported ops.
                    If False, skip unsupported ops with warning.
        """
        self.graph_ir = graph_ir
        self.device = device
        self.strict = strict

        # Create execution context
        self.context = ExecutionContext(device)

        # Execution state
        self._execution_order: List[Any] = []
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._initialized = False
        self._unsupported_ops: List[str] = []

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize interpreter state."""
        # Import operators to populate registry
        from . import operators  # noqa: F401

        # Get input/output names
        self._input_names = [inp.name for inp in self.graph_ir.inputs]
        self._output_names = [out.name for out in self.graph_ir.outputs]

        # Load constants (weights, biases)
        self._load_constants()

        # Compute execution order
        self._compute_execution_order()

        # Check for unsupported operators
        self._check_operator_support()

        self._initialized = True

    def _load_constants(self) -> None:
        """Load model constants (weights, biases) into context."""
        for name, data in self.graph_ir.constants.items():
            # Convert bytes to numpy array if needed
            if isinstance(data, bytes):
                # Assume float32 for now, actual dtype should come from graph
                arr = np.frombuffer(data, dtype=np.float32)
            else:
                arr = np.asarray(data, dtype=np.float32)

            self.context.set_constant(name, arr)

    def _compute_execution_order(self) -> None:
        """
        Topologically sort nodes for correct execution order.

        Uses Kahn's algorithm for topological sorting based on
        data dependencies between nodes.
        """
        nodes = list(self.graph_ir.nodes)

        if not nodes:
            self._execution_order = []
            return

        # Build adjacency list and in-degree count
        # Node A -> Node B means A's output is B's input
        in_degree: Dict[str, int] = defaultdict(int)
        adjacency: Dict[str, List[Any]] = defaultdict(list)
        node_by_name: Dict[str, Any] = {}

        # Map output names to producing nodes
        output_to_node: Dict[str, Any] = {}
        for node in nodes:
            node_by_name[node.name] = node
            for output in node.outputs:
                output_to_node[output.name] = node

        # Build graph
        for node in nodes:
            in_degree[node.name] = 0

        for node in nodes:
            for inp in node.inputs:
                # Check if input comes from another node
                if inp.name in output_to_node:
                    producer = output_to_node[inp.name]
                    adjacency[producer.name].append(node)
                    in_degree[node.name] += 1

        # Kahn's algorithm
        queue = [n for n in nodes if in_degree[n.name] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)

            for dependent in adjacency[node.name]:
                in_degree[dependent.name] -= 1
                if in_degree[dependent.name] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(sorted_nodes) != len(nodes):
            # Fallback to original order if cycle detected
            sorted_nodes = nodes

        self._execution_order = sorted_nodes

    def _check_operator_support(self) -> None:
        """Check which operators are supported."""
        self._unsupported_ops = []

        for node in self._execution_order:
            if not OperatorRegistry.is_supported(node.op_type):
                self._unsupported_ops.append(node.op_type)

        if self._unsupported_ops and self.strict:
            raise NotImplementedError(
                f"Unsupported operators: {set(self._unsupported_ops)}. "
                "Set strict=False to skip these operators."
            )

    @property
    def is_fully_supported(self) -> bool:
        """Check if all operators in the graph are supported."""
        return len(self._unsupported_ops) == 0

    @property
    def unsupported_operators(self) -> List[str]:
        """Get list of unsupported operator types."""
        return list(set(self._unsupported_ops))

    def __call__(
        self,
        **inputs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Execute the graph with given inputs.

        Args:
            **inputs: Named input tensors as numpy arrays.
                      Names must match graph input names.

        Returns:
            Dictionary of output tensors.

        Example:
            outputs = interpreter(input=np.random.randn(1, 3, 224, 224))
        """
        # Validate inputs
        for name in self._input_names:
            if name not in inputs:
                raise ValueError(
                    f"Missing input '{name}'. Required inputs: {self._input_names}"
                )

        # Clear previous execution state
        self.context.clear()

        # Set inputs
        for name, value in inputs.items():
            if name in self._input_names:
                self.context.set_tensor(
                    name, np.ascontiguousarray(value, dtype=np.float32)
                )

        # Execute nodes in order
        for node in self._execution_order:
            self._execute_node(node)

        # Collect outputs
        outputs = {}
        for name in self._output_names:
            if self.context.has_tensor(name):
                outputs[name] = self.context.get_tensor_numpy(name)

        return outputs

    def _execute_node(self, node: Any) -> None:
        """
        Execute a single node using the registered kernel.

        Args:
            node: Node to execute.
        """
        op_type = node.op_type

        # Check if operator is supported
        if not OperatorRegistry.is_supported(op_type):
            if self.strict:
                raise NotImplementedError(f"Operator '{op_type}' not supported")
            else:
                # Skip unsupported operator
                return

        # Get kernel function
        kernel = OperatorRegistry.get_kernel(op_type)

        # Prepare inputs (list of tensor names)
        input_names = [inp.name for inp in node.inputs]

        # Prepare outputs (list of tensor names)
        output_names = [out.name for out in node.outputs]

        # Get attributes
        attrs = node.attrs if hasattr(node, "attrs") else {}

        # Execute kernel
        kernel(self.context, input_names, output_names, attrs)

    def execute_with_timing(
        self,
        **inputs: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Execute graph with per-node timing.

        Returns:
            Tuple of (outputs, timing_dict).
        """
        import time

        timings = {}

        # Clear and set inputs
        self.context.clear()
        for name, value in inputs.items():
            if name in self._input_names:
                self.context.set_tensor(
                    name, np.ascontiguousarray(value, dtype=np.float32)
                )

        # Execute with timing
        for node in self._execution_order:
            start = time.perf_counter()
            self._execute_node(node)
            elapsed = time.perf_counter() - start
            timings[node.name] = elapsed * 1000  # Convert to ms

        # Collect outputs
        outputs = {}
        for name in self._output_names:
            if self.context.has_tensor(name):
                outputs[name] = self.context.get_tensor_numpy(name)

        return outputs, timings

    def summary(self) -> str:
        """Get a summary of the interpreter state."""
        lines = [
            f"ONNXInterpreter Summary",
            f"  Graph: {self.graph_ir.name}",
            f"  Device: {self.device}",
            f"  Nodes: {len(self._execution_order)}",
            f"  Inputs: {self._input_names}",
            f"  Outputs: {self._output_names}",
            f"  Fully Supported: {self.is_fully_supported}",
        ]

        if self._unsupported_ops:
            lines.append(f"  Unsupported Ops: {set(self._unsupported_ops)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ONNXInterpreter(graph='{self.graph_ir.name}', "
            f"device='{self.device}', "
            f"nodes={len(self._execution_order)})"
        )
