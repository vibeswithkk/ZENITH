# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Optimization Passes (PoC Implementation)

This module implements core optimization passes for the Zenith framework
as specified in CetakBiru Phase 0. These are proof-of-concept implementations
that demonstrate the optimization algorithms before full C++ implementation.

Optimization passes include:
- Constant Folding: Evaluate constant expressions at compile time
- Dead Code Elimination: Remove unused nodes from the graph
- Operator Fusion: Combine sequential operations into single optimized ops
- Conv-BN Fusion: Fold BatchNorm parameters into Conv weights
- Layout Transformation: NHWC/NCHW conversion

Profiling and Benchmarking:
- Profiler: Per-operation timing and memory tracking
- Benchmark: Performance comparison with NumPy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy

from ..core.graph_ir import GraphIR
from ..core.node import Node
from ..core.tensor import TensorDescriptor
from ..core.types import Shape

# Phase 2: Advanced optimization imports
try:
    from .fusion_pass import (
        FusionPass,
        ConvBNFusion,
        ConvBNReLUFusion,
        GemmAddFusion,
        fuse_conv_bn,
        compute_conv_bn_weights,
    )
    from .layout_pass import (
        LayoutFormat,
        LayoutTransformPass,
        optimize_layout,
        transpose_nhwc_to_nchw,
        transpose_nchw_to_nhwc,
        convert_weights_layout,
        get_layout_from_shape,
    )
    from .profiler import (
        Profiler,
        ProfileSession,
        OperationMetrics,
        get_profiler,
        reset_profiler,
    )
    from .benchmark import (
        Benchmark,
        BenchmarkResult,
        ComparisonResult,
        benchmark_matmul_vs_numpy,
    )

    _PHASE2_AVAILABLE = True
except ImportError:
    _PHASE2_AVAILABLE = False


class OptimizationPass(ABC):
    """
    Abstract base class for all optimization passes.

    Each pass transforms a GraphIR and returns a modified version.
    Passes should be idempotent - applying them multiple times
    should have the same effect as applying once.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this optimization pass."""
        pass

    @abstractmethod
    def run(self, graph: GraphIR) -> tuple[GraphIR, bool]:
        """
        Apply the optimization pass to the graph.

        Args:
            graph: The input GraphIR to optimize.

        Returns:
            Tuple of (optimized_graph, was_modified).
            was_modified is True if any changes were made.
        """
        pass

    def __repr__(self) -> str:
        return f"<OptimizationPass: {self.name}>"


class ConstantFoldingPass(OptimizationPass):
    """
    Constant Folding Optimization Pass.

    Evaluates operations with all-constant inputs at compile time,
    replacing the operation node with a constant result.

    Example:
        Input:  x = Constant(2), y = Constant(3), z = Add(x, y)
        Output: z = Constant(5)

    This reduces runtime computation by pre-computing known values.
    """

    @property
    def name(self) -> str:
        return "constant_folding"

    # Operations that can be folded when all inputs are constants
    FOLDABLE_OPS = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Neg",
        "Reshape",
        "Transpose",
        "Cast",
        "Identity",
    }

    def run(self, graph: GraphIR) -> tuple[GraphIR, bool]:
        """Apply constant folding to the graph."""
        modified = False
        nodes_to_remove = []
        constants_to_add = {}

        # Build a map of constant tensor names
        constant_names = set(graph.constants.keys())

        for node in graph.nodes:
            # Check if this operation can be folded
            if node.op_type not in self.FOLDABLE_OPS:
                continue

            # Check if all inputs are constants
            all_inputs_constant = all(inp.name in constant_names for inp in node.inputs)

            if all_inputs_constant and node.inputs:
                # This node can be folded
                # In a full implementation, we would actually compute the result
                # For PoC, we mark the node for removal and note the folding

                # Mark for removal
                nodes_to_remove.append(node.name)

                # Mark outputs as constants (placeholder - real impl would compute)
                for output in node.outputs:
                    # In real implementation, compute actual value
                    constants_to_add[output.name] = b"<folded_constant>"
                    constant_names.add(output.name)

                modified = True

        # Apply modifications
        if modified:
            # Create a new graph with modifications
            new_graph = GraphIR(name=graph.name)
            new_graph.inputs = copy.deepcopy(graph.inputs)
            new_graph.outputs = copy.deepcopy(graph.outputs)
            new_graph.constants = copy.deepcopy(graph.constants)

            # Add new constants
            for name, data in constants_to_add.items():
                new_graph.add_constant(name, data)

            # Add nodes that weren't removed
            for node in graph.nodes:
                if node.name not in nodes_to_remove:
                    new_graph.add_node(
                        op_type=node.op_type,
                        name=node.name,
                        inputs=copy.deepcopy(node.inputs),
                        outputs=copy.deepcopy(node.outputs),
                        attrs=copy.deepcopy(node.attrs),
                    )

            return new_graph, True

        return graph, False


class DeadCodeEliminationPass(OptimizationPass):
    """
    Dead Code Elimination (DCE) Optimization Pass.

    Removes nodes whose outputs are never used by any other node
    or by the graph outputs.

    Example:
        Input:  a = Input, b = Relu(a), c = Sigmoid(a), output = b
        Output: a = Input, b = Relu(a), output = b
        (Sigmoid node is removed as 'c' is never used)

    This reduces memory usage and computation by eliminating unused work.
    """

    @property
    def name(self) -> str:
        return "dead_code_elimination"

    def run(self, graph: GraphIR) -> tuple[GraphIR, bool]:
        """Apply dead code elimination to the graph."""
        # Build set of "live" tensor names (those that are actually used)
        live_tensors: set[str] = set()

        # Output tensors are always live
        for output in graph.outputs:
            live_tensors.add(output.name)

        # Work backwards from outputs to find all live tensors
        # Keep iterating until no new live tensors are found
        changed = True
        while changed:
            changed = False
            for node in graph.nodes:
                # If any output of this node is live, all inputs are live
                node_is_live = any(out.name in live_tensors for out in node.outputs)
                if node_is_live:
                    for inp in node.inputs:
                        if inp.name not in live_tensors:
                            live_tensors.add(inp.name)
                            changed = True

        # Find dead nodes (nodes with no live outputs)
        dead_nodes = []
        for node in graph.nodes:
            if not any(out.name in live_tensors for out in node.outputs):
                dead_nodes.append(node.name)

        if not dead_nodes:
            return graph, False

        # Create new graph without dead nodes
        new_graph = GraphIR(name=graph.name)
        new_graph.inputs = copy.deepcopy(graph.inputs)
        new_graph.outputs = copy.deepcopy(graph.outputs)
        new_graph.constants = copy.deepcopy(graph.constants)

        for node in graph.nodes:
            if node.name not in dead_nodes:
                new_graph.add_node(
                    op_type=node.op_type,
                    name=node.name,
                    inputs=copy.deepcopy(node.inputs),
                    outputs=copy.deepcopy(node.outputs),
                    attrs=copy.deepcopy(node.attrs),
                )

        return new_graph, True


@dataclass
class FusionPattern:
    """Defines a pattern for operator fusion."""

    name: str
    ops_sequence: list[str]  # e.g., ["Conv", "BatchNormalization", "Relu"]
    fused_op: str  # e.g., "ConvBnRelu"


class OperatorFusionPass(OptimizationPass):
    """
    Operator Fusion Optimization Pass.

    Combines sequences of operations into single fused operations
    to reduce memory bandwidth and kernel launch overhead.

    Supported fusion patterns:
    - Conv + Relu -> ConvRelu
    - Conv + BatchNormalization + Relu -> ConvBnRelu
    - MatMul + Add -> Gemm
    - BatchNormalization + Relu -> BnRelu

    Example:
        Input:  x = Conv(input), y = Relu(x)
        Output: y = ConvRelu(input)
    """

    @property
    def name(self) -> str:
        return "operator_fusion"

    # Define fusion patterns
    FUSION_PATTERNS = [
        FusionPattern("conv_relu", ["Conv", "Relu"], "ConvRelu"),
        FusionPattern(
            "conv_bn_relu", ["Conv", "BatchNormalization", "Relu"], "ConvBnRelu"
        ),
        FusionPattern("bn_relu", ["BatchNormalization", "Relu"], "BnRelu"),
        FusionPattern("matmul_add", ["MatMul", "Add"], "Gemm"),
        FusionPattern("conv_bn", ["Conv", "BatchNormalization"], "ConvBn"),
    ]

    def run(self, graph: GraphIR) -> tuple[GraphIR, bool]:
        """Apply operator fusion to the graph."""
        modified = False

        # Build output->node map for finding consumers
        producer_map: dict[str, Node] = {}
        for node in graph.nodes:
            for output in node.outputs:
                producer_map[output.name] = node

        # Build input->nodes map for finding consumers
        consumer_map: dict[str, list[Node]] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp.name not in consumer_map:
                    consumer_map[inp.name] = []
                consumer_map[inp.name].append(node)

        # Track which nodes have been fused (to skip them)
        fused_nodes: set[str] = set()
        # Track new fused nodes to add
        fused_ops: list[tuple[Node, list[str]]] = []

        for pattern in self.FUSION_PATTERNS:
            # Look for sequences matching this pattern
            for node in graph.nodes:
                if node.name in fused_nodes:
                    continue

                if node.op_type != pattern.ops_sequence[0]:
                    continue

                # Try to match the full pattern
                sequence = [node]
                current = node
                matched = True

                for i, expected_op in enumerate(pattern.ops_sequence[1:], 1):
                    # Find the consumer of current node's output
                    if not current.outputs:
                        matched = False
                        break

                    output_name = current.outputs[0].name
                    consumers = consumer_map.get(output_name, [])

                    # Must have exactly one consumer for fusion
                    if len(consumers) != 1:
                        matched = False
                        break

                    next_node = consumers[0]
                    if next_node.op_type != expected_op:
                        matched = False
                        break

                    if next_node.name in fused_nodes:
                        matched = False
                        break

                    sequence.append(next_node)
                    current = next_node

                if matched and len(sequence) == len(pattern.ops_sequence):
                    # Found a complete match - mark for fusion
                    first_node = sequence[0]
                    last_node = sequence[-1]

                    # Create fused node
                    fused_node = Node(
                        op_type=pattern.fused_op,
                        name=f"fused_{pattern.name}_{first_node.name}",
                        inputs=copy.deepcopy(first_node.inputs),
                        outputs=copy.deepcopy(last_node.outputs),
                        attrs={
                            "fused_from": [n.name for n in sequence],
                            "pattern": pattern.name,
                        },
                    )

                    # Collect attributes from all nodes
                    for seq_node in sequence:
                        for key, value in seq_node.attrs.items():
                            fused_node.attrs[f"{seq_node.op_type}_{key}"] = value

                    fused_ops.append((fused_node, [n.name for n in sequence]))

                    for n in sequence:
                        fused_nodes.add(n.name)

                    modified = True

        if not modified:
            return graph, False

        # Build new graph with fused operations
        new_graph = GraphIR(name=graph.name)
        new_graph.inputs = copy.deepcopy(graph.inputs)
        new_graph.outputs = copy.deepcopy(graph.outputs)
        new_graph.constants = copy.deepcopy(graph.constants)

        # Add non-fused nodes
        for node in graph.nodes:
            if node.name not in fused_nodes:
                new_graph.add_node(
                    op_type=node.op_type,
                    name=node.name,
                    inputs=copy.deepcopy(node.inputs),
                    outputs=copy.deepcopy(node.outputs),
                    attrs=copy.deepcopy(node.attrs),
                )

        # Add fused nodes
        for fused_node, _ in fused_ops:
            new_graph.add_node(
                op_type=fused_node.op_type,
                name=fused_node.name,
                inputs=copy.deepcopy(fused_node.inputs),
                outputs=copy.deepcopy(fused_node.outputs),
                attrs=copy.deepcopy(fused_node.attrs),
            )

        return new_graph, True


class PassManager:
    """
    Manages and orchestrates optimization passes.

    Allows registering multiple passes and running them in sequence
    with configurable iteration limits.
    """

    def __init__(self):
        self._passes: list[OptimizationPass] = []

    def add_pass(self, opt_pass: OptimizationPass) -> "PassManager":
        """Add an optimization pass to the manager."""
        self._passes.append(opt_pass)
        return self

    def run(
        self,
        graph: GraphIR,
        max_iterations: int = 10,
    ) -> tuple[GraphIR, dict[str, int]]:
        """
        Run all registered passes on the graph.

        Passes are run repeatedly until no modifications are made
        or max_iterations is reached.

        Args:
            graph: The input graph to optimize.
            max_iterations: Maximum number of full pass iterations.

        Returns:
            Tuple of (optimized_graph, stats).
            stats is a dict mapping pass names to number of times applied.
        """
        stats: dict[str, int] = {p.name: 0 for p in self._passes}
        current_graph = graph

        for iteration in range(max_iterations):
            any_modified = False

            for opt_pass in self._passes:
                new_graph, modified = opt_pass.run(current_graph)
                if modified:
                    stats[opt_pass.name] += 1
                    current_graph = new_graph
                    any_modified = True

            if not any_modified:
                break

        return current_graph, stats


def create_default_pass_manager() -> PassManager:
    """Create a PassManager with the default optimization passes."""
    manager = PassManager()
    manager.add_pass(ConstantFoldingPass())
    manager.add_pass(DeadCodeEliminationPass())
    manager.add_pass(OperatorFusionPass())
    return manager


def optimize_graph(
    graph: GraphIR,
    opt_level: int = 2,
    max_iterations: int = 10,
) -> tuple[GraphIR, dict[str, int]]:
    """
    Convenience function to optimize a graph with default passes.

    Args:
        graph: The input GraphIR to optimize.
        opt_level: Optimization level (0=none, 1=basic, 2=standard, 3=aggressive).
        max_iterations: Maximum optimization iterations.

    Returns:
        Tuple of (optimized_graph, optimization_stats).
    """
    if opt_level == 0:
        return graph, {}

    manager = PassManager()

    # Level 1: Basic optimizations
    if opt_level >= 1:
        manager.add_pass(ConstantFoldingPass())

    # Level 2: Standard optimizations
    if opt_level >= 2:
        manager.add_pass(DeadCodeEliminationPass())

    # Level 3: Aggressive optimizations
    if opt_level >= 3:
        manager.add_pass(OperatorFusionPass())

    return manager.run(graph, max_iterations)
