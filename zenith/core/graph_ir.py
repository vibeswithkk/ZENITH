# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
GraphIR (Pure Python Implementation)

The unified intermediate representation for computation graphs.
"""

from dataclasses import dataclass, field
from typing import Optional, Any

from .types import Status, StatusCode
from .tensor import TensorDescriptor
from .node import Node


@dataclass
class GraphIR:
    """
    The unified intermediate representation for computation graphs.

    This is the core data structure that all framework adapters convert to,
    and all optimization passes operate on.

    Based on the class diagram in section 4.4 of the blueprint.
    """

    name: str = ""
    _nodes: list[Node] = field(default_factory=list, init=False, repr=False)
    _name_to_node: dict[str, Node] = field(default_factory=dict, init=False, repr=False)
    inputs: list[TensorDescriptor] = field(default_factory=list)
    outputs: list[TensorDescriptor] = field(default_factory=list)
    constants: dict[str, bytes] = field(default_factory=dict)

    def add_node(
        self,
        op_type: str,
        name: str,
        inputs: list[TensorDescriptor],
        outputs: list[TensorDescriptor],
        attrs: dict[str, Any] | None = None,
    ) -> Node:
        """Add a node to the graph."""
        node = Node(
            op_type=op_type,
            name=name,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
        )
        self._nodes.append(node)
        self._name_to_node[name] = node
        return node

    def get_node(self, name: str) -> Optional[Node]:
        """Get node by name."""
        return self._name_to_node.get(name)

    @property
    def nodes(self) -> list[Node]:
        """Get all nodes."""
        return self._nodes

    def num_nodes(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)

    def remove_node(self, name: str) -> bool:
        """Remove a node by name."""
        if name not in self._name_to_node:
            return False

        node = self._name_to_node[name]
        self._nodes.remove(node)
        del self._name_to_node[name]
        return True

    def set_inputs(self, inputs: list[TensorDescriptor]) -> None:
        """Set graph input tensors."""
        self.inputs = inputs

    def set_outputs(self, outputs: list[TensorDescriptor]) -> None:
        """Set graph output tensors."""
        self.outputs = outputs

    def add_input(self, tensor: TensorDescriptor) -> None:
        """Add a graph input."""
        self.inputs.append(tensor)

    def add_output(self, tensor: TensorDescriptor) -> None:
        """Add a graph output."""
        self.outputs.append(tensor)

    def add_constant(self, name: str, data: bytes) -> None:
        """Add constant tensor data (weights, biases, etc.)."""
        self.constants[name] = data

    def get_constant(self, name: str) -> Optional[bytes]:
        """Get constant data."""
        return self.constants.get(name)

    def find_nodes_by_op(self, op_type: str) -> list[Node]:
        """Find all nodes of a specific operation type."""
        return [n for n in self._nodes if n.op_type == op_type]

    def topological_order(self) -> list[Node]:
        """Get nodes in topological order (for execution)."""
        # Simplified implementation - returns nodes in insertion order
        # Full topological sort will be implemented in optimization phase
        return list(self._nodes)

    def validate(self) -> Status:
        """Validate the graph structure."""
        if not self._nodes:
            return Status.Error(StatusCode.InvalidGraph, "Graph has no nodes")

        if not self.inputs:
            return Status.Error(StatusCode.InvalidGraph, "Graph has no inputs")

        if not self.outputs:
            return Status.Error(StatusCode.InvalidGraph, "Graph has no outputs")

        # Check for duplicate node names
        seen = set()
        for node in self._nodes:
            if node.name in seen:
                return Status.Error(
                    StatusCode.InvalidGraph, f"Duplicate node name: {node.name}"
                )
            seen.add(node.name)

        return Status.Ok()

    def count_ops(self) -> dict[str, int]:
        """Count nodes by operation type."""
        counts: dict[str, int] = {}
        for node in self._nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return counts

    def summary(self) -> str:
        """Print graph summary."""
        lines = [
            f"GraphIR: {self.name}",
            f"  Inputs: {len(self.inputs)}",
            f"  Outputs: {len(self.outputs)}",
            f"  Nodes: {len(self._nodes)}",
            f"  Constants: {len(self.constants)}",
            "  Operations:",
        ]

        for op, count in self.count_ops().items():
            lines.append(f"    {op}: {count}")

        return "\n".join(lines)

    def clone(self) -> "GraphIR":
        """Deep clone the graph."""
        new_graph = GraphIR(name=self.name)
        new_graph.inputs = list(self.inputs)
        new_graph.outputs = list(self.outputs)
        new_graph.constants = dict(self.constants)

        for node in self._nodes:
            cloned = node.clone()
            new_graph._nodes.append(cloned)
            new_graph._name_to_node[cloned.name] = cloned

        return new_graph

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"GraphIR(name='{self.name}', nodes={len(self._nodes)})"
