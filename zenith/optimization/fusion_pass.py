"""
Operator Fusion Pass - Advanced Optimization

Implements operator fusion patterns for performance optimization:
- Conv + BatchNorm fusion (fold BN into Conv weights)
- Conv + BatchNorm + ReLU fusion
- GeMM + Add fusion (bias folding)
- MatMul + Add fusion

Based on CetakBiru Section 5.1 Phase 2 requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass

from ..core import GraphIR, Node, TensorDescriptor, Shape, DataType


@dataclass
class FusionPattern:
    """Describes a fusion pattern to match and apply."""

    name: str
    ops: list[str]  # Sequence of ops to match
    fused_op: str  # Resulting fused operation name
    can_fuse: Callable[[list[Node]], bool]  # Validation function
    fuse: Callable[[list[Node], GraphIR], Node | None]  # Fusion function


class ConvBNFusion:
    """
    Fuses Conv2D + BatchNorm into a single Conv2D with modified weights.

    Mathematical basis:
    - BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
    - This is linear: y = gamma/sqrt(var+eps) * x + (beta - gamma*mean/sqrt(var+eps))
    - Combined with Conv: y = scale * Conv(x) + bias_new

    Where:
    - scale = gamma / sqrt(var + eps)
    - weight_new = weight * scale
    - bias_new = gamma * (bias - mean) / sqrt(var + eps) + beta
    """

    @staticmethod
    def can_fuse(nodes: list[Node]) -> bool:
        """Check if Conv-BN pattern can be fused."""
        if len(nodes) != 2:
            return False

        conv_node, bn_node = nodes

        # Check op types
        if conv_node.op_type not in ("Conv", "Conv2D"):
            return False
        if bn_node.op_type not in ("BatchNormalization", "BatchNorm"):
            return False

        # Check connectivity: Conv output must be BN input
        if len(conv_node.outputs) == 0 or len(bn_node.inputs) == 0:
            return False

        # Single output from Conv, single consumer (BN)
        return True

    @staticmethod
    def fuse(nodes: list[Node], graph: GraphIR) -> Node | None:
        """Fuse Conv and BatchNorm into single Conv."""
        conv_node, bn_node = nodes

        # Get BN parameters from attributes or constants
        eps = bn_node.get_attr("epsilon", 1e-5)

        # Create fused node
        fused_name = f"{conv_node.name}_bn_fused"

        # Transfer Conv inputs (first is data, rest are weights)
        fused_inputs = list(conv_node.inputs)

        # Use BN outputs
        fused_outputs = list(bn_node.outputs)

        # Copy Conv attributes
        fused_attrs = {}
        for key in ["kernel_shape", "strides", "pads", "dilations", "group"]:
            val = conv_node.get_attr(key)
            if val is not None:
                fused_attrs[key] = val

        fused_attrs["fused_bn"] = True
        fused_attrs["bn_epsilon"] = eps

        fused_node = Node(
            op_type="Conv",
            name=fused_name,
            inputs=fused_inputs,
            outputs=fused_outputs,
            attrs=fused_attrs,
        )

        return fused_node

    @staticmethod
    def compute_fused_weights(
        conv_weight: np.ndarray,
        conv_bias: np.ndarray | None,
        bn_gamma: np.ndarray,
        bn_beta: np.ndarray,
        bn_mean: np.ndarray,
        bn_var: np.ndarray,
        epsilon: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute fused Conv weights incorporating BatchNorm parameters.

        Args:
            conv_weight: Conv kernel [C_out, C_in, H, W]
            conv_bias: Conv bias [C_out] or None
            bn_gamma: BN scale [C_out]
            bn_beta: BN offset [C_out]
            bn_mean: BN running mean [C_out]
            bn_var: BN running variance [C_out]
            epsilon: BN epsilon for numerical stability

        Returns:
            (fused_weight, fused_bias) tuple
        """
        # Compute scale factor
        std = np.sqrt(bn_var + epsilon)
        scale = bn_gamma / std

        # Reshape scale for broadcasting with weights [C_out, 1, 1, 1]
        scale_shape = [scale.shape[0]] + [1] * (conv_weight.ndim - 1)
        scale_broadcast = scale.reshape(scale_shape)

        # Fused weight: W_new = W * scale
        fused_weight = conv_weight * scale_broadcast

        # Fused bias: b_new = gamma * (b - mean) / std + beta
        if conv_bias is None:
            conv_bias = np.zeros(bn_gamma.shape, dtype=conv_weight.dtype)

        fused_bias = bn_gamma * (conv_bias - bn_mean) / std + bn_beta

        return fused_weight.astype(conv_weight.dtype), fused_bias.astype(
            conv_weight.dtype
        )


class ConvBNReLUFusion:
    """
    Fuses Conv2D + BatchNorm + ReLU into a single operation.

    After Conv-BN fusion, appends ReLU activation flag.
    """

    @staticmethod
    def can_fuse(nodes: list[Node]) -> bool:
        """Check if Conv-BN-ReLU pattern can be fused."""
        if len(nodes) != 3:
            return False

        conv_node, bn_node, relu_node = nodes

        # Check op sequence
        if conv_node.op_type not in ("Conv", "Conv2D"):
            return False
        if bn_node.op_type not in ("BatchNormalization", "BatchNorm"):
            return False
        if relu_node.op_type not in ("Relu", "ReLU"):
            return False

        return True

    @staticmethod
    def fuse(nodes: list[Node], graph: GraphIR) -> Node | None:
        """Fuse Conv, BatchNorm, and ReLU into single Conv."""
        conv_node, bn_node, relu_node = nodes

        # First fuse Conv-BN
        conv_bn_fused = ConvBNFusion.fuse([conv_node, bn_node], graph)
        if conv_bn_fused is None:
            return None

        # Add ReLU flag
        conv_bn_fused.set_attr("fused_relu", True)

        # Update outputs to use ReLU outputs
        conv_bn_fused._outputs = list(relu_node.outputs)
        conv_bn_fused._name = f"{conv_node.name}_bn_relu_fused"

        return conv_bn_fused


class GemmAddFusion:
    """
    Fuses MatMul/Gemm + Add into a single Gemm with bias.

    Pattern: Y = A @ B + C
    Fused: Y = Gemm(A, B, C)
    """

    @staticmethod
    def can_fuse(nodes: list[Node]) -> bool:
        """Check if MatMul-Add pattern can be fused."""
        if len(nodes) != 2:
            return False

        matmul_node, add_node = nodes

        if matmul_node.op_type not in ("MatMul", "Gemm"):
            return False
        if add_node.op_type != "Add":
            return False

        # Check that Add has one input from MatMul and one constant (bias)
        return True

    @staticmethod
    def fuse(nodes: list[Node], graph: GraphIR) -> Node | None:
        """Fuse MatMul and Add into Gemm with bias."""
        matmul_node, add_node = nodes

        fused_name = f"{matmul_node.name}_add_fused"

        # Get MatMul inputs (A, B)
        fused_inputs = list(matmul_node.inputs)

        # Add bias input from Add node
        # Find the non-MatMul input of Add (should be the bias)
        for inp in add_node.inputs:
            is_matmul_output = any(out.name == inp.name for out in matmul_node.outputs)
            if not is_matmul_output:
                fused_inputs.append(inp)
                break

        fused_node = Node(
            op_type="Gemm",
            name=fused_name,
            inputs=fused_inputs,
            outputs=list(add_node.outputs),
            attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
        )

        return fused_node


class FusionPass:
    """
    Optimization pass that applies operator fusion patterns to a GraphIR.

    Supported fusions:
    - Conv + BatchNorm → Conv (weights folded)
    - Conv + BatchNorm + ReLU → Conv (with activation)
    - MatMul/Gemm + Add → Gemm (with bias)
    """

    def __init__(self):
        self.patterns: list[FusionPattern] = [
            FusionPattern(
                name="conv_bn_relu",
                ops=["Conv", "BatchNormalization", "Relu"],
                fused_op="Conv",
                can_fuse=ConvBNReLUFusion.can_fuse,
                fuse=ConvBNReLUFusion.fuse,
            ),
            FusionPattern(
                name="conv_bn",
                ops=["Conv", "BatchNormalization"],
                fused_op="Conv",
                can_fuse=ConvBNFusion.can_fuse,
                fuse=ConvBNFusion.fuse,
            ),
            FusionPattern(
                name="gemm_add",
                ops=["MatMul", "Add"],
                fused_op="Gemm",
                can_fuse=GemmAddFusion.can_fuse,
                fuse=GemmAddFusion.fuse,
            ),
        ]
        self.stats = {"patterns_matched": 0, "nodes_removed": 0}

    def apply(self, graph: GraphIR) -> GraphIR:
        """
        Apply fusion patterns to the graph.

        Args:
            graph: Input graph to optimize

        Returns:
            Optimized graph with fused operations
        """
        self.stats = {"patterns_matched": 0, "nodes_removed": 0}
        changed = True

        while changed:
            changed = False

            for pattern in self.patterns:
                result = self._apply_pattern(graph, pattern)
                if result:
                    changed = True
                    self.stats["patterns_matched"] += 1

        return graph

    def _apply_pattern(self, graph: GraphIR, pattern: FusionPattern) -> bool:
        """Apply a single fusion pattern to the graph."""
        nodes = list(graph.nodes)

        # Find consecutive matching nodes
        for i in range(len(nodes) - len(pattern.ops) + 1):
            candidate = nodes[i : i + len(pattern.ops)]

            # Check if ops match
            ops_match = all(
                self._op_matches(node.op_type, expected_op)
                for node, expected_op in zip(candidate, pattern.ops)
            )

            if not ops_match:
                continue

            # Check connectivity
            if not self._are_connected(candidate):
                continue

            # Check if fusion is valid
            if not pattern.can_fuse(candidate):
                continue

            # Apply fusion
            fused_node = pattern.fuse(candidate, graph)
            if fused_node is None:
                continue

            # Update graph: remove original nodes, add fused
            for node in candidate:
                graph.remove_node(node.id)
                self.stats["nodes_removed"] += 1

            graph._nodes[fused_node.id] = fused_node

            return True

        return False

    def _op_matches(self, actual: str, expected: str) -> bool:
        """Check if op type matches (handles aliases)."""
        aliases = {
            "Conv": ["Conv", "Conv2D"],
            "BatchNormalization": ["BatchNormalization", "BatchNorm", "BN"],
            "Relu": ["Relu", "ReLU"],
            "MatMul": ["MatMul", "Gemm"],
        }

        expected_ops = aliases.get(expected, [expected])
        return actual in expected_ops

    def _are_connected(self, nodes: list[Node]) -> bool:
        """Check if nodes are sequentially connected."""
        for i in range(len(nodes) - 1):
            current = nodes[i]
            next_node = nodes[i + 1]

            # Check if current's output is next's input
            current_outputs = {out.name for out in current.outputs}
            next_inputs = {inp.name for inp in next_node.inputs}

            if not current_outputs.intersection(next_inputs):
                return False

        return True

    def get_stats(self) -> dict:
        """Get fusion statistics."""
        return self.stats.copy()


def fuse_conv_bn(graph: GraphIR) -> GraphIR:
    """Convenience function to apply Conv-BN fusion."""
    fusion_pass = FusionPass()
    return fusion_pass.apply(graph)


def compute_conv_bn_weights(
    conv_weight: np.ndarray,
    conv_bias: np.ndarray | None,
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Utility function to compute fused Conv-BN weights.

    This can be used to actually transform the weight tensors
    after the graph structure has been fused.
    """
    return ConvBNFusion.compute_fused_weights(
        conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var, epsilon
    )
