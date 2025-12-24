# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Kernel Dispatcher - Routes GraphIR operations to Zenith CUDA kernels.

Inspired by:
- TensorRT: Pre-selected kernel execution
- ONNX Runtime: Kernel dispatch based on capability
- TVM: Operator implementation dispatch

This is the critical component that connects the optimized GraphIR
to the actual high-performance CUDA kernel implementations.
"""

from typing import Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from .kernel_registry import KernelRegistry, KernelSpec, Precision, get_registry
from .context import ExecutionContext


@dataclass
class DispatchTarget:
    """Target for kernel dispatch."""

    kernel_spec: KernelSpec
    input_names: list[str]
    output_names: list[str]
    weights: dict[str, Any] = None
    attributes: dict[str, Any] = None


class KernelDispatcher:
    """
    Dispatches GraphIR operations to Zenith CUDA kernels.

    This is the "bridge" between the GraphIR representation and
    the actual kernel execution.

    Example:
        dispatcher = KernelDispatcher(precision=Precision.FP16)

        for node in execution_plan:
            dispatcher.dispatch(node, context)
    """

    # Operation type normalization
    OP_TYPE_MAP = {
        # ONNX standard names
        "MatMul": "Linear",
        "Gemm": "Linear",
        "Conv": "Conv2d",
        "BatchNormalization": "BatchNorm",
        "LayerNormalization": "LayerNorm",
        "Relu": "ReLU",
        "Gelu": "GELU",
        # Already normalized
        "Linear": "Linear",
        "Attention": "Attention",
        "MultiHeadAttention": "Attention",
        "Add": "Add",
        "Mul": "Multiply",
        "Softmax": "Softmax",
        # Fused operations
        "FusedAddLayerNorm": "FusedAddLayerNorm",
        "FusedBiasReLU": "FusedBiasReLU",
        "FusedBiasGeLU": "FusedBiasGeLU",
    }

    def __init__(
        self,
        registry: KernelRegistry = None,
        precision: Precision = Precision.FP32,
        device: str = "cuda",
    ):
        """
        Initialize dispatcher.

        Args:
            registry: Kernel registry (uses global if None)
            precision: Default precision for kernel selection
            device: Target device
        """
        self.registry = registry or get_registry()
        self.precision = precision
        self.device = device

        # Dispatch statistics
        self._dispatch_count = 0
        self._kernel_usage: dict[str, int] = {}

    def dispatch(
        self,
        node: Any,  # ExecutionNode or GraphIR node
        context: ExecutionContext,
        weights: dict[str, Any] = None,
    ) -> Any:
        """
        Dispatch a single node to appropriate kernel.

        Args:
            node: The node to dispatch (has op_type, inputs, outputs)
            context: Execution context with tensor storage
            weights: Pre-loaded weights for the node

        Returns:
            Output tensor(s)
        """
        # Get operation type
        op_type = self._normalize_op_type(
            node.op_type if hasattr(node, "op_type") else str(node)
        )

        # Get input tensors from context
        raw_inputs = node.inputs if hasattr(node, "inputs") else []
        # Handle TensorDescriptor objects - extract name if needed
        input_names = []
        for inp in raw_inputs:
            if isinstance(inp, str):
                input_names.append(inp)
            elif hasattr(inp, "name"):
                input_names.append(inp.name)
            else:
                input_names.append(str(inp))

        inputs = [
            context.get_tensor(name) for name in input_names if context.has_tensor(name)
        ]

        # Get input shapes for kernel selection
        input_shapes = [
            tuple(inp.shape) if hasattr(inp, "shape") else () for inp in inputs
        ]

        # Select kernel
        kernel = self.registry.get_kernel(op_type, self.precision, input_shapes)

        if kernel is None:
            raise UnsupportedOperationError(
                f"No kernel found for operation '{op_type}' with precision {self.precision.value}. "
                f"Supported operations: {self.registry.list_supported_ops()}"
            )

        # Execute kernel
        output = self._execute_kernel(kernel, inputs, weights, node)

        # Store output in context
        output_names = node.outputs if hasattr(node, "outputs") else []
        if output_names:
            if isinstance(output, tuple):
                for i, name in enumerate(output_names):
                    if i < len(output):
                        context.set_tensor(name, output[i])
            else:
                context.set_tensor(output_names[0], output)

        # Update statistics
        self._dispatch_count += 1
        self._kernel_usage[kernel.name] = self._kernel_usage.get(kernel.name, 0) + 1

        return output

    def _normalize_op_type(self, op_type: str) -> str:
        """Normalize operation type name."""
        return self.OP_TYPE_MAP.get(op_type, op_type)

    def _execute_kernel(
        self, kernel: KernelSpec, inputs: list[Any], weights: dict[str, Any], node: Any
    ) -> Any:
        """
        Execute a kernel with given inputs.

        Args:
            kernel: Kernel specification
            inputs: Input tensors
            weights: Weight tensors
            node: Original node for attributes

        Returns:
            Output tensor(s)
        """
        if kernel.kernel_fn is None:
            raise RuntimeError(f"Kernel {kernel.name} has no implementation")

        # Get weights if needed
        weight_args = []
        if weights:
            # Order matters - typically: weight, bias
            if "weight" in weights:
                weight_args.append(weights["weight"])
            if "bias" in weights:
                weight_args.append(weights["bias"])
            # Add any other weights
            for key, value in weights.items():
                if key not in ("weight", "bias"):
                    weight_args.append(value)

        # Get node attributes (e.g., epsilon for LayerNorm)
        attrs = {}
        if hasattr(node, "attributes"):
            attrs = node.attributes

        # Prepare arguments based on operation type
        op_type = self._normalize_op_type(
            node.op_type if hasattr(node, "op_type") else ""
        )

        try:
            return self._call_kernel(kernel, op_type, inputs, weight_args, attrs)
        except Exception as e:
            raise KernelExecutionError(
                f"Kernel {kernel.name} failed: {e}. "
                f"Inputs: {[getattr(i, 'shape', 'unknown') for i in inputs]}"
            ) from e

    def _call_kernel(
        self,
        kernel: KernelSpec,
        op_type: str,
        inputs: list[Any],
        weights: list[Any],
        attrs: dict,
    ) -> Any:
        """
        Call kernel with proper argument order.

        Different operations have different argument signatures.
        """
        fn = kernel.kernel_fn

        # Handle different operation types
        if op_type in ("Linear", "MatMul", "Gemm"):
            # linear(input, weight, bias?)
            if len(weights) >= 2:
                return fn(inputs[0], weights[0], weights[1])
            elif len(weights) == 1:
                return fn(inputs[0], weights[0])
            else:
                return fn(inputs[0], inputs[1])

        elif op_type in ("LayerNorm", "LayerNormalization"):
            # layernorm(input, gamma, beta, eps)
            eps = attrs.get("epsilon", 1e-5)
            if len(weights) >= 2:
                return fn(inputs[0], weights[0], weights[1], eps)
            else:
                return fn(
                    inputs[0], weights[0], weights[1] if len(weights) > 1 else None, eps
                )

        elif op_type in ("Attention", "MultiHeadAttention"):
            # attention(Q, K, V)
            if len(inputs) >= 3:
                return fn(inputs[0], inputs[1], inputs[2])
            else:
                # Single input, need to split into Q, K, V
                return fn(inputs[0])

        elif op_type in ("ReLU", "Relu", "GELU", "Gelu", "Sigmoid", "Softmax"):
            # activation(input)
            return fn(inputs[0])

        elif op_type == "Add":
            # add(a, b)
            return fn(inputs[0], inputs[1])

        elif op_type == "FusedAddLayerNorm":
            # fused_add_layernorm(x, residual, gamma, beta, eps)
            eps = attrs.get("epsilon", 1e-5)
            return fn(inputs[0], inputs[1], weights[0], weights[1], eps)

        elif op_type == "Conv2d":
            # conv2d(input, weight, bias?, stride, padding, ...)
            stride = attrs.get("strides", [1, 1])
            padding = attrs.get("pads", [0, 0, 0, 0])
            if len(weights) >= 2:
                return fn(inputs[0], weights[0], weights[1])
            else:
                return fn(inputs[0], weights[0])

        elif op_type == "TransposeForAttention":
            # transpose_for_attention(input, batch, seq_len, num_heads, head_dim)
            batch = attrs.get(
                "batch_size", inputs[0].shape[0] if hasattr(inputs[0], "shape") else 1
            )
            seq_len = attrs.get(
                "seq_len", inputs[0].shape[1] if hasattr(inputs[0], "shape") else 1
            )
            num_heads = attrs.get("num_heads", 12)
            head_dim = attrs.get("head_dim", 64)
            return fn(inputs[0], batch, seq_len, num_heads, head_dim)

        else:
            # Generic call - try to pass all inputs and weights
            all_args = inputs + weights
            return fn(*all_args)

    def get_dispatch_stats(self) -> dict:
        """Get dispatch statistics."""
        return {
            "total_dispatches": self._dispatch_count,
            "kernel_usage": self._kernel_usage,
            "precision": self.precision.value,
            "device": self.device,
        }

    def reset_stats(self) -> None:
        """Reset dispatch statistics."""
        self._dispatch_count = 0
        self._kernel_usage.clear()


class UnsupportedOperationError(Exception):
    """Raised when an operation is not supported."""

    pass


class KernelExecutionError(Exception):
    """Raised when kernel execution fails."""

    pass
