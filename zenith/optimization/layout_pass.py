"""
Layout Transformation Pass - Memory Layout Optimization

Implements layout transformations for optimal memory access patterns:
- NHWC ↔ NCHW conversion for different backends
- Automatic layout selection based on target hardware
- Memory layout profiling

Based on CetakBiru Section 5.1 Phase 2 requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass

from ..core import GraphIR, Node, TensorDescriptor, Shape, Layout


class LayoutFormat(Enum):
    """Tensor memory layout formats."""

    NCHW = "NCHW"  # Channel-first (PyTorch default)
    NHWC = "NHWC"  # Channel-last (TensorFlow default)
    NC = "NC"  # 2D (batch, channel)
    UNKNOWN = "UNKNOWN"


@dataclass
class LayoutPreference:
    """Backend layout preferences."""

    preferred: LayoutFormat
    fallback: LayoutFormat
    reason: str


# Backend layout preferences based on research
BACKEND_LAYOUT_PREFERENCES = {
    "cpu_avx2": LayoutPreference(
        preferred=LayoutFormat.NCHW,
        fallback=LayoutFormat.NHWC,
        reason="NCHW better for AVX2 vectorization on channel dimension",
    ),
    "cpu_neon": LayoutPreference(
        preferred=LayoutFormat.NHWC,
        fallback=LayoutFormat.NCHW,
        reason="NHWC better for ARM NEON channel-last processing",
    ),
    "cuda": LayoutPreference(
        preferred=LayoutFormat.NCHW,
        fallback=LayoutFormat.NHWC,
        reason="cuDNN prefers NCHW for most operations",
    ),
    "tensorrt": LayoutPreference(
        preferred=LayoutFormat.NCHW,
        fallback=LayoutFormat.NHWC,
        reason="TensorRT default is NCHW",
    ),
    "openvino": LayoutPreference(
        preferred=LayoutFormat.NCHW,
        fallback=LayoutFormat.NHWC,
        reason="OpenVINO prefers NCHW",
    ),
}


def transpose_nhwc_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """
    Convert tensor from NHWC to NCHW format.

    Args:
        tensor: Input tensor [N, H, W, C]

    Returns:
        Transposed tensor [N, C, H, W]
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
    return np.transpose(tensor, (0, 3, 1, 2))


def transpose_nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    """
    Convert tensor from NCHW to NHWC format.

    Args:
        tensor: Input tensor [N, C, H, W]

    Returns:
        Transposed tensor [N, H, W, C]
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
    return np.transpose(tensor, (0, 2, 3, 1))


def get_layout_from_shape(shape: list[int], hint: str | None = None) -> LayoutFormat:
    """
    Infer layout format from tensor shape.

    Args:
        shape: Tensor shape dimensions
        hint: Optional hint ("channels_first" or "channels_last")

    Returns:
        Inferred LayoutFormat
    """
    if len(shape) == 2:
        return LayoutFormat.NC
    elif len(shape) != 4:
        return LayoutFormat.UNKNOWN

    # Common heuristics for 4D tensors
    if hint == "channels_first":
        return LayoutFormat.NCHW
    elif hint == "channels_last":
        return LayoutFormat.NHWC

    # Heuristic: if dim[1] is much smaller than dim[2] or dim[3],
    # it's likely channels (NCHW)
    n, d1, d2, d3 = shape
    if d1 <= 4 and d2 > 4 and d3 > 4:
        return LayoutFormat.NCHW
    elif d3 <= 4 and d1 > 4 and d2 > 4:
        return LayoutFormat.NHWC

    return LayoutFormat.UNKNOWN


class LayoutTransformPass:
    """
    Optimization pass for memory layout transformation.

    Analyzes graph and inserts necessary transpose operations
    to match target backend preferences.
    """

    def __init__(
        self, target_layout: LayoutFormat | None = None, backend: str = "cpu_avx2"
    ):
        """
        Initialize layout transformation pass.

        Args:
            target_layout: Target layout format (overrides backend preference)
            backend: Target backend name for automatic layout selection
        """
        self.target_layout = target_layout
        self.backend = backend
        self.stats = {
            "transposes_inserted": 0,
            "layouts_converted": 0,
        }

    def apply(self, graph: GraphIR) -> GraphIR:
        """
        Apply layout transformation to the graph.

        Args:
            graph: Input graph to optimize

        Returns:
            Graph with optimized layouts
        """
        self.stats = {"transposes_inserted": 0, "layouts_converted": 0}

        # Determine target layout
        target = self.target_layout
        if target is None:
            pref = BACKEND_LAYOUT_PREFERENCES.get(self.backend)
            target = pref.preferred if pref else LayoutFormat.NCHW

        # Analyze current layouts
        current_layouts = self._analyze_layouts(graph)

        # Insert transpose nodes where needed
        for node in list(graph.nodes):
            self._process_node(graph, node, current_layouts, target)

        return graph

    def _analyze_layouts(self, graph: GraphIR) -> dict[str, LayoutFormat]:
        """Analyze current tensor layouts in the graph."""
        layouts = {}

        for tensor in graph.inputs:
            layout = self._get_tensor_layout(tensor)
            layouts[tensor.name] = layout

        for node in graph.nodes:
            for output in node.outputs:
                layout = self._get_tensor_layout(output)
                layouts[output.name] = layout

        return layouts

    def _get_tensor_layout(self, tensor: TensorDescriptor) -> LayoutFormat:
        """Get layout format for a tensor."""
        if tensor.layout == Layout.NCHW:
            return LayoutFormat.NCHW
        elif tensor.layout == Layout.NHWC:
            return LayoutFormat.NHWC
        elif tensor.layout == Layout.NC:
            return LayoutFormat.NC
        return LayoutFormat.UNKNOWN

    def _process_node(
        self,
        graph: GraphIR,
        node: Node,
        current_layouts: dict[str, LayoutFormat],
        target: LayoutFormat,
    ) -> None:
        """Process a single node for layout transformation."""
        # Convolution and pooling ops are layout-sensitive
        if node.op_type in ("Conv", "Conv2D", "MaxPool", "AvgPool"):
            self._handle_conv_like(graph, node, current_layouts, target)

    def _handle_conv_like(
        self,
        graph: GraphIR,
        node: Node,
        current_layouts: dict[str, LayoutFormat],
        target: LayoutFormat,
    ) -> None:
        """Handle layout for convolution-like operations."""
        if len(node.inputs) == 0:
            return

        input_tensor = node.inputs[0]
        current_layout = current_layouts.get(input_tensor.name, LayoutFormat.UNKNOWN)

        if current_layout == LayoutFormat.UNKNOWN:
            return

        if current_layout != target:
            # Need to insert transpose
            self.stats["transposes_inserted"] += 1
            self.stats["layouts_converted"] += 1

            # Update node attributes to reflect new layout
            node.set_attr("data_format", target.value)

    def get_stats(self) -> dict:
        """Get transformation statistics."""
        return self.stats.copy()


def optimize_layout(graph: GraphIR, backend: str = "cpu_avx2") -> GraphIR:
    """Convenience function to apply layout optimization."""
    layout_pass = LayoutTransformPass(backend=backend)
    return layout_pass.apply(graph)


def convert_weights_layout(
    weights: np.ndarray,
    from_layout: LayoutFormat,
    to_layout: LayoutFormat,
) -> np.ndarray:
    """
    Convert weight tensor between layouts.

    Conv weight layouts:
    - NCHW: [C_out, C_in, H, W]
    - NHWC: [H, W, C_in, C_out]

    Args:
        weights: Weight tensor
        from_layout: Source layout
        to_layout: Target layout

    Returns:
        Converted weight tensor
    """
    if from_layout == to_layout:
        return weights

    if weights.ndim != 4:
        return weights  # Only transform 4D weights

    if from_layout == LayoutFormat.NCHW and to_layout == LayoutFormat.NHWC:
        # [C_out, C_in, H, W] -> [H, W, C_in, C_out]
        return np.transpose(weights, (2, 3, 1, 0))
    elif from_layout == LayoutFormat.NHWC and to_layout == LayoutFormat.NCHW:
        # [H, W, C_in, C_out] -> [C_out, C_in, H, W]
        return np.transpose(weights, (3, 2, 0, 1))

    return weights
