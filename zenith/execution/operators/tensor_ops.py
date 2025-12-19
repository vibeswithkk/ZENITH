# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tensor Manipulation Operators

Implements ONNX tensor operators:
- Gather: Index selection for embeddings
- Cast: Type conversion
- Split: Split tensor along axis
- Concat: Concatenate tensors
- Slice: Extract sub-tensor
- Pad: Pad tensor
- Tile: Repeat tensor
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING
import numpy as np

from ..registry import OperatorRegistry

if TYPE_CHECKING:
    from ..context import ExecutionContext


# Data type mapping for Cast operator
ONNX_DTYPE_MAP = {
    1: np.float32,  # FLOAT
    2: np.uint8,  # UINT8
    3: np.int8,  # INT8
    4: np.uint16,  # UINT16
    5: np.int16,  # INT16
    6: np.int32,  # INT32
    7: np.int64,  # INT64
    9: bool,  # BOOL
    10: np.float16,  # FLOAT16
    11: np.float64,  # DOUBLE
    12: np.uint32,  # UINT32
    13: np.uint64,  # UINT64
}


@OperatorRegistry.register("Gather")
def execute_gather(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Gather operator - selects elements by indices.

    ONNX Spec: output[i][j][k] = data[indices[i][j][k]][j][k] (for axis=0)

    Used for: Token embeddings, position embeddings

    Inputs:
        data: Input tensor
        indices: Index tensor

    Attributes:
        axis: Axis to gather on (default: 0)
    """
    data = ctx.get_tensor(inputs[0])
    indices = ctx.get_tensor(inputs[1])

    axis = attrs.get("axis", 0)

    # Normalize axis
    if axis < 0:
        axis = data.ndim + axis

    # Use numpy's take for gathering
    result = np.take(data, indices.astype(np.int64), axis=axis)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Cast")
def execute_cast(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Cast operator - converts tensor to specified type.

    ONNX Spec: output = cast(input, to_type)

    Attributes:
        to: Target ONNX data type (integer code)
    """
    x = ctx.get_tensor(inputs[0])

    to_type = attrs.get("to", 1)  # Default to FLOAT

    if to_type in ONNX_DTYPE_MAP:
        target_dtype = ONNX_DTYPE_MAP[to_type]
    else:
        target_dtype = np.float32

    result = x.astype(target_dtype)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Split")
def execute_split(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Split operator - splits tensor into multiple parts.

    ONNX Spec: outputs = split(input, axis, split_sizes)

    Used for: Splitting Q, K, V projections

    Attributes:
        axis: Axis to split on (default: 0)
        num_outputs: Number of outputs (opset 13+)
    """
    x = ctx.get_tensor(inputs[0])

    axis = attrs.get("axis", 0)

    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    # Get split sizes from input or attribute
    if len(inputs) > 1:
        split_sizes = ctx.get_tensor(inputs[1]).tolist()
    else:
        split_sizes = attrs.get("split", None)

    if split_sizes is None:
        # Split equally
        num_outputs = len(outputs)
        split_size = x.shape[axis] // num_outputs
        split_sizes = [split_size] * num_outputs

    # Compute split indices
    indices = np.cumsum(split_sizes[:-1]).tolist()

    # Split the tensor
    splits = np.split(x, indices, axis=axis)

    # Store outputs
    for i, split in enumerate(splits):
        if i < len(outputs):
            ctx.set_tensor(outputs[i], split)


@OperatorRegistry.register("Concat")
def execute_concat(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Concat operator - concatenates tensors along axis.

    ONNX Spec: output = concat(inputs, axis)

    Used for: Merging attention heads

    Attributes:
        axis: Axis to concatenate on (required)
    """
    tensors = [ctx.get_tensor(inp) for inp in inputs]

    axis = attrs.get("axis", 0)

    result = np.concatenate(tensors, axis=axis)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Slice")
def execute_slice(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Slice operator - extracts sub-tensor.

    ONNX Spec (opset 10+): Uses inputs for starts, ends, axes, steps
    """
    data = ctx.get_tensor(inputs[0])

    # Get slice parameters from inputs
    starts = (
        ctx.get_tensor(inputs[1]).tolist()
        if len(inputs) > 1
        else attrs.get("starts", [])
    )
    ends = (
        ctx.get_tensor(inputs[2]).tolist() if len(inputs) > 2 else attrs.get("ends", [])
    )
    axes = (
        ctx.get_tensor(inputs[3]).tolist()
        if len(inputs) > 3
        else attrs.get("axes", list(range(len(starts))))
    )
    steps = (
        ctx.get_tensor(inputs[4]).tolist()
        if len(inputs) > 4
        else attrs.get("steps", [1] * len(starts))
    )

    # Build slice tuple
    slices = [slice(None)] * data.ndim

    for start, end, axis, step in zip(starts, ends, axes, steps):
        if axis < 0:
            axis = data.ndim + axis

        # Handle special end values
        if isinstance(end, (int, np.integer)):
            if end > data.shape[axis]:
                end = data.shape[axis]
            elif end < -data.shape[axis]:
                end = None

        slices[axis] = slice(
            int(start), int(end) if end is not None else None, int(step)
        )

    result = data[tuple(slices)]
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Pad")
def execute_pad(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Pad operator - pads tensor with specified values.
    """
    data = ctx.get_tensor(inputs[0])
    pads = (
        ctx.get_tensor(inputs[1]).tolist() if len(inputs) > 1 else attrs.get("pads", [])
    )

    # Get constant value
    constant_value = 0.0
    if len(inputs) > 2:
        constant_value = float(ctx.get_tensor(inputs[2]).item())

    mode = attrs.get("mode", "constant")

    # Convert pads format: ONNX uses [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    # NumPy uses [(x1_begin, x1_end), (x2_begin, x2_end), ...]
    ndim = data.ndim
    pad_width = [(int(pads[i]), int(pads[i + ndim])) for i in range(ndim)]

    if mode == "constant":
        result = np.pad(
            data, pad_width, mode="constant", constant_values=constant_value
        )
    elif mode == "reflect":
        result = np.pad(data, pad_width, mode="reflect")
    elif mode == "edge":
        result = np.pad(data, pad_width, mode="edge")
    else:
        result = np.pad(
            data, pad_width, mode="constant", constant_values=constant_value
        )

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Tile")
def execute_tile(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Tile operator - repeats tensor according to repeats.
    """
    data = ctx.get_tensor(inputs[0])
    repeats = ctx.get_tensor(inputs[1]).astype(np.int64).tolist()

    result = np.tile(data, repeats)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Expand")
def execute_expand(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Expand operator - broadcasts tensor to specified shape.
    """
    data = ctx.get_tensor(inputs[0])
    shape = ctx.get_tensor(inputs[1]).astype(np.int64).tolist()

    result = np.broadcast_to(data, shape)
    ctx.set_tensor(outputs[0], np.array(result))  # Copy to make writable


@OperatorRegistry.register("Where")
def execute_where(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Where operator - conditional selection.

    ONNX Spec: output = condition ? X : Y
    """
    condition = ctx.get_tensor(inputs[0])
    x = ctx.get_tensor(inputs[1])
    y = ctx.get_tensor(inputs[2])

    result = np.where(condition, x, y)
    ctx.set_tensor(outputs[0], result)
