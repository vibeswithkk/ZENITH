# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Shape Manipulation Operators

Implements ONNX shape operators:
- Reshape: Reshape tensor
- Transpose: Permute dimensions
- Flatten: Flatten to 2D
- Squeeze: Remove dimensions of size 1
- Unsqueeze: Add dimensions of size 1
- Concat: Concatenate tensors
- Split: Split tensor
- Gather: Gather elements
- Slice: Slice tensor
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING
import numpy as np

from ..registry import OperatorRegistry

if TYPE_CHECKING:
    from ..context import ExecutionContext


@OperatorRegistry.register("Reshape", tier=2)
def execute_reshape(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Reshape operator.

    ONNX Spec: Y = Reshape(data, shape)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    # Shape can come from input or attributes
    if len(inputs) > 1:
        shape = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64)
        shape = tuple(shape.tolist())
    else:
        shape = tuple(attrs.get("shape", data.shape))

    # Handle -1 dimension
    result = data.reshape(shape)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Transpose", tier=2)
def execute_transpose(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Transpose operator.

    ONNX Spec: Y = Transpose(data, perm)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    perm = attrs.get("perm", None)

    if perm is not None:
        result = np.transpose(data, perm)
    else:
        result = np.transpose(data)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Flatten", tier=2)
def execute_flatten(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Flatten operator.

    ONNX Spec: Y = Flatten(data, axis)
    Flattens the input tensor into a 2D matrix.
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    axis = attrs.get("axis", 1)

    # Compute new shape
    shape = data.shape
    dim0 = int(np.prod(shape[:axis]))
    dim1 = int(np.prod(shape[axis:]))

    result = data.reshape(dim0, dim1)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Squeeze", tier=2)
def execute_squeeze(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Squeeze operator - remove dimensions of size 1.

    ONNX Spec: Y = Squeeze(data, axes)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    # Get axes from input or attributes
    if len(inputs) > 1:
        axes = tuple(np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64).tolist())
    else:
        axes = attrs.get("axes", None)

    if axes is not None:
        result = np.squeeze(data, axis=tuple(axes))
    else:
        result = np.squeeze(data)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Unsqueeze", tier=2)
def execute_unsqueeze(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Unsqueeze operator - insert dimensions of size 1.

    ONNX Spec: Y = Unsqueeze(data, axes)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    # Get axes from input or attributes
    if len(inputs) > 1:
        axes = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64).tolist()
    else:
        axes = attrs.get("axes", [0])

    result = data
    for axis in sorted(axes):
        result = np.expand_dims(result, axis=axis)

    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Concat", tier=2)
def execute_concat(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Concatenate operator.

    ONNX Spec: Y = Concat(inputs, axis)
    """
    axis = attrs.get("axis", 0)

    tensors = []
    for inp in inputs:
        tensors.append(np.asarray(ctx.get_tensor(inp), dtype=np.float32))

    result = np.concatenate(tensors, axis=axis)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Split", tier=3)
def execute_split(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Split operator.

    ONNX Spec: outputs = Split(input, split, axis)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    axis = attrs.get("axis", 0)

    # Get split sizes
    if len(inputs) > 1:
        split = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64).tolist()
    else:
        split = attrs.get("split", None)

    if split is not None:
        # Split at specified indices
        indices = np.cumsum(split)[:-1].tolist()
        results = np.split(data, indices, axis=axis)
    else:
        # Split into equal parts
        results = np.split(data, len(outputs), axis=axis)

    for i, result in enumerate(results):
        if i < len(outputs):
            ctx.set_tensor(outputs[i], result)


@OperatorRegistry.register("Gather", tier=3)
def execute_gather(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Gather operator - gather elements using indices.

    ONNX Spec: Y = Gather(data, indices, axis)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)
    indices = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64)
    axis = attrs.get("axis", 0)

    result = np.take(data, indices, axis=axis)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Slice", tier=2)
def execute_slice(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """
    Slice operator.

    ONNX Spec: Y = Slice(data, starts, ends, axes, steps)
    """
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    # Get slice parameters from inputs or attributes
    if len(inputs) > 1:
        starts = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64).tolist()
        ends = np.asarray(ctx.get_tensor(inputs[2]), dtype=np.int64).tolist()
        axes = None
        steps = None
        if len(inputs) > 3:
            axes = np.asarray(ctx.get_tensor(inputs[3]), dtype=np.int64).tolist()
        if len(inputs) > 4:
            steps = np.asarray(ctx.get_tensor(inputs[4]), dtype=np.int64).tolist()
    else:
        starts = attrs.get("starts", [0])
        ends = attrs.get("ends", [data.shape[0]])
        axes = attrs.get("axes", None)
        steps = attrs.get("steps", None)

    # Build slice objects
    slices = [slice(None)] * data.ndim

    if axes is None:
        axes = list(range(len(starts)))

    for i, axis in enumerate(axes):
        start = starts[i] if i < len(starts) else 0
        end = ends[i] if i < len(ends) else data.shape[axis]
        step = steps[i] if steps and i < len(steps) else 1
        slices[axis] = slice(start, end, step)

    result = data[tuple(slices)]
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Shape", tier=2)
def execute_shape(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Shape operator - returns shape of input tensor."""
    data = np.asarray(ctx.get_tensor(inputs[0]))
    result = np.array(data.shape, dtype=np.int64)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Cast", tier=2)
def execute_cast(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Cast operator - convert tensor to specified type."""
    data = np.asarray(ctx.get_tensor(inputs[0]))
    to_type = attrs.get("to", 1)  # 1 = FLOAT

    # ONNX type mapping (simplified)
    type_map = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        6: np.int32,
        7: np.int64,
        10: np.float16,
        11: np.float64,
    }

    dtype = type_map.get(to_type, np.float32)
    result = data.astype(dtype)
    ctx.set_tensor(outputs[0], result)


@OperatorRegistry.register("Pad", tier=2)
def execute_pad(
    ctx: "ExecutionContext",
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
) -> None:
    """Pad operator - pad tensor with constant value."""
    data = np.asarray(ctx.get_tensor(inputs[0]), dtype=np.float32)

    # Get pads from input or attributes
    if len(inputs) > 1:
        pads = np.asarray(ctx.get_tensor(inputs[1]), dtype=np.int64).tolist()
    else:
        pads = attrs.get("pads", [0, 0, 0, 0])

    mode = attrs.get("mode", "constant")
    constant_value = attrs.get("constant_value", 0.0)

    if len(inputs) > 2:
        constant_value = float(np.asarray(ctx.get_tensor(inputs[2])))

    # Convert ONNX pads format to numpy format
    ndim = data.ndim
    pad_width = [(pads[i], pads[i + ndim]) for i in range(ndim)]

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
