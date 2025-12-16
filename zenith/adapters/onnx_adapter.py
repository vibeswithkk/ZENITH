# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
ONNX Adapter

Core adapter for parsing ONNX models into Zenith's GraphIR format.
ONNX serves as the "lingua franca" - the common intermediate format
that all other framework adapters use.
"""

from typing import Any
from pathlib import Path

from .base import BaseAdapter
from ..core import (
    GraphIR,
    Node,
    TensorDescriptor,
    Shape,
    DataType,
    Layout,
)


class ONNXAdapter(BaseAdapter):
    """
    Adapter for ONNX models.

    This is the core adapter that converts ONNX format to GraphIR.
    All other framework adapters (PyTorch, TensorFlow, JAX) use ONNX
    as the intermediate format before converting to GraphIR.

    Example:
        adapter = ONNXAdapter()
        graph = adapter.from_file("model.onnx")
    """

    def __init__(self):
        self._onnx = None

    @property
    def name(self) -> str:
        return "onnx"

    @property
    def is_available(self) -> bool:
        try:
            import onnx

            return True
        except ImportError:
            return False

    def _get_onnx(self):
        """Lazy import onnx."""
        if self._onnx is None:
            try:
                import onnx

                self._onnx = onnx
            except ImportError as err:
                raise ImportError(
                    "ONNX is required for ONNXAdapter. "
                    "Install it with: pip install onnx"
                ) from err
        return self._onnx

    def from_model(self, model: Any, sample_input: Any = None, **kwargs) -> GraphIR:
        """
        Convert an ONNX model to GraphIR.

        Args:
            model: ONNX ModelProto, file path, or bytes.
            sample_input: Not used for ONNX (shapes are in the model).
            **kwargs: Additional options.

        Returns:
            GraphIR representation of the model.
        """
        onnx = self._get_onnx()

        if isinstance(model, bytes):
            return self.from_bytes(model)
        elif isinstance(model, (str, Path)):
            return self.from_file(str(model))
        elif hasattr(model, "SerializeToString"):
            # ONNX ModelProto
            return self._convert_model_proto(model)
        else:
            raise ValueError(
                f"Unsupported model type: {type(model)}. "
                "Expected ONNX ModelProto, file path, or bytes."
            )

    def from_file(self, path: str) -> GraphIR:
        """Load ONNX model from file and convert to GraphIR."""
        onnx = self._get_onnx()
        model = onnx.load(path)
        return self._convert_model_proto(model)

    def from_bytes(self, data: bytes) -> GraphIR:
        """Load ONNX model from bytes and convert to GraphIR."""
        onnx = self._get_onnx()
        model = onnx.load_model_from_string(data)
        return self._convert_model_proto(model)

    def _convert_model_proto(self, model) -> GraphIR:
        """
        Convert ONNX ModelProto to GraphIR.

        This is the core conversion logic that parses the ONNX graph
        and builds the equivalent GraphIR representation.
        """
        onnx = self._get_onnx()

        # Validate model
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            pass  # Continue anyway, validation is optional

        # Get graph
        onnx_graph = model.graph

        # Create GraphIR
        graph_ir = GraphIR(name=onnx_graph.name or "onnx_model")

        # Convert inputs
        for input_info in onnx_graph.input:
            # Skip initializers (they are weights, not inputs)
            if input_info.name in [init.name for init in onnx_graph.initializer]:
                continue

            tensor_desc = self._convert_value_info(input_info)
            graph_ir.add_input(tensor_desc)

        # Convert outputs
        for output_info in onnx_graph.output:
            tensor_desc = self._convert_value_info(output_info)
            graph_ir.add_output(tensor_desc)

        # Convert initializers (weights) to constants
        for initializer in onnx_graph.initializer:
            data = onnx.numpy_helper.to_array(initializer).tobytes()
            graph_ir.add_constant(initializer.name, data)

        # Convert nodes
        for onnx_node in onnx_graph.node:
            node = self._convert_node(onnx_node, onnx_graph)
            # Add node directly to internal list
            graph_ir._nodes.append(node)
            graph_ir._name_to_node[node.name] = node

        return graph_ir

    def _convert_value_info(self, value_info) -> TensorDescriptor:
        """Convert ONNX ValueInfoProto to TensorDescriptor."""
        onnx = self._get_onnx()

        name = value_info.name
        type_proto = value_info.type

        # Get tensor type
        tensor_type = type_proto.tensor_type

        # Get dtype
        elem_type = tensor_type.elem_type
        dtype = self._onnx_dtype_to_zenith(elem_type)

        # Get shape
        shape_dims = []
        if tensor_type.HasField("shape"):
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape_dims.append(dim.dim_value)
                else:
                    # Dynamic dimension
                    shape_dims.append(-1)

        return TensorDescriptor(
            name=name,
            shape=Shape(shape_dims),
            dtype=dtype,
            layout=Layout.NCHW,  # Default assumption
        )

    def _convert_node(self, onnx_node, onnx_graph) -> Node:
        """Convert ONNX NodeProto to Zenith Node."""
        # Get or generate name
        name = onnx_node.name or f"{onnx_node.op_type}_{id(onnx_node)}"

        # Build input tensors (simplified - uses names as placeholders)
        inputs = []
        for input_name in onnx_node.input:
            if input_name:  # Skip empty inputs
                inputs.append(
                    TensorDescriptor(
                        name=input_name,
                        shape=Shape([]),  # Shape will be inferred later
                        dtype=DataType.Float32,
                    )
                )

        # Build output tensors
        outputs = []
        for output_name in onnx_node.output:
            if output_name:
                outputs.append(
                    TensorDescriptor(
                        name=output_name,
                        shape=Shape([]),
                        dtype=DataType.Float32,
                    )
                )

        # Convert attributes
        attrs = {}
        for attr in onnx_node.attribute:
            attrs[attr.name] = self._convert_attribute(attr)

        return Node(
            op_type=onnx_node.op_type,
            name=name,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

    def _convert_attribute(self, attr) -> Any:
        """Convert ONNX AttributeProto to Python value."""
        onnx = self._get_onnx()

        attr_type = attr.type

        if attr_type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr_type == onnx.AttributeProto.INT:
            return attr.i
        elif attr_type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
        elif attr_type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr_type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr_type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") for s in attr.strings]
        else:
            return None

    def _onnx_dtype_to_zenith(self, onnx_dtype: int) -> DataType:
        """Convert ONNX TensorProto dtype to Zenith DataType."""
        onnx = self._get_onnx()

        mapping = {
            onnx.TensorProto.FLOAT: DataType.Float32,
            onnx.TensorProto.FLOAT16: DataType.Float16,
            onnx.TensorProto.BFLOAT16: DataType.BFloat16,
            onnx.TensorProto.DOUBLE: DataType.Float64,
            onnx.TensorProto.INT8: DataType.Int8,
            onnx.TensorProto.INT16: DataType.Int16,
            onnx.TensorProto.INT32: DataType.Int32,
            onnx.TensorProto.INT64: DataType.Int64,
            onnx.TensorProto.UINT8: DataType.UInt8,
            onnx.TensorProto.BOOL: DataType.Bool,
        }

        return mapping.get(onnx_dtype, DataType.Float32)

    def to_onnx(
        self,
        model: Any,
        sample_input: Any = None,
        output_path: str | None = None,
        **kwargs,
    ) -> bytes:
        """ONNX adapter - model is already ONNX, just serialize."""
        onnx = self._get_onnx()

        if hasattr(model, "SerializeToString"):
            return model.SerializeToString()
        elif isinstance(model, bytes):
            return model
        else:
            raise ValueError("Model must be ONNX ModelProto or bytes")
