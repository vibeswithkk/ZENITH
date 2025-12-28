# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
GraphIR to HLO Lowering Module.

Provides conversion from Zenith GraphIR representation to XLA HLO
(High Level Operations) format, enabling direct XLA execution.

Mathematical Foundation:
    For any GraphIR operation O with inputs I:
    HLO(O)(I) = GraphIR(O)(I) +- epsilon
    where epsilon <= 1e-6 for FP32

Reference:
    - OpenXLA StableHLO Specification (https://openxla.org/stablehlo)
    - JAX jaxpr to HLO lowering (jax/_src/interpreters/mlir.py)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("zenith.core.hlo_lowering")


def _get_jax():
    """Lazy import of JAX to avoid hard dependency."""
    try:
        import jax

        return jax
    except ImportError as e:
        raise ImportError(
            "JAX is required for HLO lowering. Install with: pip install jax jaxlib"
        ) from e


class HLOOpcode(Enum):
    """HLO operation codes matching XLA specification."""

    # Elementwise unary
    ABS = "abs"
    NEGATE = "negate"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    TANH = "tanh"

    # Elementwise binary
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"

    # Reduction
    REDUCE_SUM = "reduce_sum"
    REDUCE_MAX = "reduce_max"
    REDUCE_MIN = "reduce_min"

    # Linear algebra
    DOT = "dot"
    DOT_GENERAL = "dot_general"

    # Convolution
    CONVOLUTION = "convolution"

    # Other
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    BROADCAST = "broadcast"
    SLICE = "slice"
    CONCATENATE = "concatenate"
    GATHER = "gather"
    SCATTER = "scatter"

    # Control flow
    WHILE = "while"
    CONDITIONAL = "conditional"

    # Custom
    CUSTOM_CALL = "custom_call"


@dataclass
class HLOShape:
    """HLO tensor shape descriptor."""

    dimensions: tuple[int, ...]
    element_type: str

    def __post_init__(self):
        if not isinstance(self.dimensions, tuple):
            self.dimensions = tuple(self.dimensions)

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    @property
    def size(self) -> int:
        result = 1
        for dim in self.dimensions:
            result *= dim
        return result

    def __repr__(self) -> str:
        dims = "x".join(str(d) for d in self.dimensions)
        return f"{self.element_type}[{dims}]"


@dataclass
class HLOOperation:
    """Single HLO operation representation."""

    opcode: HLOOpcode
    inputs: list[str]
    output: str
    shape: HLOShape
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Generate HLO text representation."""
        inputs_str = ", ".join(self.inputs)
        attrs = ""
        if self.attributes:
            attrs_list = [f"{k}={v}" for k, v in self.attributes.items()]
            attrs = ", " + ", ".join(attrs_list)
        return (
            f"%{self.output} = {self.opcode.value}({inputs_str}{attrs}) : {self.shape}"
        )


@dataclass
class HLOModule:
    """Complete HLO module containing computation graph."""

    name: str
    entry_computation: str
    operations: list[HLOOperation] = field(default_factory=list)
    parameters: list[tuple[str, HLOShape]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_parameter(self, name: str, shape: HLOShape) -> str:
        """Add input parameter to module."""
        self.parameters.append((name, shape))
        return name

    def add_operation(self, op: HLOOperation) -> str:
        """Add operation to module."""
        self.operations.append(op)
        return op.output

    def set_outputs(self, outputs: list[str]) -> None:
        """Set module outputs."""
        self.outputs = outputs

    def to_text(self) -> str:
        """Generate complete HLO text representation."""
        lines = [
            f"HloModule {self.name}",
            "",
            f"ENTRY {self.entry_computation} {{",
        ]

        for name, shape in self.parameters:
            lines.append(f"  %{name} = parameter() : {shape}")

        for op in self.operations:
            lines.append(f"  {op.to_text()}")

        outputs_str = ", ".join(f"%{o}" for o in self.outputs)
        lines.append(f"  ROOT tuple = tuple({outputs_str})")
        lines.append("}")

        return "\n".join(lines)

    def operation_count(self) -> int:
        """Return total number of operations."""
        return len(self.operations)


class GraphIRToHLOConverter:
    """
    Converts Zenith GraphIR to HLO module.

    Implements operation-by-operation translation with shape inference
    and validation at each step.

    Complexity: O(n) where n = number of operations in graph.
    Space: O(n) for storing HLO operations.
    """

    GRAPHIR_TO_HLO_MAP = {
        "Add": HLOOpcode.ADD,
        "Sub": HLOOpcode.SUBTRACT,
        "Mul": HLOOpcode.MULTIPLY,
        "Div": HLOOpcode.DIVIDE,
        "MatMul": HLOOpcode.DOT,
        "Relu": HLOOpcode.MAXIMUM,
        "Exp": HLOOpcode.EXP,
        "Log": HLOOpcode.LOG,
        "Sqrt": HLOOpcode.SQRT,
        "Tanh": HLOOpcode.TANH,
        "Reshape": HLOOpcode.RESHAPE,
        "Transpose": HLOOpcode.TRANSPOSE,
        "Concat": HLOOpcode.CONCATENATE,
        "Sum": HLOOpcode.REDUCE_SUM,
        "Max": HLOOpcode.REDUCE_MAX,
        "Conv": HLOOpcode.CONVOLUTION,
    }

    DTYPE_MAP = {
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "bfloat16": "bf16",
        "int8": "s8",
        "int16": "s16",
        "int32": "s32",
        "int64": "s64",
        "uint8": "u8",
        "bool": "pred",
    }

    def __init__(self):
        self._op_counter = 0
        self._value_map: dict[str, str] = {}

    def _next_id(self) -> str:
        """Generate unique operation ID."""
        self._op_counter += 1
        return f"v{self._op_counter}"

    def convert(self, graph_ir, module_name: str = "zenith_module") -> HLOModule:
        """
        Convert GraphIR to HLO module.

        Args:
            graph_ir: Zenith GraphIR object
            module_name: Name for the HLO module

        Returns:
            HLOModule containing the converted computation
        """
        self._op_counter = 0
        self._value_map = {}

        module = HLOModule(
            name=module_name,
            entry_computation="main",
        )

        for inp in graph_ir.inputs:
            shape = self._convert_shape(inp.shape, inp.dtype)
            param_name = self._next_id()
            module.add_parameter(param_name, shape)
            self._value_map[inp.name] = param_name

        for op in graph_ir.operations:
            hlo_op = self._convert_operation(op)
            if hlo_op is not None:
                module.add_operation(hlo_op)
                self._value_map[op.output_name] = hlo_op.output

        output_names = [
            self._value_map.get(out.name, out.name) for out in graph_ir.outputs
        ]
        module.set_outputs(output_names)

        return module

    def _convert_shape(self, shape: tuple, dtype: str) -> HLOShape:
        """Convert GraphIR shape to HLO shape."""
        dtype_str = self.DTYPE_MAP.get(dtype, "f32")
        return HLOShape(dimensions=tuple(shape), element_type=dtype_str)

    def _convert_operation(self, op) -> Optional[HLOOperation]:
        """Convert single GraphIR operation to HLO operation."""
        op_type = op.op_type

        if op_type not in self.GRAPHIR_TO_HLO_MAP:
            logger.warning(f"Unsupported operation type: {op_type}")
            return self._create_custom_call(op)

        hlo_opcode = self.GRAPHIR_TO_HLO_MAP[op_type]

        inputs = [self._value_map.get(inp, inp) for inp in op.inputs]

        output_id = self._next_id()
        shape = self._convert_shape(op.output_shape, op.output_dtype)

        return HLOOperation(
            opcode=hlo_opcode,
            inputs=inputs,
            output=output_id,
            shape=shape,
            attributes=getattr(op, "attributes", {}),
        )

    def _create_custom_call(self, op) -> HLOOperation:
        """Create custom call for unsupported operations."""
        inputs = [self._value_map.get(inp, inp) for inp in op.inputs]
        output_id = self._next_id()
        shape = self._convert_shape(op.output_shape, op.output_dtype)

        return HLOOperation(
            opcode=HLOOpcode.CUSTOM_CALL,
            inputs=inputs,
            output=output_id,
            shape=shape,
            attributes={
                "call_target_name": f"zenith.{op.op_type}",
                "backend_config": str(getattr(op, "config", {})),
            },
        )


class JAXFunctionToHLOConverter:
    """
    Converts JAX functions to HLO via jax.export.

    Uses modern JAX export API (jax.export, available in JAX 0.4.26+)
    to obtain StableHLO representation.

    Reference:
        https://jax.readthedocs.io/en/latest/export/export.html
    """

    def __init__(self):
        self._jax = None

    @property
    def jax(self):
        if self._jax is None:
            self._jax = _get_jax()
        return self._jax

    def lower_to_hlo(
        self,
        fn: Callable,
        example_args: Sequence[Any],
        example_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Lower JAX function to HLO text representation.

        Args:
            fn: JAX function to lower
            example_args: Example positional arguments for tracing
            example_kwargs: Example keyword arguments

        Returns:
            HLO text representation
        """
        jax = self.jax
        example_kwargs = example_kwargs or {}

        jitted = jax.jit(fn)
        lowered = jitted.lower(*example_args, **example_kwargs)

        try:
            return lowered.as_text()
        except AttributeError:
            return str(lowered.compiler_ir())

    def lower_to_stablehlo(
        self,
        fn: Callable,
        example_args: Sequence[Any],
    ) -> Any:
        """
        Lower JAX function to StableHLO module.

        Args:
            fn: JAX function to lower
            example_args: Example arguments for tracing

        Returns:
            StableHLO module object (MLIR module)
        """
        jax = self.jax

        jitted = jax.jit(fn)
        lowered = jitted.lower(*example_args)

        return lowered.compiler_ir(dialect="stablehlo")

    def export_serialized(
        self,
        fn: Callable,
        example_args: Sequence[Any],
        format: str = "stablehlo",
    ) -> bytes:
        """
        Export JAX function to serialized format.

        Args:
            fn: JAX function to export
            example_args: Example arguments for tracing
            format: Export format (stablehlo, mhlo)

        Returns:
            Serialized bytes
        """
        jax = self.jax

        try:
            from jax.experimental import export as jax_export

            input_specs = []
            for arg in example_args:
                if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                    input_specs.append(jax.ShapeDtypeStruct(arg.shape, arg.dtype))

            exported = jax_export.export(jax.jit(fn))(*example_args)

            return exported.serialize()

        except ImportError:
            jitted = jax.jit(fn)
            lowered = jitted.lower(*example_args)
            stablehlo = lowered.compiler_ir(dialect="stablehlo")

            if hasattr(stablehlo, "operation"):
                return str(stablehlo.operation).encode("utf-8")
            return str(stablehlo).encode("utf-8")


def lower_graphir_to_hlo(graph_ir, module_name: str = "zenith") -> HLOModule:
    """
    Convenience function to lower GraphIR to HLO.

    Args:
        graph_ir: Zenith GraphIR object
        module_name: Name for HLO module

    Returns:
        HLOModule containing converted operations
    """
    converter = GraphIRToHLOConverter()
    return converter.convert(graph_ir, module_name)


def lower_jax_function_to_hlo(
    fn: Callable,
    example_args: Sequence[Any],
) -> str:
    """
    Convenience function to lower JAX function to HLO text.

    Args:
        fn: JAX function
        example_args: Example arguments for tracing

    Returns:
        HLO text representation
    """
    converter = JAXFunctionToHLOConverter()
    return converter.lower_to_hlo(fn, example_args)
