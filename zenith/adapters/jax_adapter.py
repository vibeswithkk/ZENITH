# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
JAX Adapter

Converts JAX functions and Flax/Haiku models to Zenith's GraphIR format.
"""

from typing import Any, Callable

from .base import BaseAdapter
from .onnx_adapter import ONNXAdapter
from ..core import GraphIR, TensorDescriptor, Shape, DataType


class JAXAdapter(BaseAdapter):
    """
    Adapter for JAX functions and Flax models.

    Converts JAX functions to GraphIR using jax2onnx or JAX's export utilities.

    Example:
        adapter = JAXAdapter()

        def forward(x):
            return jax.numpy.dot(x, x.T)

        graph = adapter.from_model(forward, sample_input=jnp.ones((4, 4)))
    """

    def __init__(self):
        self._jax = None
        self._onnx_adapter = None

    @property
    def name(self) -> str:
        return "jax"

    @property
    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("jax") is not None

    def _get_jax(self):
        """Lazy import jax."""
        if self._jax is None:
            try:
                import jax

                self._jax = jax
            except ImportError as err:
                raise ImportError(
                    "JAX is required for JAXAdapter. "
                    "Install it with: pip install jax jaxlib"
                ) from err
        return self._jax

    def _get_onnx_adapter(self) -> ONNXAdapter:
        """Get ONNX adapter instance."""
        if self._onnx_adapter is None:
            self._onnx_adapter = ONNXAdapter()
        return self._onnx_adapter

    def from_model(
        self,
        model: Any,
        sample_input: Any = None,
        params: Any = None,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a JAX function or Flax model to GraphIR.

        Args:
            model: JAX function, Flax module, or Haiku transformed function.
            sample_input: Sample input array for tracing.
            params: Model parameters (for Flax/Haiku models).
            **kwargs: Additional options.

        Returns:
            GraphIR representation of the model.
        """
        jax = self._get_jax()

        if sample_input is None:
            raise ValueError(
                "sample_input is required for JAX model conversion. "
                "Provide a sample JAX array for tracing."
            )

        # Try to export via ONNX
        try:
            onnx_bytes = self.to_onnx(model, sample_input, params=params, **kwargs)
            onnx_adapter = self._get_onnx_adapter()
            return onnx_adapter.from_bytes(onnx_bytes)
        except Exception:
            # Fallback: trace the function and build GraphIR manually
            return self._trace_to_graphir(model, sample_input, params)

    def from_stablehlo(
        self,
        model: Any,
        sample_input: Any,
        params: Any = None,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a JAX function using StableHLO export.

        StableHLO is the recommended native export format for JAX with:
        - 5-year backward compatibility guarantee
        - Portable representation for ML compilers
        - Integration with IREE, XLA, and other backends

        Args:
            model: JAX function or Flax module.
            sample_input: Sample input for tracing.
            params: Model parameters (for Flax/Haiku models).
            **kwargs: Additional options.

        Returns:
            GraphIR representation of the model.

        Raises:
            ImportError: If jax.export is not available.
        """
        jax = self._get_jax()

        # Create the function to export
        if params is not None:

            def fn(x):
                return model.apply(params, x)
        elif callable(model):
            fn = model
        else:
            raise ValueError(
                f"Unsupported model type: {type(model)}. "
                "Expected JAX function or Flax module."
            )

        # JIT compile first (required for export)
        jitted_fn = jax.jit(fn)

        # Try to use jax.export (JAX 0.4.14+)
        try:
            exported = jax.export.export(jitted_fn)(sample_input)

            # Build GraphIR from StableHLO
            return self._stablehlo_to_graphir(exported, sample_input)

        except AttributeError:
            # Fallback for older JAX versions
            return self._trace_to_graphir(model, sample_input, params)

    def _stablehlo_to_graphir(
        self,
        exported: Any,
        sample_input: Any,
    ) -> GraphIR:
        """Convert jax.export.Exported to GraphIR."""
        jax = self._get_jax()
        import jax.numpy as jnp

        # Get StableHLO text representation
        try:
            stablehlo_text = exported.mlir_module()
        except Exception:
            stablehlo_text = str(exported)

        # Build GraphIR
        graph = GraphIR(name="jax_stablehlo_model")

        # Add input
        input_shape = list(sample_input.shape)
        graph.add_input(
            TensorDescriptor(
                name="input",
                shape=Shape(input_shape),
                dtype=self._jax_dtype_to_zenith(sample_input.dtype),
            )
        )

        # Run to get output shape
        output = exported.call(sample_input)
        is_array = hasattr(output, "shape")
        output_shape = list(output.shape) if is_array else [1]
        output_dtype = (
            self._jax_dtype_to_zenith(output.dtype)
            if hasattr(output, "dtype")
            else DataType.Float32
        )

        graph.add_output(
            TensorDescriptor(
                name="output",
                shape=Shape(output_shape),
                dtype=output_dtype,
            )
        )

        # Add StableHLO node
        graph.add_node(
            op_type="StableHLO",
            name="stablehlo_computation",
            inputs=[graph.inputs[0]],
            outputs=[graph.outputs[0]],
            attrs={
                "stablehlo_version": getattr(
                    exported, "calling_convention_version", "unknown"
                ),
                "module_summary": stablehlo_text[:500]
                if len(stablehlo_text) > 500
                else stablehlo_text,
            },
        )

        return graph

    def to_onnx(
        self,
        model: Any,
        sample_input: Any,
        output_path: str = None,
        params: Any = None,
        **kwargs,
    ) -> bytes:
        """
        Export JAX function to ONNX format.

        Uses jax2onnx if available.

        Args:
            model: JAX function or Flax model.
            sample_input: Sample input for tracing.
            output_path: Optional path to save ONNX file.
            params: Model parameters.

        Returns:
            ONNX model as bytes.
        """
        try:
            import jax2onnx
        except ImportError as err:
            raise ImportError(
                "jax2onnx is required for JAX to ONNX conversion. "
                "Install it with: pip install jax2onnx"
            ) from err

        jax = self._get_jax()

        # Determine if it's a pure function or has params
        if params is not None:
            # Flax/Haiku model - wrap with params
            def fn(x):
                return model.apply(params, x)
        elif callable(model):
            fn = model
        else:
            raise ValueError(
                f"Unsupported model type: {type(model)}. "
                "Expected JAX function or Flax module."
            )

        # Convert to ONNX
        onnx_model = jax2onnx.convert(fn, sample_input)
        onnx_bytes = onnx_model.SerializeToString()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)

        return onnx_bytes

    def _trace_to_graphir(
        self, fn: Callable, sample_input: Any, params: Any = None
    ) -> GraphIR:
        """
        Trace a JAX function and build GraphIR manually.

        This is a fallback when ONNX export is not available.
        """
        jax = self._get_jax()

        # Create wrapper if params needed
        if params is not None:

            def wrapped(x):
                return fn.apply(params, x)

            fn = wrapped

        # JIT compile to get the HLO
        jitted = jax.jit(fn)
        lowered = jitted.lower(sample_input)

        # Get the HLO text for analysis
        hlo_text = lowered.as_text()

        # Build a basic GraphIR from the traced computation
        # This is a simplified implementation - full version would parse HLO
        graph = GraphIR(name="jax_traced_model")

        # Add placeholder input/output
        import jax.numpy as jnp

        input_shape = list(sample_input.shape)
        graph.add_input(
            TensorDescriptor(
                name="input",
                shape=Shape(input_shape),
                dtype=self._jax_dtype_to_zenith(sample_input.dtype),
            )
        )

        # Run to get output shape
        output = jitted(sample_input)
        output_shape = list(output.shape) if hasattr(output, "shape") else [1]
        graph.add_output(
            TensorDescriptor(
                name="output",
                shape=Shape(output_shape),
                dtype=self._jax_dtype_to_zenith(output.dtype)
                if hasattr(output, "dtype")
                else DataType.Float32,
            )
        )

        # Add a placeholder node
        graph.add_node(
            op_type="JaxTraced",
            name="traced_computation",
            inputs=[graph.inputs[0]],
            outputs=[graph.outputs[0]],
            attrs={
                "hlo_summary": (hlo_text[:500] if len(hlo_text) > 500 else hlo_text)
            },
        )

        return graph

    def _jax_dtype_to_zenith(self, jax_dtype) -> DataType:
        """Convert JAX dtype to Zenith DataType."""
        import jax.numpy as jnp

        mapping = {
            jnp.float32: DataType.Float32,
            jnp.float16: DataType.Float16,
            jnp.bfloat16: DataType.BFloat16,
            jnp.float64: DataType.Float64,
            jnp.int8: DataType.Int8,
            jnp.int16: DataType.Int16,
            jnp.int32: DataType.Int32,
            jnp.int64: DataType.Int64,
            jnp.uint8: DataType.UInt8,
            jnp.bool_: DataType.Bool,
        }

        for jax_type, zenith_type in mapping.items():
            if jax_dtype == jax_type:
                return zenith_type

        return DataType.Float32
