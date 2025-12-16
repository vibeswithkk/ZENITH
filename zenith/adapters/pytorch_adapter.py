# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
PyTorch Adapter

Converts PyTorch models (nn.Module) to Zenith's GraphIR format,
primarily through ONNX export as the intermediate representation.
"""

from typing import Any
import io

from .base import BaseAdapter
from .onnx_adapter import ONNXAdapter
from ..core import GraphIR, TensorDescriptor, DataType


class PyTorchAdapter(BaseAdapter):
    """
    Adapter for PyTorch models.

    Converts PyTorch nn.Module to GraphIR using ONNX as intermediate format.

    Example:
        adapter = PyTorchAdapter()
        model = torch.nn.Linear(10, 5)
        sample = torch.randn(1, 10)
        graph = adapter.from_model(model, sample)
    """

    def __init__(self):
        self._torch = None
        self._onnx_adapter = None

    @property
    def name(self) -> str:
        return "pytorch"

    @property
    def is_available(self) -> bool:
        try:
            import torch

            return True
        except ImportError:
            return False

    def _get_torch(self):
        """Lazy import torch."""
        if self._torch is None:
            try:
                import torch

                self._torch = torch
            except ImportError:
                raise ImportError(
                    "PyTorch is required for PyTorchAdapter. "
                    "Install it with: pip install torch"
                )
        return self._torch

    def _get_onnx_adapter(self) -> ONNXAdapter:
        """Get ONNX adapter instance."""
        if self._onnx_adapter is None:
            self._onnx_adapter = ONNXAdapter()
        return self._onnx_adapter

    def from_model(
        self,
        model: Any,
        sample_input: Any = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a PyTorch nn.Module to GraphIR.

        Args:
            model: PyTorch nn.Module to convert.
            sample_input: Sample input tensor for tracing.
            input_names: Names for input tensors.
            output_names: Names for output tensors.
            dynamic_axes: Dynamic axis specification.
            opset_version: ONNX opset version to use.
            **kwargs: Additional options passed to torch.onnx.export.

        Returns:
            GraphIR representation of the model.
        """
        torch = self._get_torch()

        if sample_input is None:
            raise ValueError(
                "sample_input is required for PyTorch model conversion. "
                "Provide a sample input tensor for tracing."
            )

        # Set default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Export to ONNX in memory
        onnx_bytes = self.to_onnx(
            model,
            sample_input,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **kwargs,
        )

        # Convert ONNX to GraphIR
        onnx_adapter = self._get_onnx_adapter()
        return onnx_adapter.from_bytes(onnx_bytes)

    def from_torch_export(
        self,
        model: Any,
        sample_input: Any,
        dynamic_shapes: dict | None = None,
        strict: bool = True,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a PyTorch model using torch.export (PyTorch 2.x).

        This method uses the newer torch.export API which provides:
        - Full graph capture with static single-graph representation
        - AOT compilation support
        - Better support for dynamic shapes

        Args:
            model: PyTorch nn.Module to convert.
            sample_input: Sample input tensor(s) for tracing.
            dynamic_shapes: Dynamic shape specifications.
            strict: If True, require full graph capture without fallbacks.
            **kwargs: Additional options passed to torch.export.

        Returns:
            GraphIR representation of the model.

        Raises:
            ImportError: If PyTorch version < 2.0
            RuntimeError: If graph capture fails
        """
        torch = self._get_torch()

        # Check PyTorch version for torch.export availability
        major_version = int(torch.__version__.split(".")[0])
        if major_version < 2:
            raise ImportError(
                "torch.export requires PyTorch 2.0 or higher. "
                f"Current version: {torch.__version__}. "
                "Use from_model() with ONNX export instead."
            )

        # Put model in eval mode
        model.eval()

        # Use torch.export
        with torch.no_grad():
            try:
                # Try using torch.export (PyTorch 2.1+)
                exported = torch.export.export(
                    model,
                    (sample_input,),
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )

                # Convert ExportedProgram to ONNX
                onnx_bytes = self._exported_program_to_onnx(exported, sample_input)
                onnx_adapter = self._get_onnx_adapter()
                return onnx_adapter.from_bytes(onnx_bytes)

            except AttributeError:
                # Fallback for PyTorch 2.0 without torch.export.export
                return self.from_model(model, sample_input, **kwargs)

    def _exported_program_to_onnx(
        self,
        exported_program: Any,
        sample_input: Any,
    ) -> bytes:
        """Convert ExportedProgram to ONNX bytes."""
        torch = self._get_torch()

        # Get the module from ExportedProgram
        module = exported_program.module()

        # Export via standard ONNX path
        return self.to_onnx(module, sample_input)

    def to_onnx(
        self,
        model: Any,
        sample_input: Any,
        output_path: str | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
        **kwargs,
    ) -> bytes:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch nn.Module.
            sample_input: Sample input for tracing.
            output_path: Optional path to save ONNX file.
            input_names: Names for inputs.
            output_names: Names for outputs.
            dynamic_axes: Dynamic axis specification.
            opset_version: ONNX opset version.

        Returns:
            ONNX model as bytes.
        """
        torch = self._get_torch()

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Put model in eval mode
        model.eval()

        # Export to buffer
        buffer = io.BytesIO()

        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                buffer,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                **kwargs,
            )

        onnx_bytes = buffer.getvalue()

        # Save to file if requested
        if output_path:
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)

        return onnx_bytes

    def get_input_shapes(self, model: Any) -> list[TensorDescriptor]:
        """Get input shapes - requires running inference to determine."""
        # This is a placeholder - full implementation would trace the model
        raise NotImplementedError(
            "get_input_shapes requires sample input for PyTorch models. "
            "Use from_model() with sample_input instead."
        )

    def _torch_dtype_to_zenith(self, torch_dtype) -> DataType:
        """Convert PyTorch dtype to Zenith DataType."""
        torch = self._get_torch()

        mapping = {
            torch.float32: DataType.Float32,
            torch.float16: DataType.Float16,
            torch.bfloat16: DataType.BFloat16,
            torch.float64: DataType.Float64,
            torch.int8: DataType.Int8,
            torch.int16: DataType.Int16,
            torch.int32: DataType.Int32,
            torch.int64: DataType.Int64,
            torch.uint8: DataType.UInt8,
            torch.bool: DataType.Bool,
        }

        return mapping.get(torch_dtype, DataType.Float32)
