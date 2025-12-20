"""
Model Export Utilities

Provides utilities for exporting Zenith models to various formats:
- ONNX (for Triton, TensorRT, ONNX Runtime)
- TorchScript (for PyTorch deployment)
- TensorFlow SavedModel (for TF Serving)

Based on CetakBiru Section 7 Framework Integration requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np


class ExportFormat(Enum):
    """Supported export formats."""

    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW_SAVED = "tensorflow_saved"
    TFLITE = "tflite"
    TENSORRT = "tensorrt"


@dataclass
class TensorSpec:
    """Specification for an input or output tensor."""

    name: str
    shape: list[int]
    dtype: str = "float32"
    dynamic_axes: list[int] | None = None  # Axes with dynamic size


@dataclass
class ExportConfig:
    """Configuration for model export."""

    format: ExportFormat = ExportFormat.ONNX
    opset_version: int = 17  # ONNX opset version
    optimize: bool = True
    quantize: bool = False
    quantization_mode: str = "int8"  # int8, fp16
    include_preprocessing: bool = False
    include_postprocessing: bool = False
    dynamic_batch: bool = True


@dataclass
class ExportResult:
    """Result of a model export operation."""

    success: bool
    output_path: Path | None
    format: ExportFormat
    message: str = ""
    file_size_bytes: int = 0
    metadata: dict = field(default_factory=dict)


class ZenithModelExporter:
    """
    Exports Zenith models to various deployment formats.

    Handles format conversion, optimization, and metadata generation
    for deployment to inference servers.
    """

    def __init__(self, config: ExportConfig = ExportConfig()):
        """
        Initialize model exporter.

        Args:
            config: Export configuration
        """
        self.config = config

    def export(
        self,
        model: Any,
        output_path: str | Path,
        input_specs: list[TensorSpec],
        output_specs: list[TensorSpec] | None = None,
    ) -> ExportResult:
        """
        Export a model to the configured format.

        Args:
            model: Model to export (Zenith, PyTorch, or ONNX model)
            output_path: Path for the exported model
            input_specs: Input tensor specifications
            output_specs: Output tensor specifications (optional)

        Returns:
            ExportResult with status and details
        """
        output_path = Path(output_path)

        if self.config.format == ExportFormat.ONNX:
            return self._export_onnx(model, output_path, input_specs)
        elif self.config.format == ExportFormat.TORCHSCRIPT:
            return self._export_torchscript(model, output_path, input_specs)
        elif self.config.format == ExportFormat.TENSORFLOW_SAVED:
            return self._export_tensorflow(model, output_path, input_specs)
        else:
            return ExportResult(
                success=False,
                output_path=None,
                format=self.config.format,
                message=f"Unsupported format: {self.config.format}",
            )

    def _export_onnx(
        self,
        model: Any,
        output_path: Path,
        input_specs: list[TensorSpec],
    ) -> ExportResult:
        """Export to ONNX format."""
        try:
            # Try PyTorch export if model is a PyTorch module
            if hasattr(model, "forward") and hasattr(model, "parameters"):
                return self._export_pytorch_to_onnx(model, output_path, input_specs)

            # Try direct ONNX save if model is already ONNX
            if hasattr(model, "graph"):
                import onnx

                onnx.save(model, str(output_path))
                return ExportResult(
                    success=True,
                    output_path=output_path,
                    format=ExportFormat.ONNX,
                    message="Exported ONNX model",
                    file_size_bytes=output_path.stat().st_size,
                )

            return ExportResult(
                success=False,
                output_path=None,
                format=ExportFormat.ONNX,
                message="Model type not supported for ONNX export",
            )

        except ImportError as e:
            return ExportResult(
                success=False,
                output_path=None,
                format=ExportFormat.ONNX,
                message=f"Missing dependency: {e}",
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=None,
                format=ExportFormat.ONNX,
                message=f"Export failed: {e}",
            )

    def _export_pytorch_to_onnx(
        self,
        model: Any,
        output_path: Path,
        input_specs: list[TensorSpec],
    ) -> ExportResult:
        """Export PyTorch model to ONNX."""
        try:
            import torch

            # Create dummy inputs
            dummy_inputs = []
            for spec in input_specs:
                shape = spec.shape.copy()
                if self.config.dynamic_batch and shape[0] == -1:
                    shape[0] = 1  # Use batch size 1 for tracing
                dtype = getattr(torch, spec.dtype.replace("float", "float"))
                dummy_inputs.append(torch.randn(*shape, dtype=dtype))

            if len(dummy_inputs) == 1:
                dummy_inputs = dummy_inputs[0]
            else:
                dummy_inputs = tuple(dummy_inputs)

            # Build dynamic axes
            dynamic_axes = {}
            for spec in input_specs:
                if spec.dynamic_axes:
                    dynamic_axes[spec.name] = {i: f"dim_{i}" for i in spec.dynamic_axes}
                elif self.config.dynamic_batch:
                    dynamic_axes[spec.name] = {0: "batch_size"}

            # Export
            model.eval()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                model,
                dummy_inputs,
                str(output_path),
                input_names=[s.name for s in input_specs],
                opset_version=self.config.opset_version,
                dynamic_axes=dynamic_axes if dynamic_axes else None,
                do_constant_folding=self.config.optimize,
            )

            # Optionally optimize with ONNX optimizer
            if self.config.optimize:
                self._optimize_onnx(output_path)

            return ExportResult(
                success=True,
                output_path=output_path,
                format=ExportFormat.ONNX,
                message="Exported PyTorch model to ONNX",
                file_size_bytes=output_path.stat().st_size,
                metadata={
                    "opset_version": self.config.opset_version,
                    "dynamic_batch": self.config.dynamic_batch,
                },
            )

        except Exception as e:
            return ExportResult(
                success=False,
                output_path=None,
                format=ExportFormat.ONNX,
                message=f"PyTorch to ONNX export failed: {e}",
            )

    def _optimize_onnx(self, model_path: Path) -> None:
        """Apply ONNX optimizer passes."""
        try:
            import onnx
            from onnx import optimizer

            model = onnx.load(str(model_path))

            # Apply standard optimization passes
            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "fuse_consecutive_transposes",
                "fuse_bn_into_conv",
            ]

            optimized = optimizer.optimize(model, passes)
            onnx.save(optimized, str(model_path))

        except ImportError:
            pass  # ONNX optimizer not available
        except Exception:
            pass  # Optimization failed, keep original

    def _export_torchscript(
        self,
        model: Any,
        output_path: Path,
        input_specs: list[TensorSpec],
    ) -> ExportResult:
        """Export to TorchScript format."""
        try:
            import torch

            # Create dummy inputs for tracing
            dummy_inputs = []
            for spec in input_specs:
                shape = spec.shape.copy()
                if shape[0] == -1:
                    shape[0] = 1
                dtype = getattr(torch, spec.dtype.replace("float", "float"))
                dummy_inputs.append(torch.randn(*shape, dtype=dtype))

            if len(dummy_inputs) == 1:
                dummy_inputs = dummy_inputs[0]

            model.eval()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Try tracing first
            try:
                scripted = torch.jit.trace(model, dummy_inputs)
            except Exception:
                # Fall back to scripting
                scripted = torch.jit.script(model)

            # Optimize for inference
            if self.config.optimize:
                scripted = torch.jit.optimize_for_inference(scripted)

            scripted.save(str(output_path))

            return ExportResult(
                success=True,
                output_path=output_path,
                format=ExportFormat.TORCHSCRIPT,
                message="Exported to TorchScript",
                file_size_bytes=output_path.stat().st_size,
            )

        except Exception as e:
            return ExportResult(
                success=False,
                output_path=None,
                format=ExportFormat.TORCHSCRIPT,
                message=f"TorchScript export failed: {e}",
            )

    def _export_tensorflow(
        self,
        model: Any,
        output_path: Path,
        input_specs: list[TensorSpec],
    ) -> ExportResult:
        """Export to TensorFlow SavedModel format."""
        try:
            import tensorflow as tf

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(model, tf.keras.Model):
                model.save(str(output_path), save_format="tf")
            else:
                tf.saved_model.save(model, str(output_path))

            return ExportResult(
                success=True,
                output_path=output_path,
                format=ExportFormat.TENSORFLOW_SAVED,
                message="Exported to TensorFlow SavedModel",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                output_path=None,
                format=ExportFormat.TENSORFLOW_SAVED,
                message=f"TensorFlow export failed: {e}",
            )

    def generate_metadata(
        self,
        model_name: str,
        input_specs: list[TensorSpec],
        output_specs: list[TensorSpec],
    ) -> dict:
        """
        Generate model metadata for deployment.

        Args:
            model_name: Name of the model
            input_specs: Input tensor specifications
            output_specs: Output tensor specifications

        Returns:
            Metadata dictionary
        """
        return {
            "model_name": model_name,
            "format": self.config.format.value,
            "opset_version": self.config.opset_version,
            "inputs": [
                {
                    "name": s.name,
                    "shape": s.shape,
                    "dtype": s.dtype,
                    "dynamic_axes": s.dynamic_axes,
                }
                for s in input_specs
            ],
            "outputs": [
                {
                    "name": s.name,
                    "shape": s.shape,
                    "dtype": s.dtype,
                }
                for s in output_specs
            ],
            "optimization": {
                "optimized": self.config.optimize,
                "quantized": self.config.quantize,
                "quantization_mode": (
                    self.config.quantization_mode if self.config.quantize else None
                ),
            },
        }

    def save_metadata(
        self,
        metadata: dict,
        output_path: str | Path,
    ) -> None:
        """Save metadata to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================


def export_to_onnx(
    model: Any,
    output_path: str | Path,
    input_specs: list[tuple[str, list[int], str]],
    opset_version: int = 17,
    optimize: bool = True,
) -> ExportResult:
    """
    Export a model to ONNX format.

    Args:
        model: Model to export
        output_path: Output file path
        input_specs: List of (name, shape, dtype) tuples
        opset_version: ONNX opset version
        optimize: Apply optimization passes

    Returns:
        ExportResult
    """
    config = ExportConfig(
        format=ExportFormat.ONNX,
        opset_version=opset_version,
        optimize=optimize,
    )
    exporter = ZenithModelExporter(config)

    specs = [
        TensorSpec(name=name, shape=shape, dtype=dtype)
        for name, shape, dtype in input_specs
    ]

    return exporter.export(model, output_path, specs)


def export_to_torchscript(
    model: Any,
    output_path: str | Path,
    input_specs: list[tuple[str, list[int], str]],
    optimize: bool = True,
) -> ExportResult:
    """
    Export a model to TorchScript format.

    Args:
        model: PyTorch model to export
        output_path: Output file path
        input_specs: List of (name, shape, dtype) tuples
        optimize: Optimize for inference

    Returns:
        ExportResult
    """
    config = ExportConfig(
        format=ExportFormat.TORCHSCRIPT,
        optimize=optimize,
    )
    exporter = ZenithModelExporter(config)

    specs = [
        TensorSpec(name=name, shape=shape, dtype=dtype)
        for name, shape, dtype in input_specs
    ]

    return exporter.export(model, output_path, specs)


def create_deployment_package(
    model: Any,
    package_dir: str | Path,
    model_name: str,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    formats: list[ExportFormat] = None,
) -> dict[ExportFormat, ExportResult]:
    """
    Create a complete deployment package with multiple formats.

    Args:
        model: Model to export
        package_dir: Directory for the package
        model_name: Name of the model
        input_specs: Input tensor specifications
        output_specs: Output tensor specifications
        formats: List of formats to export (default: ONNX, TorchScript)

    Returns:
        Dictionary mapping formats to export results
    """
    if formats is None:
        formats = [ExportFormat.ONNX, ExportFormat.TORCHSCRIPT]

    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for fmt in formats:
        config = ExportConfig(format=fmt)
        exporter = ZenithModelExporter(config)

        if fmt == ExportFormat.ONNX:
            output_path = package_dir / f"{model_name}.onnx"
        elif fmt == ExportFormat.TORCHSCRIPT:
            output_path = package_dir / f"{model_name}.pt"
        else:
            output_path = package_dir / f"{model_name}.bin"

        result = exporter.export(model, output_path, input_specs, output_specs)
        results[fmt] = result

        # Save metadata
        if result.success:
            metadata = exporter.generate_metadata(model_name, input_specs, output_specs)
            metadata_path = output_path.with_suffix(".json")
            exporter.save_metadata(metadata, metadata_path)

    return results
