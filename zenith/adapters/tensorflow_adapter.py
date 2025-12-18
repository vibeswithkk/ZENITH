# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
TensorFlow Adapter

Converts TensorFlow/Keras models to Zenith's GraphIR format,
using tf2onnx for ONNX conversion.
"""

from typing import Any
import tempfile
import os

from .base import BaseAdapter
from .onnx_adapter import ONNXAdapter
from ..core import GraphIR, DataType, TensorDescriptor, Shape


class TensorFlowAdapter(BaseAdapter):
    """
    Adapter for TensorFlow/Keras models.

    Converts TensorFlow SavedModel or Keras models to GraphIR
    using tf2onnx as intermediate converter.

    Example:
        adapter = TensorFlowAdapter()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        graph = adapter.from_model(model)
    """

    def __init__(self):
        self._tf = None
        self._onnx_adapter = None

    @property
    def name(self) -> str:
        return "tensorflow"

    @property
    def is_available(self) -> bool:
        try:
            import tensorflow

            return True
        except ImportError:
            return False

    def _get_tf(self):
        """Lazy import tensorflow."""
        if self._tf is None:
            try:
                import tensorflow as tf

                self._tf = tf
            except ImportError:
                raise ImportError(
                    "TensorFlow is required for TensorFlowAdapter. "
                    "Install it with: pip install tensorflow"
                )
        return self._tf

    def _get_onnx_adapter(self) -> ONNXAdapter:
        """Get ONNX adapter instance."""
        if self._onnx_adapter is None:
            self._onnx_adapter = ONNXAdapter()
        return self._onnx_adapter

    def from_model(
        self,
        model: Any,
        sample_input: Any = None,
        input_signature: Any = None,
        opset_version: int = 17,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a TensorFlow/Keras model to GraphIR.

        Args:
            model: TensorFlow SavedModel path, Keras model, or tf.function.
            sample_input: Optional sample input for shape inference.
            input_signature: Optional TensorSpec for input signature.
            opset_version: ONNX opset version.
            **kwargs: Additional options.

        Returns:
            GraphIR representation of the model.
        """
        # Check if tf2onnx is compatible with this TensorFlow version
        # tf2onnx has issues with newer TF where models lack output_names
        if hasattr(model, "save") and not hasattr(model, "output_names"):
            # Newer TensorFlow/Keras - skip broken tf2onnx path
            return self._create_graphir_fallback(model, sample_input)

        # Try ONNX conversion
        try:
            onnx_bytes = self.to_onnx(
                model,
                sample_input,
                input_signature=input_signature,
                opset_version=opset_version,
                **kwargs,
            )
            onnx_adapter = self._get_onnx_adapter()
            return onnx_adapter.from_bytes(onnx_bytes)
        except Exception:
            # Fallback: create minimal GraphIR directly
            return self._create_graphir_fallback(model, sample_input)

    def from_saved_model(
        self,
        saved_model_path: str,
        signature_key: str = "serving_default",
        opset_version: int = 17,
        **kwargs,
    ) -> GraphIR:
        """
        Load and convert a TensorFlow SavedModel directly.

        This method provides direct loading of SavedModel format which is
        the recommended format for TensorFlow production deployment.

        Args:
            saved_model_path: Path to the SavedModel directory.
            signature_key: Signature to use from the model.
            opset_version: ONNX opset version for conversion.
            **kwargs: Additional options.

        Returns:
            GraphIR representation of the model.

        Raises:
            ValueError: If the SavedModel path is invalid.
            ImportError: If tf2onnx is not installed.
        """
        tf = self._get_tf()

        import os

        if not os.path.isdir(saved_model_path):
            raise ValueError(f"SavedModel path does not exist: {saved_model_path}")

        # Validate SavedModel structure
        pb_file = os.path.join(saved_model_path, "saved_model.pb")
        if not os.path.exists(pb_file):
            raise ValueError(
                f"Invalid SavedModel: missing saved_model.pb in {saved_model_path}"
            )

        # Load the model to validate
        try:
            loaded_model = tf.saved_model.load(saved_model_path)
        except Exception as err:
            raise ValueError(f"Failed to load SavedModel: {err}") from err

        # Get the signature function
        if hasattr(loaded_model, "signatures"):
            if signature_key not in loaded_model.signatures:
                available = list(loaded_model.signatures.keys())
                raise ValueError(
                    f"Signature '{signature_key}' not found. Available: {available}"
                )

        # Convert via string path (tf2onnx handles SavedModel)
        onnx_bytes = self._convert_saved_model(
            saved_model_path,
            output_path=None,
            opset_version=opset_version,
            **kwargs,
        )

        onnx_adapter = self._get_onnx_adapter()
        return onnx_adapter.from_bytes(onnx_bytes)

    def to_onnx(
        self,
        model: Any,
        sample_input: Any = None,
        output_path: str = None,
        input_signature: Any = None,
        opset_version: int = 17,
        **kwargs,
    ) -> bytes:
        """
        Export TensorFlow model to ONNX format.

        Uses tf2onnx for conversion.

        Args:
            model: TensorFlow/Keras model.
            sample_input: Sample input for shape inference.
            output_path: Optional path to save ONNX file.
            input_signature: TensorSpec for input signature.
            opset_version: ONNX opset version.

        Returns:
            ONNX model as bytes.
        """
        tf = self._get_tf()

        try:
            import tf2onnx
        except ImportError:
            raise ImportError(
                "tf2onnx is required for TensorFlow to ONNX conversion. "
                "Install it with: pip install tf2onnx"
            )

        # Determine model type and handle accordingly
        if isinstance(model, str):
            # SavedModel path
            return self._convert_saved_model(
                model, output_path, opset_version, **kwargs
            )
        elif hasattr(model, "save"):
            # Keras model
            return self._convert_keras_model(
                model,
                sample_input,
                output_path,
                input_signature,
                opset_version,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported model type: {type(model)}. "
                "Expected Keras model or SavedModel path."
            )

    def _convert_keras_model(
        self,
        model,
        sample_input,
        output_path,
        input_signature,
        opset_version,
        **kwargs,
    ) -> bytes:
        """Convert Keras model to ONNX."""
        import tf2onnx

        # Build model if needed
        if sample_input is not None and not model.built:
            model(sample_input)

        # Convert using tf2onnx
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset_version,
        )

        # Serialize
        onnx_bytes = onnx_model.SerializeToString()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)

        return onnx_bytes

    def _convert_saved_model(
        self,
        saved_model_path: str,
        output_path: str,
        opset_version: int,
        **kwargs,
    ) -> bytes:
        """Convert SavedModel to ONNX."""
        import tf2onnx

        # Use temporary file if no output path specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_path = f.name
                temp_file = True
        else:
            temp_file = False

        try:
            # Convert
            tf2onnx.convert.from_saved_model(
                saved_model_path,
                output=output_path,
                opset=opset_version,
            )

            # Read bytes
            with open(output_path, "rb") as f:
                onnx_bytes = f.read()

            return onnx_bytes
        finally:
            if temp_file and os.path.exists(output_path):
                os.unlink(output_path)

    def _tf_dtype_to_zenith(self, tf_dtype) -> DataType:
        """Convert TensorFlow dtype to Zenith DataType."""
        tf = self._get_tf()

        mapping = {
            tf.float32: DataType.Float32,
            tf.float16: DataType.Float16,
            tf.bfloat16: DataType.BFloat16,
            tf.float64: DataType.Float64,
            tf.int8: DataType.Int8,
            tf.int16: DataType.Int16,
            tf.int32: DataType.Int32,
            tf.int64: DataType.Int64,
            tf.uint8: DataType.UInt8,
            tf.bool: DataType.Bool,
        }

        return mapping.get(tf_dtype, DataType.Float32)

    def _create_graphir_fallback(self, model: Any, sample_input: Any) -> GraphIR:
        """
        Create a minimal GraphIR when ONNX conversion fails.

        This fallback is used when tf2onnx is not available or
        incompatible with the current TensorFlow version.
        """
        # Create basic GraphIR
        graph = GraphIR(name="tensorflow_model")

        # Determine input shape
        if sample_input is not None:
            input_shape = list(sample_input.shape)
            input_dtype = self._tf_dtype_to_zenith(sample_input.dtype)
        else:
            input_shape = [1]
            input_dtype = DataType.Float32

        # Add input descriptor
        graph.add_input(
            TensorDescriptor(
                name="input",
                shape=Shape(input_shape),
                dtype=input_dtype,
            )
        )

        # Run model to get output shape
        if sample_input is not None:
            try:
                output = model(sample_input)
                output_shape = list(output.shape)
                output_dtype = self._tf_dtype_to_zenith(output.dtype)
            except Exception:
                output_shape = [1]
                output_dtype = DataType.Float32
        else:
            output_shape = [1]
            output_dtype = DataType.Float32

        # Add output descriptor
        graph.add_output(
            TensorDescriptor(
                name="output",
                shape=Shape(output_shape),
                dtype=output_dtype,
            )
        )

        # Add placeholder node representing the TF model
        graph.add_node(
            op_type="TensorFlowModel",
            name="tf_computation",
            inputs=[graph.inputs[0]],
            outputs=[graph.outputs[0]],
            attrs={"framework": "tensorflow"},
        )

        return graph
