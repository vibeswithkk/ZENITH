# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Base Adapter Interface

Defines the abstract base class that all framework adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..core import GraphIR, TensorDescriptor


class BaseAdapter(ABC):
    """
    Abstract base class for framework adapters.

    All adapters (PyTorch, TensorFlow, JAX) must implement this interface
    to convert their respective model formats to Zenith's GraphIR.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this adapter (e.g., 'pytorch', 'tensorflow')."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the required framework is installed."""
        pass

    @abstractmethod
    def from_model(self, model: Any, sample_input: Any = None, **kwargs) -> GraphIR:
        """
        Convert a framework-specific model to GraphIR.

        Args:
            model: The model object from the source framework.
            sample_input: Optional sample input for tracing/export.
            **kwargs: Additional framework-specific options.

        Returns:
            GraphIR representation of the model.

        Raises:
            ImportError: If the required framework is not installed.
            ValueError: If the model cannot be converted.
        """
        pass

    def to_onnx(
        self, model: Any, sample_input: Any, output_path: str = None, **kwargs
    ) -> bytes:
        """
        Export model to ONNX format.

        This is the intermediate step used by most adapters.
        ONNX serves as the "lingua franca" between frameworks.

        Args:
            model: The model object from the source framework.
            sample_input: Sample input for tracing.
            output_path: Optional path to save ONNX file.
            **kwargs: Additional export options.

        Returns:
            ONNX model as bytes.
        """
        raise NotImplementedError(f"{self.name} adapter does not implement ONNX export")

    def get_input_shapes(self, model: Any) -> list[TensorDescriptor]:
        """
        Get the input tensor descriptors for a model.

        Args:
            model: The model object.

        Returns:
            List of TensorDescriptor for model inputs.
        """
        raise NotImplementedError(
            f"{self.name} adapter does not implement get_input_shapes"
        )

    def get_output_shapes(self, model: Any) -> list[TensorDescriptor]:
        """
        Get the output tensor descriptors for a model.

        Args:
            model: The model object.

        Returns:
            List of TensorDescriptor for model outputs.
        """
        raise NotImplementedError(
            f"{self.name} adapter does not implement get_output_shapes"
        )
