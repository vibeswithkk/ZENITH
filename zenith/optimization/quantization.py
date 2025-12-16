"""
INT8 Quantization Pipeline

Implements comprehensive INT8 quantization with:
- Post-training static quantization
- Post-training dynamic quantization
- Quantization-Aware Training (QAT) simulation
- Calibration methods (MinMax, Percentile, Entropy)

Based on CetakBiru Section 5.1 Fase 3 requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..core import GraphIR, Node, DataType


class QuantizationMode(Enum):
    """Quantization modes."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    QAT = "qat"


class CalibrationMethod(Enum):
    """Calibration methods for static quantization."""

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    ENTROPY = "entropy"


@dataclass
class QuantizationParams:
    """Quantization parameters for a tensor."""

    scale: float
    zero_point: int
    dtype: str = "int8"
    symmetric: bool = True

    def quantize(self, tensor: np.ndarray) -> np.ndarray:
        """Quantize floating-point tensor to INT8."""
        scaled = tensor / self.scale + self.zero_point
        clipped = np.clip(scaled, -128, 127)
        return clipped.astype(np.int8)

    def dequantize(self, tensor: np.ndarray) -> np.ndarray:
        """Dequantize INT8 tensor to floating-point."""
        return (tensor.astype(np.float32) - self.zero_point) * self.scale


@dataclass
class TensorStats:
    """Statistics collected during calibration."""

    min_val: float = float("inf")
    max_val: float = float("-inf")
    histogram: np.ndarray | None = None
    num_bins: int = 2048

    def update(self, tensor: np.ndarray) -> None:
        """Update statistics with new tensor values."""
        self.min_val = min(self.min_val, float(np.min(tensor)))
        self.max_val = max(self.max_val, float(np.max(tensor)))

        # Update histogram for entropy calibration
        if self.histogram is None:
            self.histogram = np.zeros(self.num_bins, dtype=np.int64)

        # Create histogram with fixed range
        hist_range = (
            (self.min_val, self.max_val) if self.min_val < self.max_val else (0, 1)
        )
        hist, _ = np.histogram(tensor.flatten(), bins=self.num_bins, range=hist_range)
        self.histogram += hist


class Calibrator(ABC):
    """Abstract base class for calibration methods."""

    @abstractmethod
    def compute_params(self, stats: TensorStats) -> QuantizationParams:
        """Compute quantization parameters from statistics."""
        pass


class MinMaxCalibrator(Calibrator):
    """MinMax calibration: uses observed min/max values."""

    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric

    def compute_params(self, stats: TensorStats) -> QuantizationParams:
        """Compute scale and zero point from min/max."""
        if self.symmetric:
            # Symmetric quantization: zero_point = 0
            abs_max = max(abs(stats.min_val), abs(stats.max_val))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            return QuantizationParams(scale=scale, zero_point=0, symmetric=True)
        else:
            # Asymmetric quantization
            scale = (stats.max_val - stats.min_val) / 255.0
            scale = scale if scale > 0 else 1.0
            zero_point = int(-stats.min_val / scale)
            zero_point = np.clip(zero_point, -128, 127)
            return QuantizationParams(
                scale=scale, zero_point=int(zero_point), symmetric=False
            )


class PercentileCalibrator(Calibrator):
    """Percentile calibration: clips outliers."""

    def __init__(self, percentile: float = 99.99, symmetric: bool = True):
        self.percentile = percentile
        self.symmetric = symmetric

    def compute_params(self, stats: TensorStats) -> QuantizationParams:
        """Compute scale using percentile clipping."""
        if stats.histogram is None:
            # Fallback to minmax
            return MinMaxCalibrator(self.symmetric).compute_params(stats)

        # Find percentile from histogram
        cumsum = np.cumsum(stats.histogram)
        total = cumsum[-1]
        threshold_count = total * (self.percentile / 100.0)

        # Find bin that exceeds threshold
        idx = np.searchsorted(cumsum, threshold_count)
        bin_width = (stats.max_val - stats.min_val) / len(stats.histogram)
        clipped_max = stats.min_val + (idx + 1) * bin_width

        if self.symmetric:
            abs_max = clipped_max
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            return QuantizationParams(scale=scale, zero_point=0, symmetric=True)
        else:
            scale = clipped_max / 255.0 if clipped_max > 0 else 1.0
            return QuantizationParams(scale=scale, zero_point=0, symmetric=False)


class EntropyCalibrator(Calibrator):
    """Entropy (KL divergence) calibration for optimal range."""

    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric

    def compute_params(self, stats: TensorStats) -> QuantizationParams:
        """Compute scale minimizing KL divergence."""
        if stats.histogram is None:
            return MinMaxCalibrator(self.symmetric).compute_params(stats)

        histogram = stats.histogram.astype(np.float64)
        histogram = histogram / (np.sum(histogram) + 1e-10)

        best_divergence = float("inf")
        best_threshold = len(histogram)

        # Search for optimal threshold
        for threshold in range(128, len(histogram)):
            # Quantized reference distribution
            ref_dist = histogram[:threshold].copy()
            if np.sum(ref_dist) == 0:
                continue

            # Quantize to 128 bins
            quantized = np.zeros(128, dtype=np.float64)
            bin_size = threshold / 128.0

            for i in range(128):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                end = min(end, threshold)
                quantized[i] = np.sum(ref_dist[start:end])

            # Expand back
            expanded = np.zeros(threshold, dtype=np.float64)
            for i in range(128):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                end = min(end, threshold)
                if end > start:
                    expanded[start:end] = quantized[i] / (end - start)

            # KL divergence
            mask = (ref_dist > 0) & (expanded > 0)
            if not np.any(mask):
                continue

            kl_div = np.sum(ref_dist[mask] * np.log(ref_dist[mask] / expanded[mask]))

            if kl_div < best_divergence:
                best_divergence = kl_div
                best_threshold = threshold

        # Compute scale from best threshold
        bin_width = (stats.max_val - stats.min_val) / len(histogram)
        optimal_max = stats.min_val + best_threshold * bin_width

        if self.symmetric:
            scale = optimal_max / 127.0 if optimal_max > 0 else 1.0
            return QuantizationParams(scale=scale, zero_point=0, symmetric=True)
        else:
            scale = optimal_max / 255.0 if optimal_max > 0 else 1.0
            return QuantizationParams(scale=scale, zero_point=0, symmetric=False)


class QuantizationModel:
    """
    Quantized model representation.

    Stores quantization parameters for all quantized tensors
    and provides methods for quantized inference.
    """

    def __init__(self, name: str = "quantized_model"):
        self.name = name
        self.weight_params: dict[str, QuantizationParams] = {}
        self.activation_params: dict[str, QuantizationParams] = {}
        self.quantized_weights: dict[str, np.ndarray] = {}

    def add_weight(
        self,
        name: str,
        weight: np.ndarray,
        params: QuantizationParams,
    ) -> None:
        """Add a quantized weight."""
        self.weight_params[name] = params
        self.quantized_weights[name] = params.quantize(weight)

    def add_activation_params(
        self,
        name: str,
        params: QuantizationParams,
    ) -> None:
        """Add activation quantization parameters."""
        self.activation_params[name] = params

    def quantize_activation(self, name: str, tensor: np.ndarray) -> np.ndarray:
        """Quantize activation tensor."""
        if name not in self.activation_params:
            # Dynamic quantization fallback
            abs_max = np.max(np.abs(tensor))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            return (tensor / scale).astype(np.int8)

        params = self.activation_params[name]
        return params.quantize(tensor)

    def get_weight(self, name: str) -> np.ndarray:
        """Get quantized weight (as INT8)."""
        return self.quantized_weights.get(name)

    def get_weight_dequantized(self, name: str) -> np.ndarray | None:
        """Get dequantized weight (as FP32)."""
        if name not in self.quantized_weights:
            return None
        params = self.weight_params[name]
        return params.dequantize(self.quantized_weights[name])


class Quantizer:
    """
    Main quantization interface.

    Usage:
        quantizer = Quantizer(mode=QuantizationMode.STATIC)

        # Collect calibration data
        for batch in calibration_data:
            quantizer.collect_stats(batch, layer_name)

        # Quantize model
        quantized_model = quantizer.quantize(model, graph)
    """

    def __init__(
        self,
        mode: QuantizationMode = QuantizationMode.STATIC,
        calibration_method: CalibrationMethod = CalibrationMethod.MINMAX,
        symmetric: bool = True,
    ):
        self.mode = mode
        self.calibration_method = calibration_method
        self.symmetric = symmetric
        self.tensor_stats: dict[str, TensorStats] = {}
        self._calibrator = self._create_calibrator()

    def _create_calibrator(self) -> Calibrator:
        """Create calibrator based on method."""
        if self.calibration_method == CalibrationMethod.MINMAX:
            return MinMaxCalibrator(self.symmetric)
        elif self.calibration_method == CalibrationMethod.PERCENTILE:
            return PercentileCalibrator(symmetric=self.symmetric)
        elif self.calibration_method == CalibrationMethod.ENTROPY:
            return EntropyCalibrator(self.symmetric)
        else:
            return MinMaxCalibrator(self.symmetric)

    def collect_stats(self, tensor: np.ndarray, name: str) -> None:
        """Collect statistics for calibration."""
        if name not in self.tensor_stats:
            self.tensor_stats[name] = TensorStats()
        self.tensor_stats[name].update(tensor)

    def compute_params(self, name: str) -> QuantizationParams | None:
        """Compute quantization parameters for a tensor."""
        if name not in self.tensor_stats:
            return None
        return self._calibrator.compute_params(self.tensor_stats[name])

    def quantize_weights(
        self,
        weights: dict[str, np.ndarray],
    ) -> QuantizationModel:
        """Quantize model weights."""
        model = QuantizationModel()

        for name, weight in weights.items():
            # Collect stats for weight
            stats = TensorStats()
            stats.update(weight)

            # Compute parameters
            params = self._calibrator.compute_params(stats)

            # Add to model
            model.add_weight(name, weight, params)

        # Add activation parameters from collected stats
        for name, stats in self.tensor_stats.items():
            params = self._calibrator.compute_params(stats)
            model.add_activation_params(name, params)

        return model

    def quantize_tensor(
        self, tensor: np.ndarray, name: str | None = None
    ) -> tuple[np.ndarray, QuantizationParams]:
        """
        Quantize a single tensor.

        Args:
            tensor: Input tensor
            name: Optional tensor name for cached params

        Returns:
            (quantized_tensor, params)
        """
        if name and name in self.tensor_stats:
            params = self._calibrator.compute_params(self.tensor_stats[name])
        else:
            # Dynamic quantization
            stats = TensorStats()
            stats.update(tensor)
            params = self._calibrator.compute_params(stats)

        quantized = params.quantize(tensor)
        return quantized, params


class QATSimulator:
    """
    Quantization-Aware Training (QAT) Simulator.

    Simulates quantization effects during training by:
    - Fake-quantizing weights and activations
    - Using straight-through estimator for gradients
    """

    def __init__(
        self,
        calibration_method: CalibrationMethod = CalibrationMethod.MINMAX,
        symmetric: bool = True,
    ):
        self.quantizer = Quantizer(
            mode=QuantizationMode.QAT,
            calibration_method=calibration_method,
            symmetric=symmetric,
        )

    def fake_quantize(
        self,
        tensor: np.ndarray,
        name: str | None = None,
    ) -> np.ndarray:
        """
        Apply fake quantization (quantize then dequantize).

        This simulates quantization error during training.
        """
        quantized, params = self.quantizer.quantize_tensor(tensor, name)
        return params.dequantize(quantized)

    def simulate_forward(
        self,
        weight: np.ndarray,
        activation: np.ndarray,
        weight_name: str | None = None,
        activation_name: str | None = None,
    ) -> np.ndarray:
        """
        Simulate quantized forward pass.

        Both weight and activation are fake-quantized.
        """
        fake_weight = self.fake_quantize(weight, weight_name)
        fake_activation = self.fake_quantize(activation, activation_name)

        # Perform computation (e.g., MatMul)
        return np.matmul(fake_activation, fake_weight)

    def get_quantized_model(
        self,
        weights: dict[str, np.ndarray],
    ) -> QuantizationModel:
        """Convert trained model to quantized model."""
        return self.quantizer.quantize_weights(weights)


def quantize_model_static(
    weights: dict[str, np.ndarray],
    calibration_data: list[np.ndarray],
    activation_names: list[str] | None = None,
    method: CalibrationMethod = CalibrationMethod.MINMAX,
) -> QuantizationModel:
    """
    Convenience function for static post-training quantization.

    Args:
        weights: Dictionary of model weights
        calibration_data: List of calibration tensors
        activation_names: Names for activation tensors
        method: Calibration method

    Returns:
        Quantized model
    """
    quantizer = Quantizer(
        mode=QuantizationMode.STATIC,
        calibration_method=method,
    )

    # Collect activation statistics
    if activation_names:
        for i, data in enumerate(calibration_data):
            name = activation_names[i] if i < len(activation_names) else f"act_{i}"
            quantizer.collect_stats(data, name)

    return quantizer.quantize_weights(weights)


def measure_quantization_error(
    original: np.ndarray,
    quantized_model: QuantizationModel,
    weight_name: str,
) -> dict:
    """
    Measure quantization error for a weight tensor.

    Returns:
        Dictionary with error metrics (MSE, max_abs_error, SNR)
    """
    dequantized = quantized_model.get_weight_dequantized(weight_name)
    if dequantized is None:
        return {"error": "Weight not found"}

    # Mean Squared Error
    mse = float(np.mean((original - dequantized) ** 2))

    # Max Absolute Error
    max_error = float(np.max(np.abs(original - dequantized)))

    # Signal-to-Noise Ratio
    signal_power = np.mean(original**2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return {
        "mse": mse,
        "max_abs_error": max_error,
        "snr_db": float(snr),
    }
