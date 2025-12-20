"""
Quantization-Aware Training (QAT) Module

Implements comprehensive QAT with:
- FakeQuantize with Straight-Through Estimator (STE)
- Learnable scale and zero-point parameters
- Batch normalization folding for inference
- QAT-aware training workflow utilities

Based on CetakBiru Section 5.1 Fase 3 requirements.
References:
- PyTorch torch.quantization QAT
- TensorFlow Model Optimization Toolkit
- Jacob et al. "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference" (arXiv:1712.05877)

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
from abc import ABC, abstractmethod

from .quantization import (
    QuantizationParams,
    QuantizationMode,
    CalibrationMethod,
    TensorStats,
    Calibrator,
    MinMaxCalibrator,
    PercentileCalibrator,
    EntropyCalibrator,
)


# =============================================================================
# QAT Configuration
# =============================================================================


class QATScheme(Enum):
    """QAT quantization scheme."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    PER_CHANNEL = "per_channel"
    PER_TENSOR = "per_tensor"


class ObserverType(Enum):
    """Type of observer for collecting statistics."""

    MINMAX = "minmax"
    MOVING_AVERAGE = "moving_average"
    HISTOGRAM = "histogram"


@dataclass
class QATConfig:
    """
    Configuration for Quantization-Aware Training.

    Attributes:
        weight_bits: Bit width for weight quantization (default: 8)
        activation_bits: Bit width for activation quantization (default: 8)
        weight_scheme: Quantization scheme for weights
        activation_scheme: Quantization scheme for activations
        observer_type: Type of observer for statistics collection
        averaging_constant: EMA constant for moving average observer
        freeze_bn_after_epochs: Freeze BN statistics after N epochs (0 = never)
        start_qat_after_epochs: Start fake quantization after N epochs
        calibration_batches: Number of batches for initial calibration
        per_channel_weights: Use per-channel quantization for weights
        symmetric_weights: Use symmetric quantization for weights
        symmetric_activations: Use symmetric quantization for activations
    """

    weight_bits: int = 8
    activation_bits: int = 8
    weight_scheme: QATScheme = QATScheme.PER_CHANNEL
    activation_scheme: QATScheme = QATScheme.PER_TENSOR
    observer_type: ObserverType = ObserverType.MOVING_AVERAGE
    averaging_constant: float = 0.01
    freeze_bn_after_epochs: int = 0
    start_qat_after_epochs: int = 0
    calibration_batches: int = 100
    per_channel_weights: bool = True
    symmetric_weights: bool = True
    symmetric_activations: bool = True


# =============================================================================
# Fake Quantization with Straight-Through Estimator
# =============================================================================


class FakeQuantize:
    """
    Fake quantization module implementing the Straight-Through Estimator.

    During forward pass:
        1. Quantize input to simulated INT8
        2. Dequantize back to float
        3. Return the dequantized value

    During backward pass:
        1. Pass gradients straight through (STE)
        2. Optionally clip gradients outside quantization range

    This allows the model to learn to be robust to quantization noise
    while still allowing gradient-based optimization.

    Reference: Bengio et al. "Estimating or Propagating Gradients Through
    Stochastic Neurons for Conditional Computation"
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
        averaging_constant: float = 0.01,
    ):
        """
        Initialize FakeQuantize module.

        Args:
            num_bits: Number of bits for quantization
            symmetric: Whether to use symmetric quantization
            per_channel: Whether to use per-channel quantization
            channel_axis: Axis for per-channel quantization
            averaging_constant: EMA constant for scale/zero_point updates
        """
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.averaging_constant = averaging_constant

        # Compute quantization bounds
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**num_bits - 1

        # Learnable/tracked parameters
        self.scale: np.ndarray | None = None
        self.zero_point: np.ndarray | None = None

        # Statistics tracking
        self.min_val: np.ndarray | None = None
        self.max_val: np.ndarray | None = None
        self.enabled: bool = True
        self.observer_enabled: bool = True

    def enable(self) -> None:
        """Enable fake quantization."""
        self.enabled = True

    def disable(self) -> None:
        """Disable fake quantization (pass-through)."""
        self.enabled = False

    def enable_observer(self) -> None:
        """Enable statistics observation."""
        self.observer_enabled = True

    def disable_observer(self) -> None:
        """Disable statistics observation."""
        self.observer_enabled = False

    def reset_stats(self) -> None:
        """Reset tracked statistics."""
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None

    def observe(self, x: np.ndarray) -> None:
        """
        Observe tensor statistics for calibration.

        Args:
            x: Input tensor to observe
        """
        if not self.observer_enabled:
            return

        if self.per_channel:
            # Per-channel statistics along channel_axis
            axes = tuple(i for i in range(x.ndim) if i != self.channel_axis)
            new_min = np.min(x, axis=axes)
            new_max = np.max(x, axis=axes)
        else:
            # Per-tensor statistics
            new_min = np.array([np.min(x)])
            new_max = np.array([np.max(x)])

        # Update with exponential moving average
        if self.min_val is None:
            self.min_val = new_min
            self.max_val = new_max
        else:
            alpha = self.averaging_constant
            self.min_val = (1 - alpha) * self.min_val + alpha * new_min
            self.max_val = (1 - alpha) * self.max_val + alpha * new_max

        # Update scale and zero_point
        self._compute_qparams()

    def _compute_qparams(self) -> None:
        """Compute quantization parameters from observed statistics."""
        if self.min_val is None or self.max_val is None:
            return

        if self.symmetric:
            # Symmetric quantization: zero_point = 0
            abs_max = np.maximum(np.abs(self.min_val), np.abs(self.max_val))
            self.scale = abs_max / ((self.qmax - self.qmin) / 2)
            # Prevent division by zero
            self.scale = np.maximum(self.scale, 1e-8)
            self.zero_point = np.zeros_like(self.scale, dtype=np.int32)
        else:
            # Asymmetric quantization
            self.scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
            self.scale = np.maximum(self.scale, 1e-8)
            self.zero_point = np.round(self.qmin - self.min_val / self.scale).astype(
                np.int32
            )
            self.zero_point = np.clip(self.zero_point, self.qmin, self.qmax)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply fake quantization with straight-through estimator.

        Args:
            x: Input tensor

        Returns:
            Fake-quantized tensor (quantized then dequantized)
        """
        # Observe statistics if enabled
        self.observe(x)

        # Pass through if disabled or not calibrated
        if not self.enabled or self.scale is None:
            return x

        # Quantize
        if self.per_channel:
            # Reshape scale for broadcasting
            shape = [1] * x.ndim
            shape[self.channel_axis] = -1
            scale = self.scale.reshape(shape)
            zero_point = self.zero_point.reshape(shape)
        else:
            scale = self.scale[0]
            zero_point = self.zero_point[0]

        # Fake quantize: quantize then dequantize
        x_q = np.clip(np.round(x / scale) + zero_point, self.qmin, self.qmax)
        x_dq = (x_q - zero_point) * scale

        return x_dq

    def forward_with_ste(
        self, x: np.ndarray, grad_output: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Forward pass with STE gradient computation.

        The STE passes gradients through unchanged for values within
        the quantization range, and zeros gradients for values outside.

        Args:
            x: Input tensor
            grad_output: Gradient from next layer (for backward pass)

        Returns:
            Tuple of (output, gradient_input)
        """
        output = self.forward(x)

        if grad_output is None:
            return output, None

        # STE: Pass gradients straight through, but zero for clipped values
        if self.scale is not None:
            if self.per_channel:
                shape = [1] * x.ndim
                shape[self.channel_axis] = -1
                scale = self.scale.reshape(shape)
                zero_point = self.zero_point.reshape(shape)
            else:
                scale = self.scale[0]
                zero_point = self.zero_point[0]

            # Compute quantized value before clipping
            x_q_raw = np.round(x / scale) + zero_point

            # Mask for values within range (STE passes these through)
            mask = (x_q_raw >= self.qmin) & (x_q_raw <= self.qmax)

            # Zero gradients for clipped values
            grad_input = grad_output * mask.astype(grad_output.dtype)
        else:
            grad_input = grad_output

        return output, grad_input

    def get_quantization_params(self) -> QuantizationParams | None:
        """
        Get current quantization parameters.

        Returns:
            QuantizationParams if calibrated, None otherwise
        """
        if self.scale is None:
            return None

        # Return per-tensor params (take first for per-channel)
        # Use .item() to properly extract scalar from numpy array
        if self.per_channel:
            scale = float(self.scale.flat[0])
            zero_point = int(self.zero_point.flat[0])
        else:
            scale = float(np.asarray(self.scale).flat[0])
            zero_point = int(np.asarray(self.zero_point).flat[0])

        return QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            dtype=f"int{self.num_bits}",
            symmetric=self.symmetric,
        )


# =============================================================================
# QAT Module Wrapper
# =============================================================================


class QATModule:
    """
    Wrapper class for applying QAT to model layers.

    This wraps a layer's weights and activations with FakeQuantize modules.
    """

    def __init__(
        self,
        weight_shape: tuple,
        config: QATConfig = QATConfig(),
        has_bias: bool = True,
    ):
        """
        Initialize QAT module wrapper.

        Args:
            weight_shape: Shape of the weight tensor
            config: QAT configuration
            has_bias: Whether the layer has a bias term
        """
        self.config = config
        self.weight_shape = weight_shape
        self.has_bias = has_bias

        # Create fake quantizers
        self.weight_fake_quant = FakeQuantize(
            num_bits=config.weight_bits,
            symmetric=config.symmetric_weights,
            per_channel=config.per_channel_weights,
            channel_axis=0,  # Output channel axis for Conv/Linear
        )

        self.activation_fake_quant = FakeQuantize(
            num_bits=config.activation_bits,
            symmetric=config.symmetric_activations,
            per_channel=False,
        )

        # Bias is typically not quantized (folded into scale)
        self.bias_fake_quant = None

    def fake_quantize_weight(self, weight: np.ndarray) -> np.ndarray:
        """Apply fake quantization to weights."""
        return self.weight_fake_quant.forward(weight)

    def fake_quantize_activation(self, activation: np.ndarray) -> np.ndarray:
        """Apply fake quantization to activations."""
        return self.activation_fake_quant.forward(activation)

    def freeze_observers(self) -> None:
        """Freeze all observers (stop updating statistics)."""
        self.weight_fake_quant.disable_observer()
        self.activation_fake_quant.disable_observer()

    def enable_fake_quant(self) -> None:
        """Enable fake quantization."""
        self.weight_fake_quant.enable()
        self.activation_fake_quant.enable()

    def disable_fake_quant(self) -> None:
        """Disable fake quantization."""
        self.weight_fake_quant.disable()
        self.activation_fake_quant.disable()


# =============================================================================
# Batch Normalization Folding
# =============================================================================


def fold_bn_into_conv(
    conv_weight: np.ndarray,
    conv_bias: np.ndarray | None,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fold batch normalization parameters into convolution weights.

    This is essential for QAT and quantized inference, as it eliminates
    the batch normalization layer and fuses its effects into the
    preceding convolution.

    BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Conv: y = W * x + b
    Folded: y = W' * x + b'
    where:
        W' = gamma * W / sqrt(var + eps)
        b' = gamma * (b - mean) / sqrt(var + eps) + beta

    Args:
        conv_weight: Convolution weight tensor [out_ch, in_ch, kH, kW]
        conv_bias: Convolution bias tensor [out_ch] or None
        bn_mean: Batch norm running mean [out_ch]
        bn_var: Batch norm running variance [out_ch]
        bn_gamma: Batch norm scale (gamma) [out_ch]
        bn_beta: Batch norm shift (beta) [out_ch]
        epsilon: Batch norm epsilon for numerical stability

    Returns:
        Tuple of (folded_weight, folded_bias)
    """
    # Compute scaling factor
    std = np.sqrt(bn_var + epsilon)
    scale = bn_gamma / std

    # Fold into weights: W' = gamma * W / std
    # Scale is applied per output channel
    if conv_weight.ndim == 4:
        # Conv2D: [out_ch, in_ch, kH, kW]
        folded_weight = conv_weight * scale.reshape(-1, 1, 1, 1)
    elif conv_weight.ndim == 2:
        # Linear: [out_features, in_features]
        folded_weight = conv_weight * scale.reshape(-1, 1)
    else:
        # Generic case
        shape = [-1] + [1] * (conv_weight.ndim - 1)
        folded_weight = conv_weight * scale.reshape(shape)

    # Fold into bias: b' = gamma * (b - mean) / std + beta
    if conv_bias is None:
        conv_bias = np.zeros(bn_mean.shape, dtype=conv_weight.dtype)

    folded_bias = bn_gamma * (conv_bias - bn_mean) / std + bn_beta

    return folded_weight, folded_bias


def unfold_bn_from_conv(
    folded_weight: np.ndarray,
    folded_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unfold (reverse) batch normalization folding from convolution.

    This is useful for converting back from folded representation
    during training if needed.

    Args:
        folded_weight: Folded weight tensor
        folded_bias: Folded bias tensor
        bn_mean: Batch norm running mean
        bn_var: Batch norm running variance
        bn_gamma: Batch norm scale (gamma)
        bn_beta: Batch norm shift (beta)
        epsilon: Batch norm epsilon

    Returns:
        Tuple of (original_weight, original_bias)
    """
    std = np.sqrt(bn_var + epsilon)
    scale = bn_gamma / std

    # Reverse weight folding
    if folded_weight.ndim == 4:
        original_weight = folded_weight / scale.reshape(-1, 1, 1, 1)
    elif folded_weight.ndim == 2:
        original_weight = folded_weight / scale.reshape(-1, 1)
    else:
        shape = [-1] + [1] * (folded_weight.ndim - 1)
        original_weight = folded_weight / scale.reshape(shape)

    # Reverse bias folding
    original_bias = (folded_bias - bn_beta) * std / bn_gamma + bn_mean

    return original_weight, original_bias


# =============================================================================
# QAT Training Utilities
# =============================================================================


class QATTrainer:
    """
    Orchestrates Quantization-Aware Training workflow.

    Typical workflow:
        1. Create trainer with config
        2. Prepare model layers with QAT modules
        3. Run calibration with representative data
        4. Train with fake quantization enabled
        5. Convert to fully quantized model
    """

    def __init__(self, config: QATConfig = QATConfig()):
        """
        Initialize QAT trainer.

        Args:
            config: QAT configuration
        """
        self.config = config
        self.qat_modules: dict[str, QATModule] = {}
        self.epoch: int = 0
        self.calibration_complete: bool = False

    def register_layer(
        self,
        name: str,
        weight_shape: tuple,
        has_bias: bool = True,
    ) -> QATModule:
        """
        Register a layer for QAT.

        Args:
            name: Unique name for the layer
            weight_shape: Shape of the weight tensor
            has_bias: Whether the layer has bias

        Returns:
            QATModule wrapper for the layer
        """
        qat_module = QATModule(weight_shape, self.config, has_bias)
        self.qat_modules[name] = qat_module
        return qat_module

    def calibrate(
        self,
        layer_name: str,
        weight: np.ndarray,
        activations: list[np.ndarray],
    ) -> None:
        """
        Calibrate a layer with representative data.

        Args:
            layer_name: Name of the layer to calibrate
            weight: Layer weight tensor
            activations: List of activation tensors from calibration data
        """
        if layer_name not in self.qat_modules:
            raise ValueError(f"Layer '{layer_name}' not registered")

        qat_module = self.qat_modules[layer_name]

        # Observe weight
        qat_module.weight_fake_quant.observe(weight)

        # Observe activations
        for act in activations:
            qat_module.activation_fake_quant.observe(act)

    def start_epoch(self, epoch: int) -> None:
        """
        Called at the start of each training epoch.

        Handles enabling/disabling fake quantization and freezing BN.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self.epoch = epoch

        # Enable fake quantization after warmup epochs
        if epoch >= self.config.start_qat_after_epochs:
            for module in self.qat_modules.values():
                module.enable_fake_quant()
        else:
            for module in self.qat_modules.values():
                module.disable_fake_quant()

        # Freeze observers after calibration period
        if epoch >= self.config.freeze_bn_after_epochs > 0:
            for module in self.qat_modules.values():
                module.freeze_observers()

    def fake_quantize_weights(
        self, weights: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Apply fake quantization to all registered weights.

        Args:
            weights: Dictionary mapping layer names to weight tensors

        Returns:
            Dictionary of fake-quantized weights
        """
        result = {}
        for name, weight in weights.items():
            if name in self.qat_modules:
                result[name] = self.qat_modules[name].fake_quantize_weight(weight)
            else:
                result[name] = weight
        return result

    def fake_quantize_activation(
        self, layer_name: str, activation: np.ndarray
    ) -> np.ndarray:
        """
        Apply fake quantization to a layer's activation.

        Args:
            layer_name: Name of the layer
            activation: Activation tensor

        Returns:
            Fake-quantized activation
        """
        if layer_name in self.qat_modules:
            return self.qat_modules[layer_name].fake_quantize_activation(activation)
        return activation

    def get_quantization_params(self) -> dict[str, dict[str, QuantizationParams]]:
        """
        Get quantization parameters for all registered layers.

        Returns:
            Dictionary mapping layer names to weight/activation params
        """
        result = {}
        for name, module in self.qat_modules.items():
            weight_params = module.weight_fake_quant.get_quantization_params()
            act_params = module.activation_fake_quant.get_quantization_params()
            result[name] = {
                "weight": weight_params,
                "activation": act_params,
            }
        return result


# =============================================================================
# Convenience Functions
# =============================================================================


def prepare_model_for_qat(
    layer_shapes: dict[str, tuple],
    config: QATConfig = QATConfig(),
) -> QATTrainer:
    """
    Prepare a model for QAT training.

    Args:
        layer_shapes: Dictionary mapping layer names to weight shapes
        config: QAT configuration

    Returns:
        QATTrainer instance with layers registered
    """
    trainer = QATTrainer(config)

    for name, shape in layer_shapes.items():
        trainer.register_layer(name, shape)

    return trainer


def convert_qat_to_quantized(
    trainer: QATTrainer,
    weights: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, QuantizationParams]]:
    """
    Convert QAT-trained weights to fully quantized format.

    Args:
        trainer: QATTrainer with trained QAT modules
        weights: Dictionary of trained weight tensors

    Returns:
        Dictionary mapping layer names to (quantized_weight, params) tuples
    """
    result = {}

    for name, weight in weights.items():
        if name in trainer.qat_modules:
            module = trainer.qat_modules[name]
            params = module.weight_fake_quant.get_quantization_params()

            if params is not None:
                # Quantize to actual integer representation
                quantized_weight = params.quantize(weight)
                result[name] = (quantized_weight, params)
            else:
                # No calibration, use dynamic quantization
                abs_max = np.max(np.abs(weight))
                scale = abs_max / 127.0 if abs_max > 0 else 1.0
                params = QuantizationParams(scale=scale, zero_point=0)
                quantized_weight = params.quantize(weight)
                result[name] = (quantized_weight, params)
        else:
            # Layer not in QAT, quantize dynamically
            abs_max = np.max(np.abs(weight))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            params = QuantizationParams(scale=scale, zero_point=0)
            quantized_weight = params.quantize(weight)
            result[name] = (quantized_weight, params)

    return result


def simulate_qat_forward(
    trainer: QATTrainer,
    layer_name: str,
    weight: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    """
    Simulate a QAT forward pass for a single layer.

    Args:
        trainer: QATTrainer instance
        layer_name: Name of the layer
        weight: Weight tensor
        activation: Input activation tensor

    Returns:
        Output after fake-quantized computation
    """
    # Fake quantize weight and activation
    fq_weight = trainer.fake_quantize_weights({layer_name: weight})[layer_name]
    fq_activation = trainer.fake_quantize_activation(layer_name, activation)

    # Perform computation (example: MatMul)
    output = np.matmul(fq_activation, fq_weight.T)

    return output


def measure_qat_error(
    original_output: np.ndarray,
    qat_output: np.ndarray,
) -> dict[str, float]:
    """
    Measure the error introduced by QAT.

    Args:
        original_output: Output from full-precision forward pass
        qat_output: Output from QAT forward pass

    Returns:
        Dictionary with error metrics
    """
    diff = original_output - qat_output

    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))

    # Signal-to-noise ratio
    signal_power = np.mean(original_output**2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Relative error
    rel_error = np.mean(np.abs(diff) / (np.abs(original_output) + 1e-10))

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "snr_db": float(snr),
        "relative_error": float(rel_error),
    }
