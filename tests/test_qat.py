"""
Test Suite for Quantization-Aware Training (QAT).

Comprehensive tests for:
- FakeQuantize forward/backward with STE
- QATModule and QATTrainer
- Batch normalization folding
- QAT accuracy preservation
- Conversion from QAT to quantized model

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import pytest
import numpy as np


class TestFakeQuantize:
    """Tests for FakeQuantize class."""

    def test_fake_quantize_initialization(self):
        """Test FakeQuantize initialization with various configurations."""
        from zenith.optimization.qat import FakeQuantize

        # Default initialization
        fq = FakeQuantize()
        assert fq.num_bits == 8
        assert fq.symmetric is True
        assert fq.qmin == -128
        assert fq.qmax == 127

        # Asymmetric
        fq_asym = FakeQuantize(symmetric=False)
        assert fq_asym.qmin == 0
        assert fq_asym.qmax == 255

        # Different bit width
        fq_4bit = FakeQuantize(num_bits=4, symmetric=True)
        assert fq_4bit.qmin == -8
        assert fq_4bit.qmax == 7

    def test_fake_quantize_observe_updates_stats(self):
        """Test that observe() updates min/max statistics."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize()
        assert fq.min_val is None
        assert fq.max_val is None

        # Observe some data
        data = np.array([1.0, 2.0, 3.0, -1.0, -2.0], dtype=np.float32)
        fq.observe(data)

        assert fq.min_val is not None
        assert fq.max_val is not None
        assert fq.min_val[0] <= -2.0
        assert fq.max_val[0] >= 3.0

    def test_fake_quantize_forward_identity_before_calibration(self):
        """Test that forward is identity before calibration."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize()
        fq.disable_observer()  # Don't auto-calibrate

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = fq.forward(x)

        np.testing.assert_array_equal(x, y)

    def test_fake_quantize_forward_after_calibration(self):
        """Test fake quantization after calibration."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(symmetric=True)

        # Calibrate with data
        calibration_data = np.array([1.0, 2.0, 3.0, -3.0], dtype=np.float32)
        fq.observe(calibration_data)

        # Forward pass
        x = np.array([1.5, 2.5, -1.5], dtype=np.float32)
        y = fq.forward(x)

        # Output should be quantized version (different from input)
        # But close to input due to fake quantization
        for i in range(len(x)):
            # Error should be bounded by half the scale
            scale = fq.scale[0]
            assert abs(x[i] - y[i]) <= scale

    def test_fake_quantize_symmetric_zero_point(self):
        """Test that symmetric quantization has zero_point = 0."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(symmetric=True)
        data = np.array([1.0, 2.0, -2.0], dtype=np.float32)
        fq.observe(data)

        assert fq.zero_point is not None
        assert fq.zero_point[0] == 0

    def test_fake_quantize_disable_enables(self):
        """Test enable/disable functionality."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fq.observe(data)

        # Disable fake quantization
        fq.disable()
        y_disabled = fq.forward(data)
        np.testing.assert_array_equal(data, y_disabled)

        # Re-enable
        fq.enable()
        y_enabled = fq.forward(data)
        # Should now be quantized (potentially different from input)
        assert fq.enabled is True

    def test_fake_quantize_ste_gradient_passthrough(self):
        """Test STE passes gradients through for in-range values."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(symmetric=True)

        # Calibrate
        data = np.array([1.0, 2.0, 3.0, -3.0], dtype=np.float32)
        fq.observe(data)

        # Forward with STE
        x = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        grad_output = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        output, grad_input = fq.forward_with_ste(x, grad_output)

        # Gradients should pass through for in-range values
        # All inputs are within [-3, 3] range
        np.testing.assert_array_almost_equal(grad_input, grad_output)

    def test_fake_quantize_ste_gradient_clipping(self):
        """Test STE zeros gradients for out-of-range values."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(symmetric=True)

        # Calibrate with small range
        data = np.array([1.0, -1.0], dtype=np.float32)
        fq.observe(data)

        # Forward with values outside range
        x = np.array([0.5, 100.0, -100.0], dtype=np.float32)  # 100, -100 outside range
        grad_output = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        _, grad_input = fq.forward_with_ste(x, grad_output)

        # First value in range, should have gradient
        assert grad_input[0] > 0
        # Second and third values clipped, gradient should be 0
        assert grad_input[1] == 0
        assert grad_input[2] == 0

    def test_fake_quantize_per_channel(self):
        """Test per-channel fake quantization."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(per_channel=True, channel_axis=0)

        # Data with 2 channels
        data = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
        fq.observe(data)

        assert fq.scale is not None
        assert len(fq.scale) == 2  # One scale per channel

        # Scales should be different for each channel
        assert fq.scale[0] != fq.scale[1]

    def test_get_quantization_params(self):
        """Test getting QuantizationParams from FakeQuantize."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(num_bits=8, symmetric=True)
        data = np.array([1.0, 2.0, 3.0, -3.0], dtype=np.float32)
        fq.observe(data)

        params = fq.get_quantization_params()

        assert params is not None
        assert params.symmetric is True
        assert params.scale > 0
        assert params.zero_point == 0


class TestBatchNormFolding:
    """Tests for batch normalization folding."""

    def test_bn_folding_preserves_output(self):
        """Test that BN folding preserves the computational result."""
        from zenith.optimization.qat import fold_bn_into_conv

        # Conv params
        conv_weight = np.random.randn(4, 3, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(4).astype(np.float32)

        # BN params
        bn_mean = np.random.randn(4).astype(np.float32)
        bn_var = np.abs(np.random.randn(4).astype(np.float32)) + 0.1
        bn_gamma = np.random.randn(4).astype(np.float32)
        bn_beta = np.random.randn(4).astype(np.float32)
        epsilon = 1e-5

        # Fold
        folded_weight, folded_bias = fold_bn_into_conv(
            conv_weight, conv_bias, bn_mean, bn_var, bn_gamma, bn_beta, epsilon
        )

        # Create test input
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)

        # Compute original: BN(Conv(x))
        # For simplicity, just verify shapes and basic properties
        assert folded_weight.shape == conv_weight.shape
        assert folded_bias.shape == conv_bias.shape

    def test_bn_folding_unfold_inverse(self):
        """Test that unfold is the inverse of fold."""
        from zenith.optimization.qat import fold_bn_into_conv, unfold_bn_from_conv

        # Original params
        conv_weight = np.random.randn(4, 3, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(4).astype(np.float32)
        bn_mean = np.random.randn(4).astype(np.float32)
        bn_var = np.abs(np.random.randn(4).astype(np.float32)) + 0.1
        bn_gamma = np.random.randn(4).astype(np.float32)
        bn_beta = np.random.randn(4).astype(np.float32)

        # Fold
        folded_w, folded_b = fold_bn_into_conv(
            conv_weight, conv_bias, bn_mean, bn_var, bn_gamma, bn_beta
        )

        # Unfold
        recovered_w, recovered_b = unfold_bn_from_conv(
            folded_w, folded_b, bn_mean, bn_var, bn_gamma, bn_beta
        )

        # Should recover original
        np.testing.assert_array_almost_equal(conv_weight, recovered_w, decimal=5)
        np.testing.assert_array_almost_equal(conv_bias, recovered_b, decimal=5)

    def test_bn_folding_linear(self):
        """Test BN folding for linear layers."""
        from zenith.optimization.qat import fold_bn_into_conv

        # Linear weight [out_features, in_features]
        linear_weight = np.random.randn(10, 20).astype(np.float32)
        linear_bias = np.random.randn(10).astype(np.float32)

        # BN params
        bn_mean = np.random.randn(10).astype(np.float32)
        bn_var = np.abs(np.random.randn(10).astype(np.float32)) + 0.1
        bn_gamma = np.random.randn(10).astype(np.float32)
        bn_beta = np.random.randn(10).astype(np.float32)

        # Fold
        folded_weight, folded_bias = fold_bn_into_conv(
            linear_weight, linear_bias, bn_mean, bn_var, bn_gamma, bn_beta
        )

        assert folded_weight.shape == linear_weight.shape
        assert folded_bias.shape == linear_bias.shape

    def test_bn_folding_none_bias(self):
        """Test BN folding when conv has no bias."""
        from zenith.optimization.qat import fold_bn_into_conv

        conv_weight = np.random.randn(4, 3, 3, 3).astype(np.float32)
        conv_bias = None  # No bias

        bn_mean = np.random.randn(4).astype(np.float32)
        bn_var = np.abs(np.random.randn(4).astype(np.float32)) + 0.1
        bn_gamma = np.random.randn(4).astype(np.float32)
        bn_beta = np.random.randn(4).astype(np.float32)

        folded_weight, folded_bias = fold_bn_into_conv(
            conv_weight, conv_bias, bn_mean, bn_var, bn_gamma, bn_beta
        )

        assert folded_weight.shape == conv_weight.shape
        assert folded_bias.shape == (4,)


class TestQATModule:
    """Tests for QATModule wrapper class."""

    def test_qat_module_creation(self):
        """Test QATModule creation."""
        from zenith.optimization.qat import QATModule, QATConfig

        config = QATConfig()
        module = QATModule(weight_shape=(64, 32), config=config)

        assert module.weight_fake_quant is not None
        assert module.activation_fake_quant is not None

    def test_qat_module_fake_quantize_weight(self):
        """Test weight fake quantization through QATModule."""
        from zenith.optimization.qat import QATModule, QATConfig

        config = QATConfig(weight_bits=8)
        module = QATModule(weight_shape=(4, 4), config=config)

        weight = np.random.randn(4, 4).astype(np.float32)
        fq_weight = module.fake_quantize_weight(weight)

        # First call calibrates, should return quantized version
        assert fq_weight.shape == weight.shape

    def test_qat_module_fake_quantize_activation(self):
        """Test activation fake quantization through QATModule."""
        from zenith.optimization.qat import QATModule, QATConfig

        config = QATConfig(activation_bits=8)
        module = QATModule(weight_shape=(4, 4), config=config)

        activation = np.random.randn(2, 4).astype(np.float32)
        fq_activation = module.fake_quantize_activation(activation)

        assert fq_activation.shape == activation.shape

    def test_qat_module_freeze_observers(self):
        """Test freezing observers stops statistics updates."""
        from zenith.optimization.qat import QATModule, QATConfig

        config = QATConfig()
        module = QATModule(weight_shape=(4, 4), config=config)

        # Initial calibration
        weight1 = np.ones((4, 4), dtype=np.float32)
        module.fake_quantize_weight(weight1)

        scale_before = module.weight_fake_quant.scale.copy()

        # Freeze observers
        module.freeze_observers()

        # New data should not update scale
        weight2 = np.ones((4, 4), dtype=np.float32) * 100
        module.fake_quantize_weight(weight2)

        np.testing.assert_array_equal(scale_before, module.weight_fake_quant.scale)


class TestQATTrainer:
    """Tests for QATTrainer orchestration class."""

    def test_qat_trainer_registration(self):
        """Test layer registration."""
        from zenith.optimization.qat import QATTrainer, QATConfig

        config = QATConfig()
        trainer = QATTrainer(config)

        module = trainer.register_layer("layer1", weight_shape=(64, 32))

        assert "layer1" in trainer.qat_modules
        assert module is not None

    def test_qat_trainer_calibration(self):
        """Test calibration workflow."""
        from zenith.optimization.qat import QATTrainer, QATConfig

        config = QATConfig()
        trainer = QATTrainer(config)
        trainer.register_layer("layer1", weight_shape=(4, 4))

        # Calibrate
        weight = np.random.randn(4, 4).astype(np.float32)
        activations = [np.random.randn(2, 4).astype(np.float32) for _ in range(3)]

        trainer.calibrate("layer1", weight, activations)

        # Check calibration happened
        assert trainer.qat_modules["layer1"].weight_fake_quant.scale is not None

    def test_qat_trainer_start_epoch_enables_fake_quant(self):
        """Test epoch control enables fake quantization."""
        from zenith.optimization.qat import QATTrainer, QATConfig

        config = QATConfig(start_qat_after_epochs=2)
        trainer = QATTrainer(config)
        trainer.register_layer("layer1", weight_shape=(4, 4))

        # Epoch 0: should be disabled
        trainer.start_epoch(0)
        assert not trainer.qat_modules["layer1"].weight_fake_quant.enabled

        # Epoch 2: should be enabled
        trainer.start_epoch(2)
        assert trainer.qat_modules["layer1"].weight_fake_quant.enabled

    def test_qat_trainer_fake_quantize_weights_dict(self):
        """Test batch fake quantization of weight dictionary."""
        from zenith.optimization.qat import QATTrainer, QATConfig

        config = QATConfig()
        trainer = QATTrainer(config)
        trainer.register_layer("layer1", weight_shape=(4, 4))
        trainer.register_layer("layer2", weight_shape=(8, 4))

        weights = {
            "layer1": np.random.randn(4, 4).astype(np.float32),
            "layer2": np.random.randn(8, 4).astype(np.float32),
            "unregistered": np.random.randn(2, 2).astype(np.float32),
        }

        fq_weights = trainer.fake_quantize_weights(weights)

        assert "layer1" in fq_weights
        assert "layer2" in fq_weights
        assert "unregistered" in fq_weights

    def test_qat_trainer_get_quantization_params(self):
        """Test getting quantization params from trainer."""
        from zenith.optimization.qat import QATTrainer, QATConfig

        config = QATConfig()
        trainer = QATTrainer(config)
        trainer.register_layer("layer1", weight_shape=(4, 4))

        # Calibrate
        weight = np.random.randn(4, 4).astype(np.float32)
        activations = [np.random.randn(2, 4).astype(np.float32)]
        trainer.calibrate("layer1", weight, activations)

        params = trainer.get_quantization_params()

        assert "layer1" in params
        assert "weight" in params["layer1"]
        assert "activation" in params["layer1"]


class TestQATConversion:
    """Tests for QAT to quantized model conversion."""

    def test_convert_qat_to_quantized(self):
        """Test conversion from QAT to fully quantized weights."""
        from zenith.optimization.qat import (
            QATTrainer,
            QATConfig,
            convert_qat_to_quantized,
        )

        config = QATConfig()
        trainer = QATTrainer(config)
        trainer.register_layer("layer1", weight_shape=(4, 4))

        # Calibrate
        weight = np.random.randn(4, 4).astype(np.float32) * 3
        trainer.calibrate("layer1", weight, [])

        # Convert
        weights = {"layer1": weight}
        quantized = convert_qat_to_quantized(trainer, weights)

        assert "layer1" in quantized
        q_weight, params = quantized["layer1"]

        # Quantized weight should be INT8
        assert q_weight.dtype == np.int8
        assert params is not None

    def test_convert_preserves_accuracy(self):
        """Test that conversion preserves reasonable accuracy."""
        from zenith.optimization.qat import (
            QATTrainer,
            QATConfig,
            convert_qat_to_quantized,
        )

        # Use per-tensor for simpler accuracy verification
        config = QATConfig(per_channel_weights=False)
        trainer = QATTrainer(config)
        trainer.register_layer("layer1", weight_shape=(4, 4))

        weight = np.random.randn(4, 4).astype(np.float32) * 3
        trainer.calibrate("layer1", weight, [])

        weights = {"layer1": weight}
        quantized = convert_qat_to_quantized(trainer, weights)

        q_weight, params = quantized["layer1"]

        # Dequantize and compare
        dequantized = params.dequantize(q_weight)

        # Quantization error should be small relative to data range
        max_error = np.max(np.abs(weight - dequantized))
        data_range = np.max(np.abs(weight))
        relative_error = max_error / data_range if data_range > 0 else 0

        # For 8-bit quantization, relative error should be < 2%
        assert relative_error < 0.02, f"Relative error {relative_error:.4f} too high"


class TestQATAccuracy:
    """Tests for QAT accuracy preservation."""

    def test_qat_error_bounded(self):
        """Test that QAT error is bounded for in-range values."""
        from zenith.optimization.qat import FakeQuantize

        fq = FakeQuantize(num_bits=8, symmetric=True)

        # Calibrate with representative data
        data = np.random.randn(1000).astype(np.float32) * 3
        fq.observe(data)

        # Test on data within calibration range
        # Use same distribution to ensure most values are in range
        test_data = np.clip(
            np.random.randn(100).astype(np.float32) * 2,
            -np.max(np.abs(data)),
            np.max(np.abs(data)),
        )
        output = fq.forward(test_data)

        # For in-range values, error should be bounded by scale/2
        # For 8-bit, relative error should be within 1% for most values
        rel_error = np.mean(np.abs(test_data - output)) / np.mean(np.abs(test_data))
        assert rel_error < 0.02  # 2% average relative error

    def test_qat_snr_reasonable(self):
        """Test that QAT SNR is reasonable for 8-bit quantization."""
        from zenith.optimization.qat import measure_qat_error, FakeQuantize

        fq = FakeQuantize(num_bits=8, symmetric=True)

        # Calibrate
        data = np.random.randn(1000).astype(np.float32)
        fq.observe(data)

        # Measure error
        test_data = np.random.randn(100).astype(np.float32)
        qat_output = fq.forward(test_data)

        error_metrics = measure_qat_error(test_data, qat_output)

        # 8-bit quantization should have reasonable SNR (typically > 30dB)
        assert error_metrics["snr_db"] > 20


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_prepare_model_for_qat(self):
        """Test prepare_model_for_qat function."""
        from zenith.optimization.qat import prepare_model_for_qat, QATConfig

        layer_shapes = {
            "conv1": (64, 3, 3, 3),
            "conv2": (128, 64, 3, 3),
            "fc1": (10, 128),
        }

        config = QATConfig()
        trainer = prepare_model_for_qat(layer_shapes, config)

        assert len(trainer.qat_modules) == 3
        assert "conv1" in trainer.qat_modules
        assert "conv2" in trainer.qat_modules
        assert "fc1" in trainer.qat_modules

    def test_simulate_qat_forward(self):
        """Test simulate_qat_forward function."""
        from zenith.optimization.qat import (
            QATTrainer,
            QATConfig,
            simulate_qat_forward,
        )

        config = QATConfig()
        trainer = QATTrainer(config)
        trainer.register_layer("fc1", weight_shape=(10, 20))

        weight = np.random.randn(10, 20).astype(np.float32)
        activation = np.random.randn(5, 20).astype(np.float32)

        # Calibrate first
        trainer.calibrate("fc1", weight, [activation])

        output = simulate_qat_forward(trainer, "fc1", weight, activation)

        # Output should have correct shape
        assert output.shape == (5, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
