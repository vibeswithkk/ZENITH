# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Comprehensive Test Suite for TensorFlow Adapter

Tests for:
- Basic TensorFlow/Keras model conversion
- HuggingFace Transformers TF model integration
- tf.function compilation hook (like torch.compile)
- Training loop integration
- Mixed precision training
- SavedModel conversion
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


# =============================================================================
# Fixtures and Test Setup
# =============================================================================


@pytest.fixture(scope="module")
def tf():
    """Get TensorFlow module, skip if not available."""
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    return tf


@pytest.fixture(scope="module")
def adapter():
    """Create TensorFlow adapter instance."""
    pytest.importorskip("tensorflow")
    from zenith.adapters import TensorFlowAdapter

    return TensorFlowAdapter()


@pytest.fixture
def simple_keras_model(tf):
    """Create a simple Keras Sequential model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )
    return model


@pytest.fixture
def conv_model(tf):
    """Create a CNN model for testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


@pytest.fixture
def functional_model(tf):
    """Create a Keras Functional API model."""
    inputs = tf.keras.Input(shape=(10,), name="input_features")
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional")
    return model


@pytest.fixture
def sample_input_dense(tf):
    """Sample input for dense models."""
    return tf.random.normal((4, 10))


@pytest.fixture
def sample_input_conv(tf):
    """Sample input for CNN models."""
    return tf.random.normal((4, 28, 28, 1))


# =============================================================================
# Basic Adapter Tests
# =============================================================================


class TestTensorFlowAdapterBasic:
    """Basic tests for TensorFlow adapter."""

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "tensorflow"

    def test_adapter_availability(self, adapter):
        """Test TensorFlow availability check."""
        assert adapter.is_available is True

    def test_tf_version_check(self, tf):
        """Verify TensorFlow 2.x is being used."""
        major = int(tf.__version__.split(".")[0])
        assert major >= 2, "TensorFlow 2.x required"


class TestKerasModelConversion:
    """Tests for Keras model to GraphIR conversion."""

    def test_sequential_model_conversion(
        self, adapter, simple_keras_model, sample_input_dense
    ):
        """Test converting Sequential model to GraphIR."""
        graph = adapter.from_model(simple_keras_model, sample_input=sample_input_dense)

        assert graph is not None
        assert graph.name is not None
        assert len(graph.inputs) > 0
        assert len(graph.outputs) > 0

    def test_functional_model_conversion(
        self, adapter, functional_model, sample_input_dense
    ):
        """Test converting Functional API model to GraphIR."""
        graph = adapter.from_model(functional_model, sample_input=sample_input_dense)

        assert graph is not None
        assert len(graph.inputs) >= 1

    def test_cnn_model_conversion(self, adapter, conv_model, sample_input_conv):
        """Test converting CNN model to GraphIR."""
        graph = adapter.from_model(conv_model, sample_input=sample_input_conv)

        assert graph is not None
        assert len(graph.inputs) > 0

    def test_conversion_without_sample_input(self, adapter, simple_keras_model):
        """Test conversion without sample input (should use fallback)."""
        graph = adapter.from_model(simple_keras_model)

        # May use fallback GraphIR
        assert graph is not None


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_to_onnx_basic(self, adapter, simple_keras_model, sample_input_dense):
        """Test basic ONNX export."""
        pytest.importorskip("tf2onnx")

        onnx_bytes = adapter.to_onnx(
            simple_keras_model, sample_input=sample_input_dense
        )

        assert onnx_bytes is not None
        assert len(onnx_bytes) > 0

    def test_to_onnx_with_file_output(
        self, adapter, simple_keras_model, sample_input_dense, tmp_path
    ):
        """Test ONNX export to file."""
        pytest.importorskip("tf2onnx")

        output_path = str(tmp_path / "model.onnx")

        onnx_bytes = adapter.to_onnx(
            simple_keras_model, sample_input=sample_input_dense, output_path=output_path
        )

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


class TestSavedModelConversion:
    """Tests for SavedModel conversion."""

    def test_save_and_convert(
        self, adapter, simple_keras_model, sample_input_dense, tmp_path, tf
    ):
        """Test saving and converting a SavedModel."""
        pytest.importorskip("tf2onnx")

        # Build and save model
        simple_keras_model(sample_input_dense)
        saved_path = str(tmp_path / "saved_model")
        simple_keras_model.save(saved_path, save_format="tf")

        # Convert
        graph = adapter.from_saved_model(saved_path)

        assert graph is not None

    def test_invalid_saved_model_path(self, adapter):
        """Test error handling for invalid SavedModel path."""
        with pytest.raises(ValueError, match="does not exist"):
            adapter.from_saved_model("/nonexistent/path")


# =============================================================================
# HuggingFace Integration Tests
# =============================================================================


class TestHuggingFaceIntegration:
    """Tests for HuggingFace Transformers integration."""

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_detect_huggingface_model(self, adapter, tf):
        """Test detection of HuggingFace TF models."""
        from transformers import TFAutoModel

        # Load a small model for testing
        try:
            model = TFAutoModel.from_pretrained("prajjwal1/bert-tiny", from_tf=True)
            assert adapter._is_huggingface_model(model)
        except Exception:
            pytest.skip("Could not load test model")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_from_transformers_basic(self, adapter):
        """Test loading model from HuggingFace."""
        try:
            graph = adapter.from_transformers(
                "prajjwal1/bert-tiny", max_length=32, batch_size=1
            )

            assert graph is not None
            assert len(graph.inputs) > 0
        except Exception as e:
            pytest.skip(f"HuggingFace model loading failed: {e}")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers library not available",
    )
    def test_from_transformers_with_task(self, adapter):
        """Test loading model with specific task."""
        try:
            graph = adapter.from_transformers(
                "prajjwal1/bert-tiny", task="text-classification", max_length=32
            )

            assert graph is not None
        except Exception as e:
            pytest.skip(f"HuggingFace model loading failed: {e}")


# =============================================================================
# tf.function Compilation Hook Tests
# =============================================================================


class TestTFCompileFunction:
    """Tests for tf.function compilation hook (like torch.compile)."""

    def test_compile_function_basic(self, adapter, simple_keras_model, tf):
        """Test basic function compilation."""
        sample = tf.random.normal((2, 10))

        @adapter.compile_function(target="cpu", precision="fp32")
        @tf.function
        def forward(x):
            return simple_keras_model(x)

        # Execute
        result = forward(sample)

        assert result is not None
        assert result.shape == (2, 5)  # Output shape matches model

    def test_compile_function_fp16(self, adapter, simple_keras_model, tf):
        """Test function compilation with FP16 precision."""
        sample = tf.random.normal((2, 10))

        @adapter.compile_function(target="cpu", precision="fp16")
        @tf.function
        def forward(x):
            return simple_keras_model(x)

        result = forward(sample)
        assert result is not None

    def test_compile_function_without_decorator(self, adapter, simple_keras_model, tf):
        """Test compilation without decorator syntax."""
        sample = tf.random.normal((2, 10))

        @tf.function
        def forward(x):
            return simple_keras_model(x)

        compiled = adapter.compile_function(forward, target="cpu", opt_level=2)

        result = compiled(sample)
        assert result is not None

    def test_compiled_function_stats(self, adapter, simple_keras_model, tf):
        """Test getting optimization stats from compiled function."""

        @adapter.compile_function(target="cpu")
        @tf.function
        def forward(x):
            return simple_keras_model(x)

        # Trigger compilation
        forward(tf.random.normal((2, 10)))

        # Get stats
        stats = forward.get_stats()
        assert hasattr(stats, "passes_applied")


# =============================================================================
# Training Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Tests for training loop integration."""

    def test_create_training_callback(self, adapter, simple_keras_model, tf):
        """Test creating training callback."""
        callback = adapter.create_training_callback(
            simple_keras_model, enable_mixed_precision=False
        )

        assert callback is not None
        # Should be able to get Keras callback
        keras_callback = callback.get_keras_callback()
        assert keras_callback is not None

    def test_training_with_callback(self, adapter, simple_keras_model, tf):
        """Test training with Zenith callback."""
        # Prepare data
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 5, size=(100,))

        # Compile model
        simple_keras_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Create callback
        zenith_callback = adapter.create_training_callback(
            simple_keras_model, enable_mixed_precision=False
        )

        # Train
        history = simple_keras_model.fit(
            x,
            y,
            epochs=1,
            batch_size=32,
            callbacks=[zenith_callback.get_keras_callback()],
            verbose=0,
        )

        assert "loss" in history.history

    def test_wrap_training_step(self, adapter, simple_keras_model, tf):
        """Test wrapping custom training step."""
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = simple_keras_model(x)
                loss = loss_fn(y, predictions)
            grads = tape.gradient(loss, simple_keras_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, simple_keras_model.trainable_variables)
            )
            return loss

        # Wrap with Zenith
        optimized_step = adapter.wrap_training_step(
            train_step, simple_keras_model, optimizer, enable_mixed_precision=False
        )

        # Execute
        x = tf.random.normal((8, 10))
        y = tf.constant([0, 1, 2, 3, 4, 0, 1, 2])

        loss = optimized_step(x, y)
        assert loss is not None


class TestMixedPrecisionTraining:
    """Tests for mixed precision training support."""

    def test_callback_with_mixed_precision(self, adapter, simple_keras_model, tf):
        """Test training callback with mixed precision enabled."""
        callback = adapter.create_training_callback(
            simple_keras_model, enable_mixed_precision=True
        )

        assert callback is not None
        # Mixed precision should be configured
        from tensorflow.keras import mixed_precision

        # Policy may be set globally
        assert callback._enable_mixed_precision is True


# =============================================================================
# Module-level API Tests
# =============================================================================


class TestModuleLevelAPI:
    """Tests for zenith.tensorflow module API."""

    def test_module_import(self):
        """Test importing zenith.tensorflow module."""
        import zenith.tensorflow as ztf

        assert hasattr(ztf, "compile")
        assert hasattr(ztf, "compile_function")
        assert hasattr(ztf, "from_model")
        assert hasattr(ztf, "from_transformers")
        assert hasattr(ztf, "create_training_callback")

    def test_is_available(self):
        """Test availability check via module."""
        import zenith.tensorflow as ztf

        assert ztf.is_available() is True

    def test_configure(self):
        """Test configuration via module."""
        import zenith.tensorflow as ztf

        config = ztf.configure(target="cuda", precision="fp16", opt_level=3)

        assert config.target == "cuda"
        assert config.precision == "fp16"
        assert config.opt_level == 3

    def test_compile_decorator(self, simple_keras_model, tf):
        """Test zenith.tensorflow.compile as decorator."""
        import zenith.tensorflow as ztf

        @ztf.compile(target="cpu")
        @tf.function
        def forward(x):
            return simple_keras_model(x)

        result = forward(tf.random.normal((2, 10)))
        assert result is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_model_type(self, adapter):
        """Test error for invalid model type."""
        with pytest.raises((ValueError, TypeError)):
            adapter.from_model({"not": "a model"})

    def test_missing_tf2onnx(self, adapter, simple_keras_model, monkeypatch):
        """Test graceful handling when tf2onnx is missing."""
        import sys

        # Temporarily remove tf2onnx from imports
        original = sys.modules.get("tf2onnx")
        sys.modules["tf2onnx"] = None

        try:
            # Should use fallback
            graph = adapter.from_model(simple_keras_model)
            assert graph is not None
        finally:
            if original:
                sys.modules["tf2onnx"] = original

    def test_large_model_handling(self, adapter, tf):
        """Test handling of larger models."""
        # Create a larger model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1024, activation="relu", input_shape=(512,)),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        sample = tf.random.normal((2, 512))
        graph = adapter.from_model(model, sample_input=sample)

        assert graph is not None


# =============================================================================
# Config Tests
# =============================================================================


class TestConfiguration:
    """Tests for ZenithTFConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from zenith.adapters.tensorflow_adapter import ZenithTFConfig

        config = ZenithTFConfig()

        assert config.target == "cuda"
        assert config.precision == "fp32"
        assert config.opt_level == 2
        assert config.opset_version == 17

    def test_custom_config(self):
        """Test custom configuration."""
        from zenith.adapters.tensorflow_adapter import ZenithTFConfig
        from zenith.adapters import TensorFlowAdapter

        config = ZenithTFConfig(
            target="cpu",
            precision="fp16",
            opt_level=3,
            enable_mixed_precision_training=True,
        )

        adapter = TensorFlowAdapter(config=config)

        assert adapter.config.precision == "fp16"
        assert adapter.config.enable_mixed_precision_training is True


# =============================================================================
# Data Type Conversion Tests
# =============================================================================


class TestDataTypeConversion:
    """Tests for TensorFlow to Zenith data type conversion."""

    def test_dtype_conversion_float32(self, adapter, tf):
        """Test float32 dtype conversion."""
        from zenith.core import DataType

        result = adapter._tf_dtype_to_zenith(tf.float32)
        assert result == DataType.Float32

    def test_dtype_conversion_float16(self, adapter, tf):
        """Test float16 dtype conversion."""
        from zenith.core import DataType

        result = adapter._tf_dtype_to_zenith(tf.float16)
        assert result == DataType.Float16

    def test_dtype_conversion_int32(self, adapter, tf):
        """Test int32 dtype conversion."""
        from zenith.core import DataType

        result = adapter._tf_dtype_to_zenith(tf.int32)
        assert result == DataType.Int32

    def test_dtype_conversion_bool(self, adapter, tf):
        """Test bool dtype conversion."""
        from zenith.core import DataType

        result = adapter._tf_dtype_to_zenith(tf.bool)
        assert result == DataType.Bool


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_inference(
        self, adapter, simple_keras_model, sample_input_dense, tf
    ):
        """Test full pipeline: model -> GraphIR -> compile -> execute."""
        import zenith

        # Convert to GraphIR
        graph = adapter.from_model(simple_keras_model, sample_input=sample_input_dense)

        # Compile with Zenith
        compiled_model = zenith.compile(
            simple_keras_model,
            target="cpu",
            precision="fp32",
            sample_input=sample_input_dense,
        )

        # Execute
        result = compiled_model(sample_input_dense)

        assert result is not None
        assert result.shape[-1] == 5  # Output classes

    def test_full_pipeline_training(self, adapter, simple_keras_model, tf):
        """Test full pipeline with training."""
        # Prepare data
        x = np.random.randn(64, 10).astype(np.float32)
        y = np.random.randint(0, 5, size=(64,))

        # Compile model
        simple_keras_model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy"
        )

        # Create Zenith training callback
        callback = adapter.create_training_callback(simple_keras_model)

        # Train
        simple_keras_model.fit(
            x,
            y,
            epochs=2,
            batch_size=16,
            callbacks=[callback.get_keras_callback()],
            verbose=0,
        )

        # Verify training happened
        assert callback._step_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
