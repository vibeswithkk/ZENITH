"""
Test Suite for Triton Backend and Model Export.

Tests for:
- Triton model configuration generation
- Python backend template generation
- Model deployment utilities
- Export format conversion

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path


class TestTritonDataTypes:
    """Tests for Triton data type conversions."""

    def test_numpy_to_triton_dtype(self):
        """Test numpy dtype to Triton conversion."""
        from zenith.serving.triton_backend import DataType

        assert DataType.from_numpy(np.dtype("float32")) == DataType.FP32
        assert DataType.from_numpy(np.dtype("float16")) == DataType.FP16
        assert DataType.from_numpy(np.dtype("int8")) == DataType.INT8
        assert DataType.from_numpy(np.dtype("int32")) == DataType.INT32
        assert DataType.from_numpy(np.dtype("int64")) == DataType.INT64

    def test_triton_to_numpy_dtype(self):
        """Test Triton to numpy dtype conversion."""
        from zenith.serving.triton_backend import DataType

        assert DataType.FP32.to_numpy() == np.dtype("float32")
        assert DataType.FP16.to_numpy() == np.dtype("float16")
        assert DataType.INT8.to_numpy() == np.dtype("int8")


class TestModelConfig:
    """Tests for Triton ModelConfig."""

    def test_basic_config(self):
        """Test basic model config creation."""
        from zenith.serving.triton_backend import (
            ModelConfig,
            TensorConfig,
            DataType,
        )

        config = ModelConfig(
            name="test_model",
            platform="onnxruntime_onnx",
            max_batch_size=8,
            inputs=[
                TensorConfig("input", DataType.FP32, [3, 224, 224]),
            ],
            outputs=[
                TensorConfig("output", DataType.FP32, [1000]),
            ],
        )

        assert config.name == "test_model"
        assert config.platform == "onnxruntime_onnx"
        assert config.max_batch_size == 8
        assert len(config.inputs) == 1
        assert len(config.outputs) == 1

    def test_config_to_pbtxt(self):
        """Test config.pbtxt generation."""
        from zenith.serving.triton_backend import (
            ModelConfig,
            TensorConfig,
            DataType,
            InstanceGroup,
        )

        config = ModelConfig(
            name="bert_encoder",
            platform="python",
            max_batch_size=16,
            inputs=[
                TensorConfig("input_ids", DataType.INT64, [512]),
                TensorConfig("attention_mask", DataType.INT64, [512]),
            ],
            outputs=[
                TensorConfig("hidden_states", DataType.FP32, [512, 768]),
            ],
            instance_group=[InstanceGroup(count=2, kind="KIND_GPU", gpus=[0])],
        )

        pbtxt = config.to_pbtxt()

        assert 'name: "bert_encoder"' in pbtxt
        assert 'platform: "python"' in pbtxt
        assert "max_batch_size: 16" in pbtxt
        assert 'name: "input_ids"' in pbtxt
        assert "data_type: TYPE_INT64" in pbtxt
        assert 'name: "hidden_states"' in pbtxt
        assert "count: 2" in pbtxt
        assert "kind: KIND_GPU" in pbtxt

    def test_dynamic_batching_config(self):
        """Test dynamic batching configuration."""
        from zenith.serving.triton_backend import (
            ModelConfig,
            TensorConfig,
            DataType,
            DynamicBatching,
        )

        config = ModelConfig(
            name="model",
            inputs=[TensorConfig("x", DataType.FP32, [10])],
            outputs=[TensorConfig("y", DataType.FP32, [10])],
            dynamic_batching=DynamicBatching(
                preferred_batch_size=[8, 16, 32],
                max_queue_delay_microseconds=200,
            ),
        )

        pbtxt = config.to_pbtxt()

        assert "dynamic_batching" in pbtxt
        assert "preferred_batch_size: [ 8, 16, 32 ]" in pbtxt
        assert "max_queue_delay_microseconds: 200" in pbtxt


class TestTritonBackend:
    """Tests for TritonBackend class."""

    def test_backend_initialization(self):
        """Test backend initialization."""
        from zenith.serving.triton_backend import (
            TritonBackend,
            TritonBackendConfig,
        )

        config = TritonBackendConfig(
            model_repository="/tmp/models",
            enable_quantization=True,
        )
        backend = TritonBackend(config)

        assert backend.config.model_repository == "/tmp/models"
        assert backend.config.enable_quantization is True

    def test_python_backend_template_generation(self):
        """Test Python backend code template generation."""
        from zenith.serving.triton_backend import (
            TritonBackend,
            ModelConfig,
            TensorConfig,
            DataType,
        )

        config = ModelConfig(
            name="zenith_model",
            inputs=[TensorConfig("input", DataType.FP32, [100])],
            outputs=[TensorConfig("output", DataType.FP32, [10])],
        )

        backend = TritonBackend()
        template = backend.generate_python_backend_template(config)

        assert "class TritonPythonModel:" in template
        assert "def initialize(self, args):" in template
        assert "def execute(self, requests):" in template
        assert "input" in template
        assert "output" in template
        assert "input_names" in template
        assert "output_names" in template

    def test_deploy_model(self):
        """Test model deployment to repository."""
        from zenith.serving.triton_backend import (
            TritonBackend,
            TritonBackendConfig,
            ModelConfig,
            TensorConfig,
            DataType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy ONNX file
            model_file = Path(tmpdir) / "model.onnx"
            model_file.write_bytes(b"dummy onnx model")

            config = TritonBackendConfig(model_repository=str(Path(tmpdir) / "models"))
            backend = TritonBackend(config)

            model_config = ModelConfig(
                name="test_model",
                platform="onnxruntime_onnx",
                inputs=[TensorConfig("x", DataType.FP32, [10])],
                outputs=[TensorConfig("y", DataType.FP32, [10])],
            )

            model_dir = backend.deploy_model(
                "test_model", model_file, model_config, version=1
            )

            # Check structure
            assert model_dir.exists()
            assert (model_dir / "config.pbtxt").exists()
            assert (model_dir / "1").exists()
            assert (model_dir / "1" / "model.onnx").exists()

    def test_create_python_backend(self):
        """Test Python backend model creation."""
        from zenith.serving.triton_backend import (
            TritonBackend,
            TritonBackendConfig,
            ModelConfig,
            TensorConfig,
            DataType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TritonBackendConfig(model_repository=str(Path(tmpdir) / "models"))
            backend = TritonBackend(config)

            model_config = ModelConfig(
                name="custom_model",
                inputs=[TensorConfig("input", DataType.FP32, [100])],
                outputs=[TensorConfig("output", DataType.FP32, [10])],
            )

            code = """
class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        return []
"""

            model_dir = backend.create_python_backend(
                "custom_model", model_config, code, version=1
            )

            assert (model_dir / "config.pbtxt").exists()
            assert (model_dir / "1" / "model.py").exists()

    def test_list_models(self):
        """Test listing models in repository."""
        from zenith.serving.triton_backend import (
            TritonBackend,
            TritonBackendConfig,
            ModelConfig,
            TensorConfig,
            DataType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TritonBackendConfig(model_repository=str(Path(tmpdir) / "models"))
            backend = TritonBackend(config)

            # Create two models
            for name in ["model_a", "model_b"]:
                model_config = ModelConfig(
                    name=name,
                    inputs=[TensorConfig("x", DataType.FP32, [10])],
                    outputs=[TensorConfig("y", DataType.FP32, [10])],
                )
                backend.create_python_backend(
                    name, model_config, "# model code", version=1
                )

            models = backend.list_models()

            assert len(models) == 2
            names = {m["name"] for m in models}
            assert "model_a" in names
            assert "model_b" in names


class TestExportToTriton:
    """Tests for export_to_triton function."""

    def test_export_function(self):
        """Test export_to_triton convenience function."""
        from zenith.serving.triton_backend import export_to_triton

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model file
            model_file = Path(tmpdir) / "model.onnx"
            model_file.write_bytes(b"dummy")

            model_dir = export_to_triton(
                model_path=model_file,
                model_name="test",
                input_specs=[("input", [3, 224, 224], "float32")],
                output_specs=[("output", [1000], "float32")],
                repository_path=str(Path(tmpdir) / "models"),
                max_batch_size=8,
            )

            assert model_dir.exists()
            assert (model_dir / "config.pbtxt").exists()


class TestModelExport:
    """Tests for model export utilities."""

    def test_tensor_spec_creation(self):
        """Test TensorSpec creation."""
        from zenith.serving.model_export import TensorSpec

        spec = TensorSpec(
            name="input",
            shape=[-1, 3, 224, 224],
            dtype="float32",
            dynamic_axes=[0],
        )

        assert spec.name == "input"
        assert spec.shape == [-1, 3, 224, 224]
        assert spec.dynamic_axes == [0]

    def test_export_config(self):
        """Test ExportConfig creation."""
        from zenith.serving.model_export import ExportConfig, ExportFormat

        config = ExportConfig(
            format=ExportFormat.ONNX,
            opset_version=17,
            optimize=True,
            quantize=True,
            quantization_mode="int8",
        )

        assert config.format == ExportFormat.ONNX
        assert config.opset_version == 17
        assert config.quantize is True

    def test_zenith_model_exporter_creation(self):
        """Test ZenithModelExporter creation."""
        from zenith.serving.model_export import (
            ZenithModelExporter,
            ExportConfig,
            ExportFormat,
        )

        config = ExportConfig(format=ExportFormat.ONNX)
        exporter = ZenithModelExporter(config)

        assert exporter.config.format == ExportFormat.ONNX

    def test_metadata_generation(self):
        """Test metadata generation."""
        from zenith.serving.model_export import (
            ZenithModelExporter,
            ExportConfig,
            TensorSpec,
        )

        exporter = ZenithModelExporter()

        input_specs = [
            TensorSpec("input", [-1, 3, 224, 224], "float32"),
        ]
        output_specs = [
            TensorSpec("output", [-1, 1000], "float32"),
        ]

        metadata = exporter.generate_metadata("resnet50", input_specs, output_specs)

        assert metadata["model_name"] == "resnet50"
        assert metadata["format"] == "onnx"
        assert len(metadata["inputs"]) == 1
        assert len(metadata["outputs"]) == 1
        assert metadata["inputs"][0]["name"] == "input"

    def test_metadata_save(self):
        """Test saving metadata to file."""
        from zenith.serving.model_export import (
            ZenithModelExporter,
            TensorSpec,
        )
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ZenithModelExporter()

            metadata = exporter.generate_metadata(
                "test",
                [TensorSpec("x", [10], "float32")],
                [TensorSpec("y", [10], "float32")],
            )

            path = Path(tmpdir) / "metadata.json"
            exporter.save_metadata(metadata, path)

            assert path.exists()

            with open(path) as f:
                loaded = json.load(f)

            assert loaded["model_name"] == "test"


class TestExportFormatEnum:
    """Tests for ExportFormat enum."""

    def test_export_format_values(self):
        """Test ExportFormat enum values."""
        from zenith.serving.model_export import ExportFormat

        assert ExportFormat.ONNX.value == "onnx"
        assert ExportFormat.TORCHSCRIPT.value == "torchscript"
        assert ExportFormat.TENSORFLOW_SAVED.value == "tensorflow_saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
