"""
NVIDIA Triton Inference Server Backend

Implements a Triton-compatible Python backend for serving Zenith-optimized models.
This module provides:
- Triton model configuration generation
- Python backend implementation following Triton's TritonPythonModel interface
- Model repository management utilities
- Runtime inference integration

Based on CetakBiru Section 8.2 requirements.
Reference: https://github.com/triton-inference-server/python_backend

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np


# =============================================================================
# Triton Configuration Types
# =============================================================================


class DataType(Enum):
    """Triton data types matching proto definitions."""

    BOOL = "TYPE_BOOL"
    UINT8 = "TYPE_UINT8"
    UINT16 = "TYPE_UINT16"
    UINT32 = "TYPE_UINT32"
    UINT64 = "TYPE_UINT64"
    INT8 = "TYPE_INT8"
    INT16 = "TYPE_INT16"
    INT32 = "TYPE_INT32"
    INT64 = "TYPE_INT64"
    FP16 = "TYPE_FP16"
    FP32 = "TYPE_FP32"
    FP64 = "TYPE_FP64"
    STRING = "TYPE_STRING"

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "DataType":
        """Convert numpy dtype to Triton DataType."""
        mapping = {
            np.dtype("bool"): cls.BOOL,
            np.dtype("uint8"): cls.UINT8,
            np.dtype("uint16"): cls.UINT16,
            np.dtype("uint32"): cls.UINT32,
            np.dtype("uint64"): cls.UINT64,
            np.dtype("int8"): cls.INT8,
            np.dtype("int16"): cls.INT16,
            np.dtype("int32"): cls.INT32,
            np.dtype("int64"): cls.INT64,
            np.dtype("float16"): cls.FP16,
            np.dtype("float32"): cls.FP32,
            np.dtype("float64"): cls.FP64,
        }
        return mapping.get(dtype, cls.FP32)

    def to_numpy(self) -> np.dtype:
        """Convert Triton DataType to numpy dtype."""
        mapping = {
            DataType.BOOL: np.dtype("bool"),
            DataType.UINT8: np.dtype("uint8"),
            DataType.UINT16: np.dtype("uint16"),
            DataType.UINT32: np.dtype("uint32"),
            DataType.UINT64: np.dtype("uint64"),
            DataType.INT8: np.dtype("int8"),
            DataType.INT16: np.dtype("int16"),
            DataType.INT32: np.dtype("int32"),
            DataType.INT64: np.dtype("int64"),
            DataType.FP16: np.dtype("float16"),
            DataType.FP32: np.dtype("float32"),
            DataType.FP64: np.dtype("float64"),
        }
        return mapping.get(self, np.dtype("float32"))


class SchedulerKind(Enum):
    """Triton scheduling policies."""

    DYNAMIC = "dynamic_batching"
    SEQUENCE = "sequence_batching"
    ENSEMBLE = "ensemble_scheduling"


@dataclass
class TensorConfig:
    """Configuration for an input or output tensor."""

    name: str
    data_type: DataType
    dims: list[int]
    is_optional: bool = False
    format: str | None = None  # For images: FORMAT_NHWC, FORMAT_NCHW

    def to_dict(self) -> dict:
        """Convert to Triton config format."""
        result = {
            "name": self.name,
            "data_type": self.data_type.value,
            "dims": self.dims,
        }
        if self.is_optional:
            result["optional"] = True
        if self.format:
            result["format"] = self.format
        return result


@dataclass
class InstanceGroup:
    """Model instance configuration."""

    count: int = 1
    kind: str = "KIND_AUTO"  # KIND_CPU, KIND_GPU, KIND_AUTO
    gpus: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {"count": self.count, "kind": self.kind}
        if self.gpus:
            result["gpus"] = self.gpus
        return result


@dataclass
class DynamicBatching:
    """Dynamic batching configuration."""

    preferred_batch_size: list[int] = field(default_factory=lambda: [4, 8, 16])
    max_queue_delay_microseconds: int = 100
    preserve_ordering: bool = False

    def to_dict(self) -> dict:
        return {
            "preferred_batch_size": self.preferred_batch_size,
            "max_queue_delay_microseconds": self.max_queue_delay_microseconds,
            "preserve_ordering": self.preserve_ordering,
        }


@dataclass
class ModelConfig:
    """
    Complete Triton model configuration.

    Generates config.pbtxt format for Triton model repositories.
    """

    name: str
    platform: str = "python"  # python, tensorrt_plan, onnxruntime_onnx
    max_batch_size: int = 0
    inputs: list[TensorConfig] = field(default_factory=list)
    outputs: list[TensorConfig] = field(default_factory=list)
    instance_group: list[InstanceGroup] = field(default_factory=list)
    dynamic_batching: DynamicBatching | None = None
    version_policy: str = "latest"  # latest, all, specific
    parameters: dict[str, str] = field(default_factory=dict)

    def to_pbtxt(self) -> str:
        """Generate config.pbtxt content."""
        lines = [f'name: "{self.name}"']

        if self.platform:
            lines.append(f'platform: "{self.platform}"')

        if self.max_batch_size > 0:
            lines.append(f"max_batch_size: {self.max_batch_size}")

        # Inputs
        for inp in self.inputs:
            lines.append("input {")
            lines.append(f'  name: "{inp.name}"')
            lines.append(f"  data_type: {inp.data_type.value}")
            dims_str = ", ".join(str(d) for d in inp.dims)
            lines.append(f"  dims: [ {dims_str} ]")
            if inp.is_optional:
                lines.append("  optional: true")
            lines.append("}")

        # Outputs
        for out in self.outputs:
            lines.append("output {")
            lines.append(f'  name: "{out.name}"')
            lines.append(f"  data_type: {out.data_type.value}")
            dims_str = ", ".join(str(d) for d in out.dims)
            lines.append(f"  dims: [ {dims_str} ]")
            lines.append("}")

        # Instance groups
        for ig in self.instance_group:
            lines.append("instance_group {")
            lines.append(f"  count: {ig.count}")
            lines.append(f"  kind: {ig.kind}")
            if ig.gpus:
                gpus_str = ", ".join(str(g) for g in ig.gpus)
                lines.append(f"  gpus: [ {gpus_str} ]")
            lines.append("}")

        # Dynamic batching
        if self.dynamic_batching:
            lines.append("dynamic_batching {")
            batch_sizes = ", ".join(
                str(s) for s in self.dynamic_batching.preferred_batch_size
            )
            lines.append(f"  preferred_batch_size: [ {batch_sizes} ]")
            lines.append(
                f"  max_queue_delay_microseconds: "
                f"{self.dynamic_batching.max_queue_delay_microseconds}"
            )
            lines.append("}")

        # Parameters
        for key, value in self.parameters.items():
            lines.append("parameters {")
            lines.append(f'  key: "{key}"')
            lines.append("  value {")
            lines.append(f'    string_value: "{value}"')
            lines.append("  }")
            lines.append("}")

        return "\n".join(lines)


# =============================================================================
# Triton Backend Configuration
# =============================================================================


@dataclass
class TritonBackendConfig:
    """Configuration for Zenith Triton backend."""

    model_repository: str = "/models"
    enable_quantization: bool = False
    quantization_mode: str = "int8"  # int8, fp16
    enable_optimization: bool = True
    optimization_level: int = 2
    device: str = "cuda:0"
    max_workspace_size_mb: int = 1024


# =============================================================================
# Triton Backend Implementation
# =============================================================================


class TritonBackend:
    """
    Zenith backend for NVIDIA Triton Inference Server.

    Implements the Triton Python backend interface to serve Zenith-optimized
    models through Triton's high-performance inference infrastructure.
    """

    def __init__(self, config: TritonBackendConfig = TritonBackendConfig()):
        """
        Initialize Triton backend.

        Args:
            config: Backend configuration
        """
        self.config = config
        self.model_repository = Path(config.model_repository)
        self.loaded_models: dict[str, Any] = {}

    def deploy_model(
        self,
        model_name: str,
        model_path: str | Path,
        model_config: ModelConfig,
        version: int = 1,
    ) -> Path:
        """
        Deploy a model to the Triton model repository.

        Args:
            model_name: Name for the deployed model
            model_path: Path to the model file (ONNX, TorchScript, etc.)
            model_config: Triton model configuration
            version: Model version number

        Returns:
            Path to the deployed model directory
        """
        # Create model directory structure
        # <model_repository>/<model_name>/<version>/model.<ext>
        model_dir = self.model_repository / model_name
        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write config.pbtxt
        config_path = model_dir / "config.pbtxt"
        config_path.write_text(model_config.to_pbtxt())

        # Copy or link model file
        model_path = Path(model_path)
        if model_path.suffix == ".onnx":
            target_name = "model.onnx"
        elif model_path.suffix == ".pt":
            target_name = "model.pt"
        else:
            target_name = f"model{model_path.suffix}"

        target_path = version_dir / target_name
        if model_path.exists():
            import shutil

            shutil.copy2(model_path, target_path)

        return model_dir

    def create_python_backend(
        self,
        model_name: str,
        model_config: ModelConfig,
        inference_code: str,
        version: int = 1,
    ) -> Path:
        """
        Create a Python backend model for Triton.

        Args:
            model_name: Name for the model
            model_config: Triton model configuration
            inference_code: Python code implementing TritonPythonModel
            version: Model version number

        Returns:
            Path to the model directory
        """
        # Update config for Python backend
        model_config.platform = "python"

        model_dir = self.model_repository / model_name
        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write config.pbtxt
        config_path = model_dir / "config.pbtxt"
        config_path.write_text(model_config.to_pbtxt())

        # Write model.py
        model_py_path = version_dir / "model.py"
        model_py_path.write_text(inference_code)

        return model_dir

    def generate_python_backend_template(
        self,
        model_config: ModelConfig,
        use_zenith: bool = True,
    ) -> str:
        """
        Generate Python backend code template for Triton.

        Args:
            model_config: Model configuration
            use_zenith: Include Zenith integrations

        Returns:
            Python code for model.py
        """
        input_names = [inp.name for inp in model_config.inputs]
        output_names = [out.name for out in model_config.outputs]

        template = f'''"""
Triton Python Backend for {model_config.name}
Auto-generated by Zenith
"""

import json
import numpy as np

# Triton imports (available at runtime in Triton container)
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None  # For testing outside Triton


class TritonPythonModel:
    """
    Triton Python backend model implementing inference for {model_config.name}.
    """

    def initialize(self, args):
        """
        Initialize the model.

        Args:
            args: Dictionary with 'model_config', 'model_instance_kind',
                  'model_instance_device_id', 'model_repository',
                  'model_version', 'model_name'
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.model_name = args['model_name']
        self.model_version = args['model_version']

        # Parse input/output configuration
        self.input_names = {input_names!r}
        self.output_names = {output_names!r}

        # Load Zenith model if available
        self._load_model(args)

    def _load_model(self, args):
        """Load the Zenith-optimized model."""
        model_path = args.get('model_repository', '')
        version = args.get('model_version', '1')

        # Placeholder: Load your model here
        # Example for ONNX:
        # import onnxruntime as ort
        # self.session = ort.InferenceSession(
        #     f"{{model_path}}/{{version}}/model.onnx"
        # )
        self.model = None

    def execute(self, requests):
        """
        Execute inference on batch of requests.

        Args:
            requests: List of pb_utils.InferenceRequest

        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []

        for request in requests:
            # Extract inputs
            inputs = {{}}
            for name in self.input_names:
                tensor = pb_utils.get_input_tensor_by_name(request, name)
                inputs[name] = tensor.as_numpy()

            # Run inference
            try:
                outputs = self._infer(inputs)

                # Create output tensors
                output_tensors = []
                for name in self.output_names:
                    if name in outputs:
                        out_tensor = pb_utils.Tensor(name, outputs[name])
                        output_tensors.append(out_tensor)

                response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )

            except Exception as e:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )

            responses.append(response)

        return responses

    def _infer(self, inputs):
        """
        Perform inference on inputs.

        Args:
            inputs: Dictionary of input name -> numpy array

        Returns:
            Dictionary of output name -> numpy array
        """
        # Placeholder: Implement your inference logic
        # Example:
        # results = self.session.run(None, inputs)
        # return dict(zip(self.output_names, results))

        # Default: echo first input as output
        first_input = list(inputs.values())[0]
        return {{self.output_names[0]: first_input}}

    def finalize(self):
        """Clean up resources."""
        self.model = None
'''
        return template

    def list_models(self) -> list[dict]:
        """
        List all models in the repository.

        Returns:
            List of model info dictionaries
        """
        models = []

        if not self.model_repository.exists():
            return models

        for model_dir in self.model_repository.iterdir():
            if not model_dir.is_dir():
                continue

            model_info = {
                "name": model_dir.name,
                "versions": [],
                "config_exists": (model_dir / "config.pbtxt").exists(),
            }

            # Find versions
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.isdigit():
                    model_info["versions"].append(int(version_dir.name))

            model_info["versions"].sort()
            models.append(model_info)

        return models


# =============================================================================
# Export Utilities
# =============================================================================


def export_to_triton(
    model_path: str | Path,
    model_name: str,
    input_specs: list[tuple[str, list[int], str]],
    output_specs: list[tuple[str, list[int], str]],
    repository_path: str | Path = "/models",
    max_batch_size: int = 8,
    enable_dynamic_batching: bool = True,
) -> Path:
    """
    Export a model to Triton format.

    Args:
        model_path: Path to model file (ONNX, TorchScript)
        model_name: Name for the Triton model
        input_specs: List of (name, dims, dtype) for inputs
        output_specs: List of (name, dims, dtype) for outputs
        repository_path: Triton model repository path
        max_batch_size: Maximum batch size (0 for no batching)
        enable_dynamic_batching: Enable dynamic batching

    Returns:
        Path to deployed model directory
    """
    # Create tensor configs
    inputs = []
    for name, dims, dtype in input_specs:
        np_dtype = np.dtype(dtype)
        data_type = DataType.from_numpy(np_dtype)
        inputs.append(TensorConfig(name=name, data_type=data_type, dims=dims))

    outputs = []
    for name, dims, dtype in output_specs:
        np_dtype = np.dtype(dtype)
        data_type = DataType.from_numpy(np_dtype)
        outputs.append(TensorConfig(name=name, data_type=data_type, dims=dims))

    # Determine platform from file extension
    model_path = Path(model_path)
    if model_path.suffix == ".onnx":
        platform = "onnxruntime_onnx"
    elif model_path.suffix == ".plan":
        platform = "tensorrt_plan"
    elif model_path.suffix == ".pt":
        platform = "pytorch_libtorch"
    else:
        platform = "python"

    # Create model config
    config = ModelConfig(
        name=model_name,
        platform=platform,
        max_batch_size=max_batch_size,
        inputs=inputs,
        outputs=outputs,
        instance_group=[InstanceGroup(count=1, kind="KIND_AUTO")],
    )

    if enable_dynamic_batching and max_batch_size > 0:
        config.dynamic_batching = DynamicBatching()

    # Deploy
    backend = TritonBackend(TritonBackendConfig(model_repository=str(repository_path)))
    return backend.deploy_model(model_name, model_path, config)


def create_triton_model_config(
    model_name: str,
    inputs: list[dict],
    outputs: list[dict],
    platform: str = "onnxruntime_onnx",
    max_batch_size: int = 8,
) -> ModelConfig:
    """
    Create a ModelConfig from simple dictionaries.

    Args:
        model_name: Name of the model
        inputs: List of input specs [{"name": str, "dims": list, "dtype": str}]
        outputs: List of output specs
        platform: Triton platform
        max_batch_size: Maximum batch size

    Returns:
        ModelConfig instance
    """
    input_configs = []
    for inp in inputs:
        data_type = DataType.from_numpy(np.dtype(inp.get("dtype", "float32")))
        input_configs.append(
            TensorConfig(
                name=inp["name"],
                data_type=data_type,
                dims=inp["dims"],
            )
        )

    output_configs = []
    for out in outputs:
        data_type = DataType.from_numpy(np.dtype(out.get("dtype", "float32")))
        output_configs.append(
            TensorConfig(
                name=out["name"],
                data_type=data_type,
                dims=out["dims"],
            )
        )

    return ModelConfig(
        name=model_name,
        platform=platform,
        max_batch_size=max_batch_size,
        inputs=input_configs,
        outputs=output_configs,
        instance_group=[InstanceGroup(count=1)],
        dynamic_batching=DynamicBatching() if max_batch_size > 0 else None,
    )
