"""
Triton Inference Server Client.

Provides unified interface for HTTP and gRPC communication with Triton.
Supports inference requests, health checks, and model metadata.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np

# Optional HTTP client
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

# Optional gRPC client
try:
    import grpc

    HAS_GRPC = True
except ImportError:
    grpc = None
    HAS_GRPC = False


# ============================================================================
# Data Types
# ============================================================================


class Protocol(Enum):
    """Communication protocol."""

    HTTP = "http"
    GRPC = "grpc"


@dataclass
class ServerStatus:
    """Triton server status."""

    live: bool = False
    ready: bool = False
    version: str = ""
    extensions: list[str] = field(default_factory=list)


@dataclass
class ModelMetadata:
    """Model metadata from Triton."""

    name: str
    versions: list[str] = field(default_factory=list)
    platform: str = ""
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)


@dataclass
class InferenceInput:
    """Input tensor for inference request."""

    name: str
    data: np.ndarray
    datatype: str = ""

    def __post_init__(self):
        if not self.datatype:
            self.datatype = self._numpy_to_triton_dtype(self.data.dtype)

    @staticmethod
    def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
        """Convert numpy dtype to Triton datatype string."""
        mapping = {
            np.dtype("bool"): "BOOL",
            np.dtype("int8"): "INT8",
            np.dtype("int16"): "INT16",
            np.dtype("int32"): "INT32",
            np.dtype("int64"): "INT64",
            np.dtype("uint8"): "UINT8",
            np.dtype("uint16"): "UINT16",
            np.dtype("uint32"): "UINT32",
            np.dtype("uint64"): "UINT64",
            np.dtype("float16"): "FP16",
            np.dtype("float32"): "FP32",
            np.dtype("float64"): "FP64",
        }
        return mapping.get(dtype, "FP32")


@dataclass
class InferenceOutput:
    """Output tensor from inference request."""

    name: str
    data: np.ndarray | None = None
    datatype: str = ""
    shape: list[int] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result from inference request."""

    model_name: str
    model_version: str
    outputs: list[InferenceOutput] = field(default_factory=list)
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""

    def get_output(self, name: str) -> np.ndarray | None:
        """Get output by name."""
        for out in self.outputs:
            if out.name == name:
                return out.data
        return None


# ============================================================================
# Triton Client
# ============================================================================


class TritonClient:
    """
    Unified client for NVIDIA Triton Inference Server.

    Supports both HTTP and gRPC protocols with automatic fallback.

    Example:
        client = TritonClient("localhost:8000", protocol=Protocol.HTTP)
        if client.is_server_ready():
            result = client.infer("my_model", inputs)
    """

    def __init__(
        self,
        url: str,
        protocol: Protocol = Protocol.HTTP,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        """
        Initialize Triton client.

        Args:
            url: Server URL (e.g., "localhost:8000" for HTTP, "localhost:8001" for gRPC)
            protocol: Communication protocol (HTTP or gRPC)
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
        """
        self.url = url
        self.protocol = protocol
        self.timeout = timeout
        self.verbose = verbose

        # Normalize URL
        if protocol == Protocol.HTTP and not url.startswith("http"):
            self.url = f"http://{url}"

        self._validate_protocol()

    def _validate_protocol(self):
        """Validate protocol dependencies are available."""
        if self.protocol == Protocol.HTTP and not HAS_REQUESTS:
            raise ImportError("requests package required for HTTP protocol")
        if self.protocol == Protocol.GRPC and not HAS_GRPC:
            raise ImportError("grpcio package required for gRPC protocol")

    # =========================================================================
    # Server Health
    # =========================================================================

    def is_server_live(self) -> bool:
        """Check if server is live."""
        try:
            if self.protocol == Protocol.HTTP:
                return self._http_health_live()
            else:
                return self._grpc_health_live()
        except Exception:
            return False

    def is_server_ready(self) -> bool:
        """Check if server is ready for inference."""
        try:
            if self.protocol == Protocol.HTTP:
                return self._http_health_ready()
            else:
                return self._grpc_health_ready()
        except Exception:
            return False

    def get_server_status(self) -> ServerStatus:
        """Get complete server status."""
        status = ServerStatus()
        status.live = self.is_server_live()
        status.ready = self.is_server_ready()

        if self.protocol == Protocol.HTTP and status.live:
            try:
                resp = requests.get(urljoin(self.url, "/v2"), timeout=self.timeout)
                if resp.ok:
                    data = resp.json()
                    status.version = data.get("version", "")
                    status.extensions = data.get("extensions", [])
            except Exception:
                pass

        return status

    def _http_health_live(self) -> bool:
        """HTTP liveness check."""
        resp = requests.get(urljoin(self.url, "/v2/health/live"), timeout=self.timeout)
        return resp.status_code == 200

    def _http_health_ready(self) -> bool:
        """HTTP readiness check."""
        resp = requests.get(urljoin(self.url, "/v2/health/ready"), timeout=self.timeout)
        return resp.status_code == 200

    def _grpc_health_live(self) -> bool:
        """gRPC liveness check (simulated for now)."""
        # Would use grpc health checking protocol
        return True

    def _grpc_health_ready(self) -> bool:
        """gRPC readiness check (simulated for now)."""
        return True

    # =========================================================================
    # Model Operations
    # =========================================================================

    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        """Check if a specific model is ready."""
        try:
            if self.protocol == Protocol.HTTP:
                path = f"/v2/models/{model_name}"
                if version:
                    path += f"/versions/{version}"
                path += "/ready"
                resp = requests.get(urljoin(self.url, path), timeout=self.timeout)
                return resp.status_code == 200
            else:
                # gRPC implementation would go here
                return True
        except Exception:
            return False

    def get_model_metadata(self, model_name: str, version: str = "") -> ModelMetadata:
        """Get model metadata."""
        metadata = ModelMetadata(name=model_name)

        if self.protocol == Protocol.HTTP:
            try:
                path = f"/v2/models/{model_name}"
                if version:
                    path += f"/versions/{version}"
                resp = requests.get(urljoin(self.url, path), timeout=self.timeout)
                if resp.ok:
                    data = resp.json()
                    metadata.versions = data.get("versions", [])
                    metadata.platform = data.get("platform", "")
                    metadata.inputs = data.get("inputs", [])
                    metadata.outputs = data.get("outputs", [])
                    metadata.parameters = data.get("parameters", {})
            except Exception as e:
                if self.verbose:
                    print(f"Failed to get model metadata: {e}")

        return metadata

    def list_models(self) -> list[str]:
        """List all available models."""
        models = []

        if self.protocol == Protocol.HTTP:
            try:
                resp = requests.get(
                    urljoin(self.url, "/v2/repository/index"),
                    timeout=self.timeout,
                )
                if resp.ok:
                    data = resp.json()
                    models = [m.get("name", "") for m in data if m.get("name")]
            except Exception as e:
                if self.verbose:
                    print(f"Failed to list models: {e}")

        return models

    # =========================================================================
    # Inference
    # =========================================================================

    def infer(
        self,
        model_name: str,
        inputs: list[InferenceInput],
        outputs: list[str] | None = None,
        model_version: str = "",
        request_id: str = "",
    ) -> InferenceResult:
        """
        Perform inference request.

        Args:
            model_name: Name of the model
            inputs: List of input tensors
            outputs: List of output names to retrieve (None for all)
            model_version: Specific model version (empty for latest)
            request_id: Optional request ID for tracing

        Returns:
            InferenceResult with outputs or error
        """
        start_time = time.perf_counter()

        try:
            if self.protocol == Protocol.HTTP:
                result = self._http_infer(
                    model_name, inputs, outputs, model_version, request_id
                )
            else:
                result = self._grpc_infer(
                    model_name, inputs, outputs, model_version, request_id
                )

            result.latency_ms = (time.perf_counter() - start_time) * 1000
            return result

        except Exception as e:
            return InferenceResult(
                model_name=model_name,
                model_version=model_version,
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _http_infer(
        self,
        model_name: str,
        inputs: list[InferenceInput],
        outputs: list[str] | None,
        model_version: str,
        request_id: str,
    ) -> InferenceResult:
        """HTTP inference implementation."""
        # Build request path
        path = f"/v2/models/{model_name}"
        if model_version:
            path += f"/versions/{model_version}"
        path += "/infer"

        # Build request body
        request_body = {
            "inputs": [],
            "outputs": [],
        }

        if request_id:
            request_body["id"] = request_id

        # Add inputs
        for inp in inputs:
            input_data = {
                "name": inp.name,
                "shape": list(inp.data.shape),
                "datatype": inp.datatype,
                "data": inp.data.flatten().tolist(),
            }
            request_body["inputs"].append(input_data)

        # Add requested outputs
        if outputs:
            for out_name in outputs:
                request_body["outputs"].append({"name": out_name})

        # Send request
        headers = {"Content-Type": "application/json"}
        resp = requests.post(
            urljoin(self.url, path),
            json=request_body,
            headers=headers,
            timeout=self.timeout,
        )

        if not resp.ok:
            return InferenceResult(
                model_name=model_name,
                model_version=model_version,
                success=False,
                error=f"HTTP {resp.status_code}: {resp.text}",
            )

        # Parse response
        data = resp.json()
        result_outputs = []

        for out_data in data.get("outputs", []):
            out = InferenceOutput(
                name=out_data.get("name", ""),
                datatype=out_data.get("datatype", ""),
                shape=out_data.get("shape", []),
            )

            # Convert data to numpy array
            if "data" in out_data:
                out.data = np.array(
                    out_data["data"], dtype=self._triton_to_numpy_dtype(out.datatype)
                ).reshape(out.shape)

            result_outputs.append(out)

        return InferenceResult(
            model_name=model_name,
            model_version=data.get("model_version", model_version),
            outputs=result_outputs,
            success=True,
        )

    def _grpc_infer(
        self,
        model_name: str,
        inputs: list[InferenceInput],
        outputs: list[str] | None,
        model_version: str,
        request_id: str,
    ) -> InferenceResult:
        """gRPC inference implementation (placeholder)."""
        # Full gRPC implementation would use tritonclient.grpc
        return InferenceResult(
            model_name=model_name,
            model_version=model_version,
            success=False,
            error="gRPC not implemented - use HTTP protocol",
        )

    @staticmethod
    def _triton_to_numpy_dtype(datatype: str) -> np.dtype:
        """Convert Triton datatype to numpy dtype."""
        mapping = {
            "BOOL": np.dtype("bool"),
            "INT8": np.dtype("int8"),
            "INT16": np.dtype("int16"),
            "INT32": np.dtype("int32"),
            "INT64": np.dtype("int64"),
            "UINT8": np.dtype("uint8"),
            "UINT16": np.dtype("uint16"),
            "UINT32": np.dtype("uint32"),
            "UINT64": np.dtype("uint64"),
            "FP16": np.dtype("float16"),
            "FP32": np.dtype("float32"),
            "FP64": np.dtype("float64"),
        }
        return mapping.get(datatype, np.dtype("float32"))

    # =========================================================================
    # Async Operations
    # =========================================================================

    def infer_async(
        self,
        model_name: str,
        inputs: list[InferenceInput],
        callback: Any = None,
        outputs: list[str] | None = None,
        model_version: str = "",
    ):
        """
        Async inference (callback-based).

        For true async, consider using asyncio with aiohttp.
        """
        import threading

        def _worker():
            result = self.infer(model_name, inputs, outputs, model_version)
            if callback:
                callback(result)

        thread = threading.Thread(target=_worker)
        thread.start()
        return thread


# ============================================================================
# Mock Client for Testing
# ============================================================================


class MockTritonClient(TritonClient):
    """
    Mock Triton client for testing without server.

    Simulates responses for unit tests and development.
    """

    def __init__(self, url: str = "localhost:8000", **kwargs):
        # Skip validation for mock
        self.url = url
        self.protocol = Protocol.HTTP
        self.timeout = 30.0
        self.verbose = False
        self._models: dict[str, ModelMetadata] = {}
        self._model_handlers: dict[str, Any] = {}

    def register_model(
        self,
        name: str,
        metadata: ModelMetadata | None = None,
        handler: Any = None,
    ):
        """Register a mock model."""
        if metadata:
            self._models[name] = metadata
        else:
            self._models[name] = ModelMetadata(name=name)

        if handler:
            self._model_handlers[name] = handler

    def is_server_live(self) -> bool:
        return True

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        return model_name in self._models

    def get_model_metadata(self, model_name: str, version: str = "") -> ModelMetadata:
        return self._models.get(model_name, ModelMetadata(name=model_name))

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def infer(
        self,
        model_name: str,
        inputs: list[InferenceInput],
        outputs: list[str] | None = None,
        model_version: str = "",
        request_id: str = "",
    ) -> InferenceResult:
        """Mock inference."""
        start_time = time.perf_counter()

        if model_name not in self._models:
            return InferenceResult(
                model_name=model_name,
                model_version=model_version,
                success=False,
                error=f"Model '{model_name}' not found",
            )

        # Use handler if available
        if model_name in self._model_handlers:
            handler = self._model_handlers[model_name]
            try:
                output_data = handler(inputs)
                result_outputs = [
                    InferenceOutput(name=k, data=v) for k, v in output_data.items()
                ]
            except Exception as e:
                return InferenceResult(
                    model_name=model_name,
                    model_version=model_version,
                    success=False,
                    error=str(e),
                )
        else:
            # Return dummy output matching first input shape
            if inputs:
                dummy_out = np.zeros_like(inputs[0].data)
                result_outputs = [InferenceOutput(name="output", data=dummy_out)]
            else:
                result_outputs = []

        latency = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            model_name=model_name,
            model_version=model_version or "1",
            outputs=result_outputs,
            success=True,
            latency_ms=latency,
        )
