# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith: Cross-Platform ML Optimization Framework

A unified optimization and compilation framework for Machine Learning
that is model-agnostic and hardware-agnostic.

Example:
    import zenith
    import torch

    model = torch.nn.Linear(10, 5)
    optimized = zenith.compile(model, target="cuda", precision="fp16")
"""

__version__ = "0.2.10"
__author__ = "Wahyu Ardiansyah"

# Try to import native bindings if available
try:
    from ._zenith_core import (
        DataType,
        Layout,
        StatusCode,
        Shape,
        Status,
        TensorDescriptor,
        Node,
        GraphIR,
        get_version,
        dtype_size,
        dtype_to_string,
    )

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False
    # Fallback to pure Python implementations
    from .core.types import DataType, Layout, StatusCode, Shape, Status
    from .core.tensor import TensorDescriptor
    from .core.node import Node
    from .core.graph_ir import GraphIR

# Import adapters
from .adapters import (
    PyTorchAdapter,
    TensorFlowAdapter,
    JAXAdapter,
    ONNXAdapter,
)

# Import backends
from . import backends

# Import framework-specific modules
from . import tensorflow
from . import jax
from . import torch

# Import inference module (E2E inference pipeline)
from . import inference

# Main API functions
from .api import compile, optimize

# Observability
from .observability import set_verbosity, Verbosity

# Errors
from .errors import (
    ZenithError,
    CompilationError,
    UnsupportedOperationError,
    PrecisionError,
    KernelError,
    ZenithMemoryError,
    ValidationError,
    ConfigurationError,
)

# Serving (Triton Inference Server integration)
from .serving import (
    TritonBackend,
    TritonBackendConfig,
    ModelConfig,
    export_to_triton,
    export_to_onnx,
    export_to_torchscript,
    ZenithModelExporter,
)

# Monitoring (Production metrics)
from .monitoring import (
    start_server as start_monitoring_server,
    MetricsServer,
    PrometheusExporter,
)

# TorchDynamo Integration (auto-registers 'zenith' backend for torch.compile)
from . import integrations  # noqa: F401


def is_native() -> bool:
    """Check if native C++ bindings are available."""
    return _HAS_NATIVE


__all__ = [
    # Core types
    "DataType",
    "Layout",
    "StatusCode",
    "Shape",
    "Status",
    "TensorDescriptor",
    "Node",
    "GraphIR",
    # Utility functions (re-exported from native bindings)
    "get_version",
    "dtype_size",
    "dtype_to_string",
    # Adapters
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "JAXAdapter",
    "ONNXAdapter",
    # API
    "compile",
    "optimize",
    "is_native",
    # Observability
    "set_verbosity",
    "Verbosity",
    # Errors
    "ZenithError",
    "CompilationError",
    "UnsupportedOperationError",
    "PrecisionError",
    "KernelError",
    "ZenithMemoryError",
    "ValidationError",
    "ConfigurationError",
    # Serving (Triton)
    "TritonBackend",
    "TritonBackendConfig",
    "ModelConfig",
    "export_to_triton",
    "export_to_onnx",
    "export_to_torchscript",
    "ZenithModelExporter",
    # Monitoring
    "start_monitoring_server",
    "MetricsServer",
    "PrometheusExporter",
    # Version
    "__version__",
    "__author__",
]
