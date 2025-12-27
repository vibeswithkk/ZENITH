# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Complete E2E Inference Engine.

This module provides a unified, production-ready interface for end-to-end
inference with Zenith-optimized models.

Architectural Design (Based on CetakBiru.md Section 3.2):
- Model-agnostic: Supports PyTorch, TensorFlow, JAX, and ONNX
- Hardware-agnostic: CPU, CUDA, ROCm with automatic fallback
- Optimization pipeline: GraphIR -> Optimization -> Compilation -> Execution

Technical Foundation:
- Inspired by TensorRT Build vs Runtime phase separation
- ONNX Runtime Execution Provider pattern for hardware abstraction
- PyTorch 2.0 torch.compile for graph capture
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class InferenceBackend(Enum):
    """Supported inference backends."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    AUTO = "auto"


class InferencePrecision(Enum):
    """Supported precision levels."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class InferenceConfig:
    """
    Configuration for E2E inference.

    Attributes:
        backend: Target backend for execution.
        precision: Numerical precision level.
        batch_size: Default batch size for inference.
        enable_optimization: Whether to apply Zenith optimizations.
        enable_cuda_graphs: Use CUDA Graphs for reduced latency.
        enable_memory_optimization: Use gradient checkpointing memory opts.
        warmup_iterations: Number of warmup runs before measurement.
        verbose: Verbosity level (0=silent, 1=info, 2=debug).
        tolerance: Numerical tolerance for validation.
    """

    backend: str = "auto"
    precision: str = "fp32"
    batch_size: int = 1
    enable_optimization: bool = True
    enable_cuda_graphs: bool = False
    enable_memory_optimization: bool = False
    warmup_iterations: int = 3
    verbose: int = 1
    tolerance: float = 1e-5


@dataclass
class InferenceResult:
    """Result from inference execution."""

    outputs: dict[str, np.ndarray]
    latency_ms: float
    backend_used: str
    precision_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceStats:
    """Statistics from inference runs."""

    total_runs: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    warmup_latency_ms: float = 0.0
    _min_initialized: bool = field(default=False, repr=False)

    @property
    def mean_latency_ms(self) -> float:
        """Calculate mean latency."""
        if self.total_runs == 0:
            return 0.0
        return self.total_latency_ms / self.total_runs

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        if latency_ms < 0:
            raise ValueError(f"Latency must be non-negative, got {latency_ms}")

        self.total_runs += 1
        self.total_latency_ms += latency_ms

        if not self._min_initialized:
            self.min_latency_ms = latency_ms
            self._min_initialized = True
        else:
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)

        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "mean_latency_ms": self.mean_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "warmup_latency_ms": self.warmup_latency_ms,
        }


class InferenceSession:
    """
    E2E Inference Session for running optimized models.

    This is the main entry point for production inference with Zenith.
    It handles model loading, optimization, and execution with automatic
    backend selection and fallback.

    Example:
        session = InferenceSession(model, config)
        result = session.run({"input": input_tensor})

    Architecture:
        1. Model Conversion: Any framework -> GraphIR
        2. Optimization: Apply Zenith optimization passes
        3. Compilation: Build execution plan with kernel selection
        4. Execution: Run with optimal kernels and memory management
    """

    def __init__(
        self,
        model: Any,
        config: InferenceConfig | None = None,
        sample_input: Any = None,
    ):
        """
        Initialize inference session.

        Args:
            model: Model from any supported framework (PyTorch, TensorFlow,
                   JAX, ONNX, or pre-compiled GraphIR).
            config: Inference configuration.
            sample_input: Sample input for model tracing (required for
                          dynamic graph frameworks like PyTorch).
        """
        self._config = config or InferenceConfig()
        self._original_model = model
        self._sample_input = sample_input
        self._compiled_model: Any = None
        self._graph_ir: Any = None
        self._stats = InferenceStats()
        self._is_initialized = False
        self._backend: str | None = None
        self._framework: str | None = None

        # Initialize the session
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the inference session."""
        self._framework = self._detect_framework(self._original_model)
        self._backend = self._select_backend()
        self._graph_ir = self._convert_to_graphir()
        self._compiled_model = self._compile_model()

        if self._sample_input is not None:
            self._run_warmup()

        self._is_initialized = True

    def _detect_framework(self, model: Any) -> str:
        """Detect which framework the model comes from."""
        model_type = type(model).__module__

        if "torch" in model_type:
            return "pytorch"
        elif "tensorflow" in model_type or "keras" in model_type:
            return "tensorflow"
        elif "jax" in model_type or "flax" in model_type:
            return "jax"
        elif hasattr(model, "graph") and hasattr(model, "nodes"):
            return "graphir"
        else:
            return "unknown"

    def _select_backend(self) -> str:
        """Select optimal backend based on config and availability."""
        if self._config.backend != "auto":
            return self._config.backend

        try:
            import torch  # noqa: F401

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _convert_to_graphir(self) -> Any:
        """Convert model to GraphIR."""
        if self._framework == "graphir":
            return self._original_model

        if self._framework == "pytorch":
            return self._convert_pytorch()
        elif self._framework == "tensorflow":
            return self._convert_tensorflow()
        elif self._framework == "jax":
            return self._convert_jax()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

    def _convert_pytorch(self) -> Any:
        """Convert PyTorch model to GraphIR."""
        from zenith.adapters import PyTorchAdapter

        adapter = PyTorchAdapter()

        sample = self._sample_input
        if sample is None:
            sample = self._infer_pytorch_sample_input()

        if sample is None:
            raise ValueError(
                "sample_input is required for PyTorch model conversion. "
                "Provide a sample input tensor when creating InferenceSession."
            )

        return adapter.from_model(self._original_model, sample)

    def _infer_pytorch_sample_input(self) -> Any:
        """
        Try to infer sample input shape from PyTorch model.

        Works for Sequential models with Linear first layer.
        """
        try:
            import torch

            model = self._original_model

            if hasattr(model, "children"):
                for child in model.children():
                    if hasattr(child, "in_features"):
                        batch = self._config.batch_size
                        in_features = child.in_features
                        return torch.randn(batch, in_features)

            if hasattr(model, "input_size"):
                return torch.randn(self._config.batch_size, *model.input_size)

            return None
        except Exception:
            return None

    def _convert_tensorflow(self) -> Any:
        """Convert TensorFlow model to GraphIR."""
        from zenith.adapters import TensorFlowAdapter

        adapter = TensorFlowAdapter()
        return adapter.from_model(self._original_model, self._sample_input)

    def _convert_jax(self) -> Any:
        """Convert JAX model to GraphIR."""
        from zenith.adapters import JAXAdapter

        adapter = JAXAdapter()
        return adapter.from_model(self._original_model, self._sample_input)

    def _compile_model(self) -> Any:
        """Compile the GraphIR into an executable model."""
        from zenith.runtime import CompileConfig, ZenithEngine

        engine = ZenithEngine(backend=self._backend)

        compile_config = CompileConfig(
            precision=self._config.precision,
            use_cuda_graphs=self._config.enable_cuda_graphs,
            verbose=self._config.verbose,
            tolerance=self._config.tolerance,
        )

        try:
            compiled = engine.compile(
                self._graph_ir,
                config=compile_config,
                original_model=self._original_model,
            )
            return compiled
        except Exception as e:
            if self._config.verbose > 0:
                print(f"[Zenith] Compilation failed, using fallback: {e}")
            return self._create_fallback_model()

    def _create_fallback_model(self) -> Any:
        """Create fallback model when compilation fails."""
        from zenith.api import CompiledModelLegacy

        return CompiledModelLegacy(
            self._graph_ir,
            backend=self._backend,
            target=self._backend,
            original_model=self._original_model,
        )

    def _run_warmup(self) -> None:
        """Run warmup iterations to trigger JIT compilation."""
        if self._sample_input is None:
            return

        warmup_times = []
        for _ in range(self._config.warmup_iterations):
            start = time.perf_counter()
            self._execute_internal(self._sample_input)
            elapsed = (time.perf_counter() - start) * 1000
            warmup_times.append(elapsed)

        if warmup_times:
            self._stats.warmup_latency_ms = sum(warmup_times) / len(warmup_times)

    def _execute_internal(self, inputs: Any) -> dict[str, np.ndarray]:
        """Execute the model internally."""
        if self._compiled_model is None:
            raise RuntimeError("Session not initialized")

        if isinstance(inputs, dict):
            input_dict = inputs
        else:
            input_dict = {"input": inputs}

        if hasattr(self._compiled_model, "run"):
            outputs = self._compiled_model.run(input_dict)
        elif callable(self._compiled_model):
            result = self._compiled_model(**input_dict)
            if isinstance(result, dict):
                outputs = result
            else:
                outputs = {"output": result}
        else:
            raise RuntimeError("Compiled model is not callable")

        return outputs

    def run(
        self,
        inputs: dict[str, Any] | Any,
        return_latency: bool = False,
    ) -> InferenceResult | dict[str, np.ndarray]:
        """
        Run inference with the given inputs.

        Args:
            inputs: Input tensors (dict or single tensor).
            return_latency: If True, return InferenceResult with timing info.

        Returns:
            Output tensors or InferenceResult if return_latency=True.
        """
        start = time.perf_counter()
        outputs = self._execute_internal(inputs)
        latency_ms = (time.perf_counter() - start) * 1000

        self._stats.record(latency_ms)

        if return_latency:
            return InferenceResult(
                outputs=outputs,
                latency_ms=latency_ms,
                backend_used=self._backend,
                precision_used=self._config.precision,
            )

        return outputs

    def benchmark(
        self,
        inputs: dict[str, Any] | Any,
        num_runs: int = 100,
        num_warmup: int = 10,
    ) -> dict[str, Any]:
        """
        Benchmark inference performance.

        Args:
            inputs: Input tensors for benchmarking.
            num_runs: Number of timed runs.
            num_warmup: Number of warmup runs.

        Returns:
            Benchmark results dictionary.
        """
        for _ in range(num_warmup):
            self._execute_internal(inputs)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self._execute_internal(inputs)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        latencies_arr = np.array(latencies)

        return {
            "num_runs": num_runs,
            "mean_ms": float(np.mean(latencies_arr)),
            "std_ms": float(np.std(latencies_arr)),
            "min_ms": float(np.min(latencies_arr)),
            "max_ms": float(np.max(latencies_arr)),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p90_ms": float(np.percentile(latencies_arr, 90)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "throughput_per_sec": 1000.0 / float(np.mean(latencies_arr)),
            "backend": self._backend,
            "precision": self._config.precision,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        return self._stats.to_dict()

    @property
    def is_initialized(self) -> bool:
        """Check if session is initialized."""
        return self._is_initialized

    @property
    def backend(self) -> str | None:
        """Get the active backend."""
        return self._backend

    @property
    def framework(self) -> str | None:
        """Get the detected framework."""
        return self._framework


def create_session(
    model: Any,
    config: InferenceConfig | None = None,
    sample_input: Any = None,
) -> InferenceSession:
    """
    Create an inference session for the given model.

    This is the main entry point for E2E inference with Zenith.

    Args:
        model: Model from any supported framework.
        config: Inference configuration.
        sample_input: Sample input for model tracing.

    Returns:
        InferenceSession ready for execution.

    Example:
        import zenith.inference as zi

        session = zi.create_session(model, sample_input=torch.randn(1, 64))
        result = session.run({"input": input_tensor})
    """
    return InferenceSession(model, config=config, sample_input=sample_input)


def infer(
    model: Any,
    inputs: dict[str, Any] | Any,
    config: InferenceConfig | None = None,
    sample_input: Any = None,
) -> dict[str, np.ndarray]:
    """
    One-shot inference with automatic session management.

    Convenience function for simple inference without session reuse.

    Args:
        model: Model from any supported framework.
        inputs: Input tensors.
        config: Inference configuration.
        sample_input: Sample input for tracing (uses inputs if None).

    Returns:
        Output tensors dictionary.

    Example:
        outputs = zenith.inference.infer(model, {"input": x})
    """
    if sample_input is None:
        if isinstance(inputs, dict):
            sample_input = next(iter(inputs.values()))
        else:
            sample_input = inputs

    session = create_session(model, config=config, sample_input=sample_input)
    return session.run(inputs)


__all__ = [
    "InferenceBackend",
    "InferencePrecision",
    "InferenceConfig",
    "InferenceResult",
    "InferenceStats",
    "InferenceSession",
    "create_session",
    "infer",
]
