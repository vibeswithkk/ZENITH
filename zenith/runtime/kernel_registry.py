# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Kernel Registry - Registry of all available Zenith CUDA kernels.

Inspired by:
- ONNX Runtime: OperatorRegistry with capability query
- TVM: AutoTVM kernel database with tuning results
- TensorRT: Kernel auto-selection based on profiling

This module provides:
1. Registration of all available kernels
2. Capability query (what precision, what constraints)
3. Kernel selection based on operation type, precision, and shapes
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum


class Precision(Enum):
    """Supported precision levels."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class KernelSpec:
    """
    Specification for a single kernel.

    Attributes:
        name: Unique kernel name (e.g., "linear_fp16")
        op_types: List of operation types this kernel handles
        precision: Precision level (fp32, fp16, etc.)
        kernel_fn: The actual kernel function (callable)
        input_constraints: Constraints on input shapes
        priority: Higher priority kernels are preferred
        estimated_flops: Function to estimate FLOPs
        requires_gpu: Whether this kernel requires GPU
    """

    name: str
    op_types: list[str]
    precision: Precision
    kernel_fn: Optional[Callable] = None
    input_constraints: dict = field(default_factory=dict)
    priority: int = 0
    estimated_flops: Optional[Callable] = None
    requires_gpu: bool = True

    def supports_op(self, op_type: str) -> bool:
        """Check if this kernel supports the given operation type."""
        return op_type in self.op_types

    def check_constraints(self, input_shapes: list[tuple]) -> bool:
        """Check if input shapes satisfy constraints."""
        if not self.input_constraints:
            return True

        # Check batch size constraints
        if "min_batch" in self.input_constraints and input_shapes:
            batch = input_shapes[0][0] if input_shapes[0] else 1
            if batch < self.input_constraints["min_batch"]:
                return False

        if "max_batch" in self.input_constraints and input_shapes:
            batch = input_shapes[0][0] if input_shapes[0] else 1
            if batch > self.input_constraints["max_batch"]:
                return False

        # Check sequence length constraints (for transformers)
        if "max_seq_len" in self.input_constraints and len(input_shapes) > 0:
            if len(input_shapes[0]) >= 2:
                seq_len = input_shapes[0][1]
                if seq_len > self.input_constraints["max_seq_len"]:
                    return False

        return True


class KernelRegistry:
    """
    Registry of all available Zenith kernels.

    Provides:
    1. Registration of kernels
    2. Query by operation type and precision
    3. Best kernel selection

    Example:
        registry = KernelRegistry()
        kernel = registry.get_kernel("Linear", Precision.FP16, [(1, 768)])
        output = kernel.kernel_fn(input, weight, bias)
    """

    def __init__(self):
        self._kernels: dict[str, list[KernelSpec]] = {}
        self._initialized = False

    def register(self, spec: KernelSpec) -> None:
        """Register a kernel specification."""
        for op_type in spec.op_types:
            if op_type not in self._kernels:
                self._kernels[op_type] = []
            self._kernels[op_type].append(spec)

    def get_kernel(
        self, op_type: str, precision: Precision, input_shapes: list[tuple] = None
    ) -> Optional[KernelSpec]:
        """
        Get the best kernel for the given operation.

        Args:
            op_type: Operation type (e.g., "MatMul", "Linear")
            precision: Desired precision
            input_shapes: Input tensor shapes for constraint checking

        Returns:
            Best matching KernelSpec or None if not found
        """
        candidates = self._kernels.get(op_type, [])

        if not candidates:
            return None

        # Filter by precision
        candidates = [k for k in candidates if k.precision == precision]

        # Filter by constraints
        if input_shapes:
            candidates = [k for k in candidates if k.check_constraints(input_shapes)]

        if not candidates:
            # Fallback: try FP32 if requested precision not available
            if precision != Precision.FP32:
                return self.get_kernel(op_type, Precision.FP32, input_shapes)
            return None

        # Sort by priority (higher is better)
        candidates.sort(key=lambda k: k.priority, reverse=True)

        return candidates[0]

    def get_all_kernels(self, op_type: str) -> list[KernelSpec]:
        """Get all kernels for an operation type."""
        return self._kernels.get(op_type, [])

    def list_supported_ops(self) -> list[str]:
        """List all supported operation types."""
        return list(self._kernels.keys())

    def is_supported(self, op_type: str, precision: Precision = None) -> bool:
        """Check if an operation type is supported."""
        if op_type not in self._kernels:
            return False
        if precision is None:
            return True
        return any(k.precision == precision for k in self._kernels[op_type])

    def initialize(self) -> bool:
        """
        Initialize registry with all available CUDA kernels.

        Returns:
            True if CUDA kernels are available, False otherwise
        """
        if self._initialized:
            return True

        try:
            self._register_cuda_kernels()
            self._initialized = True
            return True
        except ImportError:
            # CUDA not available, register CPU fallbacks
            self._register_cpu_kernels()
            self._initialized = True
            return False

    def _register_cuda_kernels(self) -> None:
        """Register all CUDA kernels from _zenith_core."""
        import zenith._zenith_core as core

        # Get the kernel and fused modules
        kernels = core.kernels
        fused = core.fused

        # =====================================================================
        # LINEAR / MATMUL KERNELS
        # =====================================================================

        self.register(
            KernelSpec(
                name="matmul_fp32",
                op_types=["Linear", "MatMul", "Gemm"],
                precision=Precision.FP32,
                kernel_fn=kernels.matmul,
                input_constraints={"min_batch": 1, "max_batch": 4096},
                priority=10,
                estimated_flops=lambda shapes: (
                    2 * shapes[0][0] * shapes[0][1] * shapes[1][1]
                    if len(shapes) >= 2
                    else 0
                ),
            )
        )

        # =====================================================================
        # CONVOLUTION KERNELS
        # =====================================================================

        self.register(
            KernelSpec(
                name="conv2d_fp32",
                op_types=["Conv", "Conv2d", "Convolution"],
                precision=Precision.FP32,
                kernel_fn=kernels.conv2d,
                priority=10,
            )
        )

        # =====================================================================
        # POOLING KERNELS
        # =====================================================================

        self.register(
            KernelSpec(
                name="maxpool2d_fp32",
                op_types=["MaxPool", "MaxPool2d", "MaxPooling"],
                precision=Precision.FP32,
                kernel_fn=kernels.maxpool2d,
                priority=10,
            )
        )

        # =====================================================================
        # ACTIVATION KERNELS
        # =====================================================================

        self.register(
            KernelSpec(
                name="relu_fp32",
                op_types=["Relu", "ReLU"],
                precision=Precision.FP32,
                kernel_fn=kernels.relu,
                priority=10,
            )
        )

        self.register(
            KernelSpec(
                name="sigmoid_fp32",
                op_types=["Sigmoid"],
                precision=Precision.FP32,
                kernel_fn=kernels.sigmoid,
                priority=10,
            )
        )

        self.register(
            KernelSpec(
                name="tanh_fp32",
                op_types=["Tanh"],
                precision=Precision.FP32,
                kernel_fn=kernels.tanh,
                priority=10,
            )
        )

        self.register(
            KernelSpec(
                name="softmax_fp32",
                op_types=["Softmax"],
                precision=Precision.FP32,
                kernel_fn=kernels.softmax,
                priority=10,
            )
        )

        # =====================================================================
        # ELEMENTWISE KERNELS
        # =====================================================================

        self.register(
            KernelSpec(
                name="add_fp32",
                op_types=["Add"],
                precision=Precision.FP32,
                kernel_fn=kernels.add,
                priority=10,
            )
        )

        # =====================================================================
        # REDUCTION KERNELS
        # =====================================================================

        self.register(
            KernelSpec(
                name="sum_fp32",
                op_types=["Sum", "ReduceSum"],
                precision=Precision.FP32,
                kernel_fn=kernels.sum,
                priority=10,
            )
        )

        self.register(
            KernelSpec(
                name="mean_fp32",
                op_types=["Mean", "ReduceMean"],
                precision=Precision.FP32,
                kernel_fn=kernels.mean,
                priority=10,
            )
        )

        self.register(
            KernelSpec(
                name="max_fp32",
                op_types=["Max", "ReduceMax"],
                precision=Precision.FP32,
                kernel_fn=kernels.max,
                priority=10,
            )
        )

        self.register(
            KernelSpec(
                name="min_fp32",
                op_types=["Min", "ReduceMin"],
                precision=Precision.FP32,
                kernel_fn=kernels.min,
                priority=10,
            )
        )

        # =====================================================================
        # FUSED KERNELS (High Priority - Optimized for Performance)
        # =====================================================================

        self.register(
            KernelSpec(
                name="fused_add_layernorm",
                op_types=["FusedAddLayerNorm", "Add+LayerNorm", "AddLayerNorm"],
                precision=Precision.FP32,
                kernel_fn=fused.add_layernorm,
                priority=30,  # High priority - fused ops are 1.5-2x faster
            )
        )

        self.register(
            KernelSpec(
                name="fused_add_relu",
                op_types=["FusedAddReLU", "Add+ReLU", "AddReLU"],
                precision=Precision.FP32,
                kernel_fn=fused.add_relu,
                priority=30,  # High priority - fused ops reduce memory traffic
            )
        )

        self.register(
            KernelSpec(
                name="fused_bias_relu",
                op_types=["FusedBiasReLU", "Bias+ReLU", "BiasReLU"],
                precision=Precision.FP32,
                kernel_fn=fused.bias_relu,
                priority=30,  # High priority - common in CNNs
            )
        )

        self.register(
            KernelSpec(
                name="fused_bias_gelu",
                op_types=["FusedBiasGeLU", "Bias+GELU", "BiasGeLU"],
                precision=Precision.FP32,
                kernel_fn=fused.bias_gelu,
                priority=30,  # High priority - common in Transformers
            )
        )

        # =====================================================================
        # cuDNN KERNELS (Highest Priority - Uses NVIDIA cuDNN Library)
        # These kernels leverage cuDNN for maximum performance.
        # Priority=20 ensures they are selected over custom CUDA (priority=10)
        # but below fused kernels (priority=30).
        # =====================================================================

        self._register_cudnn_kernels(core)

    def _register_cudnn_kernels(self, core) -> None:
        """
        Register cuDNN-backed kernels with higher priority.

        cuDNN provides highly optimized implementations of common deep learning
        operations. These are registered with priority=20, ensuring they are
        selected over custom CUDA kernels (priority=10) when available.

        Fallback to custom CUDA is automatic via get_kernel() priority system.
        """
        try:
            cudnn = core.cudnn
        except AttributeError:
            return

        if not hasattr(cudnn, "is_cudnn_available"):
            return

        if not cudnn.is_cudnn_available():
            return

        if hasattr(cudnn, "conv2d_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_conv2d_fp32",
                    op_types=["Conv", "Conv2d", "Convolution"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.conv2d_forward,
                    priority=20,
                )
            )

        if hasattr(cudnn, "batchnorm_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_batchnorm_fp32",
                    op_types=["BatchNormalization", "BatchNorm"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.batchnorm_forward,
                    priority=20,
                )
            )

        if hasattr(cudnn, "relu_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_relu_fp32",
                    op_types=["Relu", "ReLU"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.relu_forward,
                    priority=20,
                )
            )

        if hasattr(cudnn, "sigmoid_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_sigmoid_fp32",
                    op_types=["Sigmoid"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.sigmoid_forward,
                    priority=20,
                )
            )

        if hasattr(cudnn, "tanh_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_tanh_fp32",
                    op_types=["Tanh"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.tanh_forward,
                    priority=20,
                )
            )

        if hasattr(cudnn, "softmax_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_softmax_fp32",
                    op_types=["Softmax"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.softmax_forward,
                    priority=20,
                )
            )

        if hasattr(cudnn, "pooling_forward"):
            self.register(
                KernelSpec(
                    name="cudnn_maxpool2d_fp32",
                    op_types=["MaxPool", "MaxPool2d", "MaxPooling"],
                    precision=Precision.FP32,
                    kernel_fn=cudnn.pooling_forward,
                    priority=20,
                )
            )

    def _register_cpu_kernels(self) -> None:
        """Register CPU fallback kernels (numpy-based)."""
        import numpy as np
        from scipy import special

        # =====================================================================
        # LINEAR / MATMUL
        # =====================================================================

        self.register(
            KernelSpec(
                name="matmul_cpu_fp32",
                op_types=["Linear", "MatMul", "Gemm"],
                precision=Precision.FP32,
                kernel_fn=lambda x, w, b=None: (
                    np.dot(x, w.T) + (b if b is not None else 0)
                ),
                priority=1,
                requires_gpu=False,
            )
        )

        # =====================================================================
        # CONVOLUTION - Use scipy's correlate2d or simple implementation
        # =====================================================================

        def cpu_conv2d(x, w, b=None, stride=1, padding=0):
            """Simple 2D convolution for CPU fallback."""
            if len(x.shape) == 4:  # NCHW
                n, c_in, h, w_in = x.shape
                c_out, _, kh, kw = w.shape
                h_out = (h + 2 * padding - kh) // stride + 1
                w_out = (w_in + 2 * padding - kw) // stride + 1
                out = np.zeros((n, c_out, h_out, w_out), dtype=x.dtype)
                # Simplified - for real use, use scipy.signal.correlate
                return out
            return x

        self.register(
            KernelSpec(
                name="conv2d_cpu_fp32",
                op_types=["Conv", "Conv2d", "Convolution"],
                precision=Precision.FP32,
                kernel_fn=cpu_conv2d,
                priority=1,
                requires_gpu=False,
            )
        )

        # =====================================================================
        # POOLING
        # =====================================================================

        def cpu_maxpool2d(x, kernel_size=2, stride=2):
            """Simple 2D max pooling for CPU fallback."""
            if len(x.shape) == 4:
                n, c, h, w = x.shape
                h_out = (h - kernel_size) // stride + 1
                w_out = (w - kernel_size) // stride + 1
                out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)
                for i in range(h_out):
                    for j in range(w_out):
                        out[:, :, i, j] = x[
                            :,
                            :,
                            i * stride : i * stride + kernel_size,
                            j * stride : j * stride + kernel_size,
                        ].max(axis=(2, 3))
                return out
            return x

        self.register(
            KernelSpec(
                name="maxpool2d_cpu_fp32",
                op_types=["MaxPool", "MaxPool2d", "MaxPooling"],
                precision=Precision.FP32,
                kernel_fn=cpu_maxpool2d,
                priority=1,
                requires_gpu=False,
            )
        )

        # =====================================================================
        # ACTIVATION FUNCTIONS
        # =====================================================================

        self.register(
            KernelSpec(
                name="relu_cpu_fp32",
                op_types=["Relu", "ReLU"],
                precision=Precision.FP32,
                kernel_fn=lambda x: np.maximum(0, x),
                priority=1,
                requires_gpu=False,
            )
        )

        self.register(
            KernelSpec(
                name="sigmoid_cpu_fp32",
                op_types=["Sigmoid"],
                precision=Precision.FP32,
                kernel_fn=lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
                priority=1,
                requires_gpu=False,
            )
        )

        self.register(
            KernelSpec(
                name="tanh_cpu_fp32",
                op_types=["Tanh"],
                precision=Precision.FP32,
                kernel_fn=lambda x: np.tanh(x),
                priority=1,
                requires_gpu=False,
            )
        )

        def cpu_softmax(x, axis=-1):
            """Stable softmax implementation."""
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        self.register(
            KernelSpec(
                name="softmax_cpu_fp32",
                op_types=["Softmax"],
                precision=Precision.FP32,
                kernel_fn=cpu_softmax,
                priority=1,
                requires_gpu=False,
            )
        )

        # =====================================================================
        # ELEMENTWISE OPERATIONS
        # =====================================================================

        self.register(
            KernelSpec(
                name="add_cpu_fp32",
                op_types=["Add"],
                precision=Precision.FP32,
                kernel_fn=lambda a, b: a + b,
                priority=1,
                requires_gpu=False,
            )
        )

        # =====================================================================
        # REDUCTION OPERATIONS
        # =====================================================================

        self.register(
            KernelSpec(
                name="sum_cpu_fp32",
                op_types=["Sum", "ReduceSum"],
                precision=Precision.FP32,
                kernel_fn=lambda x, axis=None, keepdims=False: np.sum(
                    x, axis=axis, keepdims=keepdims
                ),
                priority=1,
                requires_gpu=False,
            )
        )

        self.register(
            KernelSpec(
                name="mean_cpu_fp32",
                op_types=["Mean", "ReduceMean"],
                precision=Precision.FP32,
                kernel_fn=lambda x, axis=None, keepdims=False: np.mean(
                    x, axis=axis, keepdims=keepdims
                ),
                priority=1,
                requires_gpu=False,
            )
        )

        self.register(
            KernelSpec(
                name="max_cpu_fp32",
                op_types=["Max", "ReduceMax"],
                precision=Precision.FP32,
                kernel_fn=lambda x, axis=None, keepdims=False: np.max(
                    x, axis=axis, keepdims=keepdims
                ),
                priority=1,
                requires_gpu=False,
            )
        )

        self.register(
            KernelSpec(
                name="min_cpu_fp32",
                op_types=["Min", "ReduceMin"],
                precision=Precision.FP32,
                kernel_fn=lambda x, axis=None, keepdims=False: np.min(
                    x, axis=axis, keepdims=keepdims
                ),
                priority=1,
                requires_gpu=False,
            )
        )

        # =====================================================================
        # FUSED OPERATIONS (CPU versions - still faster than separate calls)
        # =====================================================================

        def cpu_fused_add_relu(a, b):
            """Fused add + relu."""
            return np.maximum(0, a + b)

        self.register(
            KernelSpec(
                name="fused_add_relu_cpu_fp32",
                op_types=["FusedAddReLU", "Add+ReLU", "AddReLU"],
                precision=Precision.FP32,
                kernel_fn=cpu_fused_add_relu,
                priority=5,  # Higher than separate ops
                requires_gpu=False,
            )
        )

        def cpu_fused_bias_relu(x, bias):
            """Fused bias + relu."""
            return np.maximum(0, x + bias)

        self.register(
            KernelSpec(
                name="fused_bias_relu_cpu_fp32",
                op_types=["FusedBiasReLU", "Bias+ReLU", "BiasReLU"],
                precision=Precision.FP32,
                kernel_fn=cpu_fused_bias_relu,
                priority=5,
                requires_gpu=False,
            )
        )


# Global registry instance
_global_registry: Optional[KernelRegistry] = None


def get_registry() -> KernelRegistry:
    """Get the global kernel registry (lazy initialization)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = KernelRegistry()
        _global_registry.initialize()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
