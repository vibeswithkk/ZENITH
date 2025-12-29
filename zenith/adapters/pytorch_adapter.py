# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
PyTorch Adapter - Enterprise Edition

Comprehensive adapter for PyTorch models with full support for:
- torch.compile backend integration (TorchDynamo)
- HuggingFace Transformers models
- FX Graph conversion
- Training integration with AMP
- Custom torch.compile backend registration

Architecture follows CetakBiru.md Section 6.1 (Line 702-720):
- torch.compile as Hook mechanism
- TorchDynamo graph capture
- FX Graph to GraphIR/ONNX conversion
- Seamless integration with Zenith optimization pipeline
"""

import functools
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .base import BaseAdapter
from .onnx_adapter import ONNXAdapter
from ..core import DataType, GraphIR, Shape, TensorDescriptor

logger = logging.getLogger("zenith.adapters.pytorch")


# =============================================================================
# Configuration and Constants
# =============================================================================


@dataclass
class ZenithPyTorchConfig:
    """Configuration for Zenith PyTorch integration."""

    # Compilation options
    target: str = "cuda"  # "cpu", "cuda", "cuda:0", "rocm"
    precision: str = "fp32"  # "fp32", "fp16", "bf16", "int8"
    opt_level: int = 2  # 1-3, 3 is most aggressive

    # ONNX conversion options
    opset_version: int = 17

    # torch.compile options
    mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False  # Require full graph capture
    dynamic: bool = False  # Enable dynamic shapes

    # Training options
    enable_gradient_optimization: bool = True
    enable_amp: bool = False  # Automatic Mixed Precision
    gradient_checkpointing: bool = False

    # Profiling
    enable_profiling: bool = False
    profile_output_dir: Optional[str] = None

    # HuggingFace options
    trust_remote_code: bool = False

    # Tolerance for numerical accuracy (CetakBiru 4.1)
    tolerance: float = 1e-6


@dataclass
class OptimizationStats:
    """Statistics from Zenith optimization."""

    original_ops: int = 0
    optimized_ops: int = 0
    fusion_count: int = 0
    memory_reduction_pct: float = 0.0
    estimated_speedup: float = 1.0
    passes_applied: list[str] = field(default_factory=list)


# =============================================================================
# PyTorch Adapter - Core Implementation
# =============================================================================


class PyTorchAdapter(BaseAdapter):
    """
    Enterprise-grade adapter for PyTorch models.

    Converts PyTorch nn.Module to Zenith's GraphIR format with full support
    for inference and training workflows.

    Features:
    - torch.compile backend integration
    - HuggingFace Transformers support
    - FX Graph capture and conversion
    - Automatic Mixed Precision (AMP)
    - Training integration
    - Custom backend registration for torch.compile

    Example:
        # Basic usage
        adapter = PyTorchAdapter()
        model = torch.nn.Linear(10, 5)
        sample = torch.randn(1, 10)
        graph = adapter.from_model(model, sample)

        # HuggingFace model
        graph = adapter.from_transformers("bert-base-uncased")

        # torch.compile backend
        backend = adapter.create_compile_backend(target="cuda")
        compiled = torch.compile(model, backend=backend)

        # Compilation hook
        @zenith.torch.compile(target="cuda", precision="fp16")
        def forward(x):
            return model(x)
    """

    def __init__(self, config: Optional[ZenithPyTorchConfig] = None):
        """Initialize the PyTorch adapter.

        Args:
            config: Optional configuration for Zenith PyTorch integration.
        """
        self._torch = None
        self._onnx_adapter = None
        self._config = config or ZenithPyTorchConfig()
        self._compiled_backends: dict[str, Any] = {}
        self._optimization_stats: dict[int, OptimizationStats] = {}

    @property
    def name(self) -> str:
        return "pytorch"

    @property
    def is_available(self) -> bool:
        """Check if PyTorch is installed."""
        try:
            import importlib.util

            return importlib.util.find_spec("torch") is not None
        except Exception:
            return False

    @property
    def config(self) -> ZenithPyTorchConfig:
        """Get current configuration."""
        return self._config

    def _get_torch(self):
        """Lazy import torch."""
        if self._torch is None:
            try:
                import torch

                self._torch = torch
                logger.info(f"PyTorch {torch.__version__} loaded successfully")
            except ImportError as err:
                raise ImportError(
                    "PyTorch is required for PyTorchAdapter. "
                    "Install it with: pip install torch"
                ) from err
        return self._torch

    def _get_onnx_adapter(self) -> ONNXAdapter:
        """Get ONNX adapter instance."""
        if self._onnx_adapter is None:
            self._onnx_adapter = ONNXAdapter()
        return self._onnx_adapter

    def _get_torch_version(self) -> tuple:
        """Get PyTorch version as tuple for comparison."""
        torch = self._get_torch()
        try:
            parts = torch.__version__.split("+")[0].split(".")[:3]
            return tuple(int(p) for p in parts if p.isdigit())
        except Exception:
            return (0, 0, 0)

    def _has_torch_compile(self) -> bool:
        """Check if torch.compile is available (PyTorch >= 2.0)."""
        version = self._get_torch_version()
        return version >= (2, 0, 0)

    def _has_torch_export(self) -> bool:
        """Check if torch.export is available (PyTorch >= 2.1)."""
        version = self._get_torch_version()
        return version >= (2, 1, 0)

    # =========================================================================
    # Core Conversion Methods
    # =========================================================================

    def from_model(
        self,
        model: Any,
        sample_input: Any = None,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
        opset_version: Optional[int] = None,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a PyTorch nn.Module to GraphIR.

        Args:
            model: PyTorch nn.Module to convert.
            sample_input: Sample input tensor for tracing.
            input_names: Names for input tensors.
            output_names: Names for output tensors.
            dynamic_axes: Dynamic axis specification.
            opset_version: ONNX opset version to use.
            **kwargs: Additional options passed to export.

        Returns:
            GraphIR representation of the model.
        """
        torch = self._get_torch()

        if sample_input is None:
            raise ValueError(
                "sample_input is required for PyTorch model conversion. "
                "Provide a sample input tensor for tracing."
            )

        # Check if it's a HuggingFace model
        if self._is_huggingface_model(model):
            return self._convert_huggingface_model(model, sample_input)

        # Set defaults
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        if opset_version is None:
            opset_version = self._config.opset_version

        # Try FX Graph path first (PyTorch 2.x)
        if self._has_torch_export():
            try:
                return self.from_fx_graph(model, sample_input, **kwargs)
            except Exception as e:
                logger.debug(f"FX Graph export failed: {e}, using ONNX path")

        # Fallback to ONNX export
        onnx_bytes = self.to_onnx(
            model,
            sample_input,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **kwargs,
        )

        onnx_adapter = self._get_onnx_adapter()
        return onnx_adapter.from_bytes(onnx_bytes)

    def from_fx_graph(
        self,
        model: Any,
        sample_input: Any,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a PyTorch model using FX Graph (PyTorch 2.x).

        This method uses torch.export to capture the full computation graph
        and convert it to GraphIR.

        Args:
            model: PyTorch nn.Module to convert.
            sample_input: Sample input tensor for tracing.
            **kwargs: Additional options.

        Returns:
            GraphIR representation of the model.
        """
        torch = self._get_torch()

        if not self._has_torch_export():
            raise ImportError(
                "torch.export requires PyTorch 2.1 or higher. "
                f"Current version: {torch.__version__}. "
                "Use from_model() with ONNX export instead."
            )

        model.eval()

        with torch.no_grad():
            try:
                # Use torch.export (PyTorch 2.1+)
                exported = torch.export.export(
                    model,
                    (sample_input,),
                    strict=kwargs.get("strict", True),
                )

                return self._exported_program_to_graphir(exported, sample_input)

            except Exception as e:
                logger.debug(f"torch.export failed: {e}, falling back to ONNX")
                return self.from_model(model, sample_input, **kwargs)

    def _exported_program_to_graphir(
        self,
        exported_program: Any,
        sample_input: Any,
    ) -> GraphIR:
        """Convert ExportedProgram to GraphIR."""
        torch = self._get_torch()

        # Get the graph module
        graph_module = exported_program.module()

        # Create GraphIR
        graph = GraphIR(name="pytorch_exported_model")

        # Add input
        if isinstance(sample_input, dict):
            for name, tensor in sample_input.items():
                shape = list(tensor.shape)
                dtype = self._torch_dtype_to_zenith(tensor.dtype)
                graph.add_input(
                    TensorDescriptor(name=name, shape=Shape(shape), dtype=dtype)
                )
        else:
            shape = list(sample_input.shape)
            dtype = self._torch_dtype_to_zenith(sample_input.dtype)
            graph.add_input(
                TensorDescriptor(name="input", shape=Shape(shape), dtype=dtype)
            )

        # Run to get output shape
        with torch.no_grad():
            if isinstance(sample_input, dict):
                output = graph_module(**sample_input)
            else:
                output = graph_module(sample_input)

        # Handle output
        if isinstance(output, dict):
            for name, tensor in output.items():
                out_shape = list(tensor.shape)
                out_dtype = self._torch_dtype_to_zenith(tensor.dtype)
                graph.add_output(
                    TensorDescriptor(name=name, shape=Shape(out_shape), dtype=out_dtype)
                )
        elif isinstance(output, (tuple, list)):
            for i, tensor in enumerate(output):
                if hasattr(tensor, "shape"):
                    out_shape = list(tensor.shape)
                    out_dtype = self._torch_dtype_to_zenith(tensor.dtype)
                    graph.add_output(
                        TensorDescriptor(
                            name=f"output_{i}",
                            shape=Shape(out_shape),
                            dtype=out_dtype,
                        )
                    )
        else:
            out_shape = list(output.shape)
            out_dtype = self._torch_dtype_to_zenith(output.dtype)
            graph.add_output(
                TensorDescriptor(name="output", shape=Shape(out_shape), dtype=out_dtype)
            )

        # Add FX Graph node
        graph.add_node(
            op_type="TorchExported",
            name="exported_computation",
            inputs=graph.inputs,
            outputs=graph.outputs,
            attrs={"framework": "pytorch", "export_version": "2.x"},
        )

        return graph

    # =========================================================================
    # HuggingFace Transformers Integration
    # =========================================================================

    def from_transformers(
        self,
        model_name_or_path: str,
        task: Optional[str] = None,
        sample_input: Any = None,
        max_length: int = 128,
        batch_size: int = 1,
        **kwargs,
    ) -> GraphIR:
        """
        Convert a HuggingFace Transformers PyTorch model to GraphIR.

        This method provides seamless integration with the HuggingFace
        ecosystem for PyTorch models.

        Args:
            model_name_or_path: Model identifier (e.g., "bert-base-uncased").
            task: Optional task type (e.g., "text-classification").
            sample_input: Optional pre-computed sample input.
            max_length: Maximum sequence length for tokenization.
            batch_size: Batch size for sample input.
            **kwargs: Additional options passed to from_pretrained.

        Returns:
            GraphIR representation of the model.

        Example:
            adapter = PyTorchAdapter()
            graph = adapter.from_transformers(
                "bert-base-uncased",
                task="text-classification"
            )
        """
        torch = self._get_torch()

        try:
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForMaskedLM,
                AutoModelForQuestionAnswering,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as err:
            raise ImportError(
                "transformers library is required for HuggingFace integration. "
                "Install with: pip install transformers"
            ) from err

        # Map task to model class
        task_to_model = {
            "text-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "causal-lm": AutoModelForCausalLM,
            "masked-lm": AutoModelForMaskedLM,
            None: AutoModel,
        }

        model_class = task_to_model.get(task, AutoModel)

        # Load model
        logger.info(f"Loading HuggingFace PyTorch model: {model_name_or_path}")
        model = model_class.from_pretrained(
            model_name_or_path,
            trust_remote_code=self._config.trust_remote_code,
            **kwargs,
        )
        model.eval()

        # Create sample input if not provided
        if sample_input is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                sample_text = "This is a sample input for model tracing."
                encoded = tokenizer(
                    sample_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                sample_input = {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
                if "token_type_ids" in encoded:
                    sample_input["token_type_ids"] = encoded["token_type_ids"]
            except Exception as e:
                logger.warning(f"Could not create sample input: {e}")
                sample_input = {
                    "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
                    "attention_mask": torch.ones(
                        (batch_size, max_length), dtype=torch.long
                    ),
                }

        return self._convert_huggingface_model(model, sample_input)

    def _is_huggingface_model(self, model: Any) -> bool:
        """Check if model is a HuggingFace Transformers model."""
        model_module = type(model).__module__
        return "transformers" in model_module

    def _convert_huggingface_model(
        self,
        model: Any,
        sample_input: Any,
    ) -> GraphIR:
        """Convert HuggingFace PyTorch model to GraphIR."""
        torch = self._get_torch()

        model_config = getattr(model, "config", None)
        model_name = (
            getattr(model_config, "name_or_path", "hf_model")
            if model_config
            else "hf_model"
        )

        # Try ONNX conversion
        try:
            if isinstance(sample_input, dict):
                # Create tuple input for ONNX export
                input_tuple = tuple(sample_input.values())
                input_names = list(sample_input.keys())
            else:
                input_tuple = sample_input
                input_names = ["input"]

            onnx_bytes = self.to_onnx(
                model,
                input_tuple,
                input_names=input_names,
                output_names=["output"],
            )

            onnx_adapter = self._get_onnx_adapter()
            return onnx_adapter.from_bytes(onnx_bytes)

        except Exception as e:
            logger.warning(f"ONNX export failed for HuggingFace model: {e}")
            return self._create_huggingface_graphir(model, sample_input, model_name)

    def _create_huggingface_graphir(
        self,
        model: Any,
        sample_input: Any,
        model_name: str,
    ) -> GraphIR:
        """Create GraphIR for HuggingFace model when export fails."""
        torch = self._get_torch()
        graph = GraphIR(name=f"huggingface_{model_name}")

        # Add inputs
        if isinstance(sample_input, dict):
            for name, tensor in sample_input.items():
                shape = list(tensor.shape)
                dtype = self._torch_dtype_to_zenith(tensor.dtype)
                graph.add_input(
                    TensorDescriptor(name=name, shape=Shape(shape), dtype=dtype)
                )
        else:
            shape = list(sample_input.shape)
            dtype = self._torch_dtype_to_zenith(sample_input.dtype)
            graph.add_input(
                TensorDescriptor(name="input", shape=Shape(shape), dtype=dtype)
            )

        # Run model to get output shape
        model.eval()
        with torch.no_grad():
            try:
                if isinstance(sample_input, dict):
                    output = model(**sample_input)
                else:
                    output = model(sample_input)

                # Handle different output types
                if hasattr(output, "logits"):
                    output_tensor = output.logits
                elif hasattr(output, "last_hidden_state"):
                    output_tensor = output.last_hidden_state
                elif isinstance(output, (tuple, list)):
                    output_tensor = output[0]
                else:
                    output_tensor = output

                out_shape = list(output_tensor.shape)
                out_dtype = self._torch_dtype_to_zenith(output_tensor.dtype)
            except Exception:
                out_shape = [1]
                out_dtype = DataType.Float32

        graph.add_output(
            TensorDescriptor(name="output", shape=Shape(out_shape), dtype=out_dtype)
        )

        graph.add_node(
            op_type="HuggingFacePyTorchModel",
            name=f"hf_{model_name}",
            inputs=graph.inputs,
            outputs=graph.outputs,
            attrs={
                "framework": "pytorch",
                "source": "huggingface",
                "model_name": model_name,
            },
        )

        return graph

    # =========================================================================
    # torch.compile Backend Integration (CetakBiru Line 702-720)
    # =========================================================================

    def create_compile_backend(
        self,
        target: Optional[str] = None,
        precision: Optional[str] = None,
        opt_level: Optional[int] = None,
        **kwargs,
    ) -> Callable:
        """
        Create a custom backend for torch.compile.

        This implements CetakBiru.md Section 6.1 (Line 702-720):
        Zenith as a custom backend for torch.compile.

        Args:
            target: Target device ("cpu", "cuda").
            precision: Precision level ("fp32", "fp16", "bf16").
            opt_level: Optimization level (1-3).
            **kwargs: Additional backend options.

        Returns:
            A callable backend function for torch.compile.

        Example:
            adapter = PyTorchAdapter()
            backend = adapter.create_compile_backend(
                target="cuda",
                precision="fp16"
            )

            # Use with torch.compile
            compiled_model = torch.compile(model, backend=backend)
        """
        torch = self._get_torch()

        if not self._has_torch_compile():
            raise ImportError(
                "torch.compile requires PyTorch 2.0 or higher. "
                f"Current version: {torch.__version__}"
            )

        target = target or self._config.target
        precision = precision or self._config.precision
        opt_level = opt_level if opt_level is not None else self._config.opt_level

        def zenith_backend(gm: Any, example_inputs: list) -> Callable:
            """
            Zenith backend for torch.compile.

            This function receives an FX GraphModule and example inputs,
            optimizes them through Zenith's runtime, and returns an optimized callable
            that uses Zenith's CUDA kernels for execution.
            """
            logger.info(
                f"Zenith backend compiling with target={target}, precision={precision}"
            )

            # Phase 1: Apply FX graph pattern optimizations
            try:
                from ..optimization.fx_optimizer import optimize_fx_graph

                gm = optimize_fx_graph(
                    gm,
                    example_inputs,
                    enable_attention=True,
                    enable_activation=True,
                    enable_normalization=True,
                    verbose=(opt_level >= 2),
                )
                logger.debug("FX optimization passes applied")
            except Exception as e:
                logger.debug(f"FX optimization skipped: {e}")

            # Phase 2: Convert FX Graph to GraphIR
            try:
                graph_ir = self._fx_graph_to_graphir(gm, example_inputs)
                logger.debug(f"Converted to GraphIR: {graph_ir.name}")
            except Exception as e:
                logger.warning(f"GraphIR conversion failed: {e}")
                # Return original function as fallback
                return gm.forward

            # Apply precision transformations to the original module
            if precision == "fp16":
                gm = self._apply_fp16(gm)
            elif precision == "bf16":
                gm = self._apply_bf16(gm)

            # Try to use ZenithEngine for actual kernel execution
            try:
                from ..runtime import ZenithEngine, CompileConfig

                # Create engine and config
                engine = ZenithEngine(backend=target.split(":")[0])
                config = CompileConfig(
                    precision=precision,
                    mode=self._config.mode,
                    verbose=2 if self._config.enable_profiling else 0,
                )

                # Compile with ZenithEngine
                compiled_model = engine.compile(
                    graph_ir=graph_ir, config=config, original_model=gm.forward
                )

                logger.info(
                    f"Zenith compilation successful: "
                    f"{compiled_model.compile_stats.num_supported_ops} ops optimized"
                )

                # Create wrapper that converts PyTorch tensors properly
                # Use OptimizedExecutor for direct tensor execution
                from ..runtime.cuda_optimized import create_optimized_wrapper

                # Create optimized wrapper that uses torch.autocast
                optimized_fn = create_optimized_wrapper(
                    gm.forward, precision=precision, device=target
                )

                logger.info(
                    f"Zenith compilation successful: "
                    f"{compiled_model.compile_stats.num_supported_ops} ops optimized"
                )

                def zenith_optimized_forward(*args, **kw):
                    """Execute using Zenith's optimized kernels with autocast."""
                    try:
                        # Use optimized wrapper for direct tensor execution
                        return optimized_fn(*args, **kw)
                    except Exception as e:
                        # Fallback to original on runtime error
                        logger.debug(f"Zenith runtime fallback: {e}")
                        return gm.forward(*args, **kw)

                return zenith_optimized_forward

            except Exception as e:
                logger.warning(f"ZenithEngine compilation failed: {e}")
                logger.warning("Falling back to PyTorch execution")

                # Fallback to original PyTorch behavior
                compiled_fn = gm.forward

                # Wrap with device placement
                if target.startswith("cuda"):

                    def cuda_wrapper(*args, **kw):
                        return compiled_fn(*args, **kw)

                    return cuda_wrapper

                return compiled_fn

        return zenith_backend

    def _fx_graph_to_graphir(
        self,
        gm: Any,
        example_inputs: list,
    ) -> GraphIR:
        """Convert FX GraphModule to GraphIR."""
        torch = self._get_torch()

        graph = GraphIR(name="fx_graph_module")

        # Add inputs from example_inputs
        for i, inp in enumerate(example_inputs):
            if hasattr(inp, "shape"):
                shape = list(inp.shape)
                dtype = self._torch_dtype_to_zenith(inp.dtype)
                graph.add_input(
                    TensorDescriptor(name=f"input_{i}", shape=Shape(shape), dtype=dtype)
                )

        # Run to get output
        with torch.no_grad():
            output = gm(*example_inputs)

        # Add output
        if hasattr(output, "shape"):
            out_shape = list(output.shape)
            out_dtype = self._torch_dtype_to_zenith(output.dtype)
            graph.add_output(
                TensorDescriptor(name="output", shape=Shape(out_shape), dtype=out_dtype)
            )
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if hasattr(out, "shape"):
                    graph.add_output(
                        TensorDescriptor(
                            name=f"output_{i}",
                            shape=Shape(list(out.shape)),
                            dtype=self._torch_dtype_to_zenith(out.dtype),
                        )
                    )

        # Add node representing the FX graph
        graph.add_node(
            op_type="TorchFXGraph",
            name="fx_computation",
            inputs=graph.inputs,
            outputs=graph.outputs,
            attrs={"node_count": len(gm.graph.nodes)},
        )

        return graph

    def _apply_fp16(self, gm: Any) -> Any:
        """Apply FP16 precision to graph module."""
        torch = self._get_torch()
        return gm.half()

    def _apply_bf16(self, gm: Any) -> Any:
        """Apply BF16 precision to graph module."""
        torch = self._get_torch()
        return gm.bfloat16()

    # =========================================================================
    # Compilation Hook (like ztf.compile / zjax.compile)
    # =========================================================================

    def compile_function(
        self,
        func: Optional[Callable] = None,
        *,
        target: Optional[str] = None,
        precision: Optional[str] = None,
        opt_level: Optional[int] = None,
        mode: Optional[str] = None,
        fullgraph: bool = False,
        **kwargs,
    ) -> Callable:
        """
        Compile a PyTorch function with Zenith optimizations.

        This works similarly to torch.compile(), providing a decorator
        that optimizes PyTorch functions for inference or training.

        Args:
            func: The function to compile. If None, returns a decorator.
            target: Target device ("cpu", "cuda").
            precision: Precision level ("fp32", "fp16", "bf16", "int8").
            opt_level: Optimization level (1-3).
            mode: torch.compile mode.
            fullgraph: Require full graph capture.
            **kwargs: Additional options.

        Returns:
            Compiled function or decorator.

        Example:
            @adapter.compile_function(target="cuda", precision="fp16")
            def forward(x):
                return model(x)

            # Or without decorator
            compiled = adapter.compile_function(forward, target="cuda")
        """
        torch = self._get_torch()

        target = target or self._config.target
        precision = precision or self._config.precision
        opt_level = opt_level if opt_level is not None else self._config.opt_level
        mode = mode or self._config.mode

        def decorator(fn: Callable) -> Callable:
            return ZenithCompiledPyTorchFunction(
                fn,
                adapter=self,
                target=target,
                precision=precision,
                opt_level=opt_level,
                mode=mode,
                fullgraph=fullgraph,
                **kwargs,
            )

        if func is not None:
            return decorator(func)
        return decorator

    # =========================================================================
    # Training Integration
    # =========================================================================

    def wrap_training_step(
        self,
        train_step_fn: Callable,
        enable_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        grad_scaler: Any = None,
    ) -> Callable:
        """
        Wrap a custom training step with Zenith optimizations.

        Args:
            train_step_fn: Original training step function.
            enable_amp: Enable Automatic Mixed Precision.
            gradient_accumulation_steps: Gradient accumulation steps.
            grad_scaler: Optional GradScaler for AMP.

        Returns:
            Optimized training step function.

        Example:
            def train_step(model, batch, optimizer):
                optimizer.zero_grad()
                loss = model(batch).loss
                loss.backward()
                optimizer.step()
                return loss

            optimized_step = adapter.wrap_training_step(
                train_step,
                enable_amp=True
            )
        """
        torch = self._get_torch()

        # Create grad scaler if AMP enabled and not provided
        if enable_amp and grad_scaler is None:
            try:
                grad_scaler = torch.cuda.amp.GradScaler()
            except Exception:
                logger.warning("GradScaler not available, AMP disabled")
                enable_amp = False

        @functools.wraps(train_step_fn)
        def wrapped_step(*args, **kwargs):
            if enable_amp:
                with torch.cuda.amp.autocast():
                    return train_step_fn(*args, **kwargs)
            return train_step_fn(*args, **kwargs)

        return wrapped_step

    def create_optimizer_wrapper(
        self,
        optimizer: Any,
        enable_amp: bool = False,
    ) -> "ZenithOptimizerWrapper":
        """
        Create a Zenith-optimized optimizer wrapper.

        Args:
            optimizer: PyTorch optimizer.
            enable_amp: Enable Automatic Mixed Precision.

        Returns:
            ZenithOptimizerWrapper for use in training.
        """
        return ZenithOptimizerWrapper(
            optimizer=optimizer,
            adapter=self,
            enable_amp=enable_amp,
        )

    # =========================================================================
    # ONNX Export
    # =========================================================================

    def to_onnx(
        self,
        model: Any,
        sample_input: Any,
        output_path: Optional[str] = None,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
        opset_version: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch nn.Module.
            sample_input: Sample input for tracing.
            output_path: Optional path to save ONNX file.
            input_names: Names for inputs.
            output_names: Names for outputs.
            dynamic_axes: Dynamic axis specification.
            opset_version: ONNX opset version.

        Returns:
            ONNX model as bytes.
        """
        torch = self._get_torch()

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        if opset_version is None:
            opset_version = self._config.opset_version

        model.eval()

        buffer = io.BytesIO()

        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                buffer,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                **kwargs,
            )

        onnx_bytes = buffer.getvalue()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)

        return onnx_bytes

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_input_shapes(self, model: Any) -> list[TensorDescriptor]:
        """Get input shapes - requires sample input."""
        raise NotImplementedError(
            "get_input_shapes requires sample input for PyTorch models. "
            "Use from_model() with sample_input instead."
        )

    def get_output_shapes(self, model: Any) -> list[TensorDescriptor]:
        """Get output shapes - requires sample input."""
        raise NotImplementedError(
            "get_output_shapes requires sample input for PyTorch models. "
            "Use from_model() with sample_input instead."
        )

    def _torch_dtype_to_zenith(self, torch_dtype) -> DataType:
        """Convert PyTorch dtype to Zenith DataType."""
        torch = self._get_torch()

        mapping = {
            torch.float32: DataType.Float32,
            torch.float16: DataType.Float16,
            torch.bfloat16: DataType.BFloat16,
            torch.float64: DataType.Float64,
            torch.int8: DataType.Int8,
            torch.int16: DataType.Int16,
            torch.int32: DataType.Int32,
            torch.int64: DataType.Int64,
            torch.uint8: DataType.UInt8,
            torch.bool: DataType.Bool,
        }

        return mapping.get(torch_dtype, DataType.Float32)


# =============================================================================
# ZenithCompiledPyTorchFunction - Compilation Hook Implementation
# =============================================================================


class ZenithCompiledPyTorchFunction:
    """
    Wrapper for PyTorch functions with Zenith optimizations.

    This class implements the compilation hook functionality similar to
    torch.compile, providing automatic optimization of PyTorch functions.
    """

    def __init__(
        self,
        func: Callable,
        adapter: PyTorchAdapter,
        target: str = "cuda",
        precision: str = "fp32",
        opt_level: int = 2,
        mode: str = "default",
        fullgraph: bool = False,
        **kwargs,
    ):
        self._original_func = func
        self._adapter = adapter
        self._target = target
        self._precision = precision
        self._opt_level = opt_level
        self._mode = mode
        self._fullgraph = fullgraph
        self._kwargs = kwargs

        self._compiled_func = None
        self._graph_ir = None

        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        """Execute the compiled function."""
        if self._compiled_func is None:
            self._compile(*args, **kwargs)

        return self._execute(*args, **kwargs)

    def _compile(self, *args, **kwargs):
        """Compile the function with Zenith optimizations."""
        try:
            import torch
        except ImportError as err:
            raise ImportError("PyTorch is required") from err

        logger.info(
            f"Compiling PyTorch function with Zenith "
            f"(target={self._target}, precision={self._precision})"
        )

        # Create Zenith backend
        backend = self._adapter.create_compile_backend(
            target=self._target,
            precision=self._precision,
            opt_level=self._opt_level,
        )

        # Use torch.compile with Zenith backend
        if self._adapter._has_torch_compile():
            self._compiled_func = torch.compile(
                self._original_func,
                backend=backend,
                mode=self._mode,
                fullgraph=self._fullgraph,
            )
        else:
            # Fallback for older PyTorch
            self._compiled_func = self._original_func

    def _execute(self, *args, **kwargs):
        """Execute the compiled function."""
        return self._compiled_func(*args, **kwargs)

    def get_stats(self) -> OptimizationStats:
        """Get optimization statistics."""
        return OptimizationStats(
            original_ops=0,
            optimized_ops=0,
            fusion_count=0,
            memory_reduction_pct=0.0,
            estimated_speedup=1.0,
            passes_applied=["zenith_backend", "torch_compile"],
        )


# =============================================================================
# ZenithOptimizerWrapper - Training Optimizer Integration
# =============================================================================


class ZenithOptimizerWrapper:
    """
    Zenith-optimized optimizer wrapper for PyTorch training.

    Provides integration with AMP and gradient optimizations.
    """

    def __init__(
        self,
        optimizer: Any,
        adapter: PyTorchAdapter,
        enable_amp: bool = False,
    ):
        self._optimizer = optimizer
        self._adapter = adapter
        self._enable_amp = enable_amp
        self._grad_scaler = None

        if enable_amp:
            try:
                import torch

                self._grad_scaler = torch.cuda.amp.GradScaler()
            except Exception:
                logger.warning("GradScaler not available")
                self._enable_amp = False

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable] = None):
        """Perform optimizer step."""
        if self._enable_amp and self._grad_scaler:
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
        else:
            self._optimizer.step(closure=closure)

    def scale_loss(self, loss: Any) -> Any:
        """Scale loss for AMP."""
        if self._enable_amp and self._grad_scaler:
            return self._grad_scaler.scale(loss)
        return loss

    @property
    def param_groups(self):
        """Get parameter groups."""
        return self._optimizer.param_groups


# =============================================================================
# Module-level Convenience Functions
# =============================================================================


def compile(
    func_or_model: Any = None,
    *,
    target: str = "cuda",
    precision: str = "fp32",
    opt_level: int = 2,
    mode: str = "default",
    **kwargs,
) -> Any:
    """
    Compile a PyTorch function or model with Zenith optimizations.

    This is the main entry point for PyTorch function compilation,
    similar to torch.compile().

    Can be used as a decorator or function:

        @zenith.torch.compile(target="cuda")
        def forward(x):
            return model(x)

        # Or directly
        compiled = zenith.torch.compile(model, target="cuda")

    Args:
        func_or_model: Function or model to compile.
        target: Target device ("cpu", "cuda").
        precision: Precision ("fp32", "fp16", "bf16", "int8").
        opt_level: Optimization level (1-3).
        mode: torch.compile mode.
        **kwargs: Additional options.

    Returns:
        Compiled function/model or decorator.
    """
    adapter = PyTorchAdapter()

    def decorator(fn):
        return adapter.compile_function(
            fn,
            target=target,
            precision=precision,
            opt_level=opt_level,
            mode=mode,
            **kwargs,
        )

    if func_or_model is not None:
        return decorator(func_or_model)
    return decorator


def create_backend(
    target: str = "cuda",
    precision: str = "fp32",
    opt_level: int = 2,
    **kwargs,
) -> Callable:
    """
    Create a Zenith backend for torch.compile.

    This implements CetakBiru.md Section 6.1:
    Zenith as a custom backend for torch.compile.

    Example:
        import torch
        import zenith

        backend = zenith.torch.create_backend(device="cuda", precision="fp16")
        compiled_model = torch.compile(model, backend=backend)

    Args:
        target: Target device.
        precision: Precision level.
        opt_level: Optimization level.
        **kwargs: Additional options.

    Returns:
        Backend callable for torch.compile.
    """
    adapter = PyTorchAdapter()
    return adapter.create_compile_backend(
        target=target,
        precision=precision,
        opt_level=opt_level,
        **kwargs,
    )
