# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Engine - Main compilation and execution engine.

Inspired by:
- TensorRT: Build phase creates optimized engine, runtime executes
- ONNX Runtime: Session-based execution with providers
- TVM: Module compilation and deployment

This is the main entry point for compiling and executing models with Zenith.

Example:
    from zenith.runtime import ZenithEngine, CompileConfig

    engine = ZenithEngine(backend="cuda")
    config = CompileConfig(precision="fp16", mode="max-autotune")

    compiled = engine.compile(graph_ir, config)
    output = compiled(input_tensor)
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Union
from enum import Enum
import time
import numpy as np

from .kernel_registry import KernelRegistry, Precision, get_registry
from .executor import GraphExecutor, ExecutionPlan
from .dispatcher import KernelDispatcher
from .memory_manager import MemoryManager
from .context import ExecutionContext

from zenith.observability import ZenithLogger
from zenith.errors import CompilationError, ValidationError


class CompileMode(Enum):
    """Compilation mode (inspired by torch.compile)."""

    DEFAULT = "default"  # Balanced compile time and performance
    REDUCE_OVERHEAD = "reduce-overhead"  # Minimize CPU overhead, use CUDA Graphs
    MAX_AUTOTUNE = "max-autotune"  # Maximum tuning, longer compile time


@dataclass
class CompileConfig:
    """
    Configuration for model compilation.

    Attributes:
        precision: Target precision ("fp32", "fp16", "bf16", "int8")
        mode: Compilation mode
        use_cuda_graphs: Whether to use CUDA Graphs for low latency
        auto_tune: Whether to auto-tune kernel selection
        cache_compiled: Whether to cache compiled results
        tolerance: Numerical accuracy tolerance
        verbose: Verbosity level (0-4)
    """

    precision: str = "fp32"
    mode: str = "default"
    use_cuda_graphs: bool = False
    auto_tune: bool = False
    cache_compiled: bool = True
    tolerance: float = 1e-6
    verbose: int = 2

    def get_precision(self) -> Precision:
        """Get precision enum."""
        precision_map = {
            "fp32": Precision.FP32,
            "fp16": Precision.FP16,
            "bf16": Precision.BF16,
            "int8": Precision.INT8,
        }
        return precision_map.get(self.precision.lower(), Precision.FP32)


@dataclass
class CompileStats:
    """Statistics from compilation."""

    compile_time_ms: float
    num_nodes: int
    num_supported_ops: int
    num_unsupported_ops: int
    fused_patterns: int
    estimated_speedup: float


class CompiledModel:
    """
    A compiled model ready for execution.

    This is the main interface for running inference with Zenith-optimized models.

    Example:
        compiled = engine.compile(graph_ir, config)

        # Call like a function
        output = compiled(input_tensor)

        # Or with named inputs
        output = compiled.run({"input": input_tensor})
    """

    def __init__(
        self,
        engine: "ZenithEngine",
        executor: GraphExecutor,
        graph_ir: Any,
        config: CompileConfig,
        original_model: Any = None,
        stats: CompileStats = None,
    ):
        self.engine = engine
        self.executor = executor
        self.graph_ir = graph_ir
        self.config = config
        self.original_model = original_model
        self.compile_stats = stats

        # Execution metadata
        self._execution_count = 0
        self._total_inference_time = 0

    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute the model.

        Accepts both positional args and keyword args.
        Positional args are mapped to input names in order.
        """
        # Build input dict
        inputs = {}

        # Map positional args to input names
        input_names = self.executor.plan.input_names
        for i, arg in enumerate(args):
            if i < len(input_names):
                inputs[input_names[i]] = arg

        # Add keyword args
        inputs.update(kwargs)

        return self.run(inputs)

    def run(
        self, inputs: dict[str, Any], return_dict: bool = False
    ) -> Union[Any, dict[str, Any]]:
        """
        Run inference.

        Args:
            inputs: Dictionary of input name -> tensor
            return_dict: If True, return dict of outputs; else return first output

        Returns:
            Output tensor(s)
        """
        start_time = time.perf_counter()

        outputs = self.executor.run(inputs, return_numpy=True)

        self._execution_count += 1
        self._total_inference_time += (time.perf_counter() - start_time) * 1000

        if return_dict:
            return outputs

        # Return single output
        if outputs:
            return next(iter(outputs.values()))
        return None

    def benchmark(
        self, sample_input: dict[str, Any], num_warmup: int = 5, num_runs: int = 50
    ) -> dict:
        """
        Benchmark the compiled model.

        Args:
            sample_input: Sample input for benchmarking
            num_warmup: Number of warmup runs
            num_runs: Number of timed runs

        Returns:
            Benchmark results
        """
        # Warmup
        for _ in range(num_warmup):
            self.run(sample_input)

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.run(sample_input)
            times.append((time.perf_counter() - start) * 1000)

        return {
            "num_runs": num_runs,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p90_ms": np.percentile(times, 90),
            "p99_ms": np.percentile(times, 99),
            "throughput": 1000 / np.mean(times),
        }

    def summary(self) -> str:
        """Get model summary."""
        num_nodes = (
            self.graph_ir.num_nodes() if hasattr(self.graph_ir, "num_nodes") else 0
        )

        lines = [
            "=" * 60,
            "Zenith Compiled Model",
            "=" * 60,
            f"Graph: {self.graph_ir.name if hasattr(self.graph_ir, 'name') else 'unnamed'}",
            f"Nodes: {num_nodes}",
            f"Precision: {self.config.precision}",
            f"Mode: {self.config.mode}",
            f"Device: {self.engine.backend}",
            "",
            f"Inputs: {self.executor.plan.input_names}",
            f"Outputs: {self.executor.plan.output_names}",
            "",
        ]

        if self.compile_stats:
            lines.extend(
                [
                    "Compilation Stats:",
                    f"  Compile time: {self.compile_stats.compile_time_ms:.2f} ms",
                    f"  Supported ops: {self.compile_stats.num_supported_ops}",
                    f"  Fused patterns: {self.compile_stats.fused_patterns}",
                    f"  Est. speedup: {self.compile_stats.estimated_speedup:.2f}x",
                ]
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        num_nodes = (
            self.graph_ir.num_nodes() if hasattr(self.graph_ir, "num_nodes") else 0
        )
        return (
            f"CompiledModel("
            f"nodes={num_nodes}, "
            f"precision={self.config.precision}, "
            f"device={self.engine.backend})"
        )


class ZenithEngine:
    """
    Main engine for compiling and optimizing models.

    This is the central component that:
    1. Takes an optimized GraphIR
    2. Selects optimal kernels
    3. Builds an execution plan
    4. Creates a CompiledModel for inference

    Example:
        engine = ZenithEngine(backend="cuda")

        # Compile a model
        compiled = engine.compile(graph_ir, config)

        # Run inference
        output = compiled(input)
    """

    def __init__(self, backend: str = "cuda", registry: KernelRegistry = None):
        """
        Initialize engine.

        Args:
            backend: Target backend ("cuda", "cpu", "rocm")
            registry: Kernel registry (uses global if None)
        """
        self.backend = backend
        self.registry = registry or get_registry()
        self.memory_manager = MemoryManager(device=backend)

        # Compilation cache
        self._cache: dict[str, CompiledModel] = {}

        # Statistics
        self._compile_count = 0

    def compile(
        self,
        graph_ir: Any,
        config: CompileConfig = None,
        sample_input: Any = None,
        original_model: Any = None,
    ) -> CompiledModel:
        """
        Compile a GraphIR into an executable model.

        Args:
            graph_ir: The GraphIR to compile
            config: Compilation configuration
            sample_input: Sample input for shape inference
            original_model: Original framework model (for fallback)

        Returns:
            CompiledModel ready for inference
        """
        config = config or CompileConfig()
        start_time = time.perf_counter()
        logger = ZenithLogger.get()

        # Log compilation start
        logger.info(
            f"Compiling model for {self.backend}",
            component="compiler",
            operation="compile",
        )

        # Step 1: Validate graph
        self._validate_graph(graph_ir)

        # Step 2: Build execution plan
        execution_plan = self._build_execution_plan(graph_ir, config)

        # Step 3: Load weights from graph
        self._load_weights(execution_plan, graph_ir)

        # Step 4: Create executor
        executor = GraphExecutor(
            execution_plan=execution_plan,
            precision=config.get_precision(),
            device=self.backend,
            enable_profiling=config.verbose >= 3,
        )

        # Calculate stats
        compile_time = (time.perf_counter() - start_time) * 1000
        stats = self._calculate_stats(execution_plan, compile_time)

        # Create compiled model
        compiled = CompiledModel(
            engine=self,
            executor=executor,
            graph_ir=graph_ir,
            config=config,
            original_model=original_model,
            stats=stats,
        )

        # Log completion with compile summary
        logger.compile_summary(
            {
                "model_name": (graph_ir.name if hasattr(graph_ir, "name") else "model"),
                "target": self.backend,
                "precision": config.precision,
                "compile_time": compile_time / 1000,
                "fused_ops": stats.fused_patterns,
                "dce_removed": 0,
                "estimated_speedup": stats.estimated_speedup,
            }
        )

        self._compile_count += 1

        return compiled

    def _validate_graph(self, graph_ir: Any) -> None:
        """Validate the GraphIR."""
        if graph_ir is None:
            raise ValidationError(
                "GraphIR cannot be None",
                parameter="graph_ir",
                expected="valid GraphIR object",
                received="None",
            )

        # Check if graph has nodes
        if hasattr(graph_ir, "num_nodes"):
            if graph_ir.num_nodes() == 0:
                model_name = graph_ir.name if hasattr(graph_ir, "name") else "unknown"
                raise CompilationError(
                    "GraphIR has no nodes",
                    model_name=model_name,
                )

    def _build_execution_plan(
        self, graph_ir: Any, config: CompileConfig
    ) -> ExecutionPlan:
        """
        Build execution plan from GraphIR.

        Performs topological sort and kernel selection.
        """
        # Get nodes in topological order
        if hasattr(graph_ir, "topological_sort"):
            nodes = graph_ir.topological_sort()
        elif hasattr(graph_ir, "nodes"):
            nodes = list(graph_ir.nodes)
        elif hasattr(graph_ir, "get_nodes"):
            nodes = graph_ir.get_nodes()
        else:
            nodes = []

        # Get input/output names
        input_names = []
        output_names = []

        if hasattr(graph_ir, "inputs"):
            input_names = [
                inp.name if hasattr(inp, "name") else str(inp)
                for inp in graph_ir.inputs
            ]
        if hasattr(graph_ir, "outputs"):
            output_names = [
                out.name if hasattr(out, "name") else str(out)
                for out in graph_ir.outputs
            ]

        return ExecutionPlan(
            nodes=nodes,
            input_names=input_names,
            output_names=output_names,
            node_weights={},
            memory_plan={},
            total_ops=len(nodes),
        )

    def _load_weights(self, plan: ExecutionPlan, graph_ir: Any) -> None:
        """Load weights from graph into execution plan."""
        # Get initializers (weights)
        initializers = {}
        if hasattr(graph_ir, "initializers"):
            initializers = graph_ir.initializers
        elif hasattr(graph_ir, "get_initializers"):
            initializers = graph_ir.get_initializers()

        if not initializers:
            return

        # Map weights to nodes that use them
        for node in plan.nodes:
            node_name = node.name if hasattr(node, "name") else str(node)
            node_inputs = node.inputs if hasattr(node, "inputs") else []

            weights = {}
            for inp_name in node_inputs:
                if inp_name in initializers:
                    data = initializers[inp_name]
                    # Convert to GPU
                    if self.backend == "cuda":
                        data = self.memory_manager.to_gpu(np.ascontiguousarray(data))
                    weights[inp_name] = data

            if weights:
                plan.node_weights[node_name] = weights

    def _calculate_stats(
        self, plan: ExecutionPlan, compile_time: float
    ) -> CompileStats:
        """Calculate compilation statistics."""
        num_supported = 0
        num_unsupported = 0

        precision = Precision.FP32  # Default

        for node in plan.nodes:
            op_type = node.op_type if hasattr(node, "op_type") else str(node)
            if self.registry.is_supported(op_type, precision):
                num_supported += 1
            else:
                num_unsupported += 1

        # Estimate speedup based on precision and fusion
        estimated_speedup = 1.0
        if precision == Precision.FP16:
            estimated_speedup *= 2.0  # FP16 roughly 2x faster

        return CompileStats(
            compile_time_ms=compile_time,
            num_nodes=len(plan.nodes),
            num_supported_ops=num_supported,
            num_unsupported_ops=num_unsupported,
            fused_patterns=0,  # TODO: Count fusions
            estimated_speedup=estimated_speedup,
        )

    def _log_compile_complete(self, stats: CompileStats, config: CompileConfig) -> None:
        """Log compilation completion."""
        print(f"""
┌─────────────────────────────────────────────────────────┐
│ Zenith Compilation Complete                             │
├─────────────────────────────────────────────────────────┤
│ Target:     {self.backend:<44} │
│ Precision:  {config.precision:<44} │
│ Mode:       {config.mode:<44} │
│ Time:       {stats.compile_time_ms:.2f} ms{" " * 37} │
│                                                         │
│ Graph Statistics:                                       │
│   Total nodes:      {stats.num_nodes:<35} │
│   Supported ops:    {stats.num_supported_ops:<35} │
│   Unsupported ops:  {stats.num_unsupported_ops:<35} │
│   Est. speedup:     {stats.estimated_speedup:.1f}x{" " * 33} │
└─────────────────────────────────────────────────────────┘
        """)

    def list_supported_ops(self) -> list[str]:
        """List all supported operations."""
        return self.registry.list_supported_ops()

    def is_op_supported(self, op_type: str, precision: str = "fp32") -> bool:
        """Check if an operation is supported."""
        prec = Precision.FP16 if precision == "fp16" else Precision.FP32
        return self.registry.is_supported(op_type, prec)


# Convenience function
def create_engine(backend: str = "cuda") -> ZenithEngine:
    """Create a Zenith engine."""
    return ZenithEngine(backend=backend)
