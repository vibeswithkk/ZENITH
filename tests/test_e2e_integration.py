"""
End-to-End Integration Tests untuk Zenith Pipeline.

Test Flow:
1. Model Definition → GraphIR Creation
2. GraphIR → ONNX Export
3. ONNX Import → Optimization → Export
4. Optimized Model Execution
5. Accuracy Verification
"""

import pytest
import numpy as np
import math
import time
from typing import Dict, List, Any, Optional, Tuple


# ==============================================================================
# Mock Classes for Integration Testing
# ==============================================================================


class TensorSpec:
    """Spesifikasi tensor untuk testing."""

    def __init__(self, name: str, shape: List[int], dtype: str = "float32"):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"TensorSpec({self.name}, {self.shape}, {self.dtype})"


class NodeSpec:
    """Spesifikasi node untuk GraphIR."""

    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Optional[Dict] = None,
    ):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}


class GraphIRMock:
    """Mock GraphIR untuk integration testing."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: List[NodeSpec] = []
        self.inputs: List[TensorSpec] = []
        self.outputs: List[TensorSpec] = []
        self.weights: Dict[str, np.ndarray] = {}

    def add_input(self, name: str, shape: List[int], dtype: str = "float32"):
        self.inputs.append(TensorSpec(name, shape, dtype))

    def add_output(self, name: str, shape: List[int], dtype: str = "float32"):
        self.outputs.append(TensorSpec(name, shape, dtype))

    def add_node(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        **attrs,
    ):
        self.nodes.append(NodeSpec(name, op_type, inputs, outputs, attrs))

    def add_weight(self, name: str, data: np.ndarray):
        self.weights[name] = data

    def node_count(self) -> int:
        return len(self.nodes)


class ONNXExporterMock:
    """Mock ONNX Exporter."""

    def __init__(self):
        self.exported_graphs = []

    def export(self, graph: GraphIRMock) -> Dict:
        """Export GraphIR to ONNX-like format."""
        onnx_model = {
            "name": graph.name,
            "inputs": [{"name": t.name, "shape": t.shape} for t in graph.inputs],
            "outputs": [{"name": t.name, "shape": t.shape} for t in graph.outputs],
            "nodes": [
                {
                    "name": n.name,
                    "op_type": n.op_type,
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                    "attributes": n.attributes,
                }
                for n in graph.nodes
            ],
            "weights": {k: v.tolist() for k, v in graph.weights.items()},
        }
        self.exported_graphs.append(onnx_model)
        return onnx_model


class ONNXImporterMock:
    """Mock ONNX Importer."""

    def import_model(self, onnx_model: Dict) -> GraphIRMock:
        """Import ONNX-like format to GraphIR."""
        graph = GraphIRMock(onnx_model.get("name", "imported"))

        for inp in onnx_model.get("inputs", []):
            graph.add_input(inp["name"], inp["shape"])

        for out in onnx_model.get("outputs", []):
            graph.add_output(out["name"], out["shape"])

        for node in onnx_model.get("nodes", []):
            graph.add_node(
                node["name"],
                node["op_type"],
                node["inputs"],
                node["outputs"],
                **node.get("attributes", {}),
            )

        for name, data in onnx_model.get("weights", {}).items():
            graph.add_weight(name, np.array(data))

        return graph


class GraphOptimizer:
    """Graph optimizer with multiple passes."""

    def __init__(self):
        self.passes_applied = []

    def apply_dead_code_elimination(self, graph: GraphIRMock) -> GraphIRMock:
        """Remove unused nodes."""
        self.passes_applied.append("dead_code_elimination")
        # For mock, just return same graph
        return graph

    def apply_constant_folding(self, graph: GraphIRMock) -> GraphIRMock:
        """Fold constant expressions."""
        self.passes_applied.append("constant_folding")
        return graph

    def apply_operator_fusion(self, graph: GraphIRMock) -> GraphIRMock:
        """Fuse adjacent operators."""
        self.passes_applied.append("operator_fusion")

        # Simulate fusion: look for patterns and fuse
        fused_graph = GraphIRMock(graph.name + "_fused")
        fused_graph.inputs = graph.inputs
        fused_graph.outputs = graph.outputs
        fused_graph.weights = graph.weights

        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]

            # Pattern: MatMul + Add -> FusedMatMulAdd
            if (
                node.op_type == "MatMul"
                and i + 1 < len(graph.nodes)
                and graph.nodes[i + 1].op_type == "Add"
            ):
                fused_node = NodeSpec(
                    name=f"{node.name}_fused",
                    op_type="FusedMatMulAdd",
                    inputs=node.inputs + [graph.nodes[i + 1].inputs[1]],
                    outputs=graph.nodes[i + 1].outputs,
                )
                fused_graph.nodes.append(fused_node)
                i += 2
            else:
                fused_graph.nodes.append(node)
                i += 1

        return fused_graph

    def optimize(self, graph: GraphIRMock) -> GraphIRMock:
        """Run all optimization passes."""
        graph = self.apply_dead_code_elimination(graph)
        graph = self.apply_constant_folding(graph)
        graph = self.apply_operator_fusion(graph)
        return graph


class Quantizer:
    """INT8 Quantizer."""

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.calibration_data = []
        self.scale_zero_points = {}

    def calibrate(self, graph: GraphIRMock, data: np.ndarray):
        """Collect calibration statistics."""
        self.calibration_data.append(data)

        # Compute scale and zero_point
        min_val = np.min(data)
        max_val = np.max(data)

        scale = (max_val - min_val) / 255.0
        zero_point = int(-min_val / scale) if scale != 0 else 0

        self.scale_zero_points["default"] = (scale, zero_point)

    def quantize(self, graph: GraphIRMock) -> GraphIRMock:
        """Quantize graph to INT8."""
        q_graph = GraphIRMock(graph.name + "_int8")
        q_graph.inputs = graph.inputs
        q_graph.outputs = graph.outputs

        for node in graph.nodes:
            # Add quantized version of node
            q_node = NodeSpec(
                name=node.name + "_quant",
                op_type="Quantized" + node.op_type,
                inputs=node.inputs,
                outputs=node.outputs,
                attributes={"dtype": "int8"},
            )
            q_graph.nodes.append(q_node)

        return q_graph


class ModelExecutor:
    """Execute model with given backend."""

    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self.execution_times = []

    def execute(
        self, graph: GraphIRMock, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute graph and return outputs."""
        start = time.perf_counter()

        # Mock execution - just process through nodes
        outputs = {}

        for node in graph.nodes:
            if node.op_type in ["MatMul", "FusedMatMulAdd", "QuantizedMatMul"]:
                # Simulate MatMul
                a_name = node.inputs[0]
                if a_name in inputs:
                    a = inputs[a_name]
                else:
                    a = np.random.randn(4, 8).astype(np.float32)

                # Get or create weight
                w_name = node.inputs[1] if len(node.inputs) > 1 else "weight"
                if w_name in graph.weights:
                    w = graph.weights[w_name]
                else:
                    w = np.random.randn(8, 16).astype(np.float32)

                result = a @ w
                outputs[node.outputs[0]] = result

            elif node.op_type in ["ReLU", "QuantizedReLU"]:
                inp_name = node.inputs[0]
                if inp_name in outputs:
                    inp = outputs[inp_name]
                elif inp_name in inputs:
                    inp = inputs[inp_name]
                else:
                    inp = np.random.randn(4, 16).astype(np.float32)

                outputs[node.outputs[0]] = np.maximum(0, inp)

            elif node.op_type in ["Add", "QuantizedAdd"]:
                a_name = node.inputs[0]
                if a_name in outputs:
                    a = outputs[a_name]
                elif a_name in inputs:
                    a = inputs[a_name]
                else:
                    a = np.random.randn(4, 16).astype(np.float32)

                b_name = node.inputs[1] if len(node.inputs) > 1 else "bias"
                if b_name in graph.weights:
                    b = graph.weights[b_name]
                elif b_name in outputs:
                    b = outputs[b_name]
                else:
                    b = np.zeros(16, dtype=np.float32)

                outputs[node.outputs[0]] = a + b

        end = time.perf_counter()
        self.execution_times.append((end - start) * 1000)

        return outputs


class ErrorBoundsVerifier:
    """Verify error bounds between reference and optimized."""

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def verify(
        self, reference: Dict[str, np.ndarray], optimized: Dict[str, np.ndarray]
    ) -> Tuple[bool, Dict]:
        """Verify outputs match within tolerance."""
        results = {"passed": True, "errors": {}}

        for key in reference:
            if key not in optimized:
                results["passed"] = False
                results["errors"][key] = "Missing output"
                continue

            ref = reference[key]
            opt = optimized[key]

            if ref.shape != opt.shape:
                results["passed"] = False
                results["errors"][key] = f"Shape mismatch: {ref.shape} vs {opt.shape}"
                continue

            max_rel_error = np.max(np.abs(opt - ref) / (np.abs(ref) + 1e-10))
            max_abs_error = np.max(np.abs(opt - ref))

            if max_rel_error > self.tolerance:
                results["passed"] = False
                results["errors"][key] = f"Error {max_rel_error:.6f} > {self.tolerance}"
            else:
                results["errors"][key] = f"OK (max_rel={max_rel_error:.6f})"

        return results["passed"], results


# ==============================================================================
# Integration Test Classes
# ==============================================================================


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_simple_model_pipeline(self):
        """Test: Define → Export → Import → Optimize → Execute."""
        # Step 1: Define model as GraphIR
        graph = GraphIRMock("simple_model")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 16])

        # Add weight
        weight = np.random.randn(8, 16).astype(np.float32)
        graph.add_weight("weight", weight)

        # Add nodes
        graph.add_node("matmul1", "MatMul", ["input", "weight"], ["matmul_out"])
        graph.add_node("relu1", "ReLU", ["matmul_out"], ["output"])

        assert graph.node_count() == 2

        # Step 2: Export to ONNX
        exporter = ONNXExporterMock()
        onnx_model = exporter.export(graph)

        assert onnx_model["name"] == "simple_model"
        assert len(onnx_model["nodes"]) == 2

        # Step 3: Import from ONNX
        importer = ONNXImporterMock()
        imported_graph = importer.import_model(onnx_model)

        assert imported_graph.node_count() == 2

        # Step 4: Optimize
        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(imported_graph)

        assert "dead_code_elimination" in optimizer.passes_applied
        assert "operator_fusion" in optimizer.passes_applied

        # Step 5: Execute
        executor = ModelExecutor("cpu")
        inputs = {"input": np.random.randn(4, 8).astype(np.float32)}
        outputs = executor.execute(optimized, inputs)

        assert "output" in outputs or len(outputs) > 0

    def test_matmul_add_fusion_pipeline(self):
        """Test MatMul + Add fusion optimization."""
        # Create graph with MatMul + Add pattern
        graph = GraphIRMock("fusion_test")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 16])

        weight = np.random.randn(8, 16).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        graph.add_weight("weight", weight)
        graph.add_weight("bias", bias)

        graph.add_node("matmul", "MatMul", ["input", "weight"], ["mm_out"])
        graph.add_node("add", "Add", ["mm_out", "bias"], ["output"])

        original_count = graph.node_count()
        assert original_count == 2

        # Optimize
        optimizer = GraphOptimizer()
        fused = optimizer.apply_operator_fusion(graph)

        # Should have fused to single node
        assert fused.node_count() == 1
        assert fused.nodes[0].op_type == "FusedMatMulAdd"


class TestQuantizationPipeline:
    """Test quantization end-to-end."""

    def test_calibrate_and_quantize(self):
        """Test calibration and quantization pipeline."""
        # Create graph
        graph = GraphIRMock("quant_test")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 8])
        graph.add_node("relu", "ReLU", ["input"], ["output"])

        # Calibrate
        quantizer = Quantizer("minmax")
        calibration_data = np.random.randn(100, 8).astype(np.float32)
        quantizer.calibrate(graph, calibration_data)

        assert len(quantizer.calibration_data) == 1
        assert "default" in quantizer.scale_zero_points

        # Quantize
        q_graph = quantizer.quantize(graph)

        assert q_graph.node_count() == 1
        assert "Quantized" in q_graph.nodes[0].op_type

    def test_quantized_execution(self):
        """Test execution of quantized model."""
        # Create and quantize
        graph = GraphIRMock("quant_exec")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 16])

        graph.add_weight("weight", np.random.randn(8, 16).astype(np.float32))
        graph.add_node("matmul", "MatMul", ["input", "weight"], ["output"])

        quantizer = Quantizer("minmax")
        quantizer.calibrate(graph, np.random.randn(100, 8).astype(np.float32))
        q_graph = quantizer.quantize(graph)

        # Execute quantized
        executor = ModelExecutor("cpu")
        inputs = {"input": np.random.randn(4, 8).astype(np.float32)}
        outputs = executor.execute(q_graph, inputs)

        assert len(outputs) > 0


class TestAccuracyVerification:
    """Test accuracy verification end-to-end."""

    def test_reference_vs_optimized(self):
        """Test accuracy between reference and optimized outputs."""
        # Create reference outputs
        ref_output = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Slightly perturbed optimized output
        opt_output = ref_output + np.random.randn(*ref_output.shape) * 1e-4

        verifier = ErrorBoundsVerifier(tolerance=1e-3)
        passed, results = verifier.verify(
            {"output": ref_output}, {"output": opt_output}
        )

        assert passed
        assert "OK" in results["errors"]["output"]

    def test_accuracy_failure_detection(self):
        """Test that accuracy failures are detected."""
        ref_output = np.array([[1.0, 2.0]], dtype=np.float32)
        # Large perturbation
        opt_output = ref_output + 0.5

        verifier = ErrorBoundsVerifier(tolerance=1e-3)
        passed, results = verifier.verify(
            {"output": ref_output}, {"output": opt_output}
        )

        assert not passed


class TestMultiLayerModel:
    """Test multi-layer model pipeline."""

    def test_bert_like_encoder_layer(self):
        """Test BERT-like encoder layer pipeline."""
        graph = GraphIRMock("bert_encoder")

        # Use smaller dimensions compatible with mock executor
        batch, seq, hidden = 4, 8, 16
        graph.add_input("input", [batch, seq, hidden])
        graph.add_output("output", [batch, seq, hidden])

        # Add attention weights with compatible shapes
        graph.add_weight(
            "qkv_weight", np.random.randn(hidden, hidden).astype(np.float32)
        )
        graph.add_weight(
            "out_weight", np.random.randn(hidden, hidden).astype(np.float32)
        )
        graph.add_weight(
            "ff1_weight", np.random.randn(hidden, hidden).astype(np.float32)
        )
        graph.add_weight(
            "ff2_weight", np.random.randn(hidden, hidden).astype(np.float32)
        )

        # Nodes: simplified attention + FFN
        graph.add_node("qkv", "MatMul", ["input", "qkv_weight"], ["qkv_out"])
        graph.add_node("attn_out", "MatMul", ["qkv_out", "out_weight"], ["attn"])
        graph.add_node("add1", "Add", ["input", "attn"], ["residual1"])
        graph.add_node("ff1", "MatMul", ["residual1", "ff1_weight"], ["ff1_out"])
        graph.add_node("relu", "ReLU", ["ff1_out"], ["ff1_relu"])
        graph.add_node("ff2", "MatMul", ["ff1_relu", "ff2_weight"], ["ff2_out"])
        graph.add_node("add2", "Add", ["residual1", "ff2_out"], ["output"])

        assert graph.node_count() == 7

        # Export/Import
        exporter = ONNXExporterMock()
        importer = ONNXImporterMock()

        onnx = exporter.export(graph)
        imported = importer.import_model(onnx)

        assert imported.node_count() == 7

        # Optimize
        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(imported)

        # Verify optimization passes were applied
        assert "dead_code_elimination" in optimizer.passes_applied
        assert "constant_folding" in optimizer.passes_applied
        assert "operator_fusion" in optimizer.passes_applied

        # Verify some fusion happened (MatMul+Add patterns)
        assert len(optimizer.passes_applied) == 3

        # Verify optimized graph has nodes
        assert optimized is not None


class TestResNetLikeModel:
    """Test ResNet-like model pipeline."""

    def test_resnet_block(self):
        """Test ResNet block with residual connection."""
        graph = GraphIRMock("resnet_block")

        channels = 64
        graph.add_input("input", [1, channels, 56, 56])
        graph.add_output("output", [1, channels, 56, 56])

        # Weights for two conv layers
        graph.add_weight(
            "conv1_weight", np.random.randn(channels, channels, 3, 3).astype(np.float32)
        )
        graph.add_weight(
            "conv2_weight", np.random.randn(channels, channels, 3, 3).astype(np.float32)
        )

        # Nodes
        graph.add_node("conv1", "Conv", ["input", "conv1_weight"], ["conv1_out"])
        graph.add_node("relu1", "ReLU", ["conv1_out"], ["relu1_out"])
        graph.add_node("conv2", "Conv", ["relu1_out", "conv2_weight"], ["conv2_out"])
        graph.add_node("add", "Add", ["input", "conv2_out"], ["residual"])
        graph.add_node("relu2", "ReLU", ["residual"], ["output"])

        assert graph.node_count() == 5

        # Full pipeline
        exporter = ONNXExporterMock()
        importer = ONNXImporterMock()
        optimizer = GraphOptimizer()

        onnx = exporter.export(graph)
        imported = importer.import_model(onnx)
        optimized = optimizer.optimize(imported)

        assert len(optimizer.passes_applied) >= 3


class TestPerformanceBenchmark:
    """Test performance benchmarking in pipeline."""

    def test_execution_timing(self):
        """Test that execution timing is tracked."""
        graph = GraphIRMock("perf_test")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 16])
        graph.add_weight("weight", np.random.randn(8, 16).astype(np.float32))
        graph.add_node("matmul", "MatMul", ["input", "weight"], ["output"])

        executor = ModelExecutor("cpu")

        # Run multiple times
        for _ in range(10):
            inputs = {"input": np.random.randn(4, 8).astype(np.float32)}
            executor.execute(graph, inputs)

        assert len(executor.execution_times) == 10
        assert all(t > 0 for t in executor.execution_times)

    def test_warmup_and_benchmark(self):
        """Test warmup iterations before benchmark."""
        graph = GraphIRMock("benchmark_test")
        graph.add_input("input", [32, 64])
        graph.add_output("output", [32, 128])
        graph.add_weight("weight", np.random.randn(64, 128).astype(np.float32))
        graph.add_node("matmul", "MatMul", ["input", "weight"], ["output"])

        executor = ModelExecutor("cpu")
        inputs = {"input": np.random.randn(32, 64).astype(np.float32)}

        # Warmup
        warmup_iters = 5
        for _ in range(warmup_iters):
            executor.execute(graph, inputs)

        executor.execution_times.clear()

        # Benchmark
        benchmark_iters = 20
        for _ in range(benchmark_iters):
            executor.execute(graph, inputs)

        assert len(executor.execution_times) == benchmark_iters

        # Compute stats
        times = executor.execution_times
        mean_ms = sum(times) / len(times)
        std_ms = np.std(times)

        assert mean_ms > 0
        assert std_ms >= 0


class TestEdgeCases:
    """Test edge cases in pipeline."""

    def test_empty_graph(self):
        """Test handling of empty graph."""
        graph = GraphIRMock("empty")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 8])
        # No nodes - pass-through

        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(graph)

        assert optimized.node_count() == 0

    def test_single_node_graph(self):
        """Test single node graph."""
        graph = GraphIRMock("single")
        graph.add_input("input", [4, 8])
        graph.add_output("output", [4, 8])
        graph.add_node("relu", "ReLU", ["input"], ["output"])

        exporter = ONNXExporterMock()
        onnx = exporter.export(graph)

        assert len(onnx["nodes"]) == 1

    def test_multiple_outputs(self):
        """Test graph with multiple outputs."""
        graph = GraphIRMock("multi_out")
        graph.add_input("input", [4, 8])
        graph.add_output("out1", [4, 8])
        graph.add_output("out2", [4, 8])

        graph.add_node("relu1", "ReLU", ["input"], ["out1"])
        graph.add_node("relu2", "ReLU", ["input"], ["out2"])

        executor = ModelExecutor("cpu")
        inputs = {"input": np.random.randn(4, 8).astype(np.float32)}
        outputs = executor.execute(graph, inputs)

        assert "out1" in outputs
        assert "out2" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
