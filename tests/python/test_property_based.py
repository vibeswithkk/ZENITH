"""
Property-based tests using Hypothesis.

Tests mathematical guarantees and edge cases using property-based testing.

These tests require:
    - hypothesis library: pip install hypothesis
    - zenith.core module

If hypothesis is not available, all tests will be skipped.
"""

import pytest
import numpy as np

# Check if hypothesis is available
try:
    from hypothesis import given, strategies as st, settings

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = None
    st = None
    settings = None

# Check if zenith modules are available
try:
    from zenith.core import Shape, DataType, TensorDescriptor, GraphIR
    from zenith.optimization import (
        compute_conv_bn_weights,
        PrecisionPolicy,
    )
    from zenith.optimization.quantization import QuantizationParams

    ZENITH_AVAILABLE = True
except ImportError:
    ZENITH_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (HYPOTHESIS_AVAILABLE and ZENITH_AVAILABLE),
    reason="hypothesis or zenith modules not available",
)


class TestQuantizationProperties:
    """Property-based tests for quantization."""

    @given(st.floats(min_value=1e-10, max_value=1e6, allow_nan=False))
    @settings(max_examples=100)
    def test_quantization_scale_positive(self, val):
        """Quantization scale should always be positive."""
        params = QuantizationParams(scale=val, zero_point=0)
        assert params.scale > 0

    @given(st.integers(min_value=-128, max_value=127))
    @settings(max_examples=50)
    def test_quantization_zero_point_range(self, zp):
        """Zero point should be in valid int8 range."""
        params = QuantizationParams(scale=1.0, zero_point=zp)
        assert -128 <= params.zero_point <= 127


class TestConvBNFusionProperties:
    """Property-based tests for Conv-BN fusion."""

    @given(
        st.integers(min_value=1, max_value=64),  # out_channels
        st.integers(min_value=1, max_value=64),  # in_channels
    )
    @settings(max_examples=30)
    def test_fused_weight_shape_preserved(self, out_ch, in_ch):
        """Fused weights should preserve shape."""
        conv_weight = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(out_ch).astype(np.float32)
        bn_gamma = np.ones(out_ch, dtype=np.float32)
        bn_beta = np.zeros(out_ch, dtype=np.float32)
        bn_mean = np.zeros(out_ch, dtype=np.float32)
        bn_var = np.ones(out_ch, dtype=np.float32) + 0.1

        fused_w, fused_b = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var
        )

        assert fused_w.shape == conv_weight.shape
        assert fused_b.shape == conv_bias.shape

    @given(st.floats(min_value=1e-10, max_value=1e-3))
    @settings(max_examples=20)
    def test_epsilon_effect(self, epsilon):
        """Different epsilon values should produce valid results."""
        conv_weight = np.random.randn(8, 4, 3, 3).astype(np.float32)
        conv_bias = np.random.randn(8).astype(np.float32)
        bn_gamma = np.ones(8, dtype=np.float32)
        bn_beta = np.zeros(8, dtype=np.float32)
        bn_mean = np.zeros(8, dtype=np.float32)
        bn_var = np.ones(8, dtype=np.float32) * 0.1

        fused_w, fused_b = compute_conv_bn_weights(
            conv_weight, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var, epsilon
        )

        assert not np.isnan(fused_w).any()
        assert not np.isnan(fused_b).any()


class TestShapeProperties:
    """Property-based tests for Shape."""

    @given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=6))
    @settings(max_examples=50)
    def test_shape_creation(self, dims):
        """Shape should preserve dimensions."""
        shape = Shape(dims)
        assert len(shape) == len(dims)
        for i, d in enumerate(dims):
            assert shape[i] == d


class TestTensorDescriptorProperties:
    """Property-based tests for TensorDescriptor."""

    @given(
        st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=4),
    )
    @settings(max_examples=50)
    def test_tensor_descriptor_name(self, name, dims):
        """TensorDescriptor should preserve name."""
        td = TensorDescriptor(name, Shape(dims), DataType.Float32)
        assert td.name == name


class TestGraphIRProperties:
    """Property-based tests for GraphIR."""

    @given(
        st.text(
            min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_0123456789"
        )
    )
    @settings(max_examples=30)
    def test_graphir_name(self, name):
        """GraphIR should preserve name."""
        graph = GraphIR(name=name)
        assert graph.name == name

    @given(st.integers(min_value=0, max_value=10))
    @settings(max_examples=20)
    def test_graphir_add_inputs(self, n_inputs):
        """GraphIR should track correct number of inputs."""
        graph = GraphIR(name="test")
        for i in range(n_inputs):
            td = TensorDescriptor(f"in_{i}", Shape([1, 10]), DataType.Float32)
            graph.add_input(td)
        assert len(graph.inputs) == n_inputs


class TestMixedPrecisionProperties:
    """Property-based tests for mixed precision."""

    @given(st.floats(min_value=1.0, max_value=2**20, allow_nan=False))
    @settings(max_examples=30)
    def test_loss_scale_range(self, scale):
        """Loss scale should be valid."""
        policy = PrecisionPolicy.fp16_with_loss_scale(initial_scale=scale)
        assert policy.loss_scale == scale
        assert policy.use_dynamic_loss_scaling
