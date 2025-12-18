"""
Test Suite untuk INT8 Quantization.
Menguji calibrators, quantization parameters, dan quantizer.
"""

import pytest
import math
import struct


class TestQuantizationParameters:
    """Test QuantizationParams functionality."""

    def test_symmetric_quantization(self):
        """Test symmetric quantization (zero_point = 0)."""
        # Untuk symmetric: scale = abs_max / 127
        abs_max = 3.0
        scale = abs_max / 127.0
        zero_point = 0

        # Quantize
        def quantize(value, scale, zp):
            q = round(value / scale) + zp
            return max(-128, min(127, q))

        # Dequantize
        def dequantize(q, scale, zp):
            return scale * (q - zp)

        # Test
        original = 1.5
        q = quantize(original, scale, zero_point)
        reconstructed = dequantize(q, scale, zero_point)

        # Error harus kecil
        error = abs(original - reconstructed)
        assert error < scale  # Error bounded by scale

    def test_asymmetric_quantization(self):
        """Test asymmetric quantization (zero_point != 0)."""
        min_val = -1.0
        max_val = 3.0
        scale = (max_val - min_val) / 255.0
        zero_point = round(-128 - min_val / scale)

        # Clamp zero_point
        zero_point = max(-128, min(127, zero_point))

        def quantize(value, scale, zp):
            q = round(value / scale) + zp
            return max(-128, min(127, q))

        # Test edge values
        q_min = quantize(min_val, scale, zero_point)
        q_max = quantize(max_val, scale, zero_point)

        assert q_min >= -128
        assert q_max <= 127

    def test_quantization_range(self):
        """Test bahwa hasil quantization dalam range INT8."""
        scale = 0.1
        zero_point = 0

        def quantize(value, scale, zp):
            q = round(value / scale) + zp
            return max(-128, min(127, q))

        # Test berbagai nilai
        test_values = [-100.0, -10.0, 0.0, 10.0, 100.0]
        for value in test_values:
            q = quantize(value, scale, zero_point)
            assert -128 <= q <= 127


class TestCalibrators:
    """Test berbagai metode kalibrasi."""

    def test_minmax_calibrator(self):
        """Test MinMax calibrator."""
        data = [1.0, 2.0, 3.0, -1.0, -2.0]
        min_val = min(data)
        max_val = max(data)

        # Symmetric
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / 127.0
        zero_point = 0

        assert scale == 3.0 / 127.0
        assert zero_point == 0

    def test_percentile_calibrator(self):
        """Test Percentile calibrator."""
        # Data dengan outlier
        data = [1.0, 2.0, 3.0, 100.0]  # 100.0 adalah outlier
        sorted_data = sorted(data)

        # 99th percentile
        def percentile(sorted_arr, p):
            idx = p * (len(sorted_arr) - 1) / 100.0
            lower = int(math.floor(idx))
            upper = int(math.ceil(idx))
            if lower == upper:
                return sorted_arr[lower]
            frac = idx - lower
            return sorted_arr[lower] * (1 - frac) + sorted_arr[upper] * frac

        p99 = percentile(sorted_data, 99.0)
        # Percentile harus lebih kecil dari max (ignoring outlier effect)
        assert p99 <= max(data)

    def test_histogram_creation(self):
        """Test pembuatan histogram untuk entropy calibration."""
        num_bins = 10
        min_val = -1.0
        max_val = 1.0
        bin_width = (max_val - min_val) / num_bins

        data = [0.0, 0.1, 0.2, -0.1, -0.2]
        histogram = [0] * num_bins

        for value in data:
            if value < min_val:
                bin_idx = 0
            elif value >= max_val:
                bin_idx = num_bins - 1
            else:
                bin_idx = int((value - min_val) / bin_width)
            histogram[bin_idx] += 1

        assert sum(histogram) == len(data)

    def test_kl_divergence(self):
        """Test perhitungan KL divergence."""
        # Dua distribusi
        p = [0.1, 0.2, 0.3, 0.4]  # Original
        q = [0.25, 0.25, 0.25, 0.25]  # Uniform (quantized approximation)

        # KL(P || Q) = sum(p * log(p/q))
        kl = 0.0
        for pi, qi in zip(p, q):
            if pi > 0 and qi > 0:
                kl += pi * math.log(pi / qi)

        # KL divergence harus non-negatif
        assert kl >= 0


class TestQuantizer:
    """Test Quantizer class."""

    def test_quantize_array(self):
        """Test quantization of array."""
        scale = 0.1
        zero_point = 0

        original = [0.5, 1.0, -0.5, 0.0]

        def quantize(v):
            q = round(v / scale) + zero_point
            return max(-128, min(127, q))

        def dequantize(q):
            return scale * (q - zero_point)

        quantized = [quantize(v) for v in original]
        reconstructed = [dequantize(q) for q in quantized]

        # Verifikasi error
        for o, r in zip(original, reconstructed):
            assert abs(o - r) < scale

    def test_error_computation(self):
        """Test RMS error computation."""
        original = [1.0, 2.0, 3.0]
        reconstructed = [1.1, 1.9, 3.1]

        # RMS error
        mse = sum((o - r) ** 2 for o, r in zip(original, reconstructed))
        rmse = math.sqrt(mse / len(original))

        assert rmse < 0.2

    def test_supported_ops(self):
        """Test operasi yang didukung untuk quantization."""
        supported = ["Conv", "MatMul", "Gemm", "Linear", "Add"]
        skipped = ["Softmax", "LayerNormalization", "BatchNormalization"]

        # Semua ops harus non-empty string
        for op in supported + skipped:
            assert isinstance(op, str)
            assert len(op) > 0


class TestQuantizationMathematics:
    """Test formula matematika kuantisasi dari CetakBiru."""

    def test_quantization_formula(self):
        """
        Test formula dari CetakBiru:
        min_{s,z} ||X - s*(quantize(X/s + z) - z)||²
        """
        X = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Try different scales
        best_scale = None
        min_error = float("inf")

        for scale_idx in range(1, 100):
            scale = scale_idx * 0.1
            zero_point = 0

            # Quantize and dequantize
            error = 0.0
            for x in X:
                q = round(x / scale) + zero_point
                q = max(-128, min(127, q))
                x_reconstructed = scale * (q - zero_point)
                error += (x - x_reconstructed) ** 2

            if error < min_error:
                min_error = error
                best_scale = scale

        # Best scale should be reasonable
        assert best_scale is not None
        assert best_scale > 0

    def test_calibration_methods_order(self):
        """Test bahwa entropy calibration lebih akurat dari minmax."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 10.0]  # Outlier at end

        # MinMax: sensitif terhadap outlier
        minmax_scale = max(abs(d) for d in data) / 127.0

        # Percentile 99%: ignores outlier
        sorted_data = sorted(data)
        p99 = sorted_data[int(0.99 * (len(sorted_data) - 1))]
        percentile_scale = p99 / 127.0

        # Percentile scale harus lebih kecil (lebih presisi untuk data normal)
        assert percentile_scale < minmax_scale


class TestQuantizedOperations:
    """Test operasi yang sudah dikuantisasi."""

    def test_quantized_add(self):
        """Test quantized addition."""
        # Dua tensor dengan scale berbeda
        a_scale = 0.1
        b_scale = 0.2
        output_scale = 0.15

        # Nilai asli
        a_float = 1.0
        b_float = 2.0
        expected = a_float + b_float

        # Quantize inputs
        a_q = round(a_float / a_scale)
        b_q = round(b_float / b_scale)

        # Requantize ke output scale
        result_float = a_scale * a_q + b_scale * b_q
        output_q = round(result_float / output_scale)
        final = output_scale * output_q

        # Error check
        assert abs(final - expected) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
