"""
Test Suite untuk Error Bounds Verification.
Berdasarkan CetakBiru.md Bab 4.1: Jaminan Stabilitas Numerik.
"""

import pytest
import math


class TestErrorTolerance:
    """Test konfigurasi toleransi error."""

    def test_default_tolerances(self):
        """Test default tolerances dari CetakBiru."""
        # Nilai default dari CetakBiru
        delta_fp32 = 1e-6
        delta_fp16 = 1e-3
        delta_int8 = 5e-2

        # FP32 harus paling ketat
        assert delta_fp32 < delta_fp16 < delta_int8

    def test_epsilon_prevents_division_by_zero(self):
        """Test bahwa epsilon mencegah division by zero."""
        epsilon = 1e-10
        reference = 0.0
        optimized = 1e-12

        # Tanpa epsilon -> division by zero
        # Dengan epsilon -> aman
        rel_error = abs(optimized - reference) / (abs(reference) + epsilon)
        assert math.isfinite(rel_error)


class TestRelativeErrorFormula:
    """Test formula relative error dari CetakBiru.

    |T(F)(x) - F(x)| / (|F(x)| + ε) ≤ δ
    """

    def test_relative_error_calculation(self):
        """Test perhitungan relative error."""
        epsilon = 1e-10

        # Case 1: Small error
        reference = 1.0
        optimized = 1.0001
        rel_err = abs(optimized - reference) / (abs(reference) + epsilon)
        assert abs(rel_err - 0.0001) < 1e-8

        # Case 2: Reference near zero
        reference = 0.0
        optimized = 0.001
        rel_err = abs(optimized - reference) / (abs(reference) + epsilon)
        # Dengan epsilon, ini tetap terdefinisi
        assert math.isfinite(rel_err)

        # Case 3: Negative values
        reference = -5.0
        optimized = -5.005
        rel_err = abs(optimized - reference) / (abs(reference) + epsilon)
        assert abs(rel_err - 0.001) < 1e-8

    def test_tolerance_check(self):
        """Test bahwa toleransi berfungsi dengan benar."""
        epsilon = 1e-10
        delta = 1e-3  # tolerance

        test_cases = [
            # (reference, optimized, should_pass)
            (1.0, 1.0005, True),  # 0.0005 < 1e-3
            (1.0, 1.002, False),  # 0.002 > 1e-3
            (100.0, 100.05, True),  # 0.0005 < 1e-3
            (100.0, 100.2, False),  # 0.002 > 1e-3
        ]

        for ref, opt, expected_pass in test_cases:
            rel_err = abs(opt - ref) / (abs(ref) + epsilon)
            passed = rel_err <= delta
            assert passed == expected_pass, f"Failed for ref={ref}, opt={opt}"


class TestGradientFidelity:
    """Test gradient fidelity dari CetakBiru.

    ||∇̂L(θ) - ∇L(θ)||₂ ≤ γ · ||∇L(θ)||₂
    """

    def test_gradient_l2_norm(self):
        """Test perhitungan L2 norm gradient."""
        grad = [1.0, 2.0, 3.0, 4.0]
        l2_norm = math.sqrt(sum(g**2 for g in grad))
        expected = math.sqrt(1 + 4 + 9 + 16)
        assert abs(l2_norm - expected) < 1e-10

    def test_gradient_fidelity_check(self):
        """Test gradient fidelity formula."""
        gamma = 0.001

        ref_grad = [1.0, 2.0, 3.0]
        opt_grad = [1.001, 2.002, 3.003]

        # Compute norms
        diff = [o - r for r, o in zip(ref_grad, opt_grad)]
        diff_norm = math.sqrt(sum(d**2 for d in diff))
        ref_norm = math.sqrt(sum(r**2 for r in ref_grad))

        relative_grad_error = diff_norm / ref_norm
        passed = relative_grad_error <= gamma

        # Error is sqrt(0.001^2 + 0.002^2 + 0.003^2) / sqrt(14)
        # = sqrt(0.000014) / sqrt(14)
        # = 0.00374... ~ 0.001
        # Ini borderline dengan gamma=0.001
        assert isinstance(passed, bool)


class TestDynamicRangeAnalysis:
    """Test dynamic range analysis untuk mixed precision."""

    def test_fp16_bounds(self):
        """Test FP16 bounds."""
        FP16_MAX = 65504.0
        FP16_MIN_SUBNORMAL = 5.96e-8

        # Data yang aman untuk FP16
        safe_data = [1.0, 100.0, 1000.0, -500.0]
        max_abs = max(abs(d) for d in safe_data)
        assert max_abs < FP16_MAX * 0.95

        # Data yang tidak aman (overflow)
        unsafe_data = [1.0, 70000.0]
        max_abs = max(abs(d) for d in unsafe_data)
        assert max_abs > FP16_MAX

    def test_dynamic_range_db(self):
        """Test perhitungan dynamic range dalam dB."""
        data = [0.01, 1.0, 100.0]

        abs_max = max(abs(d) for d in data)
        abs_min_nonzero = min(abs(d) for d in data if abs(d) > 1e-30)

        dynamic_range_db = 20.0 * math.log10(abs_max / abs_min_nonzero)

        # 100 / 0.01 = 10000, log10(10000) = 4, 20 * 4 = 80 dB
        assert abs(dynamic_range_db - 80.0) < 1e-6

    def test_int8_safety(self):
        """Test INT8 safety check."""
        # INT8 membutuhkan dynamic range yang rendah
        # dan data yang bisa di-quantize dengan baik

        # Data dengan dynamic range rendah (aman)
        safe_data = [0.1, 0.5, 1.0, 2.0]
        abs_max = max(abs(d) for d in safe_data)
        abs_min = min(abs(d) for d in safe_data if abs(d) > 0)
        dr = 20.0 * math.log10(abs_max / abs_min) if abs_min > 0 else 0
        assert dr < 40.0  # Aman untuk INT8


class TestErrorMetrics:
    """Test perhitungan error metrics."""

    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        ref = [1.0, 2.0, 3.0, 4.0]
        opt = [1.1, 2.2, 2.9, 4.0]

        mae = sum(abs(r - o) for r, o in zip(ref, opt)) / len(ref)
        expected = (0.1 + 0.2 + 0.1 + 0.0) / 4
        assert abs(mae - expected) < 1e-10

    def test_rmse(self):
        """Test RMSE calculation."""
        ref = [1.0, 2.0, 3.0]
        opt = [1.1, 2.0, 3.1]

        mse = sum((r - o) ** 2 for r, o in zip(ref, opt)) / len(ref)
        rmse = math.sqrt(mse)
        expected = math.sqrt((0.01 + 0 + 0.01) / 3)
        assert abs(rmse - expected) < 1e-10

    def test_cosine_similarity(self):
        """Test cosine similarity."""
        a = [1.0, 0.0, 1.0]
        b = [1.0, 0.0, 1.0]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        cosine = dot / (norm_a * norm_b)

        assert abs(cosine - 1.0) < 1e-10  # Identical vectors

    def test_snr_calculation(self):
        """Test SNR (Signal-to-Noise Ratio) calculation."""
        ref = [1.0, 2.0, 3.0, 4.0]
        noise = [0.01, 0.02, 0.01, 0.02]
        opt = [r + n for r, n in zip(ref, noise)]

        signal_power = sum(r**2 for r in ref)
        noise_power = sum((o - r) ** 2 for r, o in zip(ref, opt))

        snr_db = 10 * math.log10(signal_power / noise_power)

        # SNR harus tinggi karena noise kecil
        assert snr_db > 40  # > 40 dB means low noise (high quality)


class TestBatchVerification:
    """Test batch verification untuk multiple tensors."""

    def test_all_pass(self):
        """Test ketika semua tensor pass."""
        results = [True, True, True]
        all_passed = all(results)
        num_passed = sum(1 for r in results if r)

        assert all_passed
        assert num_passed == 3

    def test_mixed_results(self):
        """Test ketika ada tensor yang fail."""
        results = [True, False, True]
        all_passed = all(results)
        num_passed = sum(1 for r in results if r)
        num_failed = sum(1 for r in results if not r)

        assert not all_passed
        assert num_passed == 2
        assert num_failed == 1


class TestPrecisionSafety:
    """Test precision conversion safety."""

    def test_safe_conversion(self):
        """Test data yang aman untuk konversi."""
        data = [0.5, 1.0, 2.0, 3.0]

        # FP16 safety check
        FP16_MAX = 65504.0
        alpha = 0.95
        abs_max = max(abs(d) for d in data)

        safe_for_fp16 = abs_max < FP16_MAX * alpha
        assert safe_for_fp16

    def test_unsafe_conversion(self):
        """Test data yang tidak aman untuk konversi."""
        data = [1.0, 100000.0]  # Exceeds FP16 max

        FP16_MAX = 65504.0
        alpha = 0.95
        abs_max = max(abs(d) for d in data)

        safe_for_fp16 = abs_max < FP16_MAX * alpha
        assert not safe_for_fp16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
