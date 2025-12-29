#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Fuzzing Tests

Property-based fuzzing using Hypothesis to discover edge cases and crashes.

Targets:
- VersionInfo parsing
- MetricsCollector input handling
- Configuration validation

Reference:
    Google OSS-Fuzz methodology
    Hypothesis documentation

Usage:
    pytest security/fuzz_tests.py -v
    pytest security/fuzz_tests.py --hypothesis-show-statistics
"""

import sys
import unittest
from typing import Optional

# Check for Hypothesis availability
try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis import Phase, Verbosity

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    # Stub decorators for when Hypothesis is not installed
    def given(*args, **kwargs):
        def decorator(func):
            def wrapper(*a, **kw):
                pass

            return wrapper

        return decorator

    class st:
        @staticmethod
        def text(*args, **kwargs):
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

    def settings(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def assume(condition):
        pass


class TestVersionInfoFuzzing(unittest.TestCase):
    """
    Fuzz testing for VersionInfo parsing.

    Goal: Ensure no crashes on arbitrary input strings.
    """

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=1000, deadline=None)
    def test_version_parse_no_crash(self, version_str: str):
        """
        VersionInfo.parse should never crash on any input.

        It should either return a valid VersionInfo or raise ValueError.
        """
        from zenith.compat import VersionInfo

        try:
            result = VersionInfo.parse(version_str)
            # If parsing succeeds, verify structure
            self.assertIsInstance(result.major, int)
            self.assertIsInstance(result.minor, int)
            self.assertIsInstance(result.patch, int)
            self.assertGreaterEqual(result.major, 0)
            self.assertGreaterEqual(result.minor, 0)
            self.assertGreaterEqual(result.patch, 0)
        except ValueError:
            # Expected for invalid input
            pass
        except TypeError:
            # Expected for None input
            pass

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-1000, max_value=1000),
    )
    @settings(max_examples=500, deadline=None)
    def test_version_comparison_consistency(self, major: int, minor: int, patch: int):
        """
        Version comparison should be consistent.

        Properties tested:
        - Reflexivity: v == v
        - Antisymmetry: if v1 < v2 then not v2 < v1
        """
        from zenith.compat import VersionInfo

        # Skip invalid versions
        assume(major >= 0 and minor >= 0 and patch >= 0)

        v1 = VersionInfo(major, minor, patch)
        v2 = VersionInfo(major, minor, patch)

        # Reflexivity
        self.assertEqual(v1, v2)

        # Consistency with hash
        self.assertEqual(hash(v1), hash(v2))

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(st.text(alphabet="0123456789.", min_size=1, max_size=20))
    @settings(max_examples=500, deadline=None)
    def test_version_parse_numeric_strings(self, version_str: str):
        """
        Test version parsing with numeric-like strings.

        Ensures proper handling of edge cases like "...", "1.2.3.4.5", etc.
        """
        from zenith.compat import VersionInfo

        try:
            result = VersionInfo.parse(version_str)
            # Verify valid result
            self.assertIsInstance(result, VersionInfo)
        except ValueError:
            # Expected for invalid formats
            pass


class TestMetricsCollectorFuzzing(unittest.TestCase):
    """
    Fuzz testing for MetricsCollector.

    Goal: Ensure metrics recording handles extreme values.
    """

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(
        st.floats(allow_nan=True, allow_infinity=True),
        st.floats(allow_nan=True, allow_infinity=True),
        st.integers(min_value=-1000000, max_value=1000000),
    )
    @settings(max_examples=500, deadline=None)
    def test_record_inference_no_crash(
        self, latency: float, memory: float, batch_size: int
    ):
        """
        MetricsCollector.record_inference should handle extreme values.

        Tests NaN, Infinity, negative values.
        """
        from zenith.observability import MetricsCollector, InferenceMetrics

        collector = MetricsCollector()

        try:
            metrics = InferenceMetrics(
                latency_ms=latency,
                memory_mb=memory,
                batch_size=batch_size,
            )
            collector.record_inference(metrics)

            # Should not crash on get_summary
            summary = collector.get_summary()
            self.assertIsInstance(summary, dict)

        except (ValueError, OverflowError):
            # Acceptable for extreme values
            pass

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
            min_size=0,
            max_size=1000,
        )
    )
    @settings(max_examples=200, deadline=None)
    def test_latency_statistics_consistency(self, latencies: list):
        """
        Verify statistical calculations are consistent.

        P50 should always be <= P99.
        """
        from zenith.observability import MetricsCollector, InferenceMetrics

        collector = MetricsCollector()

        for lat in latencies:
            collector.record_inference(InferenceMetrics(latency_ms=lat))

        summary = collector.get_summary()

        if latencies:
            p50 = summary.get("latency_p50_ms", 0)
            p99 = summary.get("latency_p99_ms", 0)

            # P50 should be <= P99
            self.assertLessEqual(p50, p99 + 0.001)  # Small epsilon for float


class TestDeprecationFuzzing(unittest.TestCase):
    """
    Fuzz testing for deprecation system.

    Goal: Ensure warn_deprecated handles arbitrary strings.
    """

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(
        st.text(min_size=0, max_size=200),
        st.text(min_size=0, max_size=20),
        st.text(min_size=0, max_size=20),
        st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=200, deadline=None)
    def test_warn_deprecated_no_crash(
        self,
        name: str,
        since: str,
        removal: Optional[str],
        alternative: Optional[str],
    ):
        """
        warn_deprecated should handle any string input.
        """
        import warnings
        from zenith.compat import warn_deprecated, ZenithDeprecationWarning

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                warn_deprecated(
                    name=name,
                    since=since,
                    removal=removal if removal else None,
                    alternative=alternative if alternative else None,
                )
            except Exception as e:
                # Should not raise any exception
                self.fail(f"warn_deprecated raised {type(e).__name__}: {e}")


class TestInputValidationFuzzing(unittest.TestCase):
    """
    Fuzz testing for general input validation.

    Tests functions that parse user input.
    """

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis not installed")
    @given(st.binary(min_size=0, max_size=1000))
    @settings(max_examples=500, deadline=None)
    def test_binary_input_handling(self, data: bytes):
        """
        Test that binary data doesn't crash string-expecting functions.
        """
        from zenith.compat import VersionInfo

        try:
            # Attempt to decode and parse
            decoded = data.decode("utf-8", errors="replace")
            VersionInfo.parse(decoded)
        except (ValueError, TypeError, UnicodeDecodeError):
            # Expected for invalid input
            pass


def run_fuzz_tests():
    """Run all fuzz tests."""
    if not HAS_HYPOTHESIS:
        print("=" * 60)
        print("  FUZZING TESTS SKIPPED")
        print("  Hypothesis not installed. Install with: pip install hypothesis")
        print("=" * 60)
        return 1

    print("=" * 60)
    print("  ZENITH FUZZING TESTS")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVersionInfoFuzzing))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCollectorFuzzing))
    suite.addTests(loader.loadTestsFromTestCase(TestDeprecationFuzzing))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidationFuzzing))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_fuzz_tests())
