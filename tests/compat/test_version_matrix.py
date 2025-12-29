# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Version Matrix Compatibility Tests

Tests Zenith compatibility across different Python and dependency versions.
"""

import sys
import unittest
import warnings


class TestPythonVersionCompatibility(unittest.TestCase):
    """Test Python version compatibility."""

    def test_python_minimum_version(self):
        """Ensure Python >= 3.9."""
        self.assertGreaterEqual(sys.version_info[:2], (3, 9))

    def test_python_maximum_tested_version(self):
        """Check if running on tested Python version."""
        tested_versions = [(3, 9), (3, 10), (3, 11), (3, 12)]
        current = sys.version_info[:2]

        if current not in tested_versions:
            warnings.warn(
                f"Python {current[0]}.{current[1]} has not been fully tested",
                UserWarning,
            )


class TestVersionInfo(unittest.TestCase):
    """Test VersionInfo class."""

    def test_parse_version(self):
        """Test version string parsing."""
        from zenith.compat import VersionInfo

        v = VersionInfo.parse("0.3.0")
        self.assertEqual(v.major, 0)
        self.assertEqual(v.minor, 3)
        self.assertEqual(v.patch, 0)

    def test_version_comparison(self):
        """Test version comparison operators."""
        from zenith.compat import VersionInfo

        v1 = VersionInfo(0, 2, 0)
        v2 = VersionInfo(0, 3, 0)
        v3 = VersionInfo(0, 3, 0)
        v4 = VersionInfo(1, 0, 0)

        self.assertTrue(v1 < v2)
        self.assertTrue(v2 <= v3)
        self.assertTrue(v2 == v3)
        self.assertTrue(v4 > v2)
        self.assertFalse(v1 > v2)

    def test_get_current_version(self):
        """Test getting current version."""
        from zenith.compat import get_current_version

        v = get_current_version()
        self.assertIsNotNone(v)
        self.assertGreaterEqual(v.major, 0)


class TestDeprecationWarnings(unittest.TestCase):
    """Test deprecation warning system."""

    def test_deprecated_decorator(self):
        """Test @deprecated decorator emits warning."""
        from zenith.compat import deprecated, ZenithDeprecationWarning

        @deprecated(since="0.3.0", removal="0.5.0", alternative="new_func()")
        def old_func():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

            self.assertEqual(result, "result")
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, ZenithDeprecationWarning))
            self.assertIn("0.3.0", str(w[0].message))
            self.assertIn("0.5.0", str(w[0].message))

    def test_warn_deprecated(self):
        """Test warn_deprecated function."""
        from zenith.compat import warn_deprecated, ZenithDeprecationWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated(
                name="old_feature",
                since="0.2.0",
                removal="0.4.0",
            )

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, ZenithDeprecationWarning))


class TestVersionCompatibility(unittest.TestCase):
    """Test version compatibility checks."""

    def test_check_version_compatibility(self):
        """Test version range checking."""
        from zenith.compat import check_version_compatibility

        # Should always pass with no constraints
        self.assertTrue(check_version_compatibility())

    def test_deprecation_registry(self):
        """Test deprecation registry."""
        from zenith.compat import (
            register_deprecation,
            get_deprecation_list,
            _DEPRECATION_REGISTRY,
        )

        # Clear registry
        _DEPRECATION_REGISTRY.clear()

        register_deprecation(
            name="test_feature",
            since="0.3.0",
            removal="0.5.0",
            reason="Replaced with better API",
        )

        deps = get_deprecation_list()
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0]["name"], "test_feature")


class TestPyTorchCompatibility(unittest.TestCase):
    """Test PyTorch version compatibility."""

    def test_pytorch_import(self):
        """Test PyTorch can be imported."""
        try:
            import torch

            self.assertIsNotNone(torch.__version__)
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_pytorch_version_minimum(self):
        """Check PyTorch version >= 1.13."""
        try:
            import torch

            version_parts = torch.__version__.split(".")
            major = int(version_parts[0])
            minor = int(version_parts[1].split("+")[0])

            if major < 1 or (major == 1 and minor < 13):
                self.fail(f"PyTorch {torch.__version__} too old, need >= 1.13")
        except ImportError:
            self.skipTest("PyTorch not installed")


class TestJAXCompatibility(unittest.TestCase):
    """Test JAX version compatibility."""

    def test_jax_import(self):
        """Test JAX can be imported."""
        try:
            import jax

            self.assertIsNotNone(jax.__version__)
        except ImportError:
            self.skipTest("JAX not installed")


if __name__ == "__main__":
    unittest.main()
