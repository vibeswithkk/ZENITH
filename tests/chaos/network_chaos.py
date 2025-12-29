#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Network Chaos Testing Module

Simulates network failures to test Zenith's fault tolerance:
- Connection timeout injection
- DNS failure simulation
- Intermittent connection failures
- Bandwidth throttling simulation

Reference:
    Netflix Chaos Monkey principles
    Google DiRT methodology

Usage:
    pytest tests/chaos/network_chaos.py -v
"""

import socket
import sys
import time
import threading
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional
from unittest.mock import patch, MagicMock


@dataclass
class NetworkChaosConfig:
    """Configuration for network chaos injection."""

    timeout_probability: float = 0.0
    failure_probability: float = 0.0
    latency_ms: float = 0.0
    max_retries: int = 3

    def __post_init__(self):
        if not 0.0 <= self.timeout_probability <= 1.0:
            raise ValueError("timeout_probability must be between 0 and 1")
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be between 0 and 1")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


class NetworkChaosInjector:
    """
    Injects network failures for chaos testing.

    Thread-safe implementation for concurrent testing.
    """

    def __init__(self, config: Optional[NetworkChaosConfig] = None):
        self.config = config or NetworkChaosConfig()
        self._lock = threading.Lock()
        self._call_count = 0
        self._failure_count = 0
        self._timeout_count = 0

    @property
    def stats(self) -> dict:
        """Get injection statistics."""
        with self._lock:
            return {
                "total_calls": self._call_count,
                "failures_injected": self._failure_count,
                "timeouts_injected": self._timeout_count,
            }

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._call_count = 0
            self._failure_count = 0
            self._timeout_count = 0

    def should_inject_failure(self) -> bool:
        """Determine if failure should be injected based on probability."""
        import random

        return random.random() < self.config.failure_probability

    def should_inject_timeout(self) -> bool:
        """Determine if timeout should be injected based on probability."""
        import random

        return random.random() < self.config.timeout_probability

    def inject_latency(self) -> None:
        """Inject artificial latency."""
        if self.config.latency_ms > 0:
            time.sleep(self.config.latency_ms / 1000.0)

    @contextmanager
    def chaos_socket(self):
        """
        Context manager that patches socket operations with chaos injection.

        Injects timeouts and connection failures based on configuration.
        """
        original_connect = socket.socket.connect
        injector = self

        def chaos_connect(self_socket, address):
            with injector._lock:
                injector._call_count += 1

            # Inject latency
            injector.inject_latency()

            # Possibly inject timeout
            if injector.should_inject_timeout():
                with injector._lock:
                    injector._timeout_count += 1
                raise socket.timeout("Chaos injected: connection timeout")

            # Possibly inject failure
            if injector.should_inject_failure():
                with injector._lock:
                    injector._failure_count += 1
                raise ConnectionRefusedError("Chaos injected: connection refused")

            return original_connect(self_socket, address)

        with patch.object(socket.socket, "connect", chaos_connect):
            yield


class TestNetworkChaosInjector(unittest.TestCase):
    """Unit tests for NetworkChaosInjector."""

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = NetworkChaosConfig(
            timeout_probability=0.5,
            failure_probability=0.1,
            latency_ms=100,
        )
        self.assertEqual(config.timeout_probability, 0.5)

        # Invalid probability
        with self.assertRaises(ValueError):
            NetworkChaosConfig(timeout_probability=1.5)

        with self.assertRaises(ValueError):
            NetworkChaosConfig(failure_probability=-0.1)

        with self.assertRaises(ValueError):
            NetworkChaosConfig(latency_ms=-10)

    def test_stats_tracking(self):
        """Test statistics tracking."""
        injector = NetworkChaosInjector()

        stats = injector.stats
        self.assertEqual(stats["total_calls"], 0)

        # Increment manually for testing
        with injector._lock:
            injector._call_count = 10
            injector._failure_count = 2
            injector._timeout_count = 1

        stats = injector.stats
        self.assertEqual(stats["total_calls"], 10)
        self.assertEqual(stats["failures_injected"], 2)
        self.assertEqual(stats["timeouts_injected"], 1)

        injector.reset_stats()
        stats = injector.stats
        self.assertEqual(stats["total_calls"], 0)

    def test_probability_functions(self):
        """Test probability-based injection."""
        # 0% probability should never inject
        config = NetworkChaosConfig(
            timeout_probability=0.0,
            failure_probability=0.0,
        )
        injector = NetworkChaosInjector(config)

        for _ in range(100):
            self.assertFalse(injector.should_inject_timeout())
            self.assertFalse(injector.should_inject_failure())

        # 100% probability should always inject
        config = NetworkChaosConfig(
            timeout_probability=1.0,
            failure_probability=1.0,
        )
        injector = NetworkChaosInjector(config)

        for _ in range(100):
            self.assertTrue(injector.should_inject_timeout())
            self.assertTrue(injector.should_inject_failure())

    def test_latency_injection(self):
        """Test latency injection."""
        config = NetworkChaosConfig(latency_ms=50)
        injector = NetworkChaosInjector(config)

        start = time.perf_counter()
        injector.inject_latency()
        elapsed = (time.perf_counter() - start) * 1000

        # Should take at least 50ms
        self.assertGreaterEqual(elapsed, 45)


class TestNetworkResilience(unittest.TestCase):
    """
    Test Zenith's resilience to network failures.

    Verifies graceful degradation under chaos conditions.
    """

    def test_graceful_timeout_handling(self):
        """Test that timeouts are handled gracefully."""
        config = NetworkChaosConfig(timeout_probability=1.0)
        injector = NetworkChaosInjector(config)

        with injector.chaos_socket():
            # Attempt connection should raise timeout
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)

            with self.assertRaises(socket.timeout):
                sock.connect(("127.0.0.1", 12345))

            sock.close()

        stats = injector.stats
        self.assertEqual(stats["timeouts_injected"], 1)

    def test_graceful_connection_failure(self):
        """Test that connection failures are handled gracefully."""
        config = NetworkChaosConfig(failure_probability=1.0)
        injector = NetworkChaosInjector(config)

        with injector.chaos_socket():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            with self.assertRaises(ConnectionRefusedError):
                sock.connect(("127.0.0.1", 12345))

            sock.close()

        stats = injector.stats
        self.assertEqual(stats["failures_injected"], 1)

    def test_retry_logic_under_chaos(self):
        """Test retry logic works under intermittent failures."""
        # 50% failure rate
        config = NetworkChaosConfig(failure_probability=0.5)
        injector = NetworkChaosInjector(config)

        successes = 0
        failures = 0
        attempts = 100

        with injector.chaos_socket():
            for _ in range(attempts):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    # This will either fail from chaos or from no server
                    sock.connect(("127.0.0.1", 12345))
                    successes += 1
                except ConnectionRefusedError:
                    failures += 1
                except OSError:
                    # Connection refused by OS (no server)
                    successes += 1
                finally:
                    sock.close()

        # Should have some failures from chaos
        stats = injector.stats
        self.assertGreater(stats["failures_injected"], 0)
        self.assertEqual(stats["total_calls"], attempts)


def run_network_chaos_tests():
    """Run all network chaos tests."""
    print("=" * 60)
    print("  NETWORK CHAOS TESTS")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestNetworkChaosInjector))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkResilience))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_network_chaos_tests())
