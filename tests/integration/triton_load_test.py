"""
Triton Load Testing Module.

Provides tools for load testing Triton deployments with:
- Concurrent request handling
- Latency/throughput metrics
- Error rate tracking
- Configurable test scenarios

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import asyncio
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from zenith.serving.triton_client import (
    InferenceInput,
    InferenceResult,
    MockTritonClient,
    Protocol,
    TritonClient,
)


# ============================================================================
# Load Test Configuration
# ============================================================================


@dataclass
class LoadTestConfig:
    """Configuration for load test."""

    # Connection
    server_url: str = "localhost:8000"
    protocol: Protocol = Protocol.HTTP
    timeout: float = 30.0

    # Model
    model_name: str = "test_model"
    model_version: str = ""

    # Load parameters
    num_requests: int = 100
    concurrent_workers: int = 10
    warmup_requests: int = 10

    # Duration-based mode (if > 0, overrides num_requests)
    duration_seconds: float = 0.0

    # Rate limiting (requests per second, 0 = unlimited)
    target_rps: float = 0.0

    # Input generation
    input_generator: Callable[[], list[InferenceInput]] | None = None


# ============================================================================
# Load Test Metrics
# ============================================================================


@dataclass
class LatencyMetrics:
    """Latency statistics."""

    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass
class LoadTestResult:
    """Complete load test result."""

    # Basic metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # Latency
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    latencies_ms: list[float] = field(default_factory=list)

    # Throughput
    total_duration_sec: float = 0.0
    requests_per_second: float = 0.0
    actual_concurrent: int = 0

    # Errors
    errors: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 60,
            "Load Test Results",
            "=" * 60,
            "",
            f"Total Requests:    {self.total_requests}",
            f"Successful:        {self.successful_requests}",
            f"Failed:            {self.failed_requests}",
            f"Error Rate:        {self.error_rate * 100:.2f}%",
            "",
            "Latency:",
            f"  Mean:   {self.latency.mean_ms:.2f} ms",
            f"  Std:    {self.latency.std_ms:.2f} ms",
            f"  Min:    {self.latency.min_ms:.2f} ms",
            f"  Max:    {self.latency.max_ms:.2f} ms",
            f"  P50:    {self.latency.p50_ms:.2f} ms",
            f"  P90:    {self.latency.p90_ms:.2f} ms",
            f"  P95:    {self.latency.p95_ms:.2f} ms",
            f"  P99:    {self.latency.p99_ms:.2f} ms",
            "",
            f"Throughput:        {self.requests_per_second:.2f} req/s",
            f"Duration:          {self.total_duration_sec:.2f} s",
            f"Concurrent:        {self.actual_concurrent}",
            "=" * 60,
        ]

        if self.errors:
            lines.append("")
            lines.append("Error Summary:")
            for error, count in sorted(
                self.errors.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {error}: {count}")

        return "\n".join(lines)


# ============================================================================
# Load Tester
# ============================================================================


class TritonLoadTester:
    """
    Load tester for Triton Inference Server.

    Supports concurrent request generation and metrics collection.

    Example:
        tester = TritonLoadTester(config)
        result = tester.run()
        print(result.summary())
    """

    def __init__(self, config: LoadTestConfig):
        """Initialize load tester."""
        self.config = config
        self._client: TritonClient | None = None
        self._results: list[InferenceResult] = []
        self._stop_flag = False

    def _get_client(self) -> TritonClient:
        """Get or create client."""
        if self._client is None:
            self._client = TritonClient(
                url=self.config.server_url,
                protocol=self.config.protocol,
                timeout=self.config.timeout,
            )
        return self._client

    def set_mock_client(self, client: MockTritonClient):
        """Set mock client for testing."""
        self._client = client

    def _generate_input(self) -> list[InferenceInput]:
        """Generate input for request."""
        if self.config.input_generator:
            return self.config.input_generator()

        # Default: random input
        data = np.random.randn(1, 224, 224, 3).astype(np.float32)
        return [InferenceInput(name="input", data=data)]

    def _single_request(self) -> InferenceResult:
        """Execute single inference request."""
        client = self._get_client()
        inputs = self._generate_input()
        return client.infer(
            model_name=self.config.model_name,
            inputs=inputs,
            model_version=self.config.model_version,
        )

    def warmup(self) -> None:
        """Run warmup requests."""
        if self.config.warmup_requests <= 0:
            return

        for _ in range(self.config.warmup_requests):
            try:
                self._single_request()
            except Exception:
                pass

    def run(self) -> LoadTestResult:
        """
        Run load test.

        Returns:
            LoadTestResult with metrics
        """
        self._results = []
        self._stop_flag = False

        # Warmup
        self.warmup()

        # Determine number of requests
        if self.config.duration_seconds > 0:
            return self._run_duration_based()
        else:
            return self._run_count_based()

    def _run_count_based(self) -> LoadTestResult:
        """Run load test with fixed request count."""
        results: list[InferenceResult] = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
            futures = [
                executor.submit(self._single_request)
                for _ in range(self.config.num_requests)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create failed result
                    results.append(
                        InferenceResult(
                            model_name=self.config.model_name,
                            model_version=self.config.model_version,
                            success=False,
                            error=str(e),
                        )
                    )

        total_duration = time.perf_counter() - start_time
        return self._compute_metrics(results, total_duration)

    def _run_duration_based(self) -> LoadTestResult:
        """Run load test for specified duration."""
        results: list[InferenceResult] = []
        start_time = time.perf_counter()
        end_time = start_time + self.config.duration_seconds

        def worker():
            while time.perf_counter() < end_time and not self._stop_flag:
                try:
                    result = self._single_request()
                    results.append(result)
                except Exception as e:
                    results.append(
                        InferenceResult(
                            model_name=self.config.model_name,
                            model_version=self.config.model_version,
                            success=False,
                            error=str(e),
                        )
                    )

                # Rate limiting
                if self.config.target_rps > 0:
                    time.sleep(1.0 / self.config.target_rps)

        with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
            futures = [
                executor.submit(worker) for _ in range(self.config.concurrent_workers)
            ]
            for future in as_completed(futures):
                pass  # Wait for all workers

        total_duration = time.perf_counter() - start_time
        return self._compute_metrics(results, total_duration)

    def stop(self):
        """Signal stop for duration-based test."""
        self._stop_flag = True

    def _compute_metrics(
        self, results: list[InferenceResult], total_duration: float
    ) -> LoadTestResult:
        """Compute metrics from results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        # Collect latencies from successful requests
        latencies = [r.latency_ms for r in results if r.success and r.latency_ms > 0]

        # Collect errors
        errors: dict[str, int] = {}
        for r in results:
            if not r.success and r.error:
                error_key = r.error[:50]  # Truncate long errors
                errors[error_key] = errors.get(error_key, 0) + 1

        # Compute latency metrics
        latency_metrics = LatencyMetrics()
        if latencies:
            latency_metrics.mean_ms = statistics.mean(latencies)
            latency_metrics.std_ms = (
                statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            )
            latency_metrics.min_ms = min(latencies)
            latency_metrics.max_ms = max(latencies)
            latency_metrics.p50_ms = np.percentile(latencies, 50)
            latency_metrics.p90_ms = np.percentile(latencies, 90)
            latency_metrics.p95_ms = np.percentile(latencies, 95)
            latency_metrics.p99_ms = np.percentile(latencies, 99)

        return LoadTestResult(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            error_rate=failed / total if total > 0 else 0.0,
            latency=latency_metrics,
            latencies_ms=latencies,
            total_duration_sec=total_duration,
            requests_per_second=total / total_duration if total_duration > 0 else 0.0,
            actual_concurrent=self.config.concurrent_workers,
            errors=errors,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def run_load_test(
    server_url: str = "localhost:8000",
    model_name: str = "test_model",
    num_requests: int = 100,
    concurrent_workers: int = 10,
    protocol: Protocol = Protocol.HTTP,
    input_generator: Callable[[], list[InferenceInput]] | None = None,
    verbose: bool = True,
) -> LoadTestResult:
    """
    Run a simple load test.

    Args:
        server_url: Triton server URL
        model_name: Model to test
        num_requests: Total number of requests
        concurrent_workers: Number of concurrent workers
        protocol: HTTP or gRPC
        input_generator: Custom input generator function
        verbose: Print results

    Returns:
        LoadTestResult
    """
    config = LoadTestConfig(
        server_url=server_url,
        model_name=model_name,
        num_requests=num_requests,
        concurrent_workers=concurrent_workers,
        protocol=protocol,
        input_generator=input_generator,
    )

    tester = TritonLoadTester(config)
    result = tester.run()

    if verbose:
        print(result.summary())

    return result


def run_mock_load_test(
    model_name: str = "mock_model",
    num_requests: int = 100,
    concurrent_workers: int = 10,
    inference_handler: Callable[[list[InferenceInput]], dict[str, np.ndarray]]
    | None = None,
    verbose: bool = True,
) -> LoadTestResult:
    """
    Run load test with mock client for development/testing.

    Args:
        model_name: Model name
        num_requests: Total requests
        concurrent_workers: Concurrent workers
        inference_handler: Custom handler for mock inference
        verbose: Print results

    Returns:
        LoadTestResult
    """
    config = LoadTestConfig(
        model_name=model_name,
        num_requests=num_requests,
        concurrent_workers=concurrent_workers,
    )

    # Setup mock client
    mock_client = MockTritonClient()
    mock_client.register_model(model_name, handler=inference_handler)

    tester = TritonLoadTester(config)
    tester.set_mock_client(mock_client)

    result = tester.run()

    if verbose:
        print(result.summary())

    return result


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point for load testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Triton Load Testing Tool")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--model", default="test_model", help="Model name")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent workers")
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Duration in seconds (overrides --requests)",
    )
    parser.add_argument(
        "--protocol",
        choices=["http", "grpc"],
        default="http",
        help="Protocol to use",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock client")

    args = parser.parse_args()

    if args.mock:
        result = run_mock_load_test(
            model_name=args.model,
            num_requests=args.requests,
            concurrent_workers=args.concurrent,
        )
    else:
        result = run_load_test(
            server_url=args.url,
            model_name=args.model,
            num_requests=args.requests,
            concurrent_workers=args.concurrent,
            protocol=Protocol.HTTP if args.protocol == "http" else Protocol.GRPC,
        )

    return 0 if result.error_rate < 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
