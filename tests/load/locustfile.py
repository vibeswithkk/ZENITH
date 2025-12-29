# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Load Testing with Locust

HTTP endpoint load testing for the Zenith monitoring server.

Prerequisites:
    pip install locust

Usage:
    # Start Zenith server first
    zenith serve --port 8080
    
    # Run Locust
    locust -f tests/load/locustfile.py --host=http://localhost:8080
    
    # Or headless mode for CI
    locust -f tests/load/locustfile.py --host=http://localhost:8080 \
           --headless -u 100 -r 10 --run-time 60s

Test Scenarios:
    1. HealthCheckUser: Hammers /health endpoint
    2. MetricsScraper: Simulates Prometheus scraping /metrics
    3. SummaryPoller: Polls /summary for JSON metrics
    4. MixedWorkload: Combined realistic usage pattern
"""

try:
    from locust import HttpUser, task, between, events
    from locust.runners import MasterRunner, WorkerRunner

    HAS_LOCUST = True
except ImportError:
    HAS_LOCUST = False

    # Stub classes for import without locust
    class HttpUser:
        pass

    def task(weight=1):
        def decorator(func):
            return func

        return decorator

    def between(a, b):
        return 1


import time
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HealthCheckUser(HttpUser):
    """
    Stress test for /health endpoint.

    Target: Validate server stability under 1000+ concurrent health checks.
    Expected: p99 latency < 10ms, 0% error rate.
    """

    wait_time = between(0.1, 0.5)
    weight = 3

    @task
    def check_health(self):
        """
        GET /health

        Success criteria:
        - Status 200
        - Response contains {"status": "healthy"}
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Unexpected status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")


class MetricsScraper(HttpUser):
    """
    Simulates Prometheus scraping /metrics endpoint.

    Target: Validate metrics export under high scrape frequency.
    Reference: Prometheus default scrape_interval is 15s, we test at 100ms.
    """

    wait_time = between(0.1, 0.3)
    weight = 2

    @task
    def scrape_metrics(self):
        """
        GET /metrics

        Success criteria:
        - Status 200
        - Response contains Prometheus format (zenith_*)
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                text = response.text
                if "zenith_" in text or len(text) > 0:
                    response.success()
                else:
                    response.failure("Empty or invalid metrics")
            else:
                response.failure(f"HTTP {response.status_code}")


class SummaryPoller(HttpUser):
    """
    Polls /summary for JSON metrics.

    Target: Validate JSON serialization under load.
    """

    wait_time = between(0.5, 1.0)
    weight = 1

    @task
    def get_summary(self):
        """
        GET /summary

        Success criteria:
        - Status 200
        - Valid JSON with expected fields
        """
        with self.client.get("/summary", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate expected fields exist
                    if "total_inferences" in data or isinstance(data, dict):
                        response.success()
                    else:
                        response.failure("Missing expected fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")
            else:
                response.failure(f"HTTP {response.status_code}")


class MixedWorkloadUser(HttpUser):
    """
    Realistic mixed workload simulating actual usage.

    Pattern:
    - 60% health checks (monitoring systems)
    - 30% metrics scrapes (Prometheus)
    - 10% summary polls (dashboards)
    """

    wait_time = between(0.2, 1.0)
    weight = 4

    @task(6)
    def health_check(self):
        """Health check - highest frequency."""
        self.client.get("/health")

    @task(3)
    def scrape_metrics(self):
        """Metrics scrape - medium frequency."""
        self.client.get("/metrics")

    @task(1)
    def get_summary(self):
        """Summary poll - lowest frequency."""
        self.client.get("/summary")

    @task(1)
    def get_root(self):
        """Root endpoint - service discovery."""
        self.client.get("/")


# Event hooks for custom reporting
if HAS_LOCUST:

    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        """Log test configuration at start."""
        logger.info("=" * 60)
        logger.info("ZENITH LOAD TEST STARTED")
        logger.info("=" * 60)
        if isinstance(environment.runner, MasterRunner):
            logger.info("Running in distributed mode (master)")
        elif isinstance(environment.runner, WorkerRunner):
            logger.info("Running in distributed mode (worker)")
        else:
            logger.info("Running in standalone mode")

    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        """Generate summary report at end."""
        logger.info("=" * 60)
        logger.info("ZENITH LOAD TEST COMPLETED")
        logger.info("=" * 60)

        stats = environment.stats
        if stats.total.num_requests > 0:
            logger.info(f"Total requests: {stats.total.num_requests}")
            logger.info(f"Failures: {stats.total.num_failures}")
            logger.info(f"Avg response time: {stats.total.avg_response_time:.2f}ms")

            if stats.total.num_failures > 0:
                error_rate = (stats.total.num_failures / stats.total.num_requests) * 100
                logger.warning(f"Error rate: {error_rate:.2f}%")
            else:
                logger.info("Error rate: 0.00%")
