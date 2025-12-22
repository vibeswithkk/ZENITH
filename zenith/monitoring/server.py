# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Metrics Server

FastAPI-based metrics server with:
- /metrics endpoint for Prometheus
- /health endpoint for health checks
- WebSocket for live streaming
"""

import asyncio
import json
import time
from typing import Optional, Set

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import PlainTextResponse, JSONResponse
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from zenith.observability import get_metrics_collector, ZenithLogger


class MetricsServer:
    """
    FastAPI-based metrics server.

    Provides HTTP endpoints for monitoring and WebSocket for live streaming.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize server."""
        if not HAS_FASTAPI:
            raise ImportError(
                "fastapi and uvicorn not installed. "
                "Install with: pip install fastapi uvicorn"
            )

        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Zenith Metrics Server",
            description="Real-time monitoring for Zenith ML framework",
            version="1.0.0",
        )
        self._logger = ZenithLogger.get()
        self._websocket_clients: Set[WebSocket] = set()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Zenith Metrics Server",
                "version": "1.0.0",
                "endpoints": ["/metrics", "/health", "/summary", "/ws"],
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}

        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            """Prometheus metrics endpoint."""
            try:
                from .exporter import get_exporter, check_prometheus_available

                if not check_prometheus_available():
                    return PlainTextResponse(
                        "prometheus_client not installed",
                        status_code=500,
                    )
                exporter = get_exporter()
                return PlainTextResponse(
                    exporter.generate().decode("utf-8"),
                    media_type=exporter.content_type(),
                )
            except Exception as e:
                collector = get_metrics_collector()
                return PlainTextResponse(
                    collector.export_prometheus(),
                    media_type="text/plain",
                )

        @self.app.get("/summary")
        async def summary():
            """Get metrics summary as JSON."""
            collector = get_metrics_collector()
            return JSONResponse(collector.get_summary())

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for live metrics streaming."""
            await websocket.accept()
            self._websocket_clients.add(websocket)
            self._logger.info(
                "WebSocket client connected",
                component="monitoring",
            )

            try:
                while True:
                    collector = get_metrics_collector()
                    data = {
                        "timestamp": time.time(),
                        "metrics": collector.get_summary(),
                    }
                    await websocket.send_json(data)
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self._websocket_clients.discard(websocket)
                self._logger.info(
                    "WebSocket client disconnected",
                    component="monitoring",
                )

    async def broadcast(self, data: dict) -> None:
        """Broadcast data to all WebSocket clients."""
        for client in self._websocket_clients.copy():
            try:
                await client.send_json(data)
            except Exception:
                self._websocket_clients.discard(client)

    def run(self) -> None:
        """Run the server."""
        self._logger.info(
            f"Starting metrics server on {self.host}:{self.port}",
            component="monitoring",
        )
        uvicorn.run(self.app, host=self.host, port=self.port)


def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start metrics server."""
    server = MetricsServer(host=host, port=port)
    server.run()


if __name__ == "__main__":
    start_server()
