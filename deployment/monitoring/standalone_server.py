#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Standalone Zenith Metrics Server for Docker deployment

import asyncio
import time
import random
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
import numpy as np

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)


# Metrics Registry
registry = CollectorRegistry()

inference_total = Counter(
    "zenith_inference_total",
    "Total inferences",
    ["model", "precision"],
    registry=registry,
)

inference_errors = Counter(
    "zenith_inference_errors_total",
    "Total errors",
    ["model", "error_type"],
    registry=registry,
)

inference_latency = Histogram(
    "zenith_inference_latency_seconds",
    "Inference latency",
    ["model", "precision"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

memory_usage = Gauge(
    "zenith_memory_mb",
    "Memory usage MB",
    ["device"],
    registry=registry,
)

active_models = Gauge(
    "zenith_active_models",
    "Active compiled models",
    registry=registry,
)


# Metrics Storage
class MetricsStore:
    def __init__(self):
        self.latencies = []
        self.total_inferences = 0
        self.total_errors = 0

    def record(self, latency_ms: float, model: str = "bert", precision: str = "fp16"):
        self.latencies.append(latency_ms)
        self.total_inferences += 1
        inference_total.labels(model=model, precision=precision).inc()
        inference_latency.labels(model=model, precision=precision).observe(
            latency_ms / 1000
        )

    def get_summary(self):
        if not self.latencies:
            return {"total_inferences": 0, "total_errors": 0}
        arr = np.array(self.latencies[-1000:])
        return {
            "total_inferences": self.total_inferences,
            "total_errors": self.total_errors,
            "latency_mean_ms": float(np.mean(arr)),
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p90_ms": float(np.percentile(arr, 90)),
            "latency_p99_ms": float(np.percentile(arr, 99)),
        }


store = MetricsStore()

# FastAPI App
app = FastAPI(
    title="Zenith Metrics Server",
    description="Real-time monitoring for Zenith ML",
    version="1.0.0",
)

websocket_clients: Set[WebSocket] = set()


@app.get("/")
async def root():
    return {
        "name": "Zenith Metrics Server",
        "version": "1.0.0",
        "endpoints": ["/metrics", "/health", "/summary", "/ws"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(
        generate_latest(registry).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/summary")
async def summary():
    return JSONResponse(store.get_summary())


@app.post("/record")
async def record_inference(latency_ms: float = 10.0, model: str = "bert"):
    store.record(latency_ms, model)
    return {"status": "recorded"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    try:
        while True:
            data = {
                "timestamp": time.time(),
                "metrics": store.get_summary(),
            }
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)


# Demo data generator
async def generate_demo_data():
    models = ["bert", "resnet", "gpt2"]
    while True:
        model = random.choice(models)
        latency = random.uniform(5, 50)
        store.record(latency, model)
        memory_usage.labels(device="cuda:0").set(random.uniform(500, 2000))
        active_models.set(len(models))
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(generate_demo_data())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
