from __future__ import annotations
from prometheus_client import Counter, Histogram

REQS = Counter("dmr_requests_total", "Total requests", ["endpoint"])
LAT = Histogram("dmr_latency_ms", "Latency ms", ["endpoint"])


def mark(endpoint: str) -> None:
    REQS.labels(endpoint=endpoint).inc()
