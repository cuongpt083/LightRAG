"""Prometheus metrics definitions and stage instrumentation for LightRAG queries."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None

if PROMETHEUS_AVAILABLE:
    QUERY_DURATION = Histogram(
        "lightrag_query_duration_seconds",
        "Total LightRAG query request latency in seconds",
        ["mode", "status"],
        buckets=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
    )

    QUERY_TOTAL = Counter(
        "lightrag_query_requests_total",
        "Total LightRAG query requests count",
        ["mode", "status"],
    )

    QUERY_STAGE_DURATION = Histogram(
        "lightrag_query_stage_duration_seconds",
        "Duration of individual query processing stages in seconds",
        ["mode", "stage"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0],
    )

    LLM_CALL_DURATION = Histogram(
        "lightrag_llm_call_duration_seconds",
        "Duration of LLM calls made by LightRAG in seconds",
        ["role"],
        buckets=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    )
else:
    QUERY_DURATION = None
    QUERY_TOTAL = None
    QUERY_STAGE_DURATION = None
    LLM_CALL_DURATION = None


def record_query_duration(mode: str, status: str, duration: float) -> None:
    if not PROMETHEUS_AVAILABLE or QUERY_DURATION is None:
        return
    try:
        QUERY_DURATION.labels(mode=str(mode or "default"), status=str(status or "unknown")).observe(
            max(0.0, float(duration))
        )
    except Exception as exc:
        logger.debug(f"Failed to record query duration: {exc}")


def record_query_total(mode: str, status: str) -> None:
    if not PROMETHEUS_AVAILABLE or QUERY_TOTAL is None:
        return
    try:
        QUERY_TOTAL.labels(mode=str(mode or "default"), status=str(status or "unknown")).inc()
    except Exception as exc:
        logger.debug(f"Failed to record query total: {exc}")


def record_llm_call(role: str, duration: float) -> None:
    if not PROMETHEUS_AVAILABLE or LLM_CALL_DURATION is None:
        return
    try:
        LLM_CALL_DURATION.labels(role=str(role or "default")).observe(max(0.0, float(duration)))
    except Exception as exc:
        logger.debug(f"Failed to record llm call metric: {exc}")


@contextmanager
def record_query_stage(mode: str, stage: str):
    """Safely record the execution duration of a query processing stage."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if PROMETHEUS_AVAILABLE and QUERY_STAGE_DURATION is not None:
            duration = time.perf_counter() - start
            try:
                QUERY_STAGE_DURATION.labels(
                    mode=str(mode or "default"),
                    stage=str(stage or "unknown"),
                ).observe(max(0.0, duration))
            except Exception as exc:
                logger.debug(f"Failed to observe query stage '{stage}': {exc}")
