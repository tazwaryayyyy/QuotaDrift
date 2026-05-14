"""
Prometheus metric definitions for QuotaDrift.

Extracted to a dedicated module so that both main.py (API layer) and
model_manager.py (provider health tracking) can import from a single
source of truth without a circular dependency.

All three metrics are registered once at import time via the Prometheus
default registry. Re-importing this module returns the same registered
Counter/Histogram objects — no duplicate-metric panics.
"""

from prometheus_client import Counter, Histogram

MODEL_REQUESTS: Counter = Counter(
    "quotadrift_model_requests_total",
    "Requests routed to each model slot, broken down by outcome status",
    ["model", "status"],
)

MODEL_LATENCY: Histogram = Histogram(
    "quotadrift_model_latency_seconds",
    "End-to-end wall-clock latency per model slot, "
    "measured from request dispatch to final token or error",
    ["model"],
)

TOKEN_USAGE: Counter = Counter(
    "quotadrift_model_tokens_total",
    "Cumulative tokens consumed per model slot",
    ["model"],
)
