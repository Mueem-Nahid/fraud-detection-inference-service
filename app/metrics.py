# ============================================================================
# ModelServe — Prometheus Metrics
# ============================================================================
from prometheus_client import Counter, Histogram, Gauge

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received"
)

prediction_duration_seconds = Histogram(
    "prediction_duration_seconds",
    "Time taken to process each prediction (feature fetch + model inference)",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Number of failed prediction requests"
)

model_version_info = Gauge(
    "model_version_info",
    "Currently served model version",
    ["version"]
)

feast_online_store_hits_total = Counter(
    "feast_online_store_hits_total",
    "Successful feature lookups from Feast"
)

feast_online_store_misses_total = Counter(
    "feast_online_store_misses_total",
    "Failed or empty feature lookups from Feast"
)