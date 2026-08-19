"""
Prometheus metrics - FinSentinel M3 monitoring.

Owns every metric definition and the PSI drift computation. This is a
pure module: it is imported by the FastAPI app (src/api/main.py), which
calls track_prediction() on each inference and exposes the registry on
/metrics. It deliberately has no __main__ block - a monitoring module
that generates its own synthetic predictions would report metrics that
say nothing about real traffic.
"""

from collections import deque
from typing import Optional

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# --- Prediction metrics ---

PREDICTION_COUNTER = Counter(
    "finsentinel_predictions_total",
    "Total number of predictions made",
    ["label", "use_case"]
)

PREDICTION_LATENCY = Histogram(
    "finsentinel_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

CONFIDENCE_GAUGE = Gauge(
    "finsentinel_confidence_score",
    "Last prediction confidence score"
)

CONFIDENCE_SUMMARY = Summary(
    "finsentinel_confidence_summary",
    "Prediction confidence score distribution"
)

# --- Sentiment distribution ---

SENTIMENT_COUNTER = Counter(
    "finsentinel_sentiment_total",
    "Total predictions per sentiment label",
    ["label"]
)

SENTIMENT_RATIO = Gauge(
    "finsentinel_sentiment_ratio",
    "Current ratio of each sentiment label",
    ["label"]
)

# --- Model info ---

MODEL_ACCURACY = Gauge(
    "finsentinel_model_accuracy",
    "Model accuracy on test set"
)

MODEL_F1 = Gauge(
    "finsentinel_model_f1",
    "Model F1 score on test set"
)

MODEL_INFO = Gauge(
    "finsentinel_model_info",
    "Model metadata",
    ["model_name", "base_model", "version"]
)

# --- API metrics ---

API_REQUESTS_TOTAL = Counter(
    "finsentinel_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"]
)

API_ERRORS_TOTAL = Counter(
    "finsentinel_api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"]
)

API_ACTIVE_REQUESTS = Gauge(
    "finsentinel_api_active_requests",
    "Number of currently active requests"
)

# --- Data drift ---

DATA_DRIFT_SCORE = Gauge(
    "finsentinel_data_drift_score",
    "Population Stability Index between recent and reference distributions"
)

DATA_DRIFT_ALERT = Gauge(
    "finsentinel_data_drift_alert",
    "Data drift alert - 1=drift detected, 0=normal"
)

# Standard PSI reading: < 0.1 no drift, 0.1-0.25 moderate, >= 0.25
# significant. The alert threshold matches that convention rather than
# a hand-picked value.
DRIFT_THRESHOLD = 0.25
MIN_WINDOW_FOR_DRIFT = 20

# --- Internal state for drift detection ---

_prediction_window = deque(maxlen=100)
_label_counts = {"bearish": 0, "bullish": 0, "neutral": 0}
_total_predictions = 0

# Reference distribution measured on the training set.
REFERENCE_DISTRIBUTION = {
    "bearish": 0.164,
    "bullish": 0.192,
    "neutral": 0.644
}

LABELS = ("bearish", "bullish", "neutral")


# --- Core tracking functions ---

def track_prediction(
    label: str,
    score: float,
    latency_ms: float,
    use_case: str = "general"
) -> None:
    """
    Track a single prediction - updates all relevant metrics.
    Called from the FastAPI /predict endpoint after the model call.
    """
    global _total_predictions

    PREDICTION_COUNTER.labels(label=label, use_case=use_case).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    CONFIDENCE_GAUGE.set(score)
    CONFIDENCE_SUMMARY.observe(score)
    SENTIMENT_COUNTER.labels(label=label).inc()

    _prediction_window.append(label)
    if label in _label_counts:
        _label_counts[label] += 1
    _total_predictions += 1

    _update_sentiment_ratios()
    _compute_drift()


def track_api_request(endpoint: str, method: str, status_code: int) -> None:
    """Track an API request with endpoint, method and status code."""
    API_REQUESTS_TOTAL.labels(
        endpoint=endpoint,
        method=method,
        status_code=str(status_code)
    ).inc()


def track_api_error(endpoint: str, error_type: str) -> None:
    """Track an API error."""
    API_ERRORS_TOTAL.labels(endpoint=endpoint, error_type=error_type).inc()


def set_active_requests(count: int) -> None:
    """Set the number of currently active requests."""
    API_ACTIVE_REQUESTS.set(count)


def set_model_info(
    model_name: str,
    base_model: str,
    accuracy: float,
    f1_score: float,
    version: str = "1.0.0"
) -> None:
    """
    Register model metadata and performance metrics.
    Called once at API startup.
    """
    MODEL_ACCURACY.set(accuracy)
    MODEL_F1.set(f1_score)
    MODEL_INFO.labels(
        model_name=model_name,
        base_model=base_model,
        version=version
    ).set(1)

    print(f"Model registered: {model_name}")
    print(f"  Accuracy : {accuracy}")
    print(f"  F1 Score : {f1_score}")


def reset_state() -> None:
    """
    Clear the sliding window and counters. Intended for tests and for
    controlled drift experiments, never called in normal operation.
    """
    global _total_predictions
    _prediction_window.clear()
    for key in _label_counts:
        _label_counts[key] = 0
    _total_predictions = 0


# --- Internal helpers ---

def _update_sentiment_ratios() -> None:
    """
    Update sentiment ratio gauges from the sliding window of the last
    100 predictions.
    """
    if not _prediction_window:
        return

    total = len(_prediction_window)
    for label in LABELS:
        SENTIMENT_RATIO.labels(label=label).set(
            _prediction_window.count(label) / total
        )


def _compute_drift() -> None:
    """
    Compute data drift using the Population Stability Index (PSI),
    comparing the current prediction distribution against the reference
    (training) distribution.

    PSI < 0.1  : no drift
    PSI < 0.25 : moderate drift
    PSI >= 0.25: significant drift -> alert

    Below MIN_WINDOW_FOR_DRIFT predictions the index is not computed:
    on a handful of samples PSI is dominated by sampling noise.
    """
    if len(_prediction_window) < MIN_WINDOW_FOR_DRIFT:
        return

    total = len(_prediction_window)
    psi = 0.0

    for label in LABELS:
        actual = max(_prediction_window.count(label) / total, 0.0001)
        expected = max(REFERENCE_DISTRIBUTION.get(label, 0.01), 0.0001)
        psi += (actual - expected) * np.log(actual / expected)

    DATA_DRIFT_SCORE.set(round(psi, 4))

    if psi >= DRIFT_THRESHOLD:
        DATA_DRIFT_ALERT.set(1)
        print(f"  DRIFT ALERT: PSI={psi:.4f} exceeds threshold {DRIFT_THRESHOLD}")
    else:
        DATA_DRIFT_ALERT.set(0)


# --- Standalone server (unused when imported by the API) ---

def start_metrics_server(port: int = 8002) -> None:
    """
    Start a standalone Prometheus metrics HTTP server.

    Not used in the containerized stack, where the API exposes /metrics
    on port 8000 directly. Kept for local debugging.
    """
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")