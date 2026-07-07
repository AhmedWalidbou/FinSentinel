import time
import numpy as np
from collections import deque
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from typing import Optional


# ─── Prediction metrics ───────────────────────────────────────────────────────

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

# ─── Sentiment distribution ───────────────────────────────────────────────────

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

# ─── Model info ───────────────────────────────────────────────────────────────

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

# ─── API metrics ──────────────────────────────────────────────────────────────

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

# ─── Data drift ───────────────────────────────────────────────────────────────

DATA_DRIFT_SCORE = Gauge(
    "finsentinel_data_drift_score",
    "Data drift score — 0=no drift, 1=full drift"
)

DATA_DRIFT_ALERT = Gauge(
    "finsentinel_data_drift_alert",
    "Data drift alert — 1=drift detected, 0=normal"
)

DRIFT_THRESHOLD = 0.3


# ─── Internal state for drift detection ──────────────────────────────────────

_prediction_window = deque(maxlen=100)
_label_counts = {"bearish": 0, "bullish": 0, "neutral": 0}
_total_predictions = 0

# Reference distribution from training data
REFERENCE_DISTRIBUTION = {
    "bearish": 0.164,
    "bullish": 0.192,
    "neutral": 0.644
}


# ─── Core tracking functions ──────────────────────────────────────────────────

def track_prediction(
    label: str,
    score: float,
    latency_ms: float,
    use_case: str = "general"
) -> None:
    """
    Track a single prediction — updates all relevant metrics.
    Called from FastAPI /predict endpoint.
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


def track_api_request(
    endpoint: str,
    method: str,
    status_code: int
) -> None:
    """
    Track an API request with endpoint, method and status code.
    """
    API_REQUESTS_TOTAL.labels(
        endpoint=endpoint,
        method=method,
        status_code=str(status_code)
    ).inc()


def track_api_error(endpoint: str, error_type: str) -> None:
    """
    Track an API error.
    """
    API_ERRORS_TOTAL.labels(
        endpoint=endpoint,
        error_type=error_type
    ).inc()


def set_active_requests(count: int) -> None:
    """
    Set the number of currently active requests.
    """
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


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _update_sentiment_ratios() -> None:
    """
    Update sentiment ratio gauges based on recent predictions.
    Uses sliding window of last 100 predictions.
    """
    if not _prediction_window:
        return

    total = len(_prediction_window)
    for label in ["bearish", "bullish", "neutral"]:
        count = _prediction_window.count(label)
        ratio = count / total
        SENTIMENT_RATIO.labels(label=label).set(ratio)


def _compute_drift() -> None:
    """
    Compute data drift using Population Stability Index (PSI).
    Compares current prediction distribution vs reference (training) distribution.
    PSI < 0.1  : no drift
    PSI < 0.25 : moderate drift
    PSI >= 0.25: significant drift
    """
    if len(_prediction_window) < 20:
        return

    total = len(_prediction_window)
    psi = 0.0

    for label in ["bearish", "bullish", "neutral"]:
        actual = _prediction_window.count(label) / total
        expected = REFERENCE_DISTRIBUTION.get(label, 0.01)

        actual = max(actual, 0.0001)
        expected = max(expected, 0.0001)

        psi += (actual - expected) * np.log(actual / expected)

    DATA_DRIFT_SCORE.set(round(psi, 4))

    if psi >= DRIFT_THRESHOLD:
        DATA_DRIFT_ALERT.set(1)
        print(f"  DRIFT ALERT: PSI={psi:.4f} exceeds threshold {DRIFT_THRESHOLD}")
    else:
        DATA_DRIFT_ALERT.set(0)


# ─── Server ───────────────────────────────────────────────────────────────────

def start_metrics_server(port: int = 8002) -> None:
    """
    Start Prometheus metrics HTTP server on given port.
    Default port 8002 to avoid conflict with FastAPI on 8000.
    """
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
    print(f"Metrics available at: http://localhost:{port}/metrics")


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting FinSentinel Prometheus metrics server...\n")

    set_model_info(
        model_name="Walid692/finsentinel-camembert",
        base_model="camembert-base",
        accuracy=0.7808,
        f1_score=0.7759
    )

    predictions = [
        ("neutral", 0.8339, 299.66, "general"),
        ("bullish", 0.9123, 150.12, "finance"),
        ("bearish", 0.7456, 210.34, "finance"),
        ("neutral", 0.6789, 180.00, "general"),
        ("neutral", 0.7234, 195.00, "general"),
        ("bullish", 0.8567, 160.00, "finance"),
        ("neutral", 0.9012, 140.00, "general"),
        ("bearish", 0.6543, 220.00, "finance"),
        ("neutral", 0.7890, 175.00, "general"),
        ("bullish", 0.8234, 155.00, "finance"),
    ]

    for label, score, latency, use_case in predictions:
        track_prediction(label, score, latency, use_case)

    track_api_request("/predict", "POST", 200)
    track_api_request("/predict", "POST", 200)
    track_api_request("/health", "GET", 200)
    track_api_request("/predict", "POST", 422)
    track_api_error("/predict", "validation_error")

    start_metrics_server(port=8002)

    print("\nMetrics server running on http://localhost:8002/metrics")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMetrics server stopped.")