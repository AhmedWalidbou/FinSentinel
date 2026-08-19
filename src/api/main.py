"""
FinSentinel API - French financial sentiment analysis (CamemBERT).

Instrumented for Prometheus: every prediction feeds the monitoring
module (src/monitoring/prometheus_metrics.py), which owns the metric
definitions and the PSI drift computation. The API exposes them on
/metrics so Prometheus scrapes real production traffic rather than
synthetic data.
"""

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from transformers import pipeline

from src.monitoring.prometheus_metrics import (
    set_active_requests,
    set_model_info,
    track_api_error,
    track_api_request,
    track_prediction,
)

app = FastAPI(
    title="FinSentinel API",
    description="French financial sentiment analysis API - CamemBERT fine-tuned",
    version="1.0.0"
)

MODEL_NAME = "Walid692/finsentinel-camembert"
BASE_MODEL = "camembert-base"
ACCURACY = 0.7808
F1_SCORE = 0.7759

classifier = None
_active_requests = 0

LABEL_MAP = {
    "LABEL_0": "bearish",
    "LABEL_1": "bullish",
    "LABEL_2": "neutral"
}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Track every request: active count, endpoint, method and status code.
    The /metrics endpoint itself is excluded to avoid self-observation
    inflating the counters on every scrape.
    """
    global _active_requests
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)

    _active_requests += 1
    set_active_requests(_active_requests)
    try:
        response = await call_next(request)
        track_api_request(path, request.method, response.status_code)
        if response.status_code >= 400:
            track_api_error(path, f"http_{response.status_code}")
        return response
    finally:
        _active_requests -= 1
        set_active_requests(_active_requests)


@app.on_event("startup")
def load_model():
    """Load the model and register its metadata as gauges."""
    global classifier
    print(f"Loading model: {MODEL_NAME}")
    classifier = pipeline("text-classification", model=MODEL_NAME)
    print("Model loaded successfully")
    set_model_info(
        model_name=MODEL_NAME,
        base_model=BASE_MODEL,
        accuracy=ACCURACY,
        f1_score=F1_SCORE,
    )


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    label: str
    score: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint - health check."""
    return {"status": "ok", "model": MODEL_NAME, "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_NAME, "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint - exposes the monitoring registry."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict sentiment of a financial text.

    Latency is measured around the model call only, then handed to the
    monitoring module after the measurement so tracking never inflates
    the reported latency.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.time()
    result = classifier(request.text)[0]
    latency_ms = (time.time() - start) * 1000

    label = LABEL_MAP.get(result["label"], result["label"])
    score = round(result["score"], 4)

    track_prediction(label=label, score=score, latency_ms=latency_ms)

    return {
        "text": request.text,
        "label": label,
        "score": score,
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/info")
def info():
    """Model information endpoint."""
    return {
        "model": MODEL_NAME,
        "labels": ["bearish", "bullish", "neutral"],
        "language": "fr",
        "accuracy": ACCURACY,
        "f1_score": F1_SCORE,
        "base_model": BASE_MODEL,
        "train_samples": 10252
    }