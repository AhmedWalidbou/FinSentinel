from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import time


app = FastAPI(
    title="FinSentinel API",
    description="French financial sentiment analysis API — CamemBERT fine-tuned",
    version="1.0.0"
)

MODEL_NAME = "Walid692/finsentinel-camembert"
classifier = None

LABEL_MAP = {
    "LABEL_0": "bearish",
    "LABEL_1": "bullish",
    "LABEL_2": "neutral"
}


@app.on_event("startup")
def load_model():
    """
    Load model on startup.
    """
    global classifier
    print(f"Loading model: {MODEL_NAME}")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME
    )
    print("Model loaded successfully")


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
    """
    Root endpoint — health check.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict sentiment of a financial text.
    Returns label and confidence score.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.time()
    result = classifier(request.text)[0]
    latency_ms = (time.time() - start) * 1000

    return {
        "text": request.text,
        "label": LABEL_MAP.get(result["label"], result["label"]),
        "score": round(result["score"], 4),
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/info")
def info():
    """
    Model information endpoint.
    """
    return {
        "model": MODEL_NAME,
        "labels": ["bearish", "bullish", "neutral"],
        "language": "fr",
        "accuracy": 0.7808,
        "f1_score": 0.7759,
        "base_model": "camembert-base",
        "train_samples": 10252
    }