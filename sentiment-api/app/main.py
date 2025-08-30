# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib, time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

# ────────────────────────────────────────────────────────────
# Load model artifacts
# ────────────────────────────────────────────────────────────
model = joblib.load("model/model.joblib")
le    = joblib.load("model/label_encoder.joblib")

# Get labels from LabelEncoder
labels = list(le.classes_)

# ────────────────────────────────────────────────────────────
# Pydantic schemas
# ────────────────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    text: str

# Dynamically use LabelEncoder classes for response enum
class SentimentResponse(BaseModel):
    sentiment: Literal[tuple(labels)]
    confidence: float

# ────────────────────────────────────────────────────────────
# Create FastAPI app
# ────────────────────────────────────────────────────────────
app = FastAPI(title="Real‑Time Sentiment API")

@app.get("/")
def root():
    return {
        "message": "Welcome to the Real‑Time Sentiment Analysis API 🎉",
        "endpoints": ["/predict"],
        "docs": "/docs",
    }

# Optional: custom histogram for model latency
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken (seconds) to generate a single prediction"
)

# ────────────────────────────────────────────────────────────
# Prediction route
# ────────────────────────────────────────────────────────────
@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest):
    start = time.perf_counter()
    try:
        probs = model.predict_proba([req.text])[0]
        idx   = probs.argmax()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    return {
        "sentiment": le.inverse_transform([idx])[0],
        "confidence": float(probs[idx]),
    }

# ────────────────────────────────────────────────────────────
# Expose Prometheus metrics
# ────────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)
