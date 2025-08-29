# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

# ────────────────────────────────────────────────────────────
# Load model artefacts
# ────────────────────────────────────────────────────────────
model = joblib.load("model/model.joblib")
le    = joblib.load("model/label_encoder.joblib")


# ────────────────────────────────────────────────────────────
# Pydantic schemas
# ────────────────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# ────────────────────────────────────────────────────────────
# Create FastAPI app
# ────────────────────────────────────────────────────────────
app = FastAPI(title="Real‑Time Sentiment API")

# 👋 Root “welcome / health‑check” route
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
# Routes
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
# Expose Prometheus metrics (default latency, count, etc.)
# ────────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)
