from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import get_model_info, get_model_metrics, get_predictor, load_predictor
from src.api.schemas import PredictBatchRequest, PredictRequest
from src.inference.BasePredictor import BasePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictor()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict(
    payload: PredictRequest,
    predictor: BasePredictor = Depends(get_predictor),
) -> dict[str, Any]:
    return predictor.predict(payload.text)


@app.post("/predict_batch")
def predict_batch(
    payload: PredictBatchRequest,
    predictor: BasePredictor = Depends(get_predictor),
) -> dict[str, Any]:
    predictions = predictor.predict_batch(payload.texts)
    return {
        "count": len(predictions),
        "predictions": predictions,
    }


@app.get("/info")
def info() -> dict[str, Any]:
    return get_model_info()


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return get_model_metrics()
