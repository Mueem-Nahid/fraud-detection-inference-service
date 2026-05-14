import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from feature_client import FeatureClient
from metrics import prediction_duration_seconds, prediction_errors_total, prediction_requests_total
from model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ModelServe", version="1.0.0")

_model_loader: Optional[ModelLoader] = None
_feature_client: Optional[FeatureClient] = None


class PredictRequest(BaseModel):
    entity_id: int


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    timestamp: str


class PredictExplainResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    timestamp: str
    features: dict


def get_model_loader() -> ModelLoader:
    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_loader


def get_feature_client() -> FeatureClient:
    if _feature_client is None:
        raise HTTPException(status_code=503, detail="Feature client not initialized")
    return _feature_client


def init_app(model_loader: ModelLoader, feature_client: FeatureClient) -> FastAPI:
    global _model_loader, _feature_client
    _model_loader = model_loader
    _feature_client = feature_client
    return app


@app.get("/health")
def health():
    model_loader = get_model_loader()
    return {"status": "healthy", "model_version": model_loader.get_version() or "unknown"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    prediction_requests_total.inc()
    with prediction_duration_seconds.time():
        try:
            feature_client = get_feature_client()
            features = feature_client.get_features_dataframe(request.entity_id)

            if features.empty:
                prediction_errors_total.inc()
                raise HTTPException(status_code=404, detail="Features not found for entity")

            model_loader = get_model_loader()
            raw_pred = model_loader.predict(features)

            if isinstance(raw_pred, (list, pd.Series)):
                prediction = int(raw_pred[0])
                probability = float(raw_pred[0]) if len(raw_pred) > 1 else 0.5
            else:
                prediction = int(raw_pred)
                probability = float(raw_pred)

            return PredictResponse(
                prediction=prediction,
                probability=probability,
                model_version=model_loader.get_version() or "unknown",
                timestamp=datetime.utcnow().isoformat()
            )
        except HTTPException:
            raise
        except Exception as exc:
            prediction_errors_total.inc()
            logger.error(f"Prediction failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/{entity_id}", response_model=PredictExplainResponse)
def predict_get(entity_id: int, explain: bool = False):
    prediction_requests_total.inc()
    with prediction_duration_seconds.time():
        try:
            feature_client = get_feature_client()
            features = feature_client.get_features_dataframe(entity_id)

            if features.empty:
                prediction_errors_total.inc()
                raise HTTPException(status_code=404, detail="Features not found for entity")

            model_loader = get_model_loader()
            raw_pred = model_loader.predict(features)

            if isinstance(raw_pred, (list, pd.Series)):
                prediction = int(raw_pred[0])
                probability = float(raw_pred[0]) if len(raw_pred) > 1 else 0.5
            else:
                prediction = int(raw_pred)
                probability = float(raw_pred)

            return PredictExplainResponse(
                prediction=prediction,
                probability=probability,
                model_version=model_loader.get_version() or "unknown",
                timestamp=datetime.utcnow().isoformat(),
                features=features.iloc[0].to_dict()
            )
        except HTTPException:
            raise
        except Exception as exc:
            prediction_errors_total.inc()
            logger.error(f"Prediction failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)