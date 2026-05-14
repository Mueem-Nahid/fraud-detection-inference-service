import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.feature_client import FeatureClient, FEATURE_COLUMNS
from app.metrics import model_version_info, prediction_duration_seconds, prediction_errors_total, prediction_requests_total
from app.model_loader import ModelLoader

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


@app.on_event("startup")
def startup() -> None:
    global _model_loader, _feature_client
    if _model_loader is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.environ.get("MODEL_NAME", "fraud_detection_model")
        _model_loader = ModelLoader(tracking_uri, model_name)
        version = _model_loader.get_version()
        if version:
            model_version_info.labels(version=version).set(1)
    if _feature_client is None:
        _feature_client = FeatureClient()


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
            prediction, probability = _predict_values(model_loader, features)

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
            logger.error("Prediction failed: %s", exc)
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
            prediction, probability = _predict_values(model_loader, features)

            response_features = features[FEATURE_COLUMNS].iloc[0].to_dict() if explain else {}

            return PredictExplainResponse(
                prediction=prediction,
                probability=probability,
                model_version=model_loader.get_version() or "unknown",
                timestamp=datetime.utcnow().isoformat(),
                features=response_features
            )
        except HTTPException:
            raise
        except Exception as exc:
            prediction_errors_total.inc()
            logger.error("Prediction failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))


def _predict_values(model_loader: ModelLoader, features: pd.DataFrame) -> tuple[int, float]:
    model_features = features[["cc_num"] + FEATURE_COLUMNS]
    raw_pred = model_loader.predict(model_features)
    if isinstance(raw_pred, pd.Series):
        value = raw_pred.iloc[0]
    elif isinstance(raw_pred, (list, tuple)):
        value = raw_pred[0]
    else:
        try:
            value = raw_pred[0]
        except (TypeError, IndexError):
            value = raw_pred

    prediction = int(value)
    probability = float(value) if 0 <= float(value) <= 1 else float(prediction)
    return prediction, probability


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
