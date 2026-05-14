# ============================================================================
# ModelServe — MLflow Model Loader
# ============================================================================

import logging
from typing import Any, Optional

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, mlflow_tracking_uri: str, model_name: str) -> None:
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._version: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            model_uri = f"models:/{self.model_name}/Production"
            self._model = mlflow.pyfunc.load_model(model_uri)

            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(self.model_name, stages=["Production"])
            if latest_version:
                self._version = str(latest_version[0].version)
            else:
                self._version = "unknown"

            logger.info(f"Successfully loaded model '{self.model_name}' version {self._version}")
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            self._model = None
            self._version = None

    def predict(self, features: pd.DataFrame) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        return self._model.predict(features)

    def get_version(self) -> Optional[str]:
        return self._version