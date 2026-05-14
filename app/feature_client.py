import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from feast import FeatureStore

from app.metrics import feast_online_store_hits_total, feast_online_store_misses_total

logger = logging.getLogger(__name__)

FEATURE_VIEW_NAME = "fraud_features"
REPO_PATH = Path(os.environ.get("FEAST_REPO_PATH", Path(__file__).parent.parent / "feast_repo"))
FEATURE_COLUMNS = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "amt_log",
    "distance_km",
    "category_encoded",
]


class FeatureClient:
    def __init__(self, repo_path: Optional[Path] = None):
        self._repo_path = repo_path or REPO_PATH
        self._store: Optional[FeatureStore] = None
        self._hit_count: int = 0
        self._miss_count: int = 0

    @property
    def store(self) -> FeatureStore:
        if self._store is None:
            self._store = FeatureStore(repo_path=str(self._repo_path))
        return self._store

    def get_features(self, entity_id: int) -> dict[str, Any]:
        try:
            result = self.store.get_online_features(
                entity_rows=[{"cc_num": entity_id}],
                features=[f"{FEATURE_VIEW_NAME}:{name}" for name in FEATURE_COLUMNS],
            )
        except Exception as exc:
            logger.warning("Feast get_online_features failed for entity %s: %s", entity_id, exc)
            self._miss_count += 1
            feast_online_store_misses_total.inc()
            return {}

        raw_features = result.to_dict()
        features = {
            key: value[0] if isinstance(value, list) and value else value
            for key, value in raw_features.items()
        }
        features.setdefault("cc_num", entity_id)

        if all(features.get(name) is not None for name in FEATURE_COLUMNS):
            self._hit_count += 1
            feast_online_store_hits_total.inc()
            return features

        logger.warning("No complete feature row found for entity %s", entity_id)
        self._miss_count += 1
        feast_online_store_misses_total.inc()
        return {}

    def get_features_dataframe(self, entity_id: int) -> pd.DataFrame:
        features = self.get_features(entity_id)
        if not features:
            return pd.DataFrame()
        return pd.DataFrame([features])

    def get_stats(self) -> dict[str, int]:
        return {"hits": self._hit_count, "misses": self._miss_count}

    def reset_stats(self) -> None:
        self._hit_count = 0
        self._miss_count = 0
