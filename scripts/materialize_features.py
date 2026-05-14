import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from feast import FeatureStore

from feast_repo.feature_definitions import cc_num, fraud_features, source

ROOT = Path(__file__).resolve().parents[1]
FEAST_REPO = ROOT / "feast_repo"
FEATURES_FILE = ROOT / "training" / "features.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    os.environ.setdefault("REDIS_HOST", "localhost")

    if not FEATURES_FILE.exists():
        logger.error("Missing %s. Run training/train.py before materializing features.", FEATURES_FILE)
        return 1

    try:
        store = FeatureStore(repo_path=str(FEAST_REPO))
        store.apply([cc_num, source, fraud_features])
        store.materialize_incremental(end_date=datetime.now(timezone.utc))
    except Exception as exc:
        logger.error("Feature materialization failed: %s", exc)
        return 1

    logger.info("Materialized Feast features from %s into Redis online store", FEATURES_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
