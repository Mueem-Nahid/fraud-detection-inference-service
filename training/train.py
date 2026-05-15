import os
import sys
import json
import math
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from logger import get_logger


logger = get_logger(__name__)

TRAIN_DIR = os.path.dirname(__file__)
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file://{TRAIN_DIR}/mlruns")
DATA_PATH = os.path.join(TRAIN_DIR, "fraudTrain.csv")
FEATURES_PARQUET_PATH = os.path.join(TRAIN_DIR, "features.parquet")
SAMPLE_REQUEST_PATH = os.path.join(TRAIN_DIR, "sample_request.json")
MODEL_NAME = os.environ.get("MODEL_NAME", "fraud_detection_model")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")

# Ensure a writable local artifact root, overriding server-stored paths.
os.environ.setdefault("MLFLOW_DEFAULT_ARTIFACT_ROOT", os.path.join(TRAIN_DIR, "mlartifacts"))

FEATURE_COLS = [
    "cc_num", "amt", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "amt_log", "distance_km", "category_encoded"
]

LABEL_COL = "is_fraud"
FEAST_EXPORT_COLS = ["cc_num", "event_timestamp"] + [col for col in FEATURE_COLS if col != "cc_num"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def engineer_features(df):
    df = df.copy()

    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    df["amt_log"] = np.log1p(df["amt"])

    df["distance_km"] = df.apply(
        lambda r: haversine_km(r["lat"], r["long"], r["merch_lat"], r["merch_long"]), axis=1
    )

    category_mapping = {cat: idx for idx, cat in enumerate(df["category"].unique())}
    df["category_encoded"] = df["category"].map(category_mapping)

    df["event_timestamp"] = dt.dt.tz_localize("UTC")

    return df


def prepare_features(df, feature_cols):
    return df[feature_cols].copy()


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud_detection")

    logger.info("Loading data from %s ...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %s rows", len(df))

    df = engineer_features(df)

    X = prepare_features(df, FEATURE_COLS)
    y = df[LABEL_COL]

    logger.info("Target distribution: %s", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="fraud_rf_training") as run:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            logger.info("%s: %.4f", name, value)

        input_example = X_train.head(5)
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(
            model,
            "fraud_model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

    client = mlflow.MlflowClient()
    run_versions = [
        version for version in client.search_model_versions(f"name='{MODEL_NAME}'")
        if version.run_id == run.info.run_id
    ]
    if not run_versions:
        raise RuntimeError(f"No registered model version found for run {run.info.run_id}")

    latest_version = max(run_versions, key=lambda version: int(version.version)).version
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, latest_version)
    logger.info("Model version %s assigned alias '%s'", latest_version, MODEL_ALIAS)

    feast_df = df[FEAST_EXPORT_COLS].copy()
    feast_df.to_parquet(FEATURES_PARQUET_PATH, index=False)
    logger.info("Saved features.parquet to %s", FEATURES_PARQUET_PATH)

    sample_cc_num = int(df["cc_num"].iloc[0])
    sample_row = df.iloc[0]
    sample_request = {
        "entity_id": sample_cc_num,
        "cc_num": int(sample_row["cc_num"]),
        "amt": float(sample_row["amt"]),
        "lat": float(sample_row["lat"]),
        "long": float(sample_row["long"]),
        "city_pop": int(sample_row["city_pop"]),
        "unix_time": int(sample_row["unix_time"]),
        "merch_lat": float(sample_row["merch_lat"]),
        "merch_long": float(sample_row["merch_long"]),
        "hour_of_day": int(sample_row["hour_of_day"]),
        "day_of_week": int(sample_row["day_of_week"]),
        "is_weekend": int(sample_row["is_weekend"]),
        "is_night": int(sample_row["is_night"]),
        "amt_log": float(sample_row["amt_log"]),
        "distance_km": float(sample_row["distance_km"]),
        "category_encoded": int(sample_row["category_encoded"]),
        "event_timestamp": sample_row["event_timestamp"].isoformat()
    }
    with open(SAMPLE_REQUEST_PATH, "w", encoding="utf-8") as f:
        json.dump(sample_request, f, indent=4)
        f.write("\n")
    logger.info("Saved sample_request.json to %s", SAMPLE_REQUEST_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed")
        raise
