import os
import json
import math
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.path.join(os.path.dirname(__file__), "fraudTrain.csv")
FEATURES_PARQUET_PATH = os.path.join(os.path.dirname(__file__), "features.parquet")
SAMPLE_REQUEST_PATH = os.path.join(os.path.dirname(__file__), "sample_request.json")

FEATURE_COLS = [
    "cc_num", "amt", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "amt_log", "distance_km", "category_encoded"
]

LABEL_COL = "is_fraud"


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

    df["event_timestamp"] = dt.astype("int64") // 10**9

    return df


def prepare_features(df, feature_cols):
    return df[feature_cols].copy()


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud_detection")

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    df = engineer_features(df)

    X = prepare_features(df, FEATURE_COLS)
    y = df[LABEL_COL]

    print(f"Target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="fraud_rf_training"):
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
            print(f"  {name}: {value:.4f}")

        mlflow.sklearn.log_model(model, "fraud_model", registered_model_name="fraud_detection_model")

    client = mlflow.MlflowClient()
    model_name = "fraud_detection_model"
    latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version

    client.transition_model_version_stage(
        name=model_name,
        version=int(latest_version),
        stage="Production"
    )
    print(f"Model version {latest_version} transitioned to Production")

    feast_df = df[["cc_num", "event_timestamp"] + FEATURE_COLS].copy()
    feast_df.to_parquet(FEATURES_PARQUET_PATH, index=False)
    print(f"Saved features.parquet to {FEATURES_PARQUET_PATH}")

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
        "event_timestamp": int(sample_row["event_timestamp"])
    }
    with open(SAMPLE_REQUEST_PATH, "w") as f:
        json.dump(sample_request, f, indent=4)
    print(f"Saved sample_request.json to {SAMPLE_REQUEST_PATH}")


if __name__ == "__main__":
    main()