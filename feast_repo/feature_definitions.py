from pathlib import Path

from feast import Entity, FileSource, FeatureView, Field
from feast.data_format import ParquetFormat
from feast.types import Int64, Float64
from datetime import timedelta

cc_num = Entity(
    name="cc_num",
    value_type=Int64,
    description="Credit card number",
)

FEATURES_FILE = Path(__file__).resolve().parents[1] / "training" / "features.parquet"

source = FileSource(
    name="fraud_features_source",
    path=str(FEATURES_FILE),
    timestamp_field="event_timestamp",
    parquet_format=ParquetFormat(),
)

fraud_features = FeatureView(
    name="fraud_features",
    entities=[cc_num],
    ttl=timedelta(days=7),
    schema=[
        Field(name="amt", dtype=Float64),
        Field(name="lat", dtype=Float64),
        Field(name="long", dtype=Float64),
        Field(name="city_pop", dtype=Int64),
        Field(name="unix_time", dtype=Int64),
        Field(name="merch_lat", dtype=Float64),
        Field(name="merch_long", dtype=Float64),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_night", dtype=Int64),
        Field(name="amt_log", dtype=Float64),
        Field(name="distance_km", dtype=Float64),
        Field(name="category_encoded", dtype=Int64),
    ],
    online=True,
    source=source,
)
