import warnings

import pandas as pd
import numpy as np

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer

from pytorch_forecasting.data import (
    NaNLabelEncoder,
    EncoderNormalizer
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# IGNORE NON-CRITICAL WARNINGS
# ==========================================

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "dataset/Metro_Interstate_Traffic_Volume_fixed.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ==========================================
# 2. CONVERT DATE/TIME
# ==========================================

df["date_time"] = pd.to_datetime(
    df["date_time"]
)

df = df.sort_values(
    "date_time"
).reset_index(drop=True)


# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

df["holiday"] = df["holiday"].fillna("None")

df["weather_main"] = df[
    "weather_main"
].fillna("Unknown")

df["weather_description"] = df[
    "weather_description"
].fillna("Unknown")


# ==========================================
# 4. CREATE TIME FEATURES
# ==========================================

df["hour"] = df["date_time"].dt.hour

df["weekday"] = df["date_time"].dt.weekday

df["month"] = df["date_time"].dt.month

df["year"] = df["date_time"].dt.year


# ==========================================
# 5. CREATE TFT TIME INDEX
# ==========================================

df["time_idx"] = (
    (
        df["date_time"]
        - df["date_time"].min()
    ).dt.total_seconds()
    / 3600
).round().astype(int)


# ==========================================
# 6. CREATE GROUP ID
# ==========================================

df["group_id"] = "I94"


# ==========================================
# 7. CONVERT CATEGORICAL FEATURES
# ==========================================

categorical_columns = [
    "holiday",
    "weather_main",
    "weather_description",
    "group_id"
]

for column in categorical_columns:

    df[column] = df[column].astype(str)


# ==========================================
# 8. CLEAN TRAFFIC VOLUME
# ==========================================

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"],
    errors="coerce"
)

df = df.dropna(
    subset=["traffic_volume"]
)

df["traffic_volume"] = df[
    "traffic_volume"
].astype(np.float32)


# ==========================================
# 9. SETTINGS
# ==========================================

max_encoder_length = 24

max_prediction_length = 1

training_cutoff = int(
    df["time_idx"].max() * 0.8
)

print(
    "Training cutoff:",
    training_cutoff
)


# ==========================================
# 10. CREATE TRAINING DATASET
# ==========================================

training = TimeSeriesDataSet(

    df[
        lambda x:
        x.time_idx <= training_cutoff
    ],

    time_idx="time_idx",

    target="traffic_volume",

    group_ids=["group_id"],

    min_encoder_length=
        max_encoder_length,

    max_encoder_length=
        max_encoder_length,

    min_prediction_length=
        max_prediction_length,

    max_prediction_length=
        max_prediction_length,


    # --------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------

    static_categoricals=[
        "group_id"
    ],

    time_varying_known_categoricals=[
        "holiday"
    ],

    time_varying_known_reals=[
        "hour",
        "weekday",
        "month",
        "year"
    ],

    time_varying_unknown_categoricals=[
        "weather_main",
        "weather_description"
    ],

    time_varying_unknown_reals=[
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all"
    ],


    # --------------------------------------
    # CATEGORICAL ENCODERS
    # --------------------------------------

    categorical_encoders={

        "group_id":
            NaNLabelEncoder(add_nan=True),

        "holiday":
            NaNLabelEncoder(add_nan=True),

        "weather_main":
            NaNLabelEncoder(add_nan=True),

        "weather_description":
            NaNLabelEncoder(add_nan=True),
    },


    # --------------------------------------
    # CONTINUOUS TARGET
    # --------------------------------------

    target_normalizer=EncoderNormalizer(
        method="standard",
        center=True
    ),


    # --------------------------------------
    # TFT OPTIONS
    # --------------------------------------

    add_relative_time_idx=True,

    add_target_scales=True,

    add_encoder_length=True,

    allow_missing_timesteps=True
)


# ==========================================
# 11. CREATE VALIDATION DATA
# ==========================================

validation_data = df[
    df["time_idx"]
    >
    training_cutoff - max_encoder_length
].copy()


# ==========================================
# 12. USE SMALLER VALIDATION SET
# ==========================================

# First evaluation uses 2500 rows
# to make CPU evaluation faster.

validation_data = validation_data.head(
    2500
).copy()


print(
    "\nValidation rows used:",
    len(validation_data)
)


# ==========================================
# 13. CREATE VALIDATION DATASET
# ==========================================

validation = TimeSeriesDataSet.from_dataset(

    training,

    validation_data,

    predict=False,

    stop_randomization=True
)


print(
    "Validation samples:",
    len(validation)
)


# ==========================================
# 14. CREATE VALIDATION DATALOADER
# ==========================================

validation_dataloader = (
    validation.to_dataloader(

        train=False,

        batch_size=256,

        num_workers=0
    )
)


# ==========================================
# 15. LOAD TRAINED TFT MODEL
# ==========================================

MODEL_PATH = (
    "models/tft_traffic_volume.ckpt"
)

tft = TemporalFusionTransformer.load_from_checkpoint(
    MODEL_PATH
)


print("\n===================================")
print("TFT MODEL LOADED")
print("===================================")


# ==========================================
# 16. MAKE PREDICTIONS
# ==========================================

print(
    "\nMaking predictions..."
)

predictions = tft.predict(
    validation_dataloader
)

predictions = (
    predictions
    .detach()
    .cpu()
    .numpy()
)


# ==========================================
# 17. GET ACTUAL VALUES
# ==========================================

actual = []

for batch in validation_dataloader:

    x, y = batch

    target = y[0]

    actual.extend(
        target
        .detach()
        .cpu()
        .numpy()
    )


actual = np.array(actual)


# ==========================================
# 18. MATCH LENGTHS
# ==========================================

min_length = min(
    len(actual),
    len(predictions)
)

actual = actual[:min_length]

predictions = predictions[:min_length]


print(
    "\nNumber of predictions:",
    len(predictions)
)


# ==========================================
# 19. CALCULATE MAE
# ==========================================

mae = mean_absolute_error(
    actual,
    predictions
)


# ==========================================
# 20. CALCULATE RMSE
# ==========================================

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


# ==========================================
# 21. CALCULATE R²
# ==========================================

r2 = r2_score(
    actual,
    predictions
)


# ==========================================
# 22. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("TFT MODEL EVALUATION")
print("===================================")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    "\nEvaluation completed successfully."
)