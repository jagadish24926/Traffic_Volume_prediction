import pandas as pd
import numpy as np
import warnings

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.data import (
    NaNLabelEncoder,
    EncoderNormalizer
)


# ==========================================
# 1. IGNORE NON-CRITICAL WARNINGS
# ==========================================

warnings.filterwarnings("ignore")


# ==========================================
# 2. LOAD DATASET
# ==========================================

DATA_PATH = "dataset/Metro_Interstate_Traffic_Volume_fixed.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# ==========================================
# 3. CONVERT DATE/TIME
# ==========================================

df["date_time"] = pd.to_datetime(
    df["date_time"]
)

df = df.sort_values(
    "date_time"
).reset_index(drop=True)


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

df["holiday"] = df["holiday"].fillna("None")

df["weather_main"] = df[
    "weather_main"
].fillna("Unknown")

df["weather_description"] = df[
    "weather_description"
].fillna("Unknown")


# ==========================================
# 5. CREATE TIME FEATURES
# ==========================================

df["hour"] = df["date_time"].dt.hour

df["weekday"] = df["date_time"].dt.weekday

df["month"] = df["date_time"].dt.month

df["year"] = df["date_time"].dt.year


# ==========================================
# 6. CREATE TFT TIME INDEX
# ==========================================

df["time_idx"] = (
    (
        df["date_time"]
        - df["date_time"].min()
    ).dt.total_seconds()
    / 3600
).round().astype(int)


# ==========================================
# 7. CREATE GROUP ID
# ==========================================

df["group_id"] = "I94"


# ==========================================
# 8. CONVERT CATEGORICAL COLUMNS
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
# 9. CLEAN TRAFFIC VOLUME
# ==========================================

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"],
    errors="coerce"
)

df = df.dropna(
    subset=["traffic_volume"]
).reset_index(drop=True)

df["traffic_volume"] = df[
    "traffic_volume"
].astype(np.float32)


# ==========================================
# 10. TFT SETTINGS
# ==========================================

max_encoder_length = 24

max_prediction_length = 1

training_cutoff = int(
    df["time_idx"].max() * 0.8
)


# ==========================================
# 11. CREATE TRAINING DATASET
# ==========================================

training = TimeSeriesDataSet(

    df[
        lambda x:
        x.time_idx <= training_cutoff
    ],

    time_idx="time_idx",

    target="traffic_volume",

    group_ids=["group_id"],

    min_encoder_length=max_encoder_length,

    max_encoder_length=max_encoder_length,

    min_prediction_length=max_prediction_length,

    max_prediction_length=max_prediction_length,


    # --------------------------------------
    # STATIC CATEGORICALS
    # --------------------------------------

    static_categoricals=[
        "group_id"
    ],


    # --------------------------------------
    # KNOWN CATEGORICALS
    # --------------------------------------

    time_varying_known_categoricals=[
        "holiday"
    ],


    # --------------------------------------
    # KNOWN REAL VALUES
    # --------------------------------------

    time_varying_known_reals=[
        "hour",
        "weekday",
        "month",
        "year"
    ],


    # --------------------------------------
    # UNKNOWN CATEGORICALS
    # --------------------------------------

    time_varying_unknown_categoricals=[
        "weather_main",
        "weather_description"
    ],


    # --------------------------------------
    # UNKNOWN REAL VALUES
    # --------------------------------------

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
            NaNLabelEncoder(add_nan=True)
    },


    # --------------------------------------
    # TARGET NORMALIZER
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
# 12. LOAD TRAINED TFT MODEL
# ==========================================

MODEL_PATH = "models/tft_traffic_volume.ckpt"

tft = TemporalFusionTransformer.load_from_checkpoint(
    MODEL_PATH
)

print()
print("===================================")
print("TFT MODEL LOADED")
print("===================================")


# ==========================================
# 13. FIND LAST OBSERVED RECORD
# ==========================================

last_row = df.iloc[-1].copy()

last_time = last_row["date_time"]

last_time_idx = int(
    last_row["time_idx"]
)


# ==========================================
# 14. DEFINE FUTURE TIME
# ==========================================

future_time = (
    last_time +
    pd.Timedelta(hours=1)
)

future_time_idx = (
    last_time_idx + 1
)


print()
print("Last recorded time:")
print(last_time)

print()
print("Future prediction time:")
print(future_time)


# ==========================================
# 15. GET LAST 24 TIME-INDEX HOURS
# ==========================================

history_start_idx = (
    future_time_idx -
    max_encoder_length
)


encoder_data = df[
    (
        df["time_idx"] >=
        history_start_idx
    )
    &
    (
        df["time_idx"] <=
        last_time_idx
    )
].copy()


print()
print("===================================")
print("ENCODER DATA")
print("===================================")

print(
    "Required encoder length:",
    max_encoder_length
)

print(
    "Historical rows found:",
    len(encoder_data)
)

print(
    "Time index range:",
    encoder_data["time_idx"].min(),
    "to",
    encoder_data["time_idx"].max()
)


# ==========================================
# 16. CREATE FUTURE ROW
# ==========================================

future_row = last_row.copy()

future_row["date_time"] = future_time

future_row["time_idx"] = future_time_idx

future_row["hour"] = future_time.hour

future_row["weekday"] = future_time.weekday()

future_row["month"] = future_time.month

future_row["year"] = future_time.year

future_row["group_id"] = "I94"


# ==========================================
# 17. FUTURE TARGET PLACEHOLDER
# ==========================================

# The actual future traffic volume is unknown.
#
# Our installed PyTorch Forecasting version
# does not allow NaN target values while
# constructing this prediction dataset.
#
# Therefore, use the last observed value
# only as a placeholder.
#
# TFT will generate the actual prediction.

future_row["traffic_volume"] = (
    last_row["traffic_volume"]
)


# ==========================================
# 18. COMBINE HISTORY + FUTURE
# ==========================================

prediction_data = pd.concat(
    [
        encoder_data,
        pd.DataFrame([future_row])
    ],
    ignore_index=True
)


# ==========================================
# 19. REMOVE DUPLICATE TIME INDEX
# ==========================================

prediction_data = (
    prediction_data
    .sort_values("time_idx")
    .drop_duplicates(
        subset=["group_id", "time_idx"],
        keep="last"
    )
    .reset_index(drop=True)
)


print()
print("===================================")
print("PREDICTION DATA")
print("===================================")

print(
    "Total rows:",
    len(prediction_data)
)

print(
    "First time_idx:",
    prediction_data["time_idx"].min()
)

print(
    "Last time_idx:",
    prediction_data["time_idx"].max()
)


# ==========================================
# 20. CHECK TARGET VALUES
# ==========================================

if prediction_data[
    "traffic_volume"
].isna().any():

    raise ValueError(
        "traffic_volume contains missing values."
    )


if not np.isfinite(
    prediction_data["traffic_volume"]
).all():

    raise ValueError(
        "traffic_volume contains infinite values."
    )


# ==========================================
# 21. CREATE PREDICTION DATASET
# ==========================================

prediction_dataset = (
    TimeSeriesDataSet.from_dataset(

        training,

        prediction_data,

        predict=True,

        stop_randomization=True
    )
)


print()
print("===================================")
print("PREDICTION DATASET CREATED")
print("===================================")

print(
    "Prediction samples:",
    len(prediction_dataset)
)


# ==========================================
# 22. CREATE DATALOADER
# ==========================================

prediction_dataloader = (
    prediction_dataset.to_dataloader(

        train=False,

        batch_size=1,

        num_workers=0
    )
)


# ==========================================
# 23. MAKE PREDICTION
# ==========================================

print()
print("===================================")
print("MAKING TRAFFIC PREDICTION")
print("===================================")

prediction = tft.predict(
    prediction_dataloader
)


# ==========================================
# 24. CONVERT PREDICTION TO NUMPY
# ==========================================

prediction = (
    prediction
    .detach()
    .cpu()
    .numpy()
)


# ==========================================
# 25. GET PREDICTED VALUE
# ==========================================

predicted_value = float(
    prediction.flatten()[0]
)


# ==========================================
# 26. DISPLAY FINAL RESULT
# ==========================================

print()
print("===================================")
print("TRAFFIC VOLUME PREDICTION")
print("===================================")

print(
    "Last recorded time :",
    last_time
)

print(
    "Prediction time    :",
    future_time
)

print(
    "Predicted traffic  :",
    round(predicted_value, 2),
    "vehicles/hour"
)

print("===================================")

print()
print("Prediction completed successfully!")