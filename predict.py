import os
import warnings

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.data import (
    NaNLabelEncoder,
    EncoderNormalizer
)


# ==========================================
# SETTINGS
# ==========================================

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Metro_Interstate_Traffic_Volume_fixed.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tft_traffic_volume.ckpt"
)

MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 1


# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df["date_time"] = pd.to_datetime(
        df["date_time"]
    )

    df = (
        df.sort_values("date_time")
        .reset_index(drop=True)
    )

    # Missing values
    df["holiday"] = df["holiday"].fillna("None")

    df["weather_main"] = (
        df["weather_main"].fillna("Unknown")
    )

    df["weather_description"] = (
        df["weather_description"].fillna("Unknown")
    )

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["weekday"] = df["date_time"].dt.weekday
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year

    # TFT time index
    df["time_idx"] = (
        (
            df["date_time"] - df["date_time"].min()
        ).dt.total_seconds() / 3600
    ).round().astype(int)

    # Group
    df["group_id"] = "I94"

    # Categorical columns
    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description",
        "group_id"
    ]

    for column in categorical_columns:
        df[column] = df[column].astype(str)

    # Traffic volume
    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["traffic_volume"])
        .reset_index(drop=True)
    )

    df["traffic_volume"] = (
        df["traffic_volume"].astype(np.float32)
    )

    return df


# ==========================================
# CREATE TFT DATASET
# ==========================================

def create_training_dataset(df):

    training_cutoff = int(
        df["time_idx"].max() * 0.8
    )

    training = TimeSeriesDataSet(

        df[
            df["time_idx"] <= training_cutoff
        ],

        time_idx="time_idx",

        target="traffic_volume",

        group_ids=["group_id"],

        min_encoder_length=MAX_ENCODER_LENGTH,

        max_encoder_length=MAX_ENCODER_LENGTH,

        min_prediction_length=MAX_PREDICTION_LENGTH,

        max_prediction_length=MAX_PREDICTION_LENGTH,

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

        target_normalizer=EncoderNormalizer(
            method="standard",
            center=True
        ),

        add_relative_time_idx=True,

        add_target_scales=True,

        add_encoder_length=True,

        allow_missing_timesteps=True
    )

    return training


# ==========================================
# LOAD EVERYTHING ONCE
# ==========================================

print()
print("==========================================")
print(" INITIALIZING TFT TRAFFIC SYSTEM")
print("==========================================")

print("Loading dataset...")

df = load_data()

print("Dataset loaded:", df.shape)

print("Creating TFT dataset...")

training = create_training_dataset(df)

print("Loading trained TFT model...")

tft = TemporalFusionTransformer.load_from_checkpoint(
    MODEL_PATH,
    map_location="cpu"
)

tft.eval()

print("TFT model loaded successfully.")

print("==========================================")
print(" SYSTEM READY")
print("==========================================")
print()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_traffic(holiday="No Holiday"):

    # --------------------------------------
    # LAST OBSERVED RECORD
    # --------------------------------------

    last_row = df.iloc[-1].copy()

    last_time = last_row["date_time"]

    last_time_idx = int(
        last_row["time_idx"]
    )

    # --------------------------------------
    # FUTURE TIME
    # --------------------------------------

    future_time = (
        last_time + pd.Timedelta(hours=1)
    )

    future_time_idx = (
        last_time_idx + 1
    )

    # --------------------------------------
    # GET LAST 24 HOURS
    # --------------------------------------

    history_start_idx = (
        future_time_idx -
        MAX_ENCODER_LENGTH
    )

    encoder_data = df[
        (df["time_idx"] >= history_start_idx)
        &
        (df["time_idx"] <= last_time_idx)
    ].copy()

    # --------------------------------------
    # FUTURE ROW
    # --------------------------------------

    future_row = last_row.copy()

    future_row["date_time"] = future_time

    future_row["time_idx"] = (
        future_time_idx
    )

    future_row["hour"] = (
        future_time.hour
    )

    future_row["weekday"] = (
        future_time.weekday()
    )

    future_row["month"] = (
        future_time.month
    )

    future_row["year"] = (
        future_time.year
    )

    future_row["group_id"] = "I94"

    # --------------------------------------
    # HOLIDAY FROM WEBSITE
    # --------------------------------------

    if holiday == "Holiday":
        future_row["holiday"] = "Holiday"
    else:
        future_row["holiday"] = "None"

    # --------------------------------------
    # TARGET PLACEHOLDER
    # --------------------------------------

    future_row["traffic_volume"] = (
        last_row["traffic_volume"]
    )

    # --------------------------------------
    # COMBINE DATA
    # --------------------------------------

    prediction_data = pd.concat(
        [
            encoder_data,
            pd.DataFrame([future_row])
        ],
        ignore_index=True
    )

    prediction_data = (
        prediction_data
        .sort_values("time_idx")
        .drop_duplicates(
            subset=[
                "group_id",
                "time_idx"
            ],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------
    # CREATE PREDICTION DATASET
    # --------------------------------------

    prediction_dataset = (
        TimeSeriesDataSet.from_dataset(
            training,
            prediction_data,
            predict=True,
            stop_randomization=True
        )
    )

    # --------------------------------------
    # DATALOADER
    # --------------------------------------

    prediction_dataloader = (
        prediction_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0
        )
    )

    # --------------------------------------
    # PREDICT
    # --------------------------------------

    with torch.no_grad():

        prediction = tft.predict(
            prediction_dataloader
        )

    # --------------------------------------
    # GET VALUE
    # --------------------------------------

    predicted_value = float(
        prediction.detach()
        .cpu()
        .numpy()
        .flatten()[0]
    )

    return {
        "prediction": predicted_value,
        "last_time": last_time,
        "future_time": future_time
    }


# ==========================================
# TEST WHEN RUN DIRECTLY
# ==========================================

if __name__ == "__main__":

    result = predict_traffic(
        "No Holiday"
    )

    print()
    print("==========================================")
    print(" TRAFFIC VOLUME PREDICTION")
    print("==========================================")

    print(
        "Last recorded time :",
        result["last_time"]
    )

    print(
        "Prediction time    :",
        result["future_time"]
    )

    print(
        "Predicted traffic  :",
        round(result["prediction"], 2),
        "vehicles/hour"
    )

    print("==========================================")
    print()