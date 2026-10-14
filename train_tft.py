import os
import warnings
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, EncoderNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss

warnings.filterwarnings("ignore")
pl.seed_everything(42, workers=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Metro_Interstate_Traffic_Volume_fixed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tft_traffic_volume_24h.ckpt")
MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 24
BATCH_SIZE = 64
MAX_EPOCHS = 20


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}\nCopy Metro_Interstate_Traffic_Volume_fixed.csv into the dataset folder."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"date_time", "traffic_volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {sorted(required)}")
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"]).sort_values("date_time")
    df = df.drop_duplicates(subset=["date_time"], keep="last").reset_index(drop=True)
    defaults = {
        "holiday": "None", "weather_main": "Unknown", "weather_description": "Unknown",
        "temp": 273.15, "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 0.0
    }
    for col, default in defaults.items():
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["hour"] = df["date_time"].dt.hour.astype(int)
    df["weekday"] = df["date_time"].dt.weekday.astype(int)
    df["month"] = df["date_time"].dt.month.astype(int)
    df["year"] = df["date_time"].dt.year.astype(int)
    df["time_idx"] = ((df["date_time"] - df["date_time"].min()).dt.total_seconds() / 3600).round().astype(int)
    df["group_id"] = "I94"
    for col in ["holiday", "weather_main", "weather_description", "group_id"]:
        df[col] = df[col].astype(str)
    for col in ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]).reset_index(drop=True)
    df["traffic_volume"] = df["traffic_volume"].astype(np.float32)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    if len(df) < 24 * 10:
        raise ValueError("Need more historical hourly records to train a 24-hour TFT model.")

    cutoff = int(df["time_idx"].max() * 0.8)
    train_df = df[df["time_idx"] <= cutoff].copy()
    # Include enough history before validation so validation windows can encode prior observations.
    val_df = df[df["time_idx"] > cutoff - MAX_ENCODER_LENGTH].copy()

    categorical_encoders = {
        "group_id": NaNLabelEncoder(add_nan=True),
        "holiday": NaNLabelEncoder(add_nan=True),
        "weather_main": NaNLabelEncoder(add_nan=True),
        "weather_description": NaNLabelEncoder(add_nan=True),
    }

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="traffic_volume",
        group_ids=["group_id"],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["holiday"],
        time_varying_known_reals=["hour", "weekday", "month", "year"],
        time_varying_unknown_categoricals=["weather_main", "weather_description"],
        time_varying_unknown_reals=["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"],
        categorical_encoders=categorical_encoders,
        target_normalizer=EncoderNormalizer(method="standard", center=True),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=False, stop_randomization=True
    )
    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="adam",
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=True,
    )
    print(f"Training on {len(train_df):,} rows; validating on {len(val_df):,} rows.")
    print("Encoder length: 24 hours | Forecast horizon: 24 hours")
    trainer.fit(tft, train_loader, val_loader)
    trainer.save_checkpoint(MODEL_PATH)
    print(f"\nSaved 24-hour TFT checkpoint to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
