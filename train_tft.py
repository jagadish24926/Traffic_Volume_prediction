import warnings

import pandas as pd
import numpy as np

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder


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

df["date_time"] = pd.to_datetime(df["date_time"])

df = df.sort_values("date_time").reset_index(drop=True)


# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

df["holiday"] = df["holiday"].fillna("None")

df["weather_main"] = df["weather_main"].fillna("Unknown")

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
    (df["date_time"] - df["date_time"].min())
    .dt.total_seconds()
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
# 8. CLEAN TARGET
# ==========================================

df = df.dropna(subset=["traffic_volume"])

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"],
    errors="coerce"
).astype(float)

df = df.dropna(subset=["traffic_volume"])


# ==========================================
# 9. DISPLAY INFORMATION
# ==========================================

print("\nProcessed dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDate range:")
print(
    df["date_time"].min(),
    "to",
    df["date_time"].max()
)

print("\nTraffic volume statistics:")
print(df["traffic_volume"].describe())


# ==========================================
# 10. TRAIN / VALIDATION SPLIT
# ==========================================

max_encoder_length = 24
max_prediction_length = 1

training_cutoff = int(
    df["time_idx"].max() * 0.8
)

print("\nTraining cutoff:", training_cutoff)


# ==========================================
# 11. CREATE TFT TRAINING DATASET
# ==========================================

training = TimeSeriesDataSet(
    df[
        lambda x: x.time_idx <= training_cutoff
    ],

    time_idx="time_idx",

    target="traffic_volume",

    group_ids=["group_id"],

    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,

    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,

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
        "group_id": NaNLabelEncoder(add_nan=True),
        "holiday": NaNLabelEncoder(add_nan=True),
        "weather_main": NaNLabelEncoder(add_nan=True),
        "weather_description": NaNLabelEncoder(add_nan=True),
    },

    add_relative_time_idx=True,

    add_target_scales=True,

    add_encoder_length=True,

    allow_missing_timesteps=True
)


# ==========================================
# 12. CREATE VALIDATION DATASET
# ==========================================

validation_data = df[
    df["time_idx"] >
    training_cutoff - max_encoder_length
].copy()

validation = TimeSeriesDataSet.from_dataset(
    training,
    validation_data,
    predict=False,
    stop_randomization=True
)


# ==========================================
# 13. PRINT TFT DATASET INFORMATION
# ==========================================

print("\n===================================")
print("TFT DATASET CREATED SUCCESSFULLY")
print("===================================")

print("Training samples:", len(training))
print("Validation samples:", len(validation))

print("\nEncoder length:", max_encoder_length)
print("Prediction length:", max_prediction_length)


# ==========================================
# 14. CREATE DATALOADERS
# ==========================================

# Increased from 64 to 128 for faster CPU training
batch_size = 128

train_dataloader = training.to_dataloader(
    train=True,
    batch_size=batch_size,
    num_workers=0
)

val_dataloader = validation.to_dataloader(
    train=False,
    batch_size=batch_size,
    num_workers=0
)


# ==========================================
# 15. IMPORT TFT MODEL
# ==========================================

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from lightning.pytorch import Trainer

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor
)


# ==========================================
# 16. CREATE TFT MODEL
# ==========================================

tft = TemporalFusionTransformer.from_dataset(
    training,

    learning_rate=0.03,

    hidden_size=16,

    attention_head_size=4,

    dropout=0.1,

    hidden_continuous_size=8,

    loss=QuantileLoss(),

    output_size=7,

    # Disable detailed logging/interpretation
    log_interval=-1,

    reduce_on_plateau_patience=3
)


print("\n===================================")
print("TFT MODEL CREATED")
print("===================================")

print(
    "Number of parameters:",
    tft.size()
)


# ==========================================
# 17. CALLBACKS
# ==========================================

early_stop_callback = EarlyStopping(
    monitor="val_loss",

    min_delta=0.001,

    patience=3,

    verbose=True,

    mode="min"
)

lr_monitor = LearningRateMonitor(
    logging_interval="epoch"
)


# ==========================================
# 18. TRAIN MODEL
# ==========================================

trainer = Trainer(

    # Reduced from 10 to 5
    max_epochs=5,

    accelerator="auto",

    devices=1,

    gradient_clip_val=0.1,

    # Skip expensive sanity validation
    num_sanity_val_steps=0,

    # Use only 25% validation batches
    # during this first training run
    limit_val_batches=0.25,

    callbacks=[
        early_stop_callback,
        lr_monitor
    ],

    enable_model_summary=True
)


print("\n===================================")
print("STARTING TFT TRAINING")
print("===================================")


trainer.fit(
    tft,

    train_dataloaders=train_dataloader,

    val_dataloaders=val_dataloader
)


# ==========================================
# 19. SAVE MODEL
# ==========================================

import os

os.makedirs(
    "models",
    exist_ok=True
)

trainer.save_checkpoint(
    "models/tft_traffic_volume.ckpt"
)


print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print(
    "Model saved to:"
)

print(
    "models/tft_traffic_volume.ckpt"
)