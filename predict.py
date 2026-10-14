import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder, EncoderNormalizer

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Metro_Interstate_Traffic_Volume_fixed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "tft_traffic_volume_24h.ckpt")
MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 24


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\nPut Metro_Interstate_Traffic_Volume_fixed.csv inside the dataset folder."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"date_time", "traffic_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"]).sort_values("date_time")
    df = df.drop_duplicates(subset=["date_time"], keep="last").reset_index(drop=True)

    defaults = {
        "holiday": "None", "weather_main": "Unknown",
        "weather_description": "Unknown", "temp": 0.0,
        "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 0.0
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


def make_dataset(df, training=True):
    # Keep the same dataset definition in training and inference.
    return TimeSeriesDataSet(
        df,
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
        categorical_encoders={
            "group_id": NaNLabelEncoder(add_nan=True),
            "holiday": NaNLabelEncoder(add_nan=True),
            "weather_main": NaNLabelEncoder(add_nan=True),
            "weather_description": NaNLabelEncoder(add_nan=True),
        },
        target_normalizer=EncoderNormalizer(method="standard", center=True),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def weather_code_name(code):
    code = int(code or 0)
    if code == 0: return "Clear"
    if code in (1, 2, 3): return "Clouds"
    if code in (45, 48): return "Fog"
    if code in (51, 53, 55, 56, 57): return "Drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82): return "Rain"
    if code in (71, 73, 75, 77, 85, 86): return "Snow"
    if code in (95, 96, 99): return "Thunderstorm"
    return "Unknown"


def forecast_next_24_hours(holiday=False, festival=False, festival_name="", weather_payload=None):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"24-hour TFT checkpoint not found: {MODEL_PATH}. Run `python train_tft.py` first."
        )
    df = load_data()
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH, map_location="cpu")
    model.eval()

    last = df.iloc[-1].copy()
    last_time = pd.Timestamp(datetime.now()).floor("h")
    last_idx = int(df["time_idx"].max())
    future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 25)]

    # Use a contiguous hourly index after the last row in the training history.
    # The timestamps displayed are current future local hours; the learned series
    # index is extended from the end of the historical training series.
    future_rows = []
    hourly = (weather_payload or {}).get("hourly", {})
    hourly_times = hourly.get("time", [])
    hourly_temps = hourly.get("temperature_2m", [])
    hourly_rain = hourly.get("precipitation", [])
    hourly_codes = hourly.get("weather_code", [])

    for i, future_time in enumerate(future_times, start=1):
        row = last.copy()
        row["date_time"] = future_time
        row["time_idx"] = last_idx + i
        row["hour"] = future_time.hour
        row["weekday"] = future_time.weekday()
        row["month"] = future_time.month
        row["year"] = future_time.year
        row["group_id"] = "I94"
        row["holiday"] = "Holiday" if (holiday or festival) else "None"

        # Future weather comes from the Open-Meteo forecast where available.
        idx = next((j for j, value in enumerate(hourly_times) if str(value)[:13] == future_time.strftime("%Y-%m-%dT%H")), None)
        if idx is not None:
            if idx < len(hourly_temps) and hourly_temps[idx] is not None:
                # Open-Meteo temperature is Celsius; Metro dataset temp is Kelvin.
                row["temp"] = float(hourly_temps[idx]) + 273.15
            if idx < len(hourly_rain) and hourly_rain[idx] is not None:
                row["rain_1h"] = float(hourly_rain[idx])
            if idx < len(hourly_codes):
                row["weather_main"] = weather_code_name(hourly_codes[idx])
                row["weather_description"] = str(row["weather_main"])
        row["traffic_volume"] = float(last["traffic_volume"])
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([df.tail(MAX_ENCODER_LENGTH), future_df], ignore_index=True)
    combined = combined.sort_values("time_idx").drop_duplicates(["group_id", "time_idx"], keep="last").reset_index(drop=True)

    # Load the saved training dataset's encoder/normalizer metadata from checkpoint
    # by rebuilding the same dataset schema and using the checkpoint model's hparams.
    training_data = make_dataset(df.iloc[: max(MAX_ENCODER_LENGTH + 1, int(len(df) * 0.8))].copy())
    prediction_ds = TimeSeriesDataSet.from_dataset(
        training_data, combined, predict=True, stop_randomization=True
    )
    loader = prediction_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    with torch.no_grad():
        raw = model.predict(loader, mode="prediction")
    values = raw.detach().cpu().numpy().reshape(-1)
    if len(values) < 24:
        raise RuntimeError(f"Model returned {len(values)} forecast values; expected 24.")

    values = np.maximum(values[:24], 0)
    rows = []
    for i, (when, volume) in enumerate(zip(future_times, values)):
        level = "LOW" if volume < 1200 else ("MEDIUM" if volume <= 4930 else "HIGH")
        rows.append({
            "label": when.strftime("%a %I:%M %p"),
            "time": when.isoformat(timespec="minutes"),
            "volume": round(float(volume), 2),
            "level": level
        })
    return {"hourly": rows, "best_time": min(rows, key=lambda x: x["volume"])}


if __name__ == "__main__":
    result = forecast_next_24_hours()
    print("24-hour traffic forecast (historical I-94 model)")
    for row in result["hourly"]:
        print(f'{row["label"]}: {row["volume"]:.2f} vehicles/hour ({row["level"]})')
    print("Best predicted time:", result["best_time"])
