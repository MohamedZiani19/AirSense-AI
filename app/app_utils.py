"""
app_utils.py — AirSense AI

Helper functions used by app.py (the Streamlit dashboard). Keeping these
separate from app.py so the UI code stays readable and this logic can be
unit-tested or reused independently.

Covers:
- Loading deployment artifacts (model, scaler, config) produced by Notebook 8
- Generating 24-hour PM2.5 forecasts (model-type aware: LSTM/BiLSTM, SARIMA,
  or Persistence)
- Converting a PM2.5 value into an EPA AQI category
- A What-If simulation helper (only meaningful for LSTM/BiLSTM, since those
  are the only models that actually use weather inputs)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_project_paths(project_root: Path):
    """Central place to define all file paths, so app.py doesn't hardcode them."""
    return {
        "models_dir": project_root / "models",
        "processed_dir": project_root / "data" / "processed",
        "cube_path": project_root / "data" / "processed" / "airsense_cube.csv",
        "comparison_path": project_root / "models" / "model_comparison.csv",
        "config_path": project_root / "models" / "deployment_config.pkl",
        "keras_model_path": project_root / "models" / "best_model.keras",
        "pkl_model_path": project_root / "models" / "best_model.pkl",
    }


# ---------------------------------------------------------------------------
# Loading artifacts
# ---------------------------------------------------------------------------

def load_deployment_config(config_path: Path) -> dict:
    """Loads the deployment_config.pkl produced by Notebook 8."""
    with open(config_path, "rb") as f:
        return pickle.load(f)


def load_best_model(config: dict, paths: dict):
    """Loads the winning model, using the right method for its type."""
    model_type = config["model_type"]

    if model_type in ("LSTM", "BiLSTM"):
        from tensorflow import keras
        return keras.models.load_model(paths["keras_model_path"])

    elif model_type in ("SARIMA", "Persistence"):
        with open(paths["pkl_model_path"], "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unrecognized model_type '{model_type}' in deployment_config.pkl")


def load_cube(cube_path: Path) -> pd.DataFrame:
    """Loads and correctly parses the AirSense Feature Cube.

    Uses the explicit utc=True + tz_convert pattern established from
    Notebook 3 onward, since mixed DST offsets break pandas' automatic
    parse_dates.
    """
    cube = pd.read_csv(cube_path)
    cube["time"] = pd.to_datetime(cube["time"], utc=True).dt.tz_convert("America/New_York")
    return cube.sort_values("time").reset_index(drop=True)


def load_comparison_table(comparison_path: Path) -> pd.DataFrame:
    return pd.read_csv(comparison_path, index_col=0)


# ---------------------------------------------------------------------------
# Forecasting (mirrors the logic built and tested in Notebook 8)
# ---------------------------------------------------------------------------

def _encode_calendar(cube: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes Weekday/Season and converts Weekend to int, matching
    the encoding used when the LSTM/BiLSTM scaler was fit in Notebook 6."""
    encoded = pd.get_dummies(cube, columns=["Weekday", "Season"], prefix=["Weekday", "Season"])
    encoded["Weekend"] = encoded["Weekend"].astype(int)
    return encoded


def forecast_lstm_24h(model, cube: pd.DataFrame, config: dict) -> list[float]:
    """Rolling 24-hour forecast for LSTM/BiLSTM.

    Known limitation: real future weather isn't available, so weather
    columns are held at their last known values while PM2.5 and calendar
    fields (Hour, Month) are correctly advanced each step. See Notebook 8,
    Section 1, for the full explanation and a note on swapping this for a
    real weather forecast API in a production version.
    """
    input_columns = config["input_columns"]
    scaler = config["scaler"]
    window_hours = config["window_hours"]

    encoded = _encode_calendar(cube)
    history = encoded[input_columns].tail(window_hours).copy()
    last_timestamp = cube["time"].iloc[-1]

    predictions = []
    for step in range(24):
        scaled_window = scaler.transform(history[input_columns])
        X = scaled_window.reshape(1, window_hours, len(input_columns))
        pred_scaled = model.predict(X, verbose=0)[0, 0]

        pred_pm25 = pred_scaled * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
        predictions.append(float(pred_pm25))

        next_row = history.iloc[-1].copy()
        next_row["PM2.5"] = pred_pm25
        next_timestamp = last_timestamp + pd.Timedelta(hours=step + 1)
        if "Hour" in next_row.index:
            next_row["Hour"] = next_timestamp.hour
        if "Month" in next_row.index:
            next_row["Month"] = next_timestamp.month

        history = pd.concat([history.iloc[1:], next_row.to_frame().T], ignore_index=True)

    return predictions


def forecast_sarima_24h(fitted_model, steps: int = 24) -> list[float]:
    forecast = fitted_model.forecast(steps=steps)
    return forecast.values.tolist()


def forecast_persistence_24h(last_known_pm25: float, steps: int = 24) -> list[float]:
    return [last_known_pm25] * steps


def generate_forecast(model, config: dict, cube: pd.DataFrame) -> tuple[list[pd.Timestamp], list[float]]:
    """Single entry point app.py calls — dispatches to the right forecast
    function based on model_type and returns (timestamps, values)."""
    model_type = config["model_type"]
    last_time = cube["time"].iloc[-1]

    if model_type in ("LSTM", "BiLSTM"):
        values = forecast_lstm_24h(model, cube, config)
    elif model_type == "SARIMA":
        values = forecast_sarima_24h(model)
    elif model_type == "Persistence":
        values = forecast_persistence_24h(cube["PM2.5"].iloc[-1])
    else:
        raise ValueError(f"Unrecognized model_type '{model_type}'")

    timestamps = [last_time + pd.Timedelta(hours=h + 1) for h in range(24)]
    return timestamps, values


def supports_what_if(config: dict) -> bool:
    """Only LSTM/BiLSTM actually use weather as an input — SARIMA and
    Persistence are univariate, so a What-If weather simulation has nothing
    to act on for those model types."""
    return config["model_type"] in ("LSTM", "BiLSTM")


def what_if_next_hour(model, config: dict, cube: pd.DataFrame,
                       temperature: float = None, humidity: float = None,
                       wind_speed: float = None) -> float:
    """Recomputes the next-hour PM2.5 prediction with one or more weather
    values overridden — only valid when supports_what_if(config) is True.
    """
    if not supports_what_if(config):
        raise ValueError("What-If simulation isn't supported for this model type "
                          "(SARIMA and Persistence don't use weather inputs).")

    input_columns = config["input_columns"]
    scaler = config["scaler"]
    window_hours = config["window_hours"]

    encoded = _encode_calendar(cube)
    history = encoded[input_columns].tail(window_hours).copy()

    # Override the most recent hour's weather with the What-If values
    if temperature is not None and "Temperature" in history.columns:
        history.iloc[-1, history.columns.get_loc("Temperature")] = temperature
    if humidity is not None and "Relative Humidity" in history.columns:
        history.iloc[-1, history.columns.get_loc("Relative Humidity")] = humidity
    if wind_speed is not None and "Wind Speed" in history.columns:
        history.iloc[-1, history.columns.get_loc("Wind Speed")] = wind_speed

    scaled_window = scaler.transform(history[input_columns])
    X = scaled_window.reshape(1, window_hours, len(input_columns))
    pred_scaled = model.predict(X, verbose=0)[0, 0]

    return float(pred_scaled * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1])


# ---------------------------------------------------------------------------
# AQI category (EPA PM2.5 breakpoints)
# ---------------------------------------------------------------------------

_AQI_BREAKPOINTS = [
    # (pm25_low, pm25_high, category, color, risk_description)
    (0.0, 12.0, "Good", "#00E400",
     "Air quality is satisfactory, poses little or no risk."),
    (12.1, 35.4, "Moderate", "#FFFF00",
     "Acceptable; may pose a moderate risk for a very small number of unusually sensitive people."),
    (35.5, 55.4, "Unhealthy for Sensitive Groups", "#FF7E00",
     "Sensitive groups (children, elderly, people with respiratory/heart conditions) may experience effects."),
    (55.5, 150.4, "Unhealthy", "#FF0000",
     "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."),
    (150.5, 250.4, "Very Unhealthy", "#8F3F97",
     "Health alert: everyone may experience more serious health effects."),
    (250.5, float("inf"), "Hazardous", "#7E0023",
     "Health warning of emergency conditions; the entire population is more likely to be affected."),
]


def get_aqi_category(pm25: float) -> dict:
    """Maps a raw hourly PM2.5 value (μg/m³) to an EPA AQI category.

    Note: EPA's official AQI uses a 24-hour NowCast average, not a single
    hourly reading — this is a reasonable simplification for a dashboard
    showing live/forecasted hourly values, documented here so it isn't
    mistaken for the official calculation method.
    """
    pm25 = max(0.0, float(pm25))
    for low, high, category, color, description in _AQI_BREAKPOINTS:
        if low <= pm25 <= high:
            return {
                "category": category,
                "color": color,
                "description": description,
                "pm25": pm25,
            }
    # Fallback (shouldn't be reached given the inf upper bound)
    return {"category": "Unknown", "color": "#888888", "description": "", "pm25": pm25}
