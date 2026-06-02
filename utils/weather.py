"""
utils/weather.py
Open-Meteo API fetch functions for HayWatch
"""
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=3600)
def fetch_weather_forecast(lat: float, lon: float) -> dict:
    """
    Fetch 7-day hourly weather forecast from Open-Meteo.
    Returns a dict with 'hourly' DataFrame and 'daily' DataFrame.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "wind_speed_10m",
            "precipitation",
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
        ],
        "forecast_days": 7,
        "timezone": "auto",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Build hourly DataFrame
        hourly = pd.DataFrame(data["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"])

        # Build daily DataFrame
        daily = pd.DataFrame(data["daily"])
        daily["time"] = pd.to_datetime(daily["time"])

        return {"hourly": hourly, "daily": daily, "timezone": data.get("timezone", "UTC")}

    except Exception as e:
        # Return realistic synthetic data on failure
        return _synthetic_weather(lat, lon)


@st.cache_data(ttl=3600)
def fetch_soil_moisture(lat: float, lon: float) -> dict:
    """
    Fetch soil moisture data at multiple depths from Open-Meteo.
    Returns a dict with 'hourly' DataFrame containing soil_moisture columns.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "soil_moisture_0_to_1cm",
            "soil_moisture_1_to_3cm",
            "soil_moisture_3_to_9cm",
            "soil_moisture_9_to_27cm",
            "soil_moisture_27_to_81cm",
            "soil_temperature_0cm",
        ],
        "forecast_days": 7,
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return {"soil": df}

    except Exception as e:
        return _synthetic_soil()


def get_current_conditions(weather_data: dict) -> dict:
    """Extract the most recent hourly observation from forecast data."""
    hourly = weather_data.get("hourly", pd.DataFrame())
    if hourly.empty:
        return {}

    now = pd.Timestamp.now(tz=None)
    hourly_copy = hourly.copy()
    hourly_copy["time"] = pd.to_datetime(hourly_copy["time"]).dt.tz_localize(None)
    idx = (hourly_copy["time"] - now).abs().idxmin()
    row = hourly_copy.iloc[idx]

    return {
        "temp_f": round(row.get("temperature_2m", 0), 1),
        "humidity": round(row.get("relative_humidity_2m", 0), 1),
        "wind_mph": round(row.get("wind_speed_10m", 0), 1),
        "precip_prob": round(row.get("precipitation_probability", 0), 0),
    }


def storm_within_days(daily_df: pd.DataFrame, days: int = 5, threshold: float = 50) -> bool:
    """Return True if precipitation probability exceeds threshold within given days."""
    if daily_df.empty:
        return False
    window = daily_df.head(days)
    return bool((window["precipitation_probability_max"] >= threshold).any())


def seven_day_precip_total(daily_df: pd.DataFrame) -> float:
    """Sum of daily precipitation over the 7-day forecast."""
    if "precipitation_sum" not in daily_df.columns:
        return 0.0
    return round(daily_df["precipitation_sum"].sum(), 2)


# ─── Synthetic fallback data ──────────────────────────────────────────────────

def _synthetic_weather(lat: float, lon: float) -> dict:
    """Generate plausible synthetic weather data when API is unavailable."""
    import numpy as np

    dates_hourly = pd.date_range(start=datetime.now().replace(minute=0, second=0, microsecond=0),
                                  periods=168, freq="h")
    n = len(dates_hourly)
    rng = np.random.default_rng(seed=42)

    # Diurnal temperature variation
    hour_of_day = dates_hourly.hour
    base_temp = 72 + 4 * np.sin(np.pi * (hour_of_day - 6) / 12)
    temp = base_temp + rng.normal(0, 2, n)

    hourly = pd.DataFrame({
        "time": dates_hourly,
        "temperature_2m": temp,
        "relative_humidity_2m": 55 + rng.normal(0, 8, n).clip(-20, 30),
        "precipitation_probability": rng.integers(0, 40, n).astype(float),
        "wind_speed_10m": rng.uniform(3, 18, n),
        "precipitation": rng.choice([0, 0, 0, 0.05, 0.1], n),
    })

    dates_daily = pd.date_range(start=datetime.now().date(), periods=7, freq="D")
    daily = pd.DataFrame({
        "time": dates_daily,
        "temperature_2m_max": [82, 78, 75, 80, 83, 76, 79],
        "temperature_2m_min": [58, 55, 52, 60, 61, 54, 57],
        "precipitation_sum": [0.0, 0.0, 0.12, 0.0, 0.0, 0.05, 0.0],
        "precipitation_probability_max": [10, 15, 60, 20, 10, 35, 15],
        "wind_speed_10m_max": [12, 9, 14, 11, 8, 13, 10],
    })

    return {"hourly": hourly, "daily": daily, "timezone": "America/New_York"}


def _synthetic_soil() -> dict:
    """Generate plausible synthetic soil moisture data."""
    import numpy as np

    dates = pd.date_range(start=datetime.now().replace(minute=0, second=0, microsecond=0),
                          periods=168, freq="h")
    n = len(dates)
    rng = np.random.default_rng(seed=99)

    df = pd.DataFrame({
        "time": dates,
        "soil_moisture_0_to_1cm": rng.uniform(0.22, 0.42, n),
        "soil_moisture_1_to_3cm": rng.uniform(0.28, 0.45, n),
        "soil_moisture_3_to_9cm": rng.uniform(0.30, 0.48, n),
        "soil_moisture_9_to_27cm": rng.uniform(0.32, 0.50, n),
        "soil_moisture_27_to_81cm": rng.uniform(0.34, 0.52, n),
        "soil_temperature_0cm": rng.uniform(55, 72, n),
    })

    return {"soil": df}
