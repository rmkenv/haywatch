"""
utils/soil.py
Soil moisture processing and curing forecast for HayWatch.
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime


# Depth-layer metadata
DEPTH_LAYERS = [
    {"key": "soil_moisture_0_to_1cm",   "label": "0–1 cm",   "mid_cm": 0.5,  "depth_label": "Surface"},
    {"key": "soil_moisture_1_to_3cm",   "label": "1–3 cm",   "mid_cm": 2.0,  "depth_label": "Near-Surface"},
    {"key": "soil_moisture_3_to_9cm",   "label": "3–9 cm",   "mid_cm": 6.0,  "depth_label": "Shallow Root"},
    {"key": "soil_moisture_9_to_27cm",  "label": "9–27 cm",  "mid_cm": 18.0, "depth_label": "Root Zone"},
    {"key": "soil_moisture_27_to_81cm", "label": "27–81 cm", "mid_cm": 54.0, "depth_label": "Deep Root"},
]

# Aggregated layers for display
DISPLAY_LAYERS = [
    {
        "label": "0–10 cm (Surface)",
        "keys": ["soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm"],
        "color": "#1e6fa0",
    },
    {
        "label": "10–30 cm (Shallow)",
        "keys": ["soil_moisture_9_to_27cm"],
        "color": "#2e8fc0",
    },
    {
        "label": "30–80 cm (Root Zone)",
        "keys": ["soil_moisture_27_to_81cm"],
        "color": "#5ab0d8",
    },
]

# Threshold for surface moisture "dry enough to cut" (m³/m³)
CUT_THRESHOLD = 0.45  # ~45% volumetric water content


def get_current_soil_moisture(soil_df: pd.DataFrame) -> dict:
    """
    Return current soil moisture values per depth layer.
    soil_df: hourly DataFrame from fetch_soil_moisture()
    """
    if soil_df.empty:
        return {}

    now = pd.Timestamp.now(tz=None)
    df = soil_df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    idx = (df["time"] - now).abs().idxmin()
    row = df.iloc[idx]

    result = {}
    for layer in DEPTH_LAYERS:
        key = layer["key"]
        if key in row:
            result[layer["label"]] = round(float(row[key]), 4)

    return result


def get_depth_profile(soil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a depth-profile DataFrame for bar chart display.
    Returns cols: ['depth_label', 'layer', 'moisture_pct', 'color']
    """
    current = get_current_soil_moisture(soil_df)
    records = []

    for layer in DISPLAY_LAYERS:
        vals = []
        for key in layer["keys"]:
            # Find the matching depth key by prefix match to display layer keys
            for dl in DEPTH_LAYERS:
                if dl["key"] == key and dl["label"] in current:
                    vals.append(current[dl["label"]])

        if vals:
            avg = np.mean(vals) * 100  # convert to percentage
            records.append({
                "layer": layer["label"],
                "moisture_pct": round(avg, 1),
                "color": layer["color"],
            })

    return pd.DataFrame(records)


def get_soil_trend(soil_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Daily average surface soil moisture over the forecast period.
    Returns cols: ['date', 'surface_moisture_pct']
    """
    if soil_df.empty:
        return pd.DataFrame()

    df = soil_df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["date"] = df["time"].dt.date

    surface_cols = [l["key"] for l in DEPTH_LAYERS if l["mid_cm"] <= 6.0]
    available = [c for c in surface_cols if c in df.columns]

    if not available:
        return pd.DataFrame()

    df["surface_avg"] = df[available].mean(axis=1)
    daily = df.groupby("date")["surface_avg"].mean().reset_index()
    daily.columns = ["date", "surface_moisture"]
    daily["surface_moisture_pct"] = (daily["surface_moisture"] * 100).round(1)
    daily["date"] = pd.to_datetime(daily["date"])

    return daily.head(days)


def days_until_dry(soil_df: pd.DataFrame, threshold_pct: float = 45.0) -> int:
    """
    Estimate days until surface moisture falls below threshold.
    Returns -1 if already dry, or number of days (0–7), or 99 if never.
    """
    trend = get_soil_trend(soil_df)
    if trend.empty:
        return 3  # default optimistic estimate

    current = trend.iloc[0]["surface_moisture_pct"] if not trend.empty else 50.0

    if current <= threshold_pct:
        return 0  # already dry

    for i, row in trend.iterrows():
        if row["surface_moisture_pct"] <= threshold_pct:
            return int(i)

    # Estimate linear extrapolation
    if len(trend) >= 2:
        vals = trend["surface_moisture_pct"].values
        slope = np.polyfit(range(len(vals)), vals, 1)[0]
        if slope < 0:
            days_needed = (current - threshold_pct) / abs(slope)
            return min(int(np.ceil(days_needed)), 14)

    return 10  # unknown / very wet


def get_surface_moisture_now(soil_df: pd.DataFrame) -> float:
    """Return current surface (0–10 cm) volumetric water content as 0–1 fraction."""
    current = get_current_soil_moisture(soil_df)
    surface_keys = [l["label"] for l in DEPTH_LAYERS if l["mid_cm"] <= 6.0]
    vals = [current[k] for k in surface_keys if k in current]
    if not vals:
        return 0.35
    return float(np.mean(vals))


def soil_moisture_status(moisture_frac: float) -> str:
    """Return status label for soil moisture fraction."""
    pct = moisture_frac * 100
    if pct < 30:
        return "Dry — Ideal"
    elif pct < 45:
        return "Moist — Acceptable"
    elif pct < 60:
        return "Wet — Marginal"
    else:
        return "Saturated — Avoid"


def normalize_soil_moisture(moisture_frac: float,
                             dry_ref: float = 0.15,
                             wet_ref: float = 0.60) -> float:
    """Normalize soil moisture to 0–1 scale for scoring (0 = dry, 1 = saturated)."""
    return max(0.0, min(1.0, (moisture_frac - dry_ref) / (wet_ref - dry_ref)))
