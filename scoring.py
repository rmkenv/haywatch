"""
utils/scoring.py
Hay-cut suitability scoring for HayWatch.
Synthesizes weather, NDVI, and soil moisture into a daily 0–100% score.
"""
import pandas as pd
import numpy as np


# ─── Factor weight configuration ─────────────────────────────────────────────

WEIGHTS = {
    "precip": 0.35,      # precipitation probability is the #1 risk
    "wind": 0.15,        # wind aids drying; extreme wind is bad
    "temp": 0.15,        # temperature affects curing rate
    "soil": 0.25,        # soil moisture is critical for equipment and regrowth
    "ndvi": 0.10,        # crop maturity bonus
}

# Ideal cutting ranges
IDEAL_WIND_MIN = 8    # mph — enough for drying
IDEAL_WIND_MAX = 20   # mph — not damaging
IDEAL_TEMP_MIN = 65   # °F
IDEAL_TEMP_MAX = 88   # °F
NDVI_CUT_THRESHOLD = 0.65


def compute_daily_scores(
    daily_df: pd.DataFrame,
    soil_trend: pd.DataFrame,
    ndvi_current: float,
) -> pd.DataFrame:
    """
    Compute a daily hay-cut suitability score (0–100) for the 7-day forecast.

    Parameters
    ----------
    daily_df    : Open-Meteo daily forecast DataFrame
    soil_trend  : Daily soil moisture trend DataFrame from soil.get_soil_trend()
    ndvi_current: Most recent NDVI float value

    Returns
    -------
    DataFrame with columns:
        date, score, grade, precip_factor, wind_factor, temp_factor,
        soil_factor, ndvi_factor, recommendation
    """
    if daily_df.empty:
        return pd.DataFrame()

    records = []

    for i, row in daily_df.iterrows():
        date = row["time"]

        # ── Precipitation factor (lower precip_prob → higher factor) ──────
        precip_prob = row.get("precipitation_probability_max", 20)
        precip_factor = max(0.0, 1.0 - (precip_prob / 100.0) ** 0.8)

        # ── Wind factor (peaks in ideal range, penalizes calm or extreme) ──
        wind_max = row.get("wind_speed_10m_max", 10)
        if wind_max < IDEAL_WIND_MIN:
            wind_factor = 0.5 + 0.5 * (wind_max / IDEAL_WIND_MIN)
        elif wind_max <= IDEAL_WIND_MAX:
            wind_factor = 1.0
        else:
            # Above 20 mph — diminishing but still positive
            wind_factor = max(0.4, 1.0 - 0.02 * (wind_max - IDEAL_WIND_MAX))

        # ── Temperature factor ────────────────────────────────────────────
        temp_max = row.get("temperature_2m_max", 75)
        temp_min = row.get("temperature_2m_min", 55)
        temp_avg = (temp_max + temp_min) / 2

        if temp_avg < IDEAL_TEMP_MIN:
            temp_factor = max(0.3, temp_avg / IDEAL_TEMP_MIN)
        elif temp_avg <= IDEAL_TEMP_MAX:
            temp_factor = 1.0
        else:
            temp_factor = max(0.6, 1.0 - 0.015 * (temp_avg - IDEAL_TEMP_MAX))

        # ── Soil moisture factor ──────────────────────────────────────────
        soil_factor = 1.0
        if not soil_trend.empty and i < len(soil_trend):
            sm_pct = soil_trend.iloc[min(i, len(soil_trend) - 1)]["surface_moisture_pct"]
            # 0% = perfect, 45% = marginal, 60%+ = avoid
            soil_factor = max(0.0, 1.0 - (sm_pct / 60.0) ** 1.5)
        else:
            soil_factor = 0.75  # default moderate

        # ── NDVI maturity bonus ───────────────────────────────────────────
        # NDVI > threshold contributes positively; below threshold penalizes
        if ndvi_current >= NDVI_CUT_THRESHOLD:
            ndvi_factor = min(1.0, 0.7 + 0.3 * (ndvi_current - NDVI_CUT_THRESHOLD) / 0.15)
        else:
            ndvi_factor = max(0.0, ndvi_current / NDVI_CUT_THRESHOLD * 0.7)

        # ── Weighted composite score ──────────────────────────────────────
        raw_score = (
            WEIGHTS["precip"] * precip_factor
            + WEIGHTS["wind"] * wind_factor
            + WEIGHTS["temp"] * temp_factor
            + WEIGHTS["soil"] * soil_factor
            + WEIGHTS["ndvi"] * ndvi_factor
        )

        score = round(raw_score * 100, 1)

        records.append({
            "date": date,
            "score": score,
            "grade": _score_to_grade(score),
            "precip_factor": round(precip_factor * 100, 1),
            "wind_factor": round(wind_factor * 100, 1),
            "temp_factor": round(temp_factor * 100, 1),
            "soil_factor": round(soil_factor * 100, 1),
            "ndvi_factor": round(ndvi_factor * 100, 1),
            "precip_prob": precip_prob,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "wind_max": wind_max,
            "recommendation": _score_to_recommendation(score, precip_prob, ndvi_current),
        })

    return pd.DataFrame(records)


def _score_to_grade(score: float) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 45:
        return "Fair"
    elif score >= 25:
        return "Poor"
    else:
        return "Avoid"


def _score_to_recommendation(score: float, precip_prob: float, ndvi: float) -> str:
    if precip_prob >= 60:
        return "🌧️ Rain likely — delay cutting"
    elif ndvi < NDVI_CUT_THRESHOLD:
        return "🌱 Crop not mature — wait for NDVI > 0.65"
    elif score >= 80:
        return "✅ Prime window — cut today"
    elif score >= 65:
        return "✅ Good conditions — proceed"
    elif score >= 45:
        return "⚠️ Marginal — monitor closely"
    else:
        return "❌ Poor conditions — postpone"


def get_best_cut_window(scores_df: pd.DataFrame) -> dict:
    """Return the best day to cut within the 7-day window."""
    if scores_df.empty:
        return {}
    best_idx = scores_df["score"].idxmax()
    best = scores_df.iloc[best_idx]
    return {
        "date": best["date"],
        "score": best["score"],
        "grade": best["grade"],
        "recommendation": best["recommendation"],
    }


def compute_field_readiness(ndvi: float, surface_moisture_frac: float) -> dict:
    """
    Compute simple field-level readiness for the Field Schedule tab.
    Returns recommended cut date offset in days and a readiness flag.
    """
    surface_pct = surface_moisture_frac * 100
    ndvi_ready = ndvi >= NDVI_CUT_THRESHOLD
    moisture_ready = surface_pct < 55

    if ndvi_ready and moisture_ready:
        days_to_cut = 0
        status = "Ready Now"
    elif ndvi_ready and not moisture_ready:
        # Estimate days to dry
        days_to_cut = max(1, int((surface_pct - 55) / 5))
        status = f"Soil Too Wet ({surface_pct:.0f}%)"
    elif not ndvi_ready and moisture_ready:
        # Estimate days to NDVI threshold based on growth rate ~0.01/day
        days_to_cut = max(1, int((NDVI_CUT_THRESHOLD - ndvi) / 0.012))
        status = f"NDVI Too Low ({ndvi:.2f})"
    else:
        days_to_cut = max(
            int((NDVI_CUT_THRESHOLD - ndvi) / 0.012),
            int((surface_pct - 55) / 5)
        )
        status = "Not Ready"

    return {
        "ndvi_ready": ndvi_ready,
        "moisture_ready": moisture_ready,
        "days_to_cut": days_to_cut,
        "status": status,
    }
