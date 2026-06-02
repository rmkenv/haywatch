# 🌾 HayWatch — Agricultural Intelligence Dashboard

**HayWatch** is a Streamlit decision-support platform for hay farmers. It fuses modeled NDVI crop health, soil moisture profiles, and 7-day weather forecasts into a daily **hay-cut suitability score** — all from **free APIs with no API keys or accounts required**.

---

## Data Sources — All Free, No Keys

| Source | Data | Auth? |
|--------|------|-------|
| [Open-Meteo Forecast](https://open-meteo.com/) | 7-day weather + soil moisture | ❌ None |
| [Open-Meteo Archive](https://open-meteo.com/en/docs/historical-weather-api) | 30-day historical weather for NDVI model | ❌ None |

**NDVI is modeled** from real historical weather using a biophysical Beer-Lambert LAI model (Monteith 1977) — a standard precision-agriculture approach. No satellite data account needed.

---

## Quick Start

```bash
git clone https://github.com/yourorg/haywatch.git
cd haywatch
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

---

## NDVI Methodology

HayWatch derives NDVI from Open-Meteo's free historical weather archive using a biophysical crop growth model:

```
GDD_daily  = max(0, T_avg − T_base)           # Growing Degree Days (base 41°F)
Water_bal  = precipitation − ET₀              # Daily water balance (mm)
Soil_water = bucket model (0 → field capacity)
WaterStress = soil_water / field_capacity

LAI = LAI_max × sigmoid(ΣΔ GDD) × WaterStress    # Leaf Area Index
NDVI = 1 − exp(−0.45 × LAI)                       # Beer-Lambert approximation
```

This produces realistic NDVI trajectories grounded in observed temperature, precipitation, and evapotranspiration — not random noise.

---

## Project Structure

```
haywatch/
├── app.py                  # Main Streamlit application (5 tabs)
├── requirements.txt        # 7 dependencies, all free
├── README.md
└── utils/
    ├── weather.py          # Open-Meteo weather + soil moisture
    ├── ndvi.py             # Biophysical NDVI model (Open-Meteo archive)
    ├── soil.py             # Soil moisture depth profile + curing forecast
    └── scoring.py          # Hay-cut suitability score (0–100)
```

---

## Suitability Score

```
score = 0.35 × (1 − precip_prob)
      + 0.25 × soil_factor(surface_moisture)
      + 0.15 × temp_factor(temp_avg_°F)
      + 0.15 × wind_factor(wind_mph)
      + 0.10 × ndvi_maturity_factor(ndvi)
```

| Score | Grade | Action |
|-------|-------|--------|
| ≥ 80 | Excellent | Cut immediately |
| 65–80 | Good | Proceed |
| 45–65 | Fair | Monitor closely |
| < 45 | Poor/Avoid | Postpone |

---

## Deploy to Streamlit Community Cloud

1. Push to GitHub (public repo)
2. [share.streamlit.io](https://share.streamlit.io/) → New App → select `app.py`
3. No secrets or environment variables needed — all APIs are public

---

*Built with Streamlit · Open-Meteo · Beer-Lambert biophysical model*
