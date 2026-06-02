"""
app.py — HayWatch Agricultural Intelligence Dashboard
Streamlit app for hay farmer decision support.
Combines NDVI, soil moisture, and weather forecasts into a daily cut-suitability score.
"""
import sys
import os

# Ensure the repo root is on sys.path so `utils/` is importable
# regardless of the working directory Streamlit Cloud uses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

# ─── Page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="HayWatch",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Utility imports ──────────────────────────────────────────────────────────
from utils.weather import (
    fetch_weather_forecast,
    fetch_soil_moisture,
    get_current_conditions,
    storm_within_days,
    seven_day_precip_total,
)
from utils.ndvi import (
    fetch_ndvi_timeseries,
    get_latest_ndvi,
    ndvi_trend,
    ndvi_to_color,
    ndvi_status_label,
)
from utils.soil import (
    get_current_soil_moisture,
    get_depth_profile,
    get_soil_trend,
    days_until_dry,
    get_surface_moisture_now,
    soil_moisture_status,
    normalize_soil_moisture,
)
from utils.scoring import (
    compute_daily_scores,
    get_best_cut_window,
    compute_field_readiness,
    NDVI_CUT_THRESHOLD,
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #f5f2ec;
    border-right: 1px solid #e0d8cc;
  }

  /* KPI metric cards */
  [data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8e2d8;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }

  /* Alert banner */
  .storm-alert {
    background: linear-gradient(135deg, #c47b00 0%, #e09000 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 16px;
    border-left: 5px solid #8a5500;
  }

  .good-banner {
    background: linear-gradient(135deg, #3d7a2e 0%, #52a33e 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 16px;
    border-left: 5px solid #2a5520;
  }

  /* Score badge */
  .score-badge {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
  }

  /* Tab styling */
  [data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
  }
  [data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
  }

  /* Day card */
  .day-card {
    background: white;
    border-radius: 10px;
    border: 1px solid #e0d8cc;
    padding: 14px 10px;
    text-align: center;
  }

  /* Section header */
  .section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 6px;
  }

  /* Score color helpers */
  .score-excellent { color: #3d7a2e; }
  .score-good      { color: #5a9e46; }
  .score-fair      { color: #c47b00; }
  .score-poor      { color: #c44200; }
  .score-avoid     { color: #8b0000; }

  /* Folium map */
  iframe { border-radius: 10px; border: 1px solid #e0d8cc; }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Color palette ────────────────────────────────────────────────────────────
COLOR_GREEN  = "#3d7a2e"
COLOR_AMBER  = "#c47b00"
COLOR_BLUE   = "#1e6fa0"
COLOR_BG     = "#faf8f4"
COLOR_CARD   = "#ffffff"
PLOTLY_TEMPLATE = "simple_white"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h2 style="font-family:\'DM Serif Display\',serif; color:#3d7a2e; margin-bottom:2px;">🌾 HayWatch</h2>'
        '<p style="color:#888; font-size:0.82rem; margin-top:0;">Agricultural Intelligence Platform</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("**Field Settings**")
    field_name = st.text_input("Field Name", value="Home Field")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat = st.number_input("Latitude", value=38.72, format="%.4f", step=0.001)
    with col_lon:
        lon = st.number_input("Longitude", value=-77.80, format="%.4f", step=0.001)

    buffer_acres = st.slider("Field Radius (acres)", min_value=5, max_value=500, value=50, step=5)

    st.divider()
    st.markdown("**Date Range**")
    date_start = st.date_input("NDVI Start", value=date.today() - timedelta(days=30))
    date_end   = st.date_input("NDVI End",   value=date.today())

    st.divider()
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

    st.markdown("""
    <div style='font-size:0.72rem; color:#aaa; margin-top:12px; line-height:1.5;'>
    Data: Open-Meteo (weather, soil, NDVI model)<br>
    All free · No API keys required.<br>
    Scores update every hour.
    </div>
    """, unsafe_allow_html=True)

if refresh:
    st.cache_data.clear()


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data(lat, lon, days_back):
    weather = fetch_weather_forecast(lat, lon)
    soil_raw = fetch_soil_moisture(lat, lon)
    ndvi_df = fetch_ndvi_timeseries(lat, lon, days_back=days_back)
    return weather, soil_raw, ndvi_df

days_back = max(7, (date_end - date_start).days)

with st.spinner("Loading field data..."):
    weather_data, soil_raw, ndvi_df = load_all_data(lat, lon, days_back)

daily_df    = weather_data.get("daily", pd.DataFrame())
hourly_df   = weather_data.get("hourly", pd.DataFrame())
soil_df     = soil_raw.get("soil", pd.DataFrame())

current     = get_current_conditions(weather_data)
soil_trend  = get_soil_trend(soil_df)
ndvi_latest = get_latest_ndvi(lat, lon)
scores_df   = compute_daily_scores(daily_df, soil_trend, ndvi_latest)
best_window = get_best_cut_window(scores_df)
surface_sm  = get_surface_moisture_now(soil_df)
depth_prof  = get_depth_profile(soil_df)
storm_alert = storm_within_days(daily_df, days=5)
precip_7day = seven_day_precip_total(daily_df)
days_dry    = days_until_dry(soil_df)


# ─── App header ───────────────────────────────────────────────────────────────
st.markdown(
    f'<h1 style="font-family:\'DM Serif Display\',serif; color:#1a1a1a; margin-bottom:2px;">'
    f'🌾 HayWatch — {field_name}</h1>'
    f'<p style="color:#888; margin-top:0; margin-bottom:8px;">📍 {lat:.4f}, {lon:.4f} &nbsp;|&nbsp; '
    f'{date.today().strftime("%A, %B %-d, %Y")}</p>',
    unsafe_allow_html=True
)

# Storm / clear banner
if storm_alert:
    st.markdown(
        '<div class="storm-alert">⚠️ Storm System Detected Within 5 Days — '
        'Delay cutting operations and monitor forecast closely.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="good-banner">✅ No significant storm systems in the 5-day outlook — '
        'conditions look favorable for fieldwork.</div>',
        unsafe_allow_html=True
    )


# ─── Main tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🛰️ NDVI Monitor",
    "💧 Soil Moisture",
    "🌤️ Forecast",
    "📋 Field Schedule",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Current Conditions")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "🌡️ Temperature",
            f"{current.get('temp_f', '--')} °F",
            delta=None,
        )
    with c2:
        wind = current.get('wind_mph', 0)
        st.metric(
            "💨 Wind Speed",
            f"{wind} mph",
            delta="Good for drying" if 8 <= wind <= 20 else ("Too calm" if wind < 8 else "Strong"),
            delta_color="normal" if 8 <= wind <= 20 else "inverse",
        )
    with c3:
        sm_pct = round(surface_sm * 100, 1)
        st.metric(
            "💧 Surface Moisture",
            f"{sm_pct}%",
            delta=soil_moisture_status(surface_sm),
            delta_color="normal" if sm_pct < 45 else "inverse",
        )
    with c4:
        trend = ndvi_trend(ndvi_df)
        trend_arrow = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
        st.metric(
            "🌿 NDVI",
            f"{ndvi_latest:.3f}",
            delta=f"{trend_arrow} {trend.title()}",
            delta_color="normal" if ndvi_latest >= NDVI_CUT_THRESHOLD else "inverse",
        )
    with c5:
        st.metric(
            "🌧️ 7-Day Precip",
            f'{precip_7day}"',
            delta="Storm incoming!" if storm_alert else "Clear outlook",
            delta_color="inverse" if storm_alert else "normal",
        )

    st.divider()

    # Best cut window highlight
    col_score, col_detail = st.columns([1, 2])

    with col_score:
        st.markdown("### Best Cut Window")
        if best_window:
            score = best_window["score"]
            grade = best_window["grade"]
            score_class = f"score-{grade.lower()}"
            best_date_str = best_window["date"].strftime("%A, %b %-d") if hasattr(best_window["date"], "strftime") else str(best_window["date"])
            st.markdown(f"""
            <div style="background:white; border:1px solid #e0d8cc; border-radius:14px;
                        padding:24px; text-align:center; box-shadow:0 2px 12px rgba(0,0,0,0.06);">
              <div class="section-label">Optimal Day</div>
              <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; font-weight:600;
                          color:#1a1a1a; margin:4px 0;">{best_date_str}</div>
              <div class="score-badge {score_class}">{score:.0f}</div>
              <div style="font-size:0.8rem; color:#888; margin-top:2px;">/ 100 suitability score</div>
              <div style="font-size:0.95rem; font-weight:600; color:#3d7a2e; margin-top:10px;">{grade}</div>
              <div style="font-size:0.82rem; color:#555; margin-top:6px;">{best_window.get('recommendation','')}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown("### 7-Day Score Overview")
        if not scores_df.empty:
            fig = go.Figure()
            colors = []
            for s in scores_df["score"]:
                if s >= 80:   colors.append(COLOR_GREEN)
                elif s >= 65: colors.append("#5a9e46")
                elif s >= 45: colors.append(COLOR_AMBER)
                elif s >= 25: colors.append("#c44200")
                else:         colors.append("#8b0000")

            fig.add_trace(go.Bar(
                x=scores_df["date"].dt.strftime("%a %-d"),
                y=scores_df["score"],
                marker_color=colors,
                text=scores_df["score"].apply(lambda x: f"{x:.0f}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.0f}/100<extra></extra>",
            ))
            fig.add_hline(y=65, line_dash="dot", line_color=COLOR_GREEN, annotation_text="Cut threshold")
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                yaxis=dict(range=[0, 110], title="Suitability Score"),
                xaxis_title="",
                showlegend=False,
                height=280,
                margin=dict(t=20, b=20, l=40, r=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # NDVI status and soil summary row
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        status_label = ndvi_status_label(ndvi_latest)
        ndvi_color = ndvi_to_color(ndvi_latest)
        st.markdown(f"""
        <div style="background:{ndvi_color}22; border:1px solid {ndvi_color}66;
                    border-radius:10px; padding:16px;">
          <div class="section-label">Crop Status (NDVI)</div>
          <div style="font-size:1.1rem; font-weight:600; color:{ndvi_color};">{status_label}</div>
          <div style="font-size:0.82rem; color:#555; margin-top:4px;">
            Index value: <strong>{ndvi_latest:.3f}</strong> &nbsp; (threshold ≥ {NDVI_CUT_THRESHOLD})
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        dry_color = COLOR_GREEN if days_dry == 0 else (COLOR_AMBER if days_dry <= 3 else "#c44200")
        days_str = "Dry Now ✓" if days_dry == 0 else (f"{days_dry} days" if days_dry < 14 else "10+ days")
        st.markdown(f"""
        <div style="background:{dry_color}15; border:1px solid {dry_color}55;
                    border-radius:10px; padding:16px;">
          <div class="section-label">Curing Forecast</div>
          <div style="font-size:1.1rem; font-weight:600; color:{dry_color};">
            Surface dry in: {days_str}
          </div>
          <div style="font-size:0.82rem; color:#555; margin-top:4px;">
            Surface moisture: <strong>{sm_pct}%</strong> &nbsp; (target &lt; 45%)
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        humidity = current.get('humidity', 0)
        precip_prob_now = current.get('precip_prob', 0)
        st.markdown(f"""
        <div style="background:{COLOR_BLUE}10; border:1px solid {COLOR_BLUE}44;
                    border-radius:10px; padding:16px;">
          <div class="section-label">Atmosphere</div>
          <div style="font-size:1.1rem; font-weight:600; color:{COLOR_BLUE};">
            RH: {humidity}% &nbsp; | &nbsp; Rain: {precip_prob_now:.0f}%
          </div>
          <div style="font-size:0.82rem; color:#555; margin-top:4px;">
            Relative humidity and precipitation probability now
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NDVI MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 30-Day NDVI Time Series")
        if not ndvi_df.empty:
            fig_ndvi = px.line(
                ndvi_df,
                x="date",
                y="ndvi",
                markers=True,
                labels={"ndvi": "NDVI", "date": "Date"},
                color_discrete_sequence=[COLOR_GREEN],
            )
            fig_ndvi.add_hline(
                y=NDVI_CUT_THRESHOLD,
                line_dash="dash",
                line_color=COLOR_AMBER,
                annotation_text=f"Cut threshold ({NDVI_CUT_THRESHOLD})",
                annotation_font_color=COLOR_AMBER,
            )
            fig_ndvi.add_hrect(
                y0=NDVI_CUT_THRESHOLD, y1=1.0,
                fillcolor=COLOR_GREEN, opacity=0.07, line_width=0,
                annotation_text="Optimal zone", annotation_position="top left",
                annotation_font_color=COLOR_GREEN,
            )
            fig_ndvi.update_traces(
                marker=dict(size=8, color=COLOR_GREEN, line=dict(width=2, color="white")),
                line=dict(width=2.5),
            )
            fig_ndvi.update_layout(
                template=PLOTLY_TEMPLATE,
                yaxis=dict(range=[0, 1], title="NDVI"),
                xaxis_title="",
                height=340,
                margin=dict(t=30, b=20, l=50, r=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_ndvi, use_container_width=True)
        else:
            st.info("No NDVI data available for the selected period.")

    with col_right:
        st.markdown("### NDVI Field Map")
        try:
            import folium
            from streamlit_folium import st_folium

            ndvi_color_hex = ndvi_to_color(ndvi_latest)
            m = folium.Map(location=[lat, lon], zoom_start=14,
                           tiles="Esri.WorldImagery")
            folium.Circle(
                location=[lat, lon],
                radius=int(np.sqrt(buffer_acres * 4046.86 / np.pi)),
                color=ndvi_color_hex,
                fill=True,
                fill_color=ndvi_color_hex,
                fill_opacity=0.35,
                popup=folium.Popup(
                    f"<b>{field_name}</b><br>NDVI: {ndvi_latest:.3f}<br>"
                    f"Status: {ndvi_status_label(ndvi_latest)}",
                    max_width=200
                ),
                tooltip=f"NDVI: {ndvi_latest:.3f}",
            ).add_to(m)
            folium.Marker(
                [lat, lon],
                popup=f"{field_name}",
                icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
            ).add_to(m)
            st_folium(m, width=None, height=340, returned_objects=[])

        except ImportError:
            st.warning("Install `streamlit-folium` for the field map: `pip install streamlit-folium`")

        # NDVI legend
        st.markdown("""
        <div style="font-size:0.8rem; color:#666; margin-top:8px;">
          <strong>NDVI Color Scale</strong><br>
          <span style="color:#b22222;">■</span> &lt;0.3 Bare/Stressed &nbsp;
          <span style="color:#c47b00;">■</span> 0.3–0.5 Early Growth &nbsp;
          <span style="color:#5a9e46;">■</span> 0.5–0.65 Growing &nbsp;
          <span style="color:#3d7a2e;">■</span> &gt;0.65 Ready to Cut<br>
          <em>NDVI modeled from Open-Meteo weather archive — no satellite account required.</em>
        </div>
        """, unsafe_allow_html=True)

    # Data table
    st.divider()
    st.markdown("### NDVI Observation History")
    if not ndvi_df.empty:
        display_df = ndvi_df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%b %-d, %Y")
        display_df["ndvi"] = display_df["ndvi"].round(4)
        display_df["status"] = display_df["ndvi"].apply(ndvi_status_label)
        display_df.columns = ["Date", "NDVI", "Crop Status"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOIL MOISTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Current Depth Profile")
        if not depth_prof.empty:
            fig_depth = go.Figure(go.Bar(
                x=depth_prof["moisture_pct"],
                y=depth_prof["layer"],
                orientation="h",
                marker_color=[COLOR_BLUE, "#2e8fc0", "#5ab0d8"],
                text=depth_prof["moisture_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Moisture: %{x:.1f}%<extra></extra>",
            ))
            fig_depth.add_vline(x=45, line_dash="dash", line_color=COLOR_AMBER,
                                annotation_text="Cut threshold (45%)")
            fig_depth.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis=dict(range=[0, 80], title="Volumetric Water Content (%)"),
                yaxis_title="",
                height=300,
                margin=dict(t=20, b=20, l=20, r=60),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_depth, use_container_width=True)
        else:
            st.info("Soil depth profile unavailable.")

    with col_r:
        st.markdown("### 7-Day Surface Moisture Trend")
        if not soil_trend.empty:
            bar_colors = [
                COLOR_GREEN if v < 45 else (COLOR_AMBER if v < 60 else "#c44200")
                for v in soil_trend["surface_moisture_pct"]
            ]
            fig_soil = go.Figure(go.Bar(
                x=soil_trend["date"].dt.strftime("%a %-d"),
                y=soil_trend["surface_moisture_pct"],
                marker_color=bar_colors,
                text=soil_trend["surface_moisture_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Surface Moisture: %{y:.1f}%<extra></extra>",
            ))
            fig_soil.add_hline(y=45, line_dash="dot", line_color=COLOR_AMBER,
                               annotation_text="Cut threshold")
            fig_soil.add_hline(y=60, line_dash="dot", line_color="#c44200",
                               annotation_text="Avoid zone")
            fig_soil.update_layout(
                template=PLOTLY_TEMPLATE,
                yaxis=dict(range=[0, 85], title="Surface Moisture (%)"),
                xaxis_title="",
                height=300,
                margin=dict(t=20, b=20, l=50, r=60),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_soil, use_container_width=True)
        else:
            st.info("Soil trend data unavailable.")

    st.divider()

    # Curing summary cards
    st.markdown("### Curing Forecast")
    cc1, cc2, cc3, cc4 = st.columns(4)

    dry_color = COLOR_GREEN if days_dry == 0 else (COLOR_AMBER if days_dry <= 3 else "#c44200")
    days_str = "Dry Now" if days_dry == 0 else (f"{days_dry} days" if days_dry < 14 else "10+ days")

    with cc1:
        st.metric("Surface Moisture", f"{sm_pct}%",
                  delta="Dry ✓" if sm_pct < 45 else "Too Wet",
                  delta_color="normal" if sm_pct < 45 else "inverse")
    with cc2:
        st.metric("Days to Surface Dry", days_str)
    with cc3:
        sm_status = soil_moisture_status(surface_sm)
        st.metric("Soil Status", sm_status)
    with cc4:
        soil_temp_col = "soil_temperature_0cm"
        if not soil_df.empty and soil_temp_col in soil_df.columns:
            soil_temp = round(soil_df[soil_temp_col].iloc[0], 1)
        else:
            soil_temp = "—"
        st.metric("Soil Temperature", f"{soil_temp} °F" if soil_temp != "—" else "—")

    # Interpretation note
    st.info(
        "💡 **Curing Guide:** Surface moisture below 45% is ideal for cutting. "
        "Avoid cutting when soil exceeds 60% to prevent compaction and root damage. "
        "The 7-day trend accounts for Open-Meteo soil moisture forecast at 0–10 cm depth."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 7-Day Forecast & Cut Suitability")

    if not scores_df.empty and not daily_df.empty:
        # Day strip
        cols = st.columns(7)
        for i, (_, row) in enumerate(scores_df.iterrows()):
            with cols[i % 7]:
                score = row["score"]
                grade = row["grade"]
                if score >= 80:   bg, txt = "#edf7e9", COLOR_GREEN
                elif score >= 65: bg, txt = "#f2f9ee", "#5a9e46"
                elif score >= 45: bg, txt = "#fff8e6", COLOR_AMBER
                else:             bg, txt = "#fef0ec", "#c44200"

                date_label = row["date"].strftime("%a") if hasattr(row["date"], "strftime") else "—"
                day_num    = row["date"].strftime("%-d") if hasattr(row["date"], "strftime") else "—"

                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {txt}44; border-radius:10px;
                            padding:12px 6px; text-align:center; margin-bottom:8px;">
                  <div style="font-size:0.78rem; font-weight:600; color:#888;">{date_label}</div>
                  <div style="font-size:1.4rem; font-weight:700; color:{txt};">{score:.0f}</div>
                  <div style="font-size:0.7rem; color:{txt}; font-weight:600;">{grade}</div>
                  <div style="border-top:1px solid {txt}33; margin:6px 0;"></div>
                  <div style="font-size:0.72rem; color:#555;">
                    🌡️ {row['temp_max']:.0f}°/{row['temp_min']:.0f}°F<br>
                    🌧️ {row['precip_prob']:.0f}%<br>
                    💨 {row['wind_max']:.0f} mph
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Score factor breakdown
        st.markdown("### Score Factor Breakdown")
        factor_cols = ["precip_factor", "wind_factor", "temp_factor", "soil_factor", "ndvi_factor"]
        factor_labels = ["Precip (35%)", "Wind (15%)", "Temp (15%)", "Soil (25%)", "NDVI (10%)"]

        fig_factors = go.Figure()
        colors_factors = [COLOR_BLUE, "#5ab0d8", "#76c9a0", COLOR_GREEN, "#8fbc55"]

        for col, label, color in zip(factor_cols, factor_labels, colors_factors):
            if col in scores_df.columns:
                fig_factors.add_trace(go.Scatter(
                    x=scores_df["date"].dt.strftime("%a %-d"),
                    y=scores_df[col],
                    name=label,
                    mode="lines+markers",
                    line=dict(width=2, color=color),
                    marker=dict(size=6, color=color),
                    hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.0f}}%<extra></extra>",
                ))

        fig_factors.add_hline(y=65, line_dash="dot", line_color="#ccc")
        fig_factors.update_layout(
            template=PLOTLY_TEMPLATE,
            yaxis=dict(range=[0, 110], title="Factor Score (0–100)"),
            xaxis_title="",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=20, l=50, r=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_factors, use_container_width=True)

        # Recommendations table
        st.markdown("### Daily Recommendations")
        rec_df = scores_df[["date", "score", "grade", "recommendation"]].copy()
        rec_df["date"] = rec_df["date"].dt.strftime("%A, %b %-d")
        rec_df.columns = ["Date", "Score", "Grade", "Recommendation"]
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("Forecast data unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FIELD SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Field Management Schedule")

    # Editable field table
    if "field_schedule" not in st.session_state:
        today = date.today()
        st.session_state.field_schedule = pd.DataFrame({
            "Field Name": [field_name, "South Meadow", "River Bottom"],
            "Acreage": [50, 35, 80],
            "Last Cut Date": [
                (today - timedelta(days=32)).strftime("%Y-%m-%d"),
                (today - timedelta(days=28)).strftime("%Y-%m-%d"),
                (today - timedelta(days=41)).strftime("%Y-%m-%d"),
            ],
            "Notes": ["Primary field", "Tends to stay wet", "Fast draining"],
        })

    st.markdown("Edit your field roster below. Click **Calculate Readiness** to update recommendations.")
    edited = st.data_editor(
        st.session_state.field_schedule,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Field Name": st.column_config.TextColumn("Field Name", width="medium"),
            "Acreage": st.column_config.NumberColumn("Acreage (acres)", min_value=1, max_value=5000),
            "Last Cut Date": st.column_config.TextColumn("Last Cut Date (YYYY-MM-DD)"),
            "Notes": st.column_config.TextColumn("Notes"),
        },
        hide_index=True,
        key="field_editor",
    )

    if st.button("📐 Calculate Readiness", type="primary"):
        st.session_state.field_schedule = edited
        readiness = compute_field_readiness(ndvi_latest, surface_sm)

        st.markdown("### Readiness Analysis")
        for _, frow in edited.iterrows():
            fname = frow["Field Name"]
            last_cut_str = frow.get("Last Cut Date", "")
            try:
                last_cut = datetime.strptime(str(last_cut_str), "%Y-%m-%d").date()
                days_since = (date.today() - last_cut).days
            except ValueError:
                days_since = None

            days_to = readiness["days_to_cut"]
            rec_date = (date.today() + timedelta(days=days_to)).strftime("%b %-d") if days_to is not None else "TBD"
            status_color = COLOR_GREEN if readiness["status"] == "Ready Now" else COLOR_AMBER

            st.markdown(f"""
            <div style="background:white; border:1px solid #e0d8cc; border-radius:10px;
                        padding:16px 20px; margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <div style="font-family:'DM Serif Display',serif; font-size:1.05rem; color:#1a1a1a;">
                    🌾 {fname}
                  </div>
                  <div style="font-size:0.82rem; color:#888; margin-top:2px;">
                    {frow.get('Acreage', '—')} acres
                    {f' &nbsp;|&nbsp; Last cut: {days_since} days ago' if days_since else ''}
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-weight:700; color:{status_color}; font-size:1rem;">
                    {readiness['status']}
                  </div>
                  <div style="font-size:0.82rem; color:#555;">
                    Recommended cut: <strong>{rec_date}</strong>
                  </div>
                  <div style="font-size:0.75rem; color:#999; margin-top:2px;">
                    NDVI: {ndvi_latest:.3f} {'✓' if readiness['ndvi_ready'] else '✗'} &nbsp;
                    Soil: {sm_pct}% {'✓' if readiness['moisture_ready'] else '✗'}
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Scoring methodology expander
    with st.expander("ℹ️ How Suitability Scores Are Calculated"):
        st.markdown(f"""
        The **HayWatch Suitability Score** (0–100) fuses four data sources using weighted factors:

        | Factor | Weight | Notes |
        |---|---|---|
        | Precipitation Probability | 35% | Highest risk factor — rain during curing ruins hay quality |
        | Soil Moisture | 25% | Surface moisture > 55% damages soil structure and impedes equipment |
        | Temperature | 15% | Ideal range {65}–{88} °F for rapid field curing |
        | Wind Speed | 15% | Moderate winds ({8}–{20} mph) accelerate drying |
        | NDVI (Crop Maturity) | 10% | Threshold ≥ {NDVI_CUT_THRESHOLD} indicates peak nutritional value |

        **Cut recommendations:**
        - **≥ 80** — Prime window, cut immediately
        - **65–80** — Good conditions, proceed
        - **45–65** — Marginal, monitor closely
        - **< 45** — Poor conditions, postpone

        **NDVI** is modeled from Open-Meteo historical weather using a biophysical Beer-Lambert LAI model
        (Monteith 1977) — the same approach used in precision agriculture research.
        Soil moisture and weather data from Open-Meteo (free, no API key required).
        """)
