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

# ─── Custom CSS — aggressively override dark mode ────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ── Force light mode on every Streamlit surface ── */
  .stApp,
  .stApp > div,
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewBlockContainer"],
  [data-testid="block-container"],
  .main,
  .main > div,
  section.main > div {
    background-color: #f8fafc !important;
    color: #1e293b !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
  }
  [data-testid="stSidebar"] * {
    color: #1e293b !important;
  }
  [data-testid="stSidebar"] .stSlider > div > div {
    background: #e2e8f0 !important;
  }

  /* All text */
  p, span, div, label, li, td, th, h1, h2, h3, h4 {
    color: #1e293b !important;
  }

  /* Override Streamlit's default dark text colors */
  .stMarkdown, .stText, .stWrite {
    color: #1e293b !important;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
  }
  [data-testid="metric-container"] label,
  [data-testid="metric-container"] [data-testid="stMetricLabel"] div {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] div,
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] div {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
  }

  /* Input widgets — force light */
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input,
  .stSelectbox select {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
  }

  /* Date inputs */
  [data-testid="stDateInput"] input {
    background: #ffffff !important;
    color: #1e293b !important;
  }

  /* Buttons */
  [data-testid="stButton"] button {
    background: #166534 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
  }
  [data-testid="stButton"] button:hover {
    background: #14532d !important;
  }

  /* Tabs */
  [data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 2px solid #e2e8f0 !important;
  }
  [data-testid="stTabs"] [role="tab"] {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    background: transparent !important;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #166534 !important;
    border-bottom: 2px solid #166534 !important;
  }

  /* Data tables */
  [data-testid="stDataFrame"],
  .stDataFrame {
    background: #ffffff !important;
  }
  [data-testid="stDataFrame"] th {
    background: #f1f5f9 !important;
    color: #374151 !important;
    font-weight: 700 !important;
  }
  [data-testid="stDataFrame"] td {
    color: #1e293b !important;
    background: #ffffff !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] summary {
    color: #1e293b !important;
    font-weight: 600 !important;
  }

  /* Info / warning boxes */
  [data-testid="stAlert"] {
    background: #f0f9ff !important;
    border: 1px solid #bae6fd !important;
    color: #0c4a6e !important;
    border-radius: 8px !important;
  }

  /* Plotly charts — force white canvas */
  .js-plotly-plot .plotly,
  .js-plotly-plot .plotly .bg {
    fill: #ffffff !important;
  }

  /* Folium iframe */
  iframe {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
  }

  /* Dividers */
  hr {
    border-color: #e2e8f0 !important;
  }

  /* Custom component classes */
  .storm-alert {
    background: #b45309 !important;
    color: #ffffff !important;
    padding: 13px 20px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 16px;
    border-left: 5px solid #78350f;
  }
  .good-banner {
    background: #166534 !important;
    color: #ffffff !important;
    padding: 13px 20px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 16px;
    border-left: 5px solid #14532d;
  }
  .section-label {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b !important;
    margin-bottom: 6px;
  }
  .score-badge { font-size: 3rem; font-weight: 800; line-height: 1; }
  .score-excellent { color: #166534 !important; }
  .score-good      { color: #15803d !important; }
  .score-fair      { color: #92400e !important; }
  .score-poor      { color: #9a3412 !important; }
  .score-avoid     { color: #7f1d1d !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Global font */
  html, body, * { font-family: 'Inter', -apple-system, sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ─── Color palette — all text values pass WCAG AA (≥4.5:1) on white ──────────
COLOR_GREEN  = "#166534"   # 8.2:1
COLOR_GREEN2 = "#15803d"   # 5.9:1
COLOR_AMBER  = "#92400e"   # 7.1:1
COLOR_RED    = "#9a3412"   # 6.8:1
COLOR_BLUE   = "#1d4ed8"   # 7.6:1
COLOR_BLUE2  = "#1e40af"   # 9.1:1
FILL_GREEN   = "#4ade80"
FILL_AMBER   = "#fb923c"
FILL_RED     = "#f87171"
FILL_BLUE    = "#60a5fa"
COLOR_CARD   = "#ffffff"
PLOTLY_TEMPLATE = "simple_white"

# ─── Plotly layout defaults (explicit light theme) ───────────────────────────
def chart_layout(**kwargs):
    defaults = dict(
        template="simple_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#212529", family="Inter, sans-serif"),
        margin=dict(t=30, b=20, l=50, r=20),
    )
    defaults.update(kwargs)
    return defaults



# ─── Sidebar ──────────────────────────────────────────────────────────────────
# ─── Session state defaults ──────────────────────────────────────────────────
if "lat" not in st.session_state:
    st.session_state["lat"] = 38.72
if "lon" not in st.session_state:
    st.session_state["lon"] = -77.80
if "field_name" not in st.session_state:
    st.session_state["field_name"] = "Home Field"

with st.sidebar:
    st.markdown(
        '<h2 style="color:#166534; font-weight:800; margin-bottom:2px;">🌾 HayWatch</h2>'
        '<p style="color:#6c757d; font-size:0.82rem; margin-top:0;">Agricultural Intelligence Platform</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("**Field Name**")
    field_name = st.text_input("", value=st.session_state["field_name"],
                                placeholder="e.g. North Meadow",
                                label_visibility="collapsed")
    st.session_state["field_name"] = field_name

    st.divider()
    st.markdown("**Active Location**")
    st.markdown(f"""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px;
                padding:10px 12px; font-size:0.83rem; color:#166534; font-weight:600;">
      📍 {st.session_state['lat']:.5f}, {st.session_state['lon']:.5f}
    </div>
    """, unsafe_allow_html=True)
    st.caption("Drop a pin on the map to update.")

    st.divider()
    buffer_acres = st.slider("Field Buffer (acres)", min_value=5, max_value=500, value=50, step=5)

    st.divider()
    st.markdown("**Analysis Window**")
    date_start = st.date_input("From", value=date.today() - timedelta(days=30))
    date_end   = st.date_input("To",   value=date.today())

    st.divider()
    refresh = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

    st.markdown("""
    <div style='font-size:0.72rem; color:#94a3b8; margin-top:12px; line-height:1.6;'>
    📡 Open-Meteo · No API keys needed
    </div>
    """, unsafe_allow_html=True)

if refresh:
    st.cache_data.clear()

lat = st.session_state["lat"]
lon = st.session_state["lon"]
lon = st.session_state["lon"]


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data(lat, lon, days_back):
    weather = fetch_weather_forecast(lat, lon)
    soil_raw = fetch_soil_moisture(lat, lon)
    ndvi_df = fetch_ndvi_timeseries(lat, lon, days_back=days_back)
    return weather, soil_raw, ndvi_df

days_back = max(7, (date_end - date_start).days)

def make_field_map(lat, lon, buffer_acres, ndvi_latest, field_name, height=420, map_key="map"):
    """
    Render an interactive Folium map with a draggable marker.
    Returns (new_lat, new_lon) if the user moved the pin, else None.
    """
    import folium
    from streamlit_folium import st_folium

    radius_m = int(np.sqrt(buffer_acres * 4046.86 / np.pi))
    ndvi_color_hex = ndvi_to_color(ndvi_latest)

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles=None)

    # Satellite base layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    # AOI buffer circle
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="#166534",
        weight=2,
        fill=True,
        fill_color=ndvi_color_hex,
        fill_opacity=0.25,
        tooltip=f"Field AOI — {buffer_acres} ac buffer",
    ).add_to(m)

    # Draggable pin
    folium.Marker(
        location=[lat, lon],
        draggable=True,
        popup=folium.Popup(
            f"<b>{field_name}</b><br>"
            f"{lat:.5f}, {lon:.5f}<br>"
            f"NDVI: {ndvi_latest:.3f} — {ndvi_status_label(ndvi_latest)}",
            max_width=200,
        ),
        icon=folium.Icon(color="green", icon="map-pin", prefix="fa"),
        tooltip="Drag me to your field",
    ).add_to(m)

    result = st_folium(m, width=None, height=height,
                       returned_objects=["last_active_drawing", "last_clicked", "center"],
                       key=map_key)

    # Capture drag-end or click
    new_coords = None
    if result:
        # last_active_drawing captures marker drag
        drawing = result.get("last_active_drawing")
        if drawing:
            geom = drawing.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [])
                if len(coords) == 2:
                    new_coords = (round(coords[1], 5), round(coords[0], 5))

        # Fallback: plain map click
        if not new_coords and result.get("last_clicked"):
            c = result["last_clicked"]
            new_coords = (round(c["lat"], 5), round(c["lng"], 5))

    return new_coords


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
    f'<p style="color:#495057; margin-top:0; margin-bottom:8px;">📍 {lat:.4f}, {lon:.4f} &nbsp;|&nbsp; '
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🛰️ NDVI Monitor",
    "💧 Soil Moisture",
    "🌤️ Forecast",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Compact field selector at top of dashboard ────────────────────────────
    # ── Field picker map ─────────────────────────────────────────────────────
    with st.expander("🗺️ Set Field Location — pan & drop pin", expanded=True):
        try:
            coords = make_field_map(lat, lon, buffer_acres, ndvi_latest,
                                    field_name, height=380, map_key="dash_map")
            if coords and (coords[0] != lat or coords[1] != lon):
                st.session_state["lat"] = coords[0]
                st.session_state["lon"] = coords[1]
                st.rerun()
            st.caption(
                f"📍 **{lat:.5f}, {lon:.5f}** — drag the green pin or click to relocate. "
                f"Switch to Street Map layer to find roads."
            )
        except ImportError:
            st.warning("streamlit-folium not installed.")

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
            <div style="background:#ffffff; border:2px solid #dee2e6; border-radius:12px;
                        padding:24px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
              <div class="section-label">Optimal Cut Window</div>
              <div style="font-size:1.3rem; font-weight:700; color:#212529; margin:6px 0;">{best_date_str}</div>
              <div class="score-badge {score_class}">{score:.0f}</div>
              <div style="font-size:0.82rem; color:#6c757d; margin-top:4px;">/ 100 suitability score</div>
              <div style="font-size:1rem; font-weight:700; color:#166534; margin-top:12px;">{grade}</div>
              <div style="font-size:0.85rem; color:#374151; margin-top:6px; font-weight:500;">{best_window.get('recommendation','')}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown("### 7-Day Score Overview")
        if not scores_df.empty:
            fig = go.Figure()
            colors = []
            for s in scores_df["score"]:
                if s >= 80:   colors.append("#166534")
                elif s >= 65: colors.append("#15803d")
                elif s >= 45: colors.append("#92400e")
                elif s >= 25: colors.append("#9a3412")
                else:         colors.append("#7f1d1d")

            fig.add_trace(go.Bar(
                x=scores_df["date"].dt.strftime("%a %-d"),
                y=scores_df["score"],
                marker_color=colors,
                text=scores_df["score"].apply(lambda x: f"{x:.0f}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.0f}/100<extra></extra>",
            ))
            fig.add_hline(y=65, line_dash="dot", line_color="#166534", annotation_text="Cut threshold")
            fig.update_layout(**chart_layout(
                yaxis=dict(range=[0, 110], title="Suitability Score", color="#212529"),
                xaxis=dict(title="", color="#212529"),
                showlegend=False,
                height=280,
                margin=dict(t=20, b=20, l=40, r=20),
            ))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # NDVI status and soil summary row
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        status_label = ndvi_status_label(ndvi_latest)
        ndvi_color = ndvi_to_color(ndvi_latest)
        st.markdown(f"""
        <div style="background:#f0fdf4; border-left:4px solid #166534;
                    border-radius:6px; padding:16px;">
          <div class="section-label">Crop Status (NDVI)</div>
          <div style="font-size:1.1rem; font-weight:700; color:#166534;">{status_label}</div>
          <div style="font-size:0.85rem; color:#374151; margin-top:6px;">
            Index: <strong style="color:#212529;">{ndvi_latest:.3f}</strong>
            &nbsp;(cut threshold ≥ {NDVI_CUT_THRESHOLD})
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        dry_color = "#166534" if days_dry == 0 else ("#92400e" if days_dry <= 3 else "#9a3412")
        days_str = "Dry Now ✓" if days_dry == 0 else (f"{days_dry} days" if days_dry < 14 else "10+ days")
        st.markdown(f"""
        <div style="background:#fffbeb; border-left:4px solid {dry_color};
                    border-radius:6px; padding:16px;">
          <div class="section-label">Curing Forecast</div>
          <div style="font-size:1.1rem; font-weight:700; color:{dry_color};">
            Surface dry in: {days_str}
          </div>
          <div style="font-size:0.85rem; color:#374151; margin-top:6px;">
            Surface moisture: <strong style="color:#212529;">{sm_pct}%</strong>
            &nbsp;(target &lt; 45%)
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        humidity = current.get('humidity', 0)
        precip_prob_now = current.get('precip_prob', 0)
        st.markdown(f"""
        <div style="background:#eff6ff; border-left:4px solid #1d4ed8;
                    border-radius:6px; padding:16px;">
          <div class="section-label">Atmosphere</div>
          <div style="font-size:1.1rem; font-weight:700; color:#1e40af;">
            RH: {humidity}% &nbsp;|&nbsp; Rain: {precip_prob_now:.0f}%
          </div>
          <div style="font-size:0.85rem; color:#374151; margin-top:6px;">
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
                color_discrete_sequence=["#166534"],
            )
            fig_ndvi.add_hline(
                y=NDVI_CUT_THRESHOLD,
                line_dash="dash",
                line_color="#92400e",
                annotation_text=f"Cut threshold ({NDVI_CUT_THRESHOLD})",
                annotation_font_color="#92400e",
            )
            fig_ndvi.add_hrect(
                y0=NDVI_CUT_THRESHOLD, y1=1.0,
                fillcolor="#166534", opacity=0.06, line_width=0,
                annotation_text="Optimal zone", annotation_position="top left",
                annotation_font_color="#166534",
            )
            fig_ndvi.update_traces(
                marker=dict(size=8, color="#166534", line=dict(width=2, color="white")),
                line=dict(width=2.5),
            )
            fig_ndvi.update_layout(**chart_layout(
                yaxis=dict(range=[0, 1], title="NDVI", color="#212529"),
                xaxis=dict(title="", color="#212529"),
                height=340,
                margin=dict(t=30, b=20, l=50, r=20),
            ))
            st.plotly_chart(fig_ndvi, use_container_width=True)
        else:
            st.info("No NDVI data available for the selected period.")

    with col_right:
        st.markdown("### 📍 Select Your Field")
        st.caption("Pan to your farm, then **drag the green pin** or click to place it. "
                   "Toggle satellite/street in the top-right corner.")
        try:
            coords = make_field_map(lat, lon, buffer_acres, ndvi_latest,
                                    field_name, height=400, map_key="ndvi_map")
            if coords and (coords[0] != lat or coords[1] != lon):
                st.session_state["lat"] = coords[0]
                st.session_state["lon"] = coords[1]
                st.rerun()
        except ImportError:
            st.warning("Install streamlit-folium: `pip install streamlit-folium`")

        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;
                    padding:10px 14px; margin-top:8px; font-size:0.83rem;">
          <strong style="color:#1e293b;">Active AOI:</strong>
          <span style="color:#166534; font-weight:700;">{lat:.5f}, {lon:.5f}</span>
          &nbsp;·&nbsp; {buffer_acres} ac buffer &nbsp;·&nbsp;
          NDVI <strong style="color:#166534;">{ndvi_latest:.3f}</strong>
          — {ndvi_status_label(ndvi_latest)}
        </div>
        <div style="font-size:0.73rem; color:#94a3b8; margin-top:6px;">
          <span style="color:#7f1d1d;">■</span> &lt;0.3 Bare &nbsp;
          <span style="color:#92400e;">■</span> 0.3–0.5 Early &nbsp;
          <span style="color:#15803d;">■</span> 0.5–0.65 Growing &nbsp;
          <span style="color:#166534;">■</span> &gt;0.65 Ready
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
                marker_color=["#1d4ed8", "#2563eb", "#3b82f6"],
                text=depth_prof["moisture_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Moisture: %{x:.1f}%<extra></extra>",
            ))
            fig_depth.add_vline(x=45, line_dash="dash", line_color="#92400e",
                                annotation_text="Cut threshold (45%)")
            fig_depth.update_layout(**chart_layout(
                xaxis=dict(range=[0, 80], title="Volumetric Water Content (%)", color="#212529"),
                yaxis=dict(title="", color="#212529"),
                height=300,
                margin=dict(t=20, b=20, l=20, r=60),
            ))
            st.plotly_chart(fig_depth, use_container_width=True)
        else:
            st.info("Soil depth profile unavailable.")

    with col_r:
        st.markdown("### 7-Day Surface Moisture Trend")
        if not soil_trend.empty:
            bar_colors = [
                "#166534" if v < 45 else ("#92400e" if v < 60 else "#9a3412")
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
            fig_soil.add_hline(y=45, line_dash="dot", line_color="#92400e",
                               annotation_text="Cut threshold")
            fig_soil.add_hline(y=60, line_dash="dot", line_color="#9a3412",
                               annotation_text="Avoid zone")
            fig_soil.update_layout(**chart_layout(
                yaxis=dict(range=[0, 85], title="Surface Moisture (%)", color="#212529"),
                xaxis=dict(title="", color="#212529"),
                height=300,
                margin=dict(t=20, b=20, l=50, r=60),
            ))
            st.plotly_chart(fig_soil, use_container_width=True)
        else:
            st.info("Soil trend data unavailable.")

    st.divider()

    # Curing summary cards
    st.markdown("### Curing Forecast")
    cc1, cc2, cc3, cc4 = st.columns(4)

    dry_color = "#166534" if days_dry == 0 else ("#92400e" if days_dry <= 3 else "#9a3412")
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
                if score >= 80:   bg, txt = "#f0fdf4", "#166534"
                elif score >= 65: bg, txt = "#dcfce7", "#15803d"
                elif score >= 45: bg, txt = "#fef3c7", "#92400e"
                else:             bg, txt = "#fee2e2", "#9a3412"

                date_label = row["date"].strftime("%a") if hasattr(row["date"], "strftime") else "—"
                day_num    = row["date"].strftime("%-d") if hasattr(row["date"], "strftime") else "—"

                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #dee2e6; border-top:4px solid {txt};
                            border-radius:8px; padding:12px 6px; text-align:center; margin-bottom:8px;
                            box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                  <div style="font-size:0.75rem; font-weight:700; color:#6c757d; text-transform:uppercase;
                              letter-spacing:0.05em;">{date_label}</div>
                  <div style="font-size:1.6rem; font-weight:800; color:{txt}; line-height:1.1;
                              margin:4px 0;">{score:.0f}</div>
                  <div style="font-size:0.7rem; font-weight:700; color:{txt}; margin-bottom:6px;">{grade}</div>
                  <div style="border-top:1px solid #e9ecef; padding-top:6px;
                              font-size:0.75rem; color:#495057; line-height:1.6;">
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
        colors_factors = ["#1d4ed8", "#2563eb", "#059669", "#166534", "#15803d"]

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
        fig_factors.update_layout(**chart_layout(
            yaxis=dict(range=[0, 110], title="Factor Score (0–100)", color="#212529"),
            xaxis=dict(title="", color="#212529"),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#212529")),
            margin=dict(t=40, b=20, l=50, r=20),
        ))
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
