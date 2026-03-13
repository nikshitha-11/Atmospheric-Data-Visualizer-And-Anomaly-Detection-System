import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import fetch_atmospheric_data
from anomaly_detector import detect_anomalies
from forecaster import forecast_temperature
from utils import calculate_aqi_category, calculate_risk_level, save_to_csv, load_historical_data

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atmospheric Data Visualizer & Anomaly Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Definitions ─────────────────────────────────────────────────────────
THEMES = {
    "🌑 Dark (Default)": {
        "bg":           "#0e1117",
        "sidebar_bg":   "linear-gradient(180deg,#131722,#1a1f2e)",
        "card_bg":      "linear-gradient(135deg,#1e2130,#252a3a)",
        "card_border":  "#2d3448",
        "text":         "#e0e6f0",
        "subtext":      "#8892a4",
        "accent":       "#00d4ff",
        "accent2":      "#0088cc",
        "accent3":      "#00ff88",
        "hr":           "#2d3448",
        "plotly_tpl":   "plotly_dark",
        "plot_bg":      "rgba(30,33,48,0.8)",
    },
    "☀️ Light": {
        "bg":           "#f5f7fa",
        "sidebar_bg":   "linear-gradient(180deg,#e8edf5,#dde3ee)",
        "card_bg":      "linear-gradient(135deg,#ffffff,#f0f4fb)",
        "card_border":  "#c8d0e0",
        "card_border_solid": "#c8d0e0",
        "text":         "#1a2035",
        "subtext":      "#5a6478",
        "accent":       "#0066cc",
        "accent2":      "#004499",
        "accent3":      "#00aa55",
        "hr":           "#c8d0e0",
        "plotly_tpl":   "plotly_white",
        "plot_bg":      "rgba(248,250,255,0.9)",
        "tick_color":   "#1a2035",
        "grid_color":   "#dde3ee",
    },
    "🌊 Ocean Blue": {
        "bg":           "#040d1a",
        "sidebar_bg":   "linear-gradient(180deg,#050f20,#071428)",
        "card_bg":      "linear-gradient(135deg,#071428,#0a1d38)",
        "card_border":  "#0d2a50",
        "text":         "#c8e6ff",
        "subtext":      "#6a9cc0",
        "accent":       "#00aaff",
        "accent2":      "#0077cc",
        "accent3":      "#00ffcc",
        "hr":           "#0d2a50",
        "plotly_tpl":   "plotly_dark",
        "plot_bg":      "rgba(7,20,40,0.9)",
        "card_border_solid": "#0d2a50",
        "tick_color":   "#c8e6ff",
        "grid_color":   "#0d2a50",
    },
    "🌿 Forest Green": {
        "bg":           "#050e08",
        "sidebar_bg":   "linear-gradient(180deg,#061009,#08160b)",
        "card_bg":      "linear-gradient(135deg,#0a1e0d,#0e2611)",
        "card_border":  "#1a4020",
        "text":         "#c8f0d0",
        "subtext":      "#6aaa80",
        "accent":       "#00dd66",
        "accent2":      "#00aa44",
        "accent3":      "#88ff44",
        "hr":           "#1a4020",
        "plotly_tpl":   "plotly_dark",
        "plot_bg":      "rgba(10,30,13,0.9)",
        "card_border_solid": "#1a4020",
        "tick_color":   "#c8f0d0",
        "grid_color":   "#1a4020",
    },
    "🌸 Sunset Pink": {
        "bg":           "#1a0a12",
        "sidebar_bg":   "linear-gradient(180deg,#1f0c16,#250e1a)",
        "card_bg":      "linear-gradient(135deg,#2a0f1e,#321228)",
        "card_border":  "#5a1e38",
        "text":         "#ffe0f0",
        "subtext":      "#cc88aa",
        "accent":       "#ff6699",
        "accent2":      "#cc3366",
        "accent3":      "#ffaa44",
        "hr":           "#5a1e38",
        "plotly_tpl":   "plotly_dark",
        "plot_bg":      "rgba(42,15,30,0.9)",
        "card_border_solid": "#5a1e38",
        "tick_color":   "#ffe0f0",
        "grid_color":   "#5a1e38",
    },
    "🟣 Violet Storm": {
        "bg":           "#0d0618",
        "sidebar_bg":   "linear-gradient(180deg,#100720,#140828)",
        "card_bg":      "linear-gradient(135deg,#180a2e,#1e0d38)",
        "card_border":  "#3a1860",
        "text":         "#e8d8ff",
        "subtext":      "#9970cc",
        "accent":       "#bb66ff",
        "accent2":      "#8833cc",
        "accent3":      "#ff66bb",
        "hr":           "#3a1860",
        "plotly_tpl":   "plotly_dark",
        "plot_bg":      "rgba(24,10,46,0.9)",
        "card_border_solid": "#3a1860",
        "tick_color":   "#e8d8ff",
        "grid_color":   "#3a1860",
    },
}

# ── Location Groups (Cities + Rural Districts) ───────────────────────────────
LOCATIONS = {
    "🌆 Major Cities — Asia": [
        "Tokyo", "Delhi", "Mumbai", "Shanghai", "Beijing",
        "Seoul", "Bangkok", "Jakarta", "Kolkata", "Osaka",
        "Kuala Lumpur", "Singapore", "Manila", "Karachi", "Dhaka",
    ],
    "🌆 Major Cities — Europe": [
        "London", "Paris", "Berlin", "Madrid", "Rome",
        "Amsterdam", "Stockholm", "Moscow", "Istanbul", "Warsaw",
        "Vienna", "Prague", "Zurich", "Brussels", "Lisbon",
    ],
    "🌆 Major Cities — Americas": [
        "New York", "Los Angeles", "Chicago", "Houston", "Toronto",
        "Vancouver", "Sao Paulo", "Mexico City", "Buenos Aires", "Lima",
        "Bogota", "Miami", "Montreal", "Santiago", "Caracas",
    ],
    "🌆 Major Cities — Africa & Middle East": [
        "Cairo", "Lagos", "Nairobi", "Cape Town", "Casablanca",
        "Addis Ababa", "Accra", "Johannesburg", "Dubai", "Riyadh",
        "Doha", "Abu Dhabi", "Tehran", "Baghdad", "Beirut",
    ],
    "🌆 Major Cities — Oceania": [
        "Sydney", "Melbourne", "Brisbane", "Auckland", "Perth",
        "Adelaide", "Wellington", "Canberra", "Christchurch", "Hobart",
    ],
    "🌾 Rural Districts — India": [
        "Wayanad", "Coorg", "Chikkamagaluru", "Kodaikanal", "Munnar",
        "Spiti Valley", "Lahaul", "Ziro", "Majuli", "Mawsynram",
        "Sundarbans", "Rann of Kutch", "Chambal", "Thar Desert", "Araku Valley",
    ],
    "🌾 Rural Districts — Asia": [
        "Banaue", "Sagada", "Mrauk U", "Sapa", "Tam Coc",
        "Khao Yai", "Pai", "Luang Namtha", "Bandarban", "Sylhet",
        "Nuwara Eliya", "Ella", "Pokhara", "Ilam", "Trongsa",
    ],
    "🌾 Rural Districts — Europe": [
        "Cotswolds", "Dordogne", "Tuscany", "Transylvania", "Lofoten",
        "Faroe Islands", "Azores", "Douro Valley", "Alsace", "Black Forest",
        "Mosel Valley", "Dalmatian Coast", "Wachau", "Vikos Gorge", "Snowdonia",
    ],
    "🌾 Rural Districts — Americas": [
        "Patagonia", "Amazon Basin", "Oaxaca Valley", "Vermont", "Appalachian",
        "Yellowstone", "Banff", "Prince Edward Island", "Atacama", "Pantanal",
        "Mato Grosso", "Lake District Chile", "Galapagos", "Yucatan", "Caribou",
    ],
    "🌾 Rural Districts — Africa & Oceania": [
        "Serengeti", "Okavango Delta", "Drakensberg", "Sahel Region", "Nile Delta",
        "Sossusvlei", "Bwindi", "Kafue", "Kimberley Region", "Daintree",
        "Outback", "Fiordland", "Marlborough", "Arnhem Land", "Nullarbor",
    ],
}

# ── Session State Init ────────────────────────────────────────────────────────
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "🌑 Dark (Default)"
if "cached_df" not in st.session_state:
    st.session_state.cached_df = None
if "cached_forecast" not in st.session_state:
    st.session_state.cached_forecast = None
if "cached_city" not in st.session_state:
    st.session_state.cached_city = ""
if "anomaly_count" not in st.session_state:
    st.session_state.anomaly_count = 0
if "total_records" not in st.session_state:
    st.session_state.total_records = 0

# ── Active Theme ──────────────────────────────────────────────────────────────
T = THEMES[st.session_state.theme_name]

# ── Dynamic CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* App background */
    .stApp {{ background-color: {T['bg']}; }}
    .main {{ background-color: {T['bg']}; }}

    /* Metric cards */
    .metric-card {{
        background: {T['card_bg']};
        border: 1px solid {T['card_border']};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        margin-bottom: 8px;
    }}
    .metric-value {{ font-size: 1.7rem; font-weight: 700; color: {T['accent']}; }}
    .metric-label {{ font-size: 0.85rem; color: {T['subtext']}; margin-top: 4px; }}

    /* Section headers (Summary Dashboard, Interactive Visualizations) */
    .section-header {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: {T['text']} !important;
        border-left: 5px solid {T['accent']};
        padding-left: 14px;
        margin: 24px 0 16px 0;
    }}

    /* Tab labels — bigger and bolder */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: {T['subtext']} !important;
        padding: 10px 20px !important;
        border-radius: 8px 8px 0 0 !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {T['accent']} !important;
        border-bottom-color: {T['accent']} !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }}

    /* ══ FETCH BUTTON — force visible on ALL themes ══ */
    /* Target every possible Streamlit button selector */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] div.stButton > button {{
        background: {T['accent']} !important;
        background-color: {T['accent']} !important;
        color: #ffffff !important;
        border: 2px solid {T['accent2']} !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
        text-shadow: none !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {T['accent2']} !important;
        background-color: {T['accent2']} !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        transform: translateY(-2px) !important;
    }}
    /* Force ALL child text nodes white */
    [data-testid="stSidebar"] button *,
    [data-testid="stSidebar"] .stButton button *,
    [data-testid="stSidebar"] .stButton > button * {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
    }}

    /* ── Sidebar: background ── */
    div[data-testid="stSidebar"],
    div[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {{
        background: {T['sidebar_bg']} !important;
        background-color: {T['bg']} !important;
    }}

    /* ── Sidebar: all text ── */
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stRadio label span,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] b,
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span {{
        color: {T['text']} !important;
        font-size: 0.92rem !important;
    }}

    /* ── Sidebar: text input box ── */
    div[data-testid="stSidebar"] input {{
        color: {T['text']} !important;
        background-color: {T['card_border']} !important;
        border: 1px solid {T['accent']} !important;
        border-radius: 6px !important;
    }}
    div[data-testid="stSidebar"] input::placeholder {{
        color: {T['subtext']} !important;
        opacity: 1 !important;
    }}

    /* ── Sidebar: remaining selectboxes (city/region) ── */
    div[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        border: 1px solid {T['accent']} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stSidebar"] [data-baseweb="select"] span,
    div[data-testid="stSidebar"] [data-baseweb="select"] div {{
        color: {T['text']} !important;
        -webkit-text-fill-color: {T['text']} !important;
    }}

    /* ── Sidebar: slider accent value ── */
    div[data-testid="stSidebar"] .stSlider p {{
        color: {T['accent']} !important;
        font-weight: 600 !important;
    }}

    /* ── Sidebar: radio text ── */
    div[data-testid="stSidebar"] .stRadio p {{
        color: {T['text']} !important;
    }}

    /* ── Sidebar: dividers ── */
    div[data-testid="stSidebar"] hr {{
        border-color: {T['card_border']} !important;
        opacity: 0.6 !important;
    }}

    /* Main headings and text — strong specificity for all themes */
    h1, h2, h3, h4, h5, h6 {{
        color: {T['text']} !important;
    }}
    .stApp p, .stApp span, .stApp div {{
        color: {T['text']};
    }}
    /* Section header override for full visibility on all themes */
    .section-header {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: {T['text']} !important;
        background-color: transparent !important;
        border-left: 5px solid {T['accent']} !important;
        padding: 10px 0 10px 14px !important;
        margin: 24px 0 16px 0 !important;
        display: block !important;
    }}
    /* Air quality banner text */
    .stMarkdownContainer span {{
        color: {T['text']};
    }}
    /* Plotly chart backgrounds match theme */
    .js-plotly-plot .plotly .bg {{
        fill: {T['bg']} !important;
    }}
    /* Streamlit metric labels */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] {{
        color: {T['text']} !important;
    }}
    /* Tab content area background */
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: transparent !important;
    }}
    /* Expander text */
    .streamlit-expanderHeader {{
        color: {T['text']} !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }}

    /* Download buttons and all main-area buttons */
    .stDownloadButton > button {{
        background: {T['accent']} !important;
        color: #ffffff !important;
        border: 2px solid {T['accent2']} !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 8px 20px !important;
        transition: all 0.3s !important;
    }}
    .stDownloadButton > button:hover {{
        background: {T['accent2']} !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }}
    .stDownloadButton > button p,
    .stDownloadButton > button span {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{T['accent']};margin:0 0 8px 0'>🔧 Configuration</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # City Selection FIRST
    st.markdown(f"<b style='color:{T['text']}'>🏙️ City Selection</b>", unsafe_allow_html=True)
    city_mode = st.radio(
        "City pick mode",
        ["✏️ Type a city", "📍 Browse cities"],
        label_visibility="collapsed",
    )

    if city_mode == "✏️ Type a city":
        city_name = st.text_input(
            "Enter City",
            value="",
            placeholder="e.g. New York",
        )
    else:
        region = st.selectbox(
            "Region",
            options=list(LOCATIONS.keys()),
            label_visibility="collapsed",
        )
        city_pick = st.selectbox(
            "City",
            options=["— Select a city —"] + LOCATIONS[region],
            label_visibility="collapsed",
        )
        city_name = "" if city_pick == "— Select a city —" else city_pick

    st.markdown("<br>", unsafe_allow_html=True)
    # Inline style injected directly before button to guarantee override
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] button {{
        background: {T["accent"]} !important;
        background-color: {T["accent"]} !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
        border: 2px solid {T["accent2"]} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stSidebar"] button p,
    [data-testid="stSidebar"] button span,
    [data-testid="stSidebar"] button div,
    [data-testid="stSidebar"] button * {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    fetch_btn = st.button("🚀 Fetch & Analyze Data", use_container_width=True)

    st.markdown("---")

    st.markdown(f"<b style='color:{T['text']}'>📅 Data Range</b>", unsafe_allow_html=True)
    days_back = st.slider("Historical Days", min_value=1, max_value=7, value=3)

    st.markdown(f"<b style='color:{T['text']}'>⚙️ Model Settings</b>", unsafe_allow_html=True)
    contamination = st.slider(
        "Anomaly Sensitivity", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Higher = more anomalies detected"
    )
    forecast_hours = st.slider("Forecast Hours", min_value=6, max_value=48, value=24)

    st.markdown("---")

    # Theme Picker — single selectbox styled exactly like fetch button
    cur     = st.session_state.theme_name
    options = list(THEMES.keys())
    cur_idx = options.index(cur)

    st.markdown(f"<b style='color:{T['text']}'>🎨 Theme</b>", unsafe_allow_html=True)

    # Inject style directly before the selectbox (same trick as fetch button)
    st.markdown(f"""
    <style>
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {{
        background-color: {T['accent']} !important;
        border: 2px solid {T['accent2']} !important;
        border-radius: 10px !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div *,
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] input {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {{
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    chosen_theme = st.selectbox(
        "Theme", options=options, index=cur_idx,
        label_visibility="collapsed", key="theme_selectbox"
    )
    if chosen_theme != st.session_state.theme_name:
        st.session_state.theme_name = chosen_theme
        st.rerun()

    st.markdown(f"""
    <div style='font-size:0.78rem; text-align:center; color:{T['subtext']}; margin-top:12px;'>
        <p>📡 Data: Open-Meteo APIs</p>
        <p>🤖 ML: Isolation Forest</p>
        <p>📈 Forecast: Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='font-size:2.5rem; font-weight:800; color:{T['accent']}; margin:0; line-height:1.2;'>
        🌍 Atmospheric Data Visualizer and Anomaly Detection System
    </h1>
    <p style='color:{T['subtext']}; font-size:1.05rem; margin-top:8px;'>
        Real-time Monitoring &nbsp;·&nbsp; Anomaly Detection &nbsp;·&nbsp; Temperature Forecasting
    </p>
</div>
<hr style='border:none; border-top:1px solid {T['hr']}; margin-bottom:20px;'>
""", unsafe_allow_html=True)

# ── Main Logic ────────────────────────────────────────────────────────────────
if fetch_btn:
    if not city_name or city_name.strip() == "":
        st.warning("⚠️ Please enter or select a city name before fetching data.")
        st.stop()

    with st.spinner(f"🌐 Fetching atmospheric data for **{city_name}**..."):
        df = fetch_atmospheric_data(city_name, days_back=days_back)

    if df is None or df.empty:
        st.error("❌ Failed to fetch data. Please check the city name and try again.")
        st.stop()

    df = calculate_aqi_category(df)
    df = calculate_risk_level(df)
    save_to_csv(df, city_name)

    with st.spinner("🤖 Running anomaly detection..."):
        df = detect_anomalies(df, contamination=contamination)

    with st.spinner("📈 Forecasting temperatures..."):
        forecast_df = forecast_temperature(df, hours=forecast_hours)

    # Cache results so theme switch doesn't clear data
    st.session_state.cached_df       = df
    st.session_state.cached_forecast = forecast_df
    st.session_state.cached_city     = city_name

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📊 Summary Dashboard — {city_name}</div>', unsafe_allow_html=True)

    latest        = df.iloc[-1]
    total_records = len(df)
    anomaly_count = int((df['anomaly'] == -1).sum()) if 'anomaly' in df.columns else 0
    anomaly_pct   = round(anomaly_count / total_records * 100, 1)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, icon, val, label in [
        (c1, "🌡️", f"{latest['temperature_2m']:.1f}°C", "Temperature"),
        (c2, "💧", f"{latest['relativehumidity_2m']:.0f}%", "Humidity"),
        (c3, "🌬️", f"{latest['pressure_msl']:.0f} hPa", "Pressure"),
        (c4, "💨", f"{latest['windspeed_10m']:.1f} km/h", "Wind Speed"),
        (c5, "📋", f"{total_records}", "Total Records"),
        (c6, "⚠️", f"{anomaly_pct}%", f"Anomaly Rate ({anomaly_count})"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:1.5rem'>{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Air Quality Banner ────────────────────────────────────────────────────
    aqi_cat  = latest.get('aqi_category', 'N/A')
    risk_lvl = latest.get('risk_level', 'N/A')
    pm25     = latest.get('pm2_5', 0)
    pm10     = latest.get('pm10', 0)

    risk_color = {"Low Risk": "#00c851", "Moderate Risk": "#ffbb33", "High Risk": "#ff4444"}.get(risk_lvl, T['subtext'])
    aqi_color  = {
        "Good": "#00c851", "Moderate": "#adff2f",
        "Unhealthy for Sensitive Groups": "#ffbb33",
        "Unhealthy": "#ff8800", "Very Unhealthy": "#cc0000", "Hazardous": "#7d0023"
    }.get(aqi_cat, T['subtext'])

    st.markdown(f"""
    <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                border-radius:12px; padding:20px; margin-bottom:20px;'>
        <h4 style='color:{T['text']}; margin:0 0 15px 0;'>🌫️ Current Air Quality — {city_name}</h4>
        <div style='display:flex; gap:30px; flex-wrap:wrap;'>
            <div><span style='color:{T['subtext']}'>AQI Category:</span>
                 <span style='color:{aqi_color}; font-weight:700; margin-left:8px;'>{aqi_cat}</span></div>
            <div><span style='color:{T['subtext']}'>Risk Level:</span>
                 <span style='color:{risk_color}; font-weight:700; margin-left:8px;'>{risk_lvl}</span></div>
            <div><span style='color:{T['subtext']}'>PM2.5:</span>
                 <span style='color:{T['accent']}; font-weight:700; margin-left:8px;'>{pm25:.1f} µg/m³</span></div>
            <div><span style='color:{T['subtext']}'>PM10:</span>
                 <span style='color:{T['accent']}; font-weight:700; margin-left:8px;'>{pm10:.1f} µg/m³</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌡️ Temperature", "🌫️ Air Quality",
        "📊 AQI Distribution", "⚠️ Anomalies", "🔮 Forecast"
    ])

    PTpl = T['plotly_tpl']
    PBg  = T['plot_bg']
    PApp = "rgba(0,0,0,0)"

    # Shared chart font/color settings — ensures titles & axes are always visible
    _tick = T.get('tick_color', T['text'])
    _grid = T.get('grid_color', T.get('card_border_solid', '#444444'))
    chart_font = dict(color=_tick, size=13)
    title_font = dict(color=_tick, size=16, family="sans-serif")
    axis_style = dict(
        color=_tick, tickfont=dict(color=_tick, size=12),
        title_font=dict(color=_tick, size=13),
        showgrid=True, gridcolor=_grid, zeroline=False,
        linecolor=_grid, showline=True
    )

    with tab1:
        normal    = df[df['anomaly'] != -1] if 'anomaly' in df.columns else df
        anomalies = df[df['anomaly'] == -1] if 'anomaly' in df.columns else pd.DataFrame()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal['time'], y=normal['temperature_2m'],
            mode='markers', name='Normal',
            marker=dict(color=T['accent'], size=5, opacity=0.7)))
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['time'], y=anomalies['temperature_2m'],
                mode='markers', name='Anomaly',
                marker=dict(color='#ff4444', size=10, symbol='x', line=dict(width=2))))
        fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'].rolling(6).mean(),
            mode='lines', name='6h Moving Avg', line=dict(color=T['accent3'], width=2)))
        fig.update_layout(title=f"Temperature Anomaly Scatter — {city_name}",
            template=PTpl, paper_bgcolor=PApp, plot_bgcolor=PBg,
            xaxis_title="Time", yaxis_title="Temperature (°C)", height=420,
            font=chart_font, title_font=title_font,
            xaxis=dict(**axis_style),
            yaxis=dict(**axis_style),
            legend=dict(font=dict(color=T['text'])))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Relative Humidity (%)", "Atmospheric Pressure (hPa)"))
        fig2.add_trace(go.Scatter(x=df['time'], y=df['relativehumidity_2m'],
            fill='tozeroy', fillcolor="rgba(0,136,204,0.15)",
            line=dict(color=T['accent2']), name='Humidity'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df['time'], y=df['pressure_msl'],
            line=dict(color='#ff8c00'), name='Pressure'), row=2, col=1)
        fig2.update_layout(template=PTpl, paper_bgcolor=PApp,
                           plot_bgcolor=PBg, height=380, showlegend=False,
                           font=chart_font,
                           xaxis=dict(**axis_style), yaxis=dict(**axis_style),
                           xaxis2=dict(**axis_style), yaxis2=dict(**axis_style))
        fig2.update_annotations(font=dict(color=T['text'], size=14))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['time'], y=df['pm2_5'],
            mode='lines+markers', name='PM2.5',
            line=dict(color='#ff6b6b', width=2), marker=dict(size=4)))
        fig3.add_trace(go.Scatter(x=df['time'], y=df['pm10'],
            mode='lines+markers', name='PM10',
            line=dict(color='#ffd93d', width=2), marker=dict(size=4)))
        fig3.add_hline(y=15, line_dash="dash", line_color="#ff4444",
                       annotation_text="WHO PM2.5 Annual Limit (15 µg/m³)")
        fig3.add_hline(y=45, line_dash="dash", line_color="#ffbb33",
                       annotation_text="WHO PM10 Annual Limit (45 µg/m³)")
        fig3.update_layout(title=f"PM2.5 & PM10 Trend — {city_name}",
            template=PTpl, paper_bgcolor=PApp, plot_bgcolor=PBg,
            xaxis_title="Time", yaxis_title="Concentration (µg/m³)", height=420,
            font=chart_font, title_font=title_font,
            xaxis=dict(**axis_style),
            yaxis=dict(**axis_style),
            legend=dict(font=dict(color=T['text'])))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.area(df, x='time', y='windspeed_10m',
            title=f"Wind Speed — {city_name}", template=PTpl,
            color_discrete_sequence=[T['accent3']])
        fig4.update_layout(paper_bgcolor=PApp, plot_bgcolor=PBg,
                           xaxis_title="Time", yaxis_title="Wind Speed (km/h)", height=300,
                           font=chart_font, title_font=title_font,
                           xaxis=dict(**axis_style),
                           yaxis=dict(**axis_style))
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        aqi_counts = df['aqi_category'].value_counts().reset_index()
        aqi_counts.columns = ['AQI Category', 'Count']
        color_map = {
            "Good": "#00c851", "Moderate": "#adff2f",
            "Unhealthy for Sensitive Groups": "#ffbb33",
            "Unhealthy": "#ff8800", "Very Unhealthy": "#cc0000", "Hazardous": "#7d0023"
        }
        col_a, col_b = st.columns(2)
        with col_a:
            fig5 = px.bar(aqi_counts, x='AQI Category', y='Count',
                title="AQI Category Distribution", template=PTpl,
                color='AQI Category', color_discrete_map=color_map)
            fig5.update_layout(paper_bgcolor=PApp, plot_bgcolor=PBg, height=380,
                               font=chart_font, title_font=title_font,
                               xaxis=dict(**axis_style), yaxis=dict(**axis_style),
                               legend=dict(font=dict(color=T['text'])))
            st.plotly_chart(fig5, use_container_width=True)
        with col_b:
            fig6 = px.pie(aqi_counts, names='AQI Category', values='Count',
                title="AQI Category Share", template=PTpl,
                color='AQI Category', color_discrete_map=color_map, hole=0.4)
            fig6.update_layout(paper_bgcolor=PApp, height=380,
                               font=chart_font, title_font=title_font,
                               legend=dict(font=dict(color=T['text'])))
            st.plotly_chart(fig6, use_container_width=True)

        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            risk_cm = {"Low Risk": "#00c851", "Moderate Risk": "#ffbb33", "High Risk": "#ff4444"}
            fig7 = px.bar(risk_counts, x='Risk Level', y='Count',
                title="Risk Level Distribution", template=PTpl,
                color='Risk Level', color_discrete_map=risk_cm)
            fig7.update_layout(paper_bgcolor=PApp, plot_bgcolor=PBg, height=320,
                               font=chart_font, title_font=title_font,
                               xaxis=dict(**axis_style), yaxis=dict(**axis_style),
                               legend=dict(font=dict(color=T['text'])))
            st.plotly_chart(fig7, use_container_width=True)

    with tab4:
        if 'anomaly' in df.columns:
            anomaly_df = df[df['anomaly'] == -1].copy()
            st.markdown(f"**{len(anomaly_df)} anomalies** detected out of {total_records} records ({anomaly_pct}%)")

            if not anomaly_df.empty:
                feature_cols = ['temperature_2m','relativehumidity_2m','pressure_msl','windspeed_10m','pm2_5','pm10']
                avail_cols   = [c for c in feature_cols if c in df.columns]
                corr = df[avail_cols].corr()
                fig8 = px.imshow(corr, title="Feature Correlation Heatmap",
                    template=PTpl, color_continuous_scale='RdBu_r', aspect='auto')
                fig8.update_layout(paper_bgcolor=PApp, height=380,
                                   font=chart_font, title_font=title_font,
                                   xaxis=dict(color=T['text']),
                                   yaxis=dict(color=T['text']))
                st.plotly_chart(fig8, use_container_width=True)

                display_cols = ['time','temperature_2m','relativehumidity_2m','pressure_msl',
                                'windspeed_10m','pm2_5','pm10','aqi_category','risk_level']
                display_cols = [c for c in display_cols if c in anomaly_df.columns]
                st.dataframe(anomaly_df[display_cols].reset_index(drop=True), use_container_width=True)
                st.download_button("⬇️ Download Anomaly Report (CSV)",
                    anomaly_df.to_csv(index=False),
                    file_name=f"anomaly_report_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv")
            else:
                st.success("✅ No anomalies detected in the current dataset!")

    with tab5:
        if forecast_df is not None and not forecast_df.empty:
            fig9 = go.Figure()
            fig9.add_trace(go.Scatter(x=df['time'].tail(48), y=df['temperature_2m'].tail(48),
                mode='lines', name='Historical', line=dict(color=T['accent'], width=2)))
            fig9.add_trace(go.Scatter(x=forecast_df['time'], y=forecast_df['temperature_forecast'],
                mode='lines', name='Forecast',
                line=dict(color='#ff6b6b', width=2, dash='dash')))

            if 'temp_upper' in forecast_df.columns:
                fig9.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['time'], forecast_df['time'][::-1]]),
                    y=pd.concat([forecast_df['temp_upper'], forecast_df['temp_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(255,107,107,0.12)',
                    line=dict(color='rgba(255,255,255,0)'), name='Uncertainty Band'))

            fig9.update_layout(
                title=f"Temperature Forecast — Next {forecast_hours}h — {city_name}",
                template=PTpl, paper_bgcolor=PApp, plot_bgcolor=PBg,
                xaxis_title="Time", yaxis_title="Temperature (°C)", height=420,
                font=chart_font, title_font=title_font,
                xaxis=dict(**axis_style),
                yaxis=dict(**axis_style),
                legend=dict(font=dict(color=T['text'])))
            st.plotly_chart(fig9, use_container_width=True)

            fc1, fc2, fc3 = st.columns(3)
            with fc1: st.metric("Min Forecast Temp", f"{forecast_df['temperature_forecast'].min():.1f}°C")
            with fc2: st.metric("Max Forecast Temp", f"{forecast_df['temperature_forecast'].max():.1f}°C")
            with fc3: st.metric("Avg Forecast Temp", f"{forecast_df['temperature_forecast'].mean():.1f}°C")

            st.download_button("⬇️ Download Forecast Data (CSV)",
                forecast_df.to_csv(index=False),
                file_name=f"forecast_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        else:
            st.warning("⚠️ Not enough data to generate forecast. Try fetching more days.")

    with st.expander("📄 View Raw Data"):
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download Full Dataset (CSV)",
            df.to_csv(index=False),
            file_name=f"atmospheric_data_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv")

# ── Show cached data on theme switch (no re-fetch needed) ────────────────────
elif st.session_state.cached_df is not None:
    df          = st.session_state.cached_df
    forecast_df = st.session_state.cached_forecast
    city_name   = st.session_state.cached_city

    # Re-render full dashboard from cached data
    st.markdown(f'<div class="section-header">📊 Summary Dashboard — {city_name}</div>', unsafe_allow_html=True)

    latest        = df.iloc[-1]
    total_records = len(df)
    anomaly_count = int((df['anomaly'] == -1).sum()) if 'anomaly' in df.columns else 0
    anomaly_pct   = round(anomaly_count / total_records * 100, 1)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, icon, val, label in [
        (c1, "🌡️", f"{latest['temperature_2m']:.1f}°C", "Temperature"),
        (c2, "💧", f"{latest['relativehumidity_2m']:.0f}%", "Humidity"),
        (c3, "🌬️", f"{latest['pressure_msl']:.0f} hPa", "Pressure"),
        (c4, "💨", f"{latest['windspeed_10m']:.1f} km/h", "Wind Speed"),
        (c5, "📋", f"{total_records}", "Total Records"),
        (c6, "⚠️", f"{anomaly_pct}%", f"Anomaly Rate ({anomaly_count})"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:1.5rem'>{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    aqi_cat  = latest.get('aqi_category', 'N/A')
    risk_lvl = latest.get('risk_level', 'N/A')
    pm25     = latest.get('pm2_5', 0)
    pm10     = latest.get('pm10', 0)
    risk_color = {"Low Risk": "#00c851", "Moderate Risk": "#ffbb33", "High Risk": "#ff4444"}.get(risk_lvl, T['subtext'])
    aqi_color  = {"Good": "#00c851", "Moderate": "#adff2f", "Unhealthy for Sensitive Groups": "#ffbb33",
                  "Unhealthy": "#ff8800", "Very Unhealthy": "#cc0000", "Hazardous": "#7d0023"}.get(aqi_cat, T['subtext'])

    st.markdown(f"""
    <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                border-radius:12px; padding:20px; margin-bottom:20px;'>
        <h4 style='color:{T['text']}; margin:0 0 15px 0;'>🌫️ Current Air Quality — {city_name}</h4>
        <div style='display:flex; gap:30px; flex-wrap:wrap;'>
            <div><span style='color:{T['subtext']}'>AQI Category:</span>
                 <span style='color:{aqi_color}; font-weight:700; margin-left:8px;'>{aqi_cat}</span></div>
            <div><span style='color:{T['subtext']}'>Risk Level:</span>
                 <span style='color:{risk_color}; font-weight:700; margin-left:8px;'>{risk_lvl}</span></div>
            <div><span style='color:{T['subtext']}'>PM2.5:</span>
                 <span style='color:{T['accent']}; font-weight:700; margin-left:8px;'>{pm25:.1f} µg/m³</span></div>
            <div><span style='color:{T['subtext']}'>PM10:</span>
                 <span style='color:{T['accent']}; font-weight:700; margin-left:8px;'>{pm10:.1f} µg/m³</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)
    st.info("✅ Showing cached data. Change settings and click **Fetch & Analyze Data** to refresh.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌡️ Temperature", "🌫️ Air Quality",
        "📊 AQI Distribution", "⚠️ Anomalies", "🔮 Forecast"
    ])
    PTpl = T['plotly_tpl']; PBg = T['plot_bg']; PApp = "rgba(0,0,0,0)"
    _tick = T.get('tick_color', T['text'])
    _grid = T.get('grid_color', T.get('card_border_solid', '#444444'))
    chart_font = dict(color=_tick, size=13)
    title_font = dict(color=_tick, size=16, family="sans-serif")
    axis_style = dict(
        color=_tick, tickfont=dict(color=_tick, size=12),
        title_font=dict(color=_tick, size=13),
        showgrid=True, gridcolor=_grid, zeroline=False,
        linecolor=_grid, showline=True
    )

    with tab1:
        normal    = df[df['anomaly'] != -1] if 'anomaly' in df.columns else df
        anomalies = df[df['anomaly'] == -1] if 'anomaly' in df.columns else pd.DataFrame()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal['time'], y=normal['temperature_2m'], mode='markers', name='Normal', marker=dict(color=T['accent'], size=5, opacity=0.7)))
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['time'], y=anomalies['temperature_2m'], mode='markers', name='Anomaly', marker=dict(color='#ff4444', size=10, symbol='x', line=dict(width=2))))
        fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'].rolling(6).mean(), mode='lines', name='6h Moving Avg', line=dict(color=T['accent3'], width=2)))
        fig.update_layout(title=f"Temperature Anomaly Scatter — {city_name}", template=PTpl, paper_bgcolor=PApp, plot_bgcolor=PBg, xaxis_title="Time", yaxis_title="Temperature (°C)", height=420, font=chart_font, title_font=title_font, xaxis=dict(**axis_style), yaxis=dict(**axis_style), legend=dict(font=dict(color=T['text'])))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['time'], y=df['pm2_5'], mode='lines+markers', name='PM2.5', line=dict(color='#ff6b6b', width=2), marker=dict(size=4)))
        fig3.add_trace(go.Scatter(x=df['time'], y=df['pm10'], mode='lines+markers', name='PM10', line=dict(color='#ffd93d', width=2), marker=dict(size=4)))
        fig3.update_layout(title=f"PM2.5 & PM10 — {city_name}", template=PTpl, paper_bgcolor=PApp, plot_bgcolor=PBg, height=420, font=chart_font, title_font=title_font, xaxis=dict(**axis_style), yaxis=dict(**axis_style), legend=dict(font=dict(color=T['text'])))
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        aqi_counts = df['aqi_category'].value_counts().reset_index()
        aqi_counts.columns = ['AQI Category', 'Count']
        color_map = {"Good": "#00c851", "Moderate": "#adff2f", "Unhealthy for Sensitive Groups": "#ffbb33", "Unhealthy": "#ff8800", "Very Unhealthy": "#cc0000", "Hazardous": "#7d0023"}
        fig5 = px.bar(aqi_counts, x='AQI Category', y='Count', title="AQI Distribution", template=PTpl, color='AQI Category', color_discrete_map=color_map)
        fig5.update_layout(paper_bgcolor=PApp, plot_bgcolor=PBg, height=380, font=chart_font, title_font=title_font, xaxis=dict(**axis_style), yaxis=dict(**axis_style))
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        if 'anomaly' in df.columns:
            anomaly_df = df[df['anomaly'] == -1].copy()
            st.markdown(f"**{len(anomaly_df)} anomalies** detected out of {total_records} records ({anomaly_pct}%)")
            if not anomaly_df.empty:
                display_cols = [c for c in ['time','temperature_2m','relativehumidity_2m','pressure_msl','windspeed_10m','pm2_5','pm10','aqi_category','risk_level'] if c in anomaly_df.columns]
                st.dataframe(anomaly_df[display_cols].reset_index(drop=True), use_container_width=True)

    with tab5:
        if forecast_df is not None and not forecast_df.empty:
            fig9 = go.Figure()
            fig9.add_trace(go.Scatter(x=df['time'].tail(48), y=df['temperature_2m'].tail(48), mode='lines', name='Historical', line=dict(color=T['accent'], width=2)))
            fig9.add_trace(go.Scatter(x=forecast_df['time'], y=forecast_df['temperature_forecast'], mode='lines', name='Forecast', line=dict(color='#ff6b6b', width=2, dash='dash')))
            fig9.update_layout(title=f"Temperature Forecast — {city_name}", template=PTpl, paper_bgcolor=PApp, plot_bgcolor=PBg, height=420, font=chart_font, title_font=title_font, xaxis=dict(**axis_style), yaxis=dict(**axis_style), legend=dict(font=dict(color=T['text'])))
            st.plotly_chart(fig9, use_container_width=True)

# ── Welcome Screen ────────────────────────────────────────────────────────────
else:
    st.markdown(f"""
    <div style='text-align:center; padding:50px 20px;'>
        <div style='font-size:5rem; margin-bottom:20px;'>🌍</div>
        <h2 style='color:{T['text']}; font-weight:700;'>Welcome to Atmospheric Data Visualizer</h2>
        <p style='color:{T['subtext']}; font-size:1.1rem; max-width:620px; margin:0 auto 30px auto;'>
            Choose a city from the sidebar using the
            <strong style='color:{T['accent']};'>Popular Cities</strong> picker or type your own,
            then click <strong style='color:{T['accent']};'>Fetch &amp; Analyze Data</strong>
            to start monitoring real-time atmospheric conditions.
        </p>
        <div style='display:flex; gap:18px; justify-content:center; flex-wrap:wrap; margin-top:30px;'>
            <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; padding:25px; width:175px;'>
                <div style='font-size:2rem'>📡</div>
                <div style='color:{T['accent']}; font-weight:600; margin-top:10px;'>Real-time Data</div>
                <div style='color:{T['subtext']}; font-size:0.85rem; margin-top:5px;'>Open-Meteo API</div>
            </div>
            <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; padding:25px; width:175px;'>
                <div style='font-size:2rem'>🤖</div>
                <div style='color:{T['accent']}; font-weight:600; margin-top:10px;'>ML Detection</div>
                <div style='color:{T['subtext']}; font-size:0.85rem; margin-top:5px;'>Isolation Forest</div>
            </div>
            <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; padding:25px; width:175px;'>
                <div style='font-size:2rem'>📈</div>
                <div style='color:{T['accent']}; font-weight:600; margin-top:10px;'>Forecasting</div>
                <div style='color:{T['subtext']}; font-size:0.85rem; margin-top:5px;'>Random Forest</div>
            </div>
            <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; padding:25px; width:175px;'>
                <div style='font-size:2rem'>🌫️</div>
                <div style='color:{T['accent']}; font-weight:600; margin-top:10px;'>Air Quality</div>
                <div style='color:{T['subtext']}; font-size:0.85rem; margin-top:5px;'>AQI &amp; PM2.5/PM10</div>
            </div>
            <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; padding:25px; width:175px;'>
                <div style='font-size:2rem'>⚠️</div>
                <div style='color:{T['accent']}; font-weight:600; margin-top:10px;'>Risk Levels</div>
                <div style='color:{T['subtext']}; font-size:0.85rem; margin-top:5px;'>Environmental Risk</div>
            </div>
            <div style='background:{T['card_bg']}; border:1px solid {T['card_border']};
                        border-radius:12px; padding:25px; width:175px;'>
                <div style='font-size:2rem'>🎨</div>
                <div style='color:{T['accent']}; font-weight:600; margin-top:10px;'>6 Themes</div>
                <div style='color:{T['subtext']}; font-size:0.85rem; margin-top:5px;'>Dark, Light &amp; Colors</div>
            </div>
        </div>
        <p style='color:{T['subtext']}; font-size:0.9rem; margin-top:40px;'>
            🌆 50+ cities across 5 regions — or type any city worldwide
        </p>
    </div>
    """, unsafe_allow_html=True)
