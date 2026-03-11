import streamlit as st
import pandas as pd
import pydeck as pdk
from prophet import Prophet
import plotly.express as px

# ---------------------- Load CSS ----------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

# ---------------------- Hero Section ----------------------
st.markdown("""
<div style="text-align:center; padding:20px; background: linear-gradient(90deg,#00c9a7,#00ffd5);
            border-radius:15px; color:#000000; font-size:28px; font-weight:bold;">
    🌆 Atmospheric Data Visualizer & Anomaly Detection System
</div>
""", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ffffff;'>Real-time pollution monitoring & 24-hour predictions</p>", unsafe_allow_html=True)

# ---------------------- Sidebar Navigation ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "24-Hour Forecast", "Alerts", "History"])

# ---------------------- Load Data ----------------------
if "data" not in st.session_state:
    st.warning("⚠ Load city data first from Dashboard")
    st.stop()

df = st.session_state["data"]
lat = st.session_state["lat"]
lon = st.session_state["lon"]

# ---------------------- Helper Functions ----------------------
def get_aqi_label(pm):
    if pm <= 50:
        return "Good", "#06d6a0"
    elif pm <= 100:
        return "Moderate", "#ffd166"
    elif pm <= 150:
        return "Unhealthy for Sensitive Groups", "#f07b3f"
    elif pm <= 200:
        return "Unhealthy", "#ff4d6d"
    else:
        return "Hazardous", "#9d0191"

# ---------------------- Pages ----------------------
if page == "Dashboard":
    st.subheader("📍 Current Pollution Map")
    pm = df["pm2_5"].iloc[-1]
    map_data = pd.DataFrame({"lat":[lat], "lon":[lon], "pm":[pm]})

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=8, pitch=50),
        layers=[pdk.Layer("HeatmapLayer", data=map_data, get_position='[lon, lat]', get_weight="pm", radiusPixels=60)]
    ))

    st.metric("Current PM2.5 Level", pm)

elif page == "24-Hour Forecast":
    st.subheader("Next 24-Hour Pollution Prediction")

    # Prepare data for Prophet
    df_forecast = df[['timestamp', 'pm2_5']].rename(columns={'timestamp':'ds','pm2_5':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_forecast)

    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    predicted_pm = forecast[['ds','yhat']].tail(24)

    # Slider for interactive selection
    hour = st.slider("Select hour for 24-hour forecast", 0, 23, 0)
    predicted_pm_hour = predicted_pm['yhat'].iloc[hour]
    label, color = get_aqi_label(predicted_pm_hour)
    st.metric(f"Predicted PM2.5 at hour {hour}", f"{predicted_pm_hour:.2f}")
    st.markdown(f"<h3 style='color:{color}; text-align:center;'>{label}</h3>", unsafe_allow_html=True)

    # Forecast chart
    st.line_chart(forecast[['ds','yhat']].set_index('ds'))

elif page == "Alerts":
    st.subheader("⚠️ Alerts for Next 24 Hours")
    df_forecast = df[['timestamp', 'pm2_5']].rename(columns={'timestamp':'ds','pm2_5':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    predicted_pm = forecast[['ds','yhat']].tail(24)

    THRESHOLDS = {'pm2.5': 60}
    high_hours = predicted_pm[predicted_pm['yhat'] > THRESHOLDS['pm2.5']]
    if not high_hours.empty:
        for idx, row in high_hours.iterrows():
            st.markdown(f"""
            <div class="alert-card">
                ⚠ PM2.5 may exceed safe levels at {row['ds']}: {row['yhat']:.2f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No pollutant is expected to exceed safe levels in next 24 hours.")

elif page == "History":
    st.subheader("📈 Historical PM2.5 Levels")
    fig = px.line(df, x='timestamp', y='pm2_5', title="Historical PM2.5 Levels",
                  template="plotly_dark", markers=True)
    fig.update_layout(
        title_font=dict(size=22, color='gold'),
        xaxis_title='Time',
        yaxis_title='PM2.5',
        font=dict(color='white', family='Arial'),
        plot_bgcolor='#141e30',
        paper_bgcolor='#141e30'
    )
    st.plotly_chart(fig)

# ---------------------- Footer ----------------------
st.markdown("<hr><p style='text-align:center; color:#ffffff;'>Developed by Rachana Anugu | Data from OpenAQ & Local Sensors</p>", unsafe_allow_html=True)