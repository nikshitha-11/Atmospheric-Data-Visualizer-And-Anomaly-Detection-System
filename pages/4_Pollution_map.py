import streamlit as st
import pydeck as pdk
import pandas as pd
from prophet import Prophet

st.title("🗺 Pollution Heatmap")

# ----------------------------
# Check if data loaded
# ----------------------------
if "lat" not in st.session_state or "data" not in st.session_state:
    st.warning("⚠ Load city data first from Dashboard")
    st.stop()

lat = st.session_state["lat"]
lon = st.session_state["lon"]
df = st.session_state["data"]

# ----------------------------
# Determine timestamp column dynamically
# ----------------------------
# Common names for timestamp
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        timestamp_col = col
        break
else:
    st.error("No datetime column found in your data!")
    st.stop()

# ----------------------------
# Current PM2.5 value
# ----------------------------
if 'pm2_5' in df.columns:
    pm_col = 'pm2_5'
elif 'pm2.5' in df.columns:
    pm_col = 'pm2.5'
else:
    st.error("No PM2.5 column found in your data!")
    st.stop()

pm = df[pm_col].iloc[-1]

# ----------------------------
# Current Pollution Heatmap
# ----------------------------
map_data = pd.DataFrame({
    "lat": [lat],
    "lon": [lon],
    "pm": [pm]
})

st.subheader("Pollution Intensity")

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v9",
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=8,
        pitch=50
    ),
    layers=[
        pdk.Layer(
            "HeatmapLayer",
            data=map_data,
            get_position='[lon, lat]',
            get_weight="pm",
            radiusPixels=60
        )
    ]
))

st.metric("Current PM2.5 Level", pm)

# ----------------------------
# 1️⃣ Forecast Next 24 Hours
# ----------------------------
st.subheader("Next 24-Hour Pollution Prediction")

# Prepare data for Prophet
df_forecast = df[[timestamp_col, pm_col]].rename(columns={timestamp_col:'ds', pm_col:'y'})
df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])  # ensure datetime type

model = Prophet(daily_seasonality=True)
model.fit(df_forecast)

future = model.make_future_dataframe(periods=24, freq='H')
forecast = model.predict(future)

# Display predictions
st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

# ----------------------------
# 2️⃣ Alert System
# ----------------------------
st.subheader("⚠️ Alerts for Next 24 Hours")

THRESHOLDS = {
    'pm2.5': 60,
    'pm10': 100,
    'no2': 80
}

alerts = []
predicted_pm = forecast[['ds', 'yhat']].tail(24)

if predicted_pm['yhat'].max() > THRESHOLDS['pm2.5']:
    alerts.append(f"PM2.5 may exceed safe levels in next 24 hours! Max predicted: {predicted_pm['yhat'].max():.2f}")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("No pollutant is expected to exceed safe levels in the next 24 hours.")

# ----------------------------
# 3️⃣ Predicted Heatmap
# ----------------------------
st.subheader("📍 Predicted Pollution Heatmap")

predicted_value = predicted_pm['yhat'].iloc[-1]

map_data = pd.DataFrame({
    "lat": [lat],
    "lon": [lon],
    "pm": [predicted_value]
})

layer = pdk.Layer(
    "HeatmapLayer",
    data=map_data,
    get_position='[lon, lat]',
    get_weight='pm',
    radius=5000
)

view_state = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=8,
    pitch=50
)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(r)