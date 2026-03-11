import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.api_fetch import fetch_city_data
from utils.aqi_calculator import calculate_aqi
from io import BytesIO

st.title("📊 Atmospheric Monitoring Dashboard")

# City input
city = st.text_input("Enter City")

# Load button
if st.button("Load Data"):

    if city == "":
        st.warning("Please enter a city name")
        st.stop()

    df, lat, lon = fetch_city_data(city)

    if df is None:
        st.error("City not found")
        st.stop()

    # Save data
    st.session_state["data"] = df
    st.session_state["lat"] = lat
    st.session_state["lon"] = lon

    # Latest values
    temp = df["temperature_2m"].iloc[-1]
    pm = df["pm2_5"].iloc[-1]
    wind = df["wind_speed_10m"].iloc[-1]

    # AQI calculation
    aqi = calculate_aqi(pm)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌡 Temperature", f"{temp} °C")
    col2.metric("💨 PM2.5", pm)
    col3.metric("🌬 Wind Speed", wind)
    col4.metric("🏭 AQI", aqi)

    # AQI Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={'text': "Air Quality Index"},
        gauge={
            'axis': {'range': [0, 200]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("## 📈 Environmental Trends")

    # Pollution alert
    if pm > 100:
        st.error("⚠ High Pollution Alert")
    elif pm > 60:
        st.warning("⚠ Moderate Pollution")
    else:
        st.success("✅ Air Quality Normal")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            df,
            x="time",
            y="temperature_2m",
            title="🌡 Temperature Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.line(
            df,
            x="time",
            y="pm2_5",
            title="💨 Pollution Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)


# -------------------------------
# Download Section
# -------------------------------

if "data" in st.session_state:

    st.markdown("## 📥 Download Environmental Data")

    df = st.session_state["data"]

    col1, col2, col3 = st.columns(3)

    # CSV
    with col1:
        csv = df.to_csv(index=False)

        st.download_button(
            "⬇ Download CSV",
            csv,
            "atmospheric_data.csv",
            "text/csv"
        )

    # JSON
    with col2:
        json_data = df.to_json()

        st.download_button(
            "⬇ Download JSON",
            json_data,
            "atmospheric_data.json"
        )

    # Excel
    with col3:
        buffer = BytesIO()
        df.to_excel(buffer, index=False)

        st.download_button(
            "⬇ Download Excel",
            buffer.getvalue(),
            "atmospheric_data.xlsx"
        )