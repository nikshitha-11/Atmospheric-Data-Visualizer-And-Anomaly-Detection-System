import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.title("📈 Pollution Forecast")

if "data" not in st.session_state:
    st.warning("Load data from Dashboard")
    st.stop()

df = st.session_state["data"]
st.markdown("## ⚠ Anomaly Detection")

features = df[["temperature_2m","pm2_5","wind_speed_10m"]]

model = IsolationForest(contamination=0.05)
df["anomaly"] = model.fit_predict(features)

anomaly_df = df[df["anomaly"] == -1]

fig_anomaly = px.scatter(
    df,
    x="time",
    y="pm2_5",
    color=df["anomaly"].map({1:"Normal",-1:"Anomaly"}),
    title="Pollution Anomaly Detection"
)

st.plotly_chart(fig_anomaly, use_container_width=True)

future = df.tail(24).copy()
future["pm2_5"] = future["pm2_5"] * 1.05

fig = px.line(future,x="time",y="pm2_5",title="Predicted Pollution Trend")

st.plotly_chart(fig,use_container_width=True)