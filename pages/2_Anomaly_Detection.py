import streamlit as st
import plotly.graph_objects as go
from utils.anomaly_model import detect_anomalies

st.title("⚠ Atmospheric Anomaly Detection")

if "data" not in st.session_state:
    st.warning("Load data in Dashboard first")
    st.stop()

df = st.session_state["data"]

df = detect_anomalies(df)

anomalies = df[df["anomaly"]==-1]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["pm2_5"],
    mode="lines",
    name="PM2.5"
))

fig.add_trace(go.Scatter(
    x=anomalies["time"],
    y=anomalies["pm2_5"],
    mode="markers",
    marker=dict(color="red",size=10),
    name="Anomaly"
))

st.plotly_chart(fig,use_container_width=True)

st.write(f"Detected {len(anomalies)} anomalies")