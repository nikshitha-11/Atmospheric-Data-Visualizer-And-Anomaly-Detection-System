import streamlit as st

st.set_page_config(
    page_title="Atmospheric AI Platform",
    page_icon="🌍",
    layout="wide"
)
st.title("🌍 Atmospheric Data Visualizer and Anomaly Detection System")

st.sidebar.title("Navigation")
st.sidebar.info("Use the sidebar to navigate between pages.")

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
with st.sidebar:

    st.title("🌍 Atmospheric AI")

    st.markdown("---")

    st.subheader("Navigation")

    st.markdown("---")

    st.info("AI Powered Environmental Monitoring")

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)

    st.title("Atmospheric AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Anomaly Detection",
            "Forecast",
            "Pollution Map",
            "City Comparison",
            "Reports"
        ]
    )

    st.markdown("---")

    st.subheader("⚙ Settings")

    auto_refresh = st.checkbox("Auto Refresh Data")

    refresh_time = st.slider("Refresh Interval (sec)",10,120,30)

    theme = st.selectbox(
        "Theme",
        ["Dark Mode","Ocean Blue","Forest Green"]
    )

    st.markdown("---")

    st.info("Atmospheric Monitoring System\n\nAI Powered Environmental Insights")