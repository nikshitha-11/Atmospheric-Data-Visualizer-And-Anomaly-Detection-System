import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_fetch import fetch_city_data

st.title("🌆 City Pollution Comparison")

city1 = st.text_input("City 1")
city2 = st.text_input("City 2")

if st.button("Compare"):

    df1,_,_ = fetch_city_data(city1)
    df2,_,_ = fetch_city_data(city2)

    df1["city"]=city1
    df2["city"]=city2

    df = pd.concat([df1,df2])

    fig = px.line(df,x="time",y="pm2_5",color="city")

    st.plotly_chart(fig)