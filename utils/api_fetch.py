import requests
import pandas as pd

def fetch_city_data(city):

    try:
        # Get coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        geo = requests.get(geo_url, timeout=10).json()

        if "results" not in geo:
            return None, None, None

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        )

        air_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=pm10,pm2_5"
        )

        weather = requests.get(weather_url, timeout=10, verify=False).json()
        air = requests.get(air_url, timeout=10, verify=False).json()

        df_weather = pd.DataFrame(weather["hourly"])
        df_air = pd.DataFrame(air["hourly"])

        df = pd.merge(df_weather, df_air, on="time")

        df["time"] = pd.to_datetime(df["time"])

        return df, lat, lon

    except Exception as e:
        print("API Error:", e)
        return None, None, None