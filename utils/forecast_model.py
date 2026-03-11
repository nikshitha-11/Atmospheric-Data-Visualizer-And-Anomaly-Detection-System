from sklearn.ensemble import RandomForestRegressor
import numpy as np

def forecast_pm(df):

    df["hour"] = df["time"].dt.hour

    X = df[["hour"]]
    y = df["pm2_5"]

    model = RandomForestRegressor()
    model.fit(X,y)

    future = np.arange(0,24).reshape(-1,1)

    pred = model.predict(future)

    return pred