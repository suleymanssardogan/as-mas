import pandas as pd
import numpy as np

def detect_anomalies_zscore(
    csv_path:str,
    window: int =30,
    threshold: float = 3.0

):
    """
    Simple baseline anomaly detection using rolling z-score.
    """

    df = pd.read_csv(csv_path,parse_dates=["timestamp"])
    df["rolling_mean"] = df["residual_magnitude"].rolling(window).mean()
    df["rolling_std"] = df["residual_magnitude"].rolling(window).std()

    df["z_score"] = (
        df["residual_magnitude"] - df["rolling_mean"]
        ) / df["rolling_std"]

    df["is_anomaly"] = df["z_score"].abs() > threshold

    return df
