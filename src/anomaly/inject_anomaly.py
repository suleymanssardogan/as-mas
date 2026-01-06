import pandas as pd
import numpy as np


def inject_bias_anomaly(
    csv_path="data/residuals.csv",
    start_idx=120,
    bias=15.0,
    output_path="data/residuals_bias.csv"
):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    df_anom = df.copy()
    df_anom.loc[start_idx:, "residual_magnitude"] += bias

    df_anom.to_csv(output_path, index=False)
    return output_path

def inject_spike_anomaly(
    csv_path="data/residuals.csv",
    spike_idx=50,
    spike_value=100.0,
    output_path="data/residuals_spike.csv"
):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    df_spike = df.copy()
    spike_idx = min(spike_idx, len(df_spike) - 1)

    df_spike.loc[spike_idx, "residual_magnitude"] += spike_value

    df_spike.to_csv(output_path, index=False)
    return output_path
if __name__ == "__main__":
    out = inject_spike_anomaly(
        csv_path="data/residuals.csv",
        spike_idx=50,
        spike_value=100.0,
        output_path="data/residuals_spike.csv"
    )
    print(f"Saved spike data to {out}")

