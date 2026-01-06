import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ResidualWindowDataset(Dataset):
    """
    Builds sliding windows from residual_magnitude time-series.
    Returns windows shaped (seq_len, 1)
    """
    def __init__(self, csv_path: str, seq_len: int = 60):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        values = df["residual_magnitude"].astype(float).values

        # basic scaling: robust-ish normalization (median/IQR)
        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = max(q3 - q1, 1e-6)
        scaled = (values - median) / iqr

        self.seq_len = seq_len
        self.series = scaled.astype(np.float32)

        if len(self.series) < self.seq_len + 1:
            raise ValueError(f"Not enough data. Need at least {self.seq_len + 1} points.")

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        window = self.series[idx : idx + self.seq_len]
        # shape: (seq_len, 1)
        return window.reshape(-1, 1)
