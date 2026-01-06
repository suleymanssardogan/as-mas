import pandas as pd
from pathlib import Path


def save_residuals(timestamps, residual_magnitude, output_dir="data"):
    df = pd.DataFrame({
        "timestamp": timestamps,
        "residual_magnitude": residual_magnitude
    })

    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / "residuals.csv"
    df.to_csv(output_path, index=False)

    return output_path
