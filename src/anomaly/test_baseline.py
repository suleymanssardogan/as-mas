from anomaly.baseline import detect_anomalies_zscore    


df = detect_anomalies_zscore(
    csv_path="data/residuals.csv",
    window=30,
    threshold=3.0
)

print(df.head(50)[["timestamp", "residual_magnitude", "z_score", "is_anomaly"]])
