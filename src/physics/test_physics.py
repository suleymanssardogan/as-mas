from physics.tle import load_satellite_from_tle
from physics.propagation import propagate_satellite
from physics.residual import compute_residual
from data.writer import save_residuals
from datetime import datetime

import numpy as np

# --- TLE DATA ---
ISS_TLE = (
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.51858796  .00016717  00000+0  10270-3 0  9990",
    "2 25544  51.6403  96.1191 0005503  36.6418  72.5065 15.50164182430734"
)

# --- LOAD SATELLITE ---
satellite = load_satellite_from_tle(*ISS_TLE)

# --- PHYSICS PROPAGATION ---
result = propagate_satellite(
    satellite,
    start_time=datetime.utcnow()
)
print("Position (first 5 timesteps):")
print(result["position_km"][:, :5])

# --- SIMULATED OBSERVED DATA (FAKE ANOMALY) ---
observed_position = result["position_km"] + np.random.normal(
    0, 5, result["position_km"].shape
)

# --- RESIDUAL COMPUTATION ---
residuals = compute_residual(
    reference_position=result["position_km"],
    observed_position=observed_position
)
# We did TLE --> orbit propagation
# based on physics we expect orbit to be like this
# residual = real orbit - expected orbit
# like a time series digital signal
# Anomly = abnormal increase in residual magnitude
print("\nResidual magnitude (first 5 timesteps):")
print(residuals["residual_magnitude"][:5])

output_path = save_residuals(
    timestamps=result["times"],
    residual_magnitude=residuals["residual_magnitude"]
)

print(f"\nResiduals saved to: {output_path}")
