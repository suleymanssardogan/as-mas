from physics.tle import load_satellite_from_tle
from physics.propagation import propagate_satellite

ISS_TLE =(
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.51858796  .00016717  00000+0  10270-3 0  9990",
    "2 25544  51.6403  96.1191 0005503  36.6418  72.5065 15.50164182430734"

)

sat = load_satellite_from_tle(*ISS_TLE)
result = propagate_satellite(sat)

print(result["position_km"][:,:5])
