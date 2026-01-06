from skyfield.api import load


def propagate_satellite(satellite, minutes=90, step=1):
    ts = load.timescale()
    times = ts.utc(2024, 1, 1, 0, range(0, minutes, step))

    geocentric = satellite.at(times)

    position = geocentric.position.km        # shape: (3, N)
    velocity = geocentric.velocity.km_per_s  # shape: (3, N)

    return {
        "times": times.utc_datetime(),
        "position_km": position,
        "velocity_km_s": velocity
    }
