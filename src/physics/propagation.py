from skyfield.api import load
from datetime import datetime


def propagate_satellite(
    satellite,
    start_time: datetime,
    minutes=90,
    step=1
):
    ts = load.timescale()
    times = ts.utc(
        start_time.year,
        start_time.month,
        start_time.day,
        start_time.hour,
        start_time.minute,
        range(0, minutes, step)
    )

    geocentric = satellite.at(times)

    return {
        "times": times.utc_datetime(),
        "position_km": geocentric.position.km,
        "velocity_km_s": geocentric.velocity.km_per_s
    }
