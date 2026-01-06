from skyfield.api import EarthSatellite, load


def load_satellite_from_tle(name: str, line1: str, line2: str):
    ts = load.timescale()
    satellite = EarthSatellite(line1, line2, name, ts)
    return satellite
