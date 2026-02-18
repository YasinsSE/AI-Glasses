# geo_utils.py
# Pure mathematical / geographic helper functions.
# No side effects, no imports from other project modules.

import math


EARTH_RADIUS_M = 6_371_000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points in metres.

    Args:
        lat1, lon1: Origin in decimal degrees.
        lat2, lon2: Destination in decimal degrees.

    Returns:
        Distance in metres.
    """
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Forward azimuth (bearing) from point 1 to point 2 in degrees [0, 360).

    Args:
        lat1, lon1: Origin in decimal degrees.
        lat2, lon2: Destination in decimal degrees.

    Returns:
        Bearing in degrees.
    """
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    d_lon = rlon2 - rlon1
    y = math.sin(d_lon) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def get_turn_instruction(bearing_diff: float) -> str:
    """
    Human-readable turn instruction derived from the change in bearing.

    Args:
        bearing_diff: Difference between consecutive bearings in degrees.

    Returns:
        Turn instruction string.
    """
    diff = (bearing_diff + 180) % 360 - 180
    if diff > 45:
        return "Turn sharp right"
    elif diff > 10:
        return "Turn right"
    elif diff < -45:
        return "Turn sharp left"
    elif diff < -10:
        return "Turn left"
    return "Go straight"
