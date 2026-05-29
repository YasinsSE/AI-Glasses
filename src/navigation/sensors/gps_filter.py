"""Median-based GPS fix filtering.

Takes a list of raw fixes, rejects outliers, and returns the averaged Coord.
"""

from typing import List, Optional, Tuple

from ..router.geo_utils import haversine_distance


def filter_fixes(
    fixes: List[Tuple[float, float]],
    max_deviation_m: float = 15.0,
) -> Optional[Tuple[float, float]]:
    """Filter a list of raw GPS fixes and return a single (lat, lon).

    Median-based fixed-threshold filter:
      1. Fewer than 3 fixes -> simple average.
      2. Find the median centre.
      3. Drop fixes farther than max_deviation_m from the centre.
      4. Average the remaining fixes.
    """
    if not fixes:
        return None

    if len(fixes) == 1:
        return fixes[0]

    if len(fixes) == 2:
        return _average(fixes)

    med_lat = _median([f[0] for f in fixes])
    med_lon = _median([f[1] for f in fixes])

    distances = [
        haversine_distance(med_lat, med_lon, lat, lon)
        for lat, lon in fixes
    ]

    clean = [
        (lat, lon)
        for i, (lat, lon) in enumerate(fixes)
        if distances[i] <= max_deviation_m
    ]

    if not clean:
        return (med_lat, med_lon)

    return _average(clean)


def _average(fixes: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(fixes)
    return (
        sum(f[0] for f in fixes) / n,
        sum(f[1] for f in fixes) / n,
    )


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
