# gps_filter.py
# Ham fix listesi alır → outlier atar → ortalama Coord döndürür.

from typing import List, Optional, Tuple

from ..router.geo_utils import haversine_distance


def filter_fixes(
    fixes: List[Tuple[float, float]],
    max_deviation_m: float = 15.0,
) -> Optional[Tuple[float, float]]:
    """
    Ham GPS fix listesini filtrele ve tek bir (lat, lon) döndür.

    Medyan tabanlı sabit eşik filtresi:
      1. Fix < 3 → basit ortalama
      2. Medyan merkez bul
      3. Merkeze mesafesi > max_deviation_m olanları at
      4. Kalan fixlerin ortalaması
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
