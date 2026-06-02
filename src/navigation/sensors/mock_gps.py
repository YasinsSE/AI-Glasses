"""
MockGPSReader — fixed-position fake GPS for desktop testing.

Duck-types ``GPSReader`` so the rest of the system can stay completely
unaware that a hardware GPS is not connected. Implements the same
``start / stop / get_coord / get_health`` surface.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from .gps_reader import GPSHealth, GPSStatus

logger = logging.getLogger("ALAS.mock_gps")


class MockGPSReader:
    """Always returns the same coordinate. Use only with ``--mock``."""

    def __init__(
        self,
        lat: float = 39.9245,
        lon: float = 32.8465,
        speed_kmh: float = 0.0,
    ) -> None:
        self._lat = lat
        self._lon = lon
        self._speed_kmh = speed_kmh  # 0.0 == stationary (exercises auto-STANDBY).

    def start(self) -> None:
        logger.info(f"[MockGPS] Fixed position: ({self._lat}, {self._lon})")

    def stop(self) -> None:
        return None

    def get_coord(self) -> Optional[Tuple[float, float, float]]:
        return (self._lat, self._lon, 0.0)

    def get_speed_kmh(self) -> float:
        return self._speed_kmh

    def get_utc(self) -> Optional[Tuple[datetime, float]]:
        # The desktop clock is trustworthy under --mock, so expose it as the
        # "satellite" UTC so the recorder's clock_sync path is exercised.
        return datetime.now(timezone.utc), time.monotonic()

    def get_health(self) -> GPSHealth:
        return GPSHealth(
            status=GPSStatus.OK,
            satellites=10,
            hdop=1.0,
            fix_age_sec=0.0,
            fix_count=5,
            serial_ok=True,
        )