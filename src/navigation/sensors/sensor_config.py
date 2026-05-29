"""GPS sensor configuration.

Tunables for the UART NMEA reader (:mod:`navigation.sensors.gps_reader`) and the
navigation loop's GPS polling cadence. ``CANDIDATE_PORTS`` is the single source
of truth for port auto-detection, shared by the reader and the GPS test.

Composed by :class:`main.config.ALASConfig`.
"""

from dataclasses import dataclass

# Serial ports probed when auto-detecting the GPS module, in priority order.
CANDIDATE_PORTS = [
    "/dev/ttyTHS1",   # Jetson Nano 40-pin UART.
    "/dev/ttyTHS2",
    "/dev/ttyUSB0",   # USB-to-serial adapter.
    "/dev/ttyACM0",
    "/dev/ttyAMA0",   # Raspberry Pi compatibility.
    "/dev/ttyAMA10",  # Raspberry Pi 5.
]


@dataclass
class GPSConfig:
    port: str = "/dev/ttyTHS1"
    baudrate: int = 9600
    warmup_sec: float = 60.0           # Cold-start window before fixes are trusted.
    update_interval: float = 4.0       # Poll GPS every N seconds.
    stale_threshold_sec: float = 10.0  # Treat a fix older than N s as stale.
