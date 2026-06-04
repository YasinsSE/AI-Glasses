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
    # Cold-start settle before fixes are trusted. The NEO module already only
    # emits a fix once it has a real lock, so this extra guard is just a short
    # settle. 60 s was far too long: get_coord()/get_health() returned None for
    # a whole minute even with a SOLID lock, so navigation said "GPS bulunamadı"
    # right after boot despite the module's blue LED being on.
    warmup_sec: float = 5.0
    # Sliding window for fix smoothing AND freshness: get_coord() returns a
    # position if a fix arrived within this many seconds. 5 s was too tight for
    # urban/marginal reception (fixes every 6–10 s emptied the window →
    # intermittent "GPS bulunamadı"). 10 s tolerates sporadic fixes with only a
    # small (~6 m) smoothing lag while walking.
    fix_window_sec: float = 10.0
    update_interval: float = 4.0       # Poll GPS every N seconds.
    stale_threshold_sec: float = 10.0  # Treat a fix older than N s as stale.
