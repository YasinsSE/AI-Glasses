# gps_reader.py
# Background-thread NMEA reader for the NEO-7M UART module.
# get_coord()  -> (lat, lon, age_sec) or None
# get_health() -> GPSHealth

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Dict

import serial

from .gps_filter import filter_fixes

logger = logging.getLogger(__name__)


class GPSStatus(Enum):
    WARMING_UP   = "warming_up"
    NO_FIX       = "no_fix"
    LOW_ACCURACY = "low_accuracy"
    OK           = "ok"
    SERIAL_ERROR = "serial_error"


@dataclass(frozen=True)
class GPSHealth:
    status: GPSStatus
    satellites: int
    hdop: float
    fix_age_sec: float
    fix_count: int
    serial_ok: bool


# ── NMEA parse helpers ────────────────────────────────────────

def _checksum_ok(sentence: str) -> bool:
    try:
        if not sentence.startswith("$") or "*" not in sentence:
            return False
        body, expected = sentence[1:].split("*", 1)
        calc = 0
        for ch in body:
            calc ^= ord(ch)
        return f"{calc:02X}" == expected[:2].upper()
    except (ValueError, IndexError):
        return False


def _to_decimal(raw: str, hemi: str) -> Optional[float]:
    try:
        if not raw or not hemi:
            return None
        if hemi in ("N", "S"):
            deg, mins = float(raw[:2]), float(raw[2:])
        else:
            deg, mins = float(raw[:3]), float(raw[3:])
        dec = deg + mins / 60.0
        if hemi in ("S", "W"):
            dec = -dec
        return dec
    except (ValueError, IndexError):
        return None


# Constants
MAX_SERIAL_ERRORS = 10
RECONNECT_WAIT_SEC = 3.0
MAX_BACKOFF_SEC = 5.0


class GPSReader:
    """NEO-7M UART NMEA reader.

    Args:
        port:          Serial port (Jetson Nano: /dev/ttyTHS1).
        baudrate:      NEO-7M default 9600.
        timeout:       Serial read timeout in seconds.
        window:        Sliding window over which fixes are accumulated.
        min_sats:      Minimum satellites required for an OK fix.
        max_hdop:      Maximum HDOP accepted as OK.
        max_speed_kmh: Walking-speed limit used by the outlier filter.
        warmup_sec:    Cold-start wait before publishing health.
    """

    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        baudrate: int = 9600,
        timeout: float = 2.0,
        window: float = 5.0,
        min_sats: int = 4,
        max_hdop: float = 5.0,
        max_speed_kmh: float = 15.0,
        warmup_sec: float = 60.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.window = window
        self.min_sats = min_sats
        self.max_hdop = max_hdop
        self.max_speed_kmh = max_speed_kmh
        self.warmup_sec = warmup_sec

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._fixes: List[Tuple[float, float, float]] = []

        self._confirmed_meta: Dict[str, object] = {
            "sats": 0,
            "hdop": 99.9,
        }
        self._pending_gga: Optional[Dict[str, object]] = None

        self._start_time: float = 0.0
        self._serial_ok: bool = False
        self._last_fix_time: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        self._serial_ok = True
        logger.info(f"GPS opened: {self.port} @ {self.baudrate}")

        self._start_time = time.monotonic()

        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name="GPS-Reader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._serial_ok = False
        logger.info("GPS closed.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ── Public API ────────────────────────────────────────────

    def get_coord(self) -> Optional[Tuple[float, float, float]]:
        """Return the latest filtered coordinate or None.

        Returns:
            (lat, lon, age_sec) or None when no fix is available yet.
        """
        now = time.monotonic()

        if now - self._start_time < self.warmup_sec:
            return None

        with self._lock:
            cutoff = now - self.window
            self._fixes = [f for f in self._fixes if f[0] >= cutoff]

            if not self._fixes:
                return None

            newest_ts = max(f[0] for f in self._fixes)
            age_sec = now - newest_ts
            raw = [(f[1], f[2]) for f in self._fixes]

        coord = filter_fixes(raw)
        if coord is None:
            return None

        return (coord[0], coord[1], age_sec)

    def get_health(self) -> GPSHealth:
        now = time.monotonic()

        with self._lock:
            sats = self._confirmed_meta["sats"]
            hdop = self._confirmed_meta["hdop"]

            cutoff = now - self.window
            active_fixes = [f for f in self._fixes if f[0] >= cutoff]
            fix_count = len(active_fixes)

            last_fix = self._last_fix_time

        if last_fix > 0:
            fix_age = now - last_fix
        else:
            fix_age = float("inf")

        if not self._serial_ok:
            status = GPSStatus.SERIAL_ERROR
        elif now - self._start_time < self.warmup_sec:
            status = GPSStatus.WARMING_UP
        elif fix_count == 0:
            status = GPSStatus.NO_FIX
        elif sats < self.min_sats or hdop > self.max_hdop:
            status = GPSStatus.LOW_ACCURACY
        else:
            status = GPSStatus.OK

        return GPSHealth(
            status=status,
            satellites=sats,
            hdop=hdop,
            fix_age_sec=round(fix_age, 2),
            fix_count=fix_count,
            serial_ok=self._serial_ok,
        )

    @property
    def fix_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window
            return sum(1 for f in self._fixes if f[0] >= cutoff)

    @property
    def satellites(self) -> int:
        with self._lock:
            return self._confirmed_meta["sats"]

    @property
    def hdop(self) -> float:
        with self._lock:
            return self._confirmed_meta["hdop"]

    # ── Background thread ─────────────────────────────────────

    def _read_loop(self) -> None:
        consecutive_errors = 0

        while self._running:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
                consecutive_errors = 0
                self._serial_ok = True
            except (serial.SerialException, OSError) as e:
                consecutive_errors += 1
                logger.error(f"Serial hata ({consecutive_errors}/{MAX_SERIAL_ERRORS}): {e}")

                if consecutive_errors >= MAX_SERIAL_ERRORS:
                    self._reconnect()
                    consecutive_errors = 0
                else:
                    backoff = min(consecutive_errors * 0.5, MAX_BACKOFF_SEC)
                    time.sleep(backoff)
                continue

            if not line:
                continue

            if line.startswith(("$GPGGA", "$GNGGA")):
                self._parse_gga(line)
                continue

            if line.startswith(("$GPRMC", "$GNRMC")):
                self._parse_rmc(line)

    def _reconnect(self) -> None:
        self._serial_ok = False

        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass

        time.sleep(RECONNECT_WAIT_SEC)

        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            self._serial_ok = True
            logger.info(f"Serial reconnect succeeded: {self.port}")
        except (serial.SerialException, OSError) as e:
            logger.error(f"Serial reconnect failed: {e}")
            self._serial_ok = False

    def _parse_gga(self, line: str) -> None:
        if not _checksum_ok(line):
            return

        try:
            p = line.split(",")
            if len(p) < 10:
                return

            utc = p[1][:6] if len(p[1]) >= 6 else ""
            sats = int(p[7]) if p[7] else 0
            hdop = float(p[8]) if p[8] else 99.9

            with self._lock:
                    self._confirmed_meta["sats"] = sats
                    self._confirmed_meta["hdop"] = hdop
        except (ValueError, IndexError):
            pass

    def _parse_rmc(self, line: str) -> None:
        if not _checksum_ok(line):
            return

        p = line.split(",")
        if len(p) < 10:
            return

        if p[2] != "A":
            return

        lat = _to_decimal(p[3], p[4])
        lon = _to_decimal(p[5], p[6])
        if lat is None or lon is None:
            return

        rmc_utc = p[1][:6] if len(p[1]) >= 6 else ""

        try:
            speed = float(p[7]) * 1.852 if p[7] else 0.0
        except ValueError:
            speed = 0.0

        with self._lock:
            current_sats = self._confirmed_meta["sats"]
            current_hdop = self._confirmed_meta["hdop"]

        if current_sats < self.min_sats:
            return
        if current_hdop > self.max_hdop:
            return
        if speed > self.max_speed_kmh:
            return

        now = time.monotonic()
        with self._lock:
            self._fixes.append((now, lat, lon))
            self._last_fix_time = now

            max_len = int(self.window * 2) + 10
            if len(self._fixes) > max_len:
                self._fixes = self._fixes[-max_len:]
