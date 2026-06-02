"""
ActivityMonitor — automatic power-saving STANDBY detection.
============================================================
Watches motion signals and, after sustained inactivity, transitions the system
into STANDBY (``SystemMode.SLEEP``) so the camera + U-Net inference can shut
down and save LiPo battery. Wake-up is handled elsewhere — a PTT button press
(see ``tts_stt/voice_commands.py``) restores ACTIVE and the perception loop
re-acquires the camera.

Modular by design
-----------------
Motion evidence comes from pluggable :class:`MotionSource` objects. Today we
fuse:

    * :class:`VisualMotionSource` — frame-diff fed by ``PerceptionService``.
    * :class:`GpsMotionSource`    — RMC speed + displacement fed by
                                    ``NavigationService``.

A future **MPU9250 IMU** only needs a new ``ImuMotionSource`` implementing
``is_moving()`` plus one ``register()`` call — the fusion/loop below stays the
same. That is the whole point of the source abstraction.

Fusion rule (reliability by context)
------------------------------------
If GPS has a healthy fix we trust GPS (handles "cars pass by while you wait at
a red light" — the scene moves but you do not). Otherwise we trust vision
(handles indoor GPS drift — GPS jitters but the scene is still). The chosen
source must report "not moving" continuously for ``idle_enter_sec`` before we
sleep. When an IMU is added it slots in as the most-reliable source.
"""

import abc
import logging
import math
import threading
import time
from typing import List, Optional, Tuple

from main.lifecycle import ModeManager, SystemMode

logger = logging.getLogger("ALAS.standby")


# ═══════════════════════════════════════════════════════════════════
#  MOTION SOURCES (pluggable)
# ═══════════════════════════════════════════════════════════════════

class MotionSource(abc.ABC):
    """One source of 'is the user moving?' evidence."""

    name: str = "source"

    @abc.abstractmethod
    def is_moving(self) -> Optional[bool]:
        """Return True (moving), False (still), or None (no reliable reading)."""
        raise NotImplementedError


class VisualMotionSource(MotionSource):
    """Frame-to-frame visual change. Fed by ``PerceptionService``."""

    name = "visual"

    def __init__(self, threshold: float, stale_sec: float) -> None:
        self._threshold = threshold
        self._stale = stale_sec
        self._metric: Optional[float] = None
        self._ts: float = 0.0
        self._lock = threading.Lock()

    def report(self, metric: float) -> None:
        with self._lock:
            self._metric = metric
            self._ts = time.monotonic()

    def is_moving(self) -> Optional[bool]:
        with self._lock:
            if self._metric is None or (time.monotonic() - self._ts) > self._stale:
                return None
            return self._metric >= self._threshold


class GpsMotionSource(MotionSource):
    """Ground speed + displacement. Fed by ``NavigationService``.

    ``is_moving`` returns None unless GPS currently has a healthy, fresh fix —
    so the fusion layer knows when GPS is unreliable (e.g. indoors) and should
    defer to vision.
    """

    name = "gps"

    def __init__(self, moving_kmh: float, radius_m: float, stale_sec: float) -> None:
        self._moving_kmh = moving_kmh
        self._radius_m = radius_m
        self._stale = stale_sec
        self._speed: Optional[float] = None
        self._coord: Optional[Tuple[float, float]] = None
        self._health = None
        self._ts: float = 0.0
        self._anchor: Optional[Tuple[float, float]] = None
        self._lock = threading.Lock()

    def report(self, speed_kmh: Optional[float], coord, health) -> None:
        with self._lock:
            self._speed = speed_kmh
            self._coord = coord
            self._health = health
            self._ts = time.monotonic()

    def healthy(self) -> bool:
        from navigation.sensors import GPSStatus
        with self._lock:
            if self._health is None or (time.monotonic() - self._ts) > self._stale:
                return False
            return self._health.status in (GPSStatus.OK, GPSStatus.LOW_ACCURACY)

    def is_moving(self) -> Optional[bool]:
        if not self.healthy():
            return None
        with self._lock:
            if self._speed is not None and self._speed >= self._moving_kmh:
                self._anchor = self._coord
                return True
            if self._coord is not None:
                if self._anchor is None:
                    self._anchor = self._coord
                elif _haversine_m(self._anchor, self._coord) >= self._radius_m:
                    self._anchor = self._coord
                    return True
            return False


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points, in metres."""
    r = 6_371_000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


# ═══════════════════════════════════════════════════════════════════
#  ACTIVITY MONITOR
# ═══════════════════════════════════════════════════════════════════

class ActivityMonitor:
    """Daemon that drops the system into STANDBY after sustained inactivity."""

    name = "ActivityMonitor"

    def __init__(self, config, modes: ModeManager, voice, stop_event: threading.Event) -> None:
        self._cfg = config.idle
        self._modes = modes
        self._voice = voice
        self._stop = stop_event
        self._nav = None  # set via set_nav() — guards against sleeping mid-route.

        self.visual = VisualMotionSource(self._cfg.visual_motion_threshold, self._cfg.source_stale_sec)
        self.gps = GpsMotionSource(
            self._cfg.gps_moving_kmh, self._cfg.gps_stationary_radius_m, self._cfg.source_stale_sec,
        )
        # Future IMU slots in here as the most-reliable source.
        self._sources: List[MotionSource] = [self.gps, self.visual]

        self._still_since: Optional[float] = None
        self._thread: Optional[threading.Thread] = None

    def set_nav(self, nav) -> None:
        self._nav = nav

    # ── Report hooks (called by services; cheap no-ops if disabled) ───────
    def report_visual(self, metric: float) -> None:
        self.visual.report(metric)

    def report_gps(self, speed_kmh, coord, health) -> None:
        self.gps.report(speed_kmh, coord, health)

    def notify_wake(self) -> None:
        """Reset the idle timer so a just-woken system does not re-sleep at once."""
        self._still_since = None

    # ── Thread-like interface (for orderly shutdown join) ─────────────────
    def start(self) -> None:
        if not self._cfg.enabled:
            logger.info("[Standby] auto-standby disabled (pass --auto-standby to enable).")
            return
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── Fusion + decision ─────────────────────────────────────────────────
    def _decide_moving(self) -> Optional[bool]:
        # Prefer GPS when it has a healthy fix (outdoor); otherwise vision (indoor).
        if self.gps.healthy():
            primary = self.gps.is_moving()
            if primary is not None:
                return primary
        v = self.visual.is_moving()
        if v is not None:
            return v
        # Last resort: any source with a reading.
        for src in self._sources:
            m = src.is_moving()
            if m is not None:
                return m
        return None

    def _can_sleep(self) -> bool:
        if self._modes.mode != SystemMode.ACTIVE:
            return False
        if self._nav is not None and getattr(self._nav, "is_active", False):
            return False  # never sleep mid-route
        if self._voice.is_speaking_priority():
            return False
        return True

    def _loop(self) -> None:
        logger.info("[Standby] auto-standby active (idle_enter=%.0fs).", self._cfg.idle_enter_sec)
        while not self._stop.is_set():
            self._stop.wait(self._cfg.poll_interval_sec)
            if self._stop.is_set():
                break

            if not self._can_sleep():
                self._still_since = None
                continue

            moving = self._decide_moving()
            if moving is None or moving:
                self._still_since = None
                continue

            now = time.monotonic()
            if self._still_since is None:
                self._still_since = now
            elif (now - self._still_since) >= self._cfg.idle_enter_sec:
                self._enter_standby()
                self._still_since = None

        logger.info("[Standby] monitor stopped.")

    def _enter_standby(self) -> None:
        logger.info("[Standby] sustained inactivity — entering STANDBY to save power.")
        try:
            self._voice.announce_sleep()
        except Exception:  # noqa: BLE001
            logger.exception("[Standby] announce_sleep failed")
        self._modes.transition_to(SystemMode.SLEEP)
