"""
NavigationService — GPS polling + route progress + TTS turn-by-turn.
====================================================================
Owns the long-running navigation thread. Reads GPS at a fixed interval,
feeds positions to the existing ``NavigationSystem`` (which we do not touch),
and decides when to speak.

Three-tier announcement strategy (the user's "GPS nodes / 30 s fallback" note):

    (a) Approach pre-warning   — distance to next waypoint < APPROACH (30 m).
                                 Fires **once per step**, with proximity
                                 wording: "20 metre sonra sağa dönün".
    (b) Long-stretch reminder  — distance > LONG_STRETCH (100 m). The
                                 existing 30 s "Hedefe X metre" ping fires.
    (c) Mid-range silence      — between thresholds, no announcement. The
                                 user already heard the upcoming instruction
                                 (either at WAYPOINT_HIT or at the 30 m
                                 pre-warning) and the distance is too long
                                 to be useful.

Event statuses (WAYPOINT_HIT, OFF_ROUTE, FINISHED) are spoken immediately,
deduped on text so the same message does not repeat across GPS ticks.
"""

import logging
import threading
import time
from typing import Optional

from main.config import ALASConfig
from main.lifecycle import ModeManager, SystemMode
from navigation.router import Coord, NavigationSystem, ProgressResult, RouteStatus
from navigation.sensors import GPSStatus
from tts_stt.voice_policy import VoicePolicy

logger = logging.getLogger("ALAS.navigation_service")


class NavigationService(threading.Thread):
    """Polls GPS, drives NavigationSystem, and speaks the right thing at the right time."""

    def __init__(
        self,
        config: ALASConfig,
        nav: NavigationSystem,
        gps,
        voice: VoicePolicy,
        modes: ModeManager,
        stop_event: threading.Event,
        recorder=None,
        monitor=None,
    ) -> None:
        super().__init__(name="NavigationService", daemon=True)
        self._config = config
        self._nav = nav
        self._gps = gps
        self._voice = voice
        self._modes = modes
        self._stop_event = stop_event
        self._monitor = monitor  # ActivityMonitor (auto-STANDBY) or None.
        from main.session_recorder import NullRecorder
        self._rec = recorder or NullRecorder()  # field-test black-box recorder

        self._last_spoken: str = ""
        self._last_progress_time: float = 0.0
        self._prewarned_step_id: Optional[int] = None
        self._stale_announced: bool = False

    # ── Thread entry point ───────────────────────────────────────

    def run(self) -> None:
        logger.info("[Navigation] Service started.")
        while not self._stop_event.is_set():
            if self._modes.mode != SystemMode.ACTIVE:
                self._stop_event.wait(self._config.gps.update_interval)
                continue

            # When ACTIVE with no active route (environment-awareness mode), we
            # still poll GPS so the field-test recorder captures the track and
            # the satellite-UTC clock anchor even without a destination set.
            if not self._nav.is_active:
                idle_fix = self._gps.get_coord()
                if idle_fix is not None:
                    self._record_gps(*idle_fix)
                    self._report_activity((idle_fix[0], idle_fix[1]))
                else:
                    self._log_gps_state()
                    self._report_activity(None)
                self._stop_event.wait(self._config.gps.update_interval)
                continue

            fix = self._gps.get_coord()
            if fix is None:
                self._log_gps_state()
                self._report_activity(None)
                self._stop_event.wait(self._config.gps.update_interval)
                continue

            lat, lon, age = fix
            self._record_gps(lat, lon, age)
            self._report_activity((lat, lon))
            if age > self._config.gps.stale_threshold_sec:
                if not self._stale_announced:
                    self._voice.say_nav("GPS sinyali zayıf, konum güncellenemiyor.")
                    self._stale_announced = True
                self._stop_event.wait(self._config.gps.update_interval)
                continue
            self._stale_announced = False
            try:
                result = self._nav.update(Coord(lat, lon))
            except Exception:  # noqa: BLE001
                logger.exception("[Navigation] nav.update failed")
                self._stop_event.wait(self._config.gps.update_interval)
                continue

            self._announce(result)
            self._stop_event.wait(self._config.gps.update_interval)

        logger.info("[Navigation] Stopped.")

    # ── Announcement logic ───────────────────────────────────────

    def _report_activity(self, coord) -> None:
        """Feed the auto-STANDBY monitor with GPS speed/displacement + health."""
        if self._monitor is None:
            return
        try:
            speed = self._gps.get_speed_kmh()
        except Exception:  # noqa: BLE001
            speed = None
        try:
            health = self._gps.get_health()
        except Exception:  # noqa: BLE001
            health = None
        self._monitor.report_gps(speed, coord, health)

    def _record_gps(self, lat: float, lon: float, age: float) -> None:
        """Log the GPS fix and (once) anchor absolute time from satellite UTC."""
        try:
            health = self._gps.get_health()
            self._rec.log_gps(lat, lon, round(age, 2), health.satellites,
                              health.hdop, health.status.value)
        except Exception:  # noqa: BLE001
            pass
        get_utc = getattr(self._gps, "get_utc", None)
        if get_utc is not None:
            utc = get_utc()
            if utc is not None:
                self._rec.note_gps_utc(utc[0], utc[1])

    def _announce(self, result: ProgressResult) -> None:
        step = result.current_step
        self._rec.log_nav(
            result.status.value if hasattr(result.status, "value") else result.status,
            distance_to_next_m=getattr(result, "distance_to_next", None),
            step_text=(step.text if step is not None else None),
        )
        if result.status == RouteStatus.WAYPOINT_HIT:
            self._speak_event(result.message)
            self._prewarned_step_id = None  # next step not yet pre-warned
            return

        if result.status == RouteStatus.OFF_ROUTE:
            self._speak_event("Rotadan çıktınız. Lütfen geri dönün.")
            return

        if result.status == RouteStatus.FINISHED:
            self._speak_event("Hedefinize ulaştınız, iyi günler.")
            return

        if result.status != RouteStatus.PROGRESSING:
            return

        # PROGRESSING — three-tier delivery
        dist = result.distance_to_next
        step = result.current_step
        if dist is None or step is None:
            return

        # (a) Approach pre-warning — fires once per step
        if (
            dist <= self._config.nav.approach_threshold_m
            and self._prewarned_step_id != step.step_id
        ):
            # The "start" step is an intro ("Rota başlıyor. X üzerindesiniz"),
            # not a turn — prefixing it with "N metre sonra" is nonsensical.
            if step.action == "start":
                self._voice.say_nav(step.text)
            else:
                self._voice.say_nav(f"{int(dist)} metre sonra {step.text}")
            self._prewarned_step_id = step.step_id
            self._last_spoken = step.text
            return

        # (b) Long-stretch reminder — periodic distance ping
        if dist >= self._config.nav.long_stretch_threshold_m:
            now = time.monotonic()
            if (now - self._last_progress_time) > self._config.nav.progress_announce_interval:
                self._voice.say_progress(f"Hedefe {int(dist)} metre.")
                self._last_progress_time = now
        # (c) Mid-range silence — fall through, no speech.

    def _speak_event(self, text: str) -> None:
        if not text or text == self._last_spoken:
            return
        self._voice.say_nav(text)
        self._last_spoken = text

    def _log_gps_state(self) -> None:
        try:
            health = self._gps.get_health()
        except Exception:  # noqa: BLE001
            return
        if health.status == GPSStatus.WARMING_UP:
            logger.debug("[Navigation] GPS warming up...")
        elif health.status == GPSStatus.NO_FIX:
            logger.debug("[Navigation] No GPS fix yet.")