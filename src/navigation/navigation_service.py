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
    ) -> None:
        super().__init__(name="NavigationService", daemon=True)
        self._config = config
        self._nav = nav
        self._gps = gps
        self._voice = voice
        self._modes = modes
        self._stop = stop_event

        self._last_spoken: str = ""
        self._last_progress_time: float = 0.0
        self._prewarned_step_id: Optional[int] = None
        self._stale_announced: bool = False

    # ── Thread entry point ───────────────────────────────────────

    def run(self) -> None:
        logger.info("[Navigation] Service started.")
        while not self._stop.is_set():
            if self._modes.mode != SystemMode.ACTIVE or not self._nav.is_active:
                self._stop.wait(self._config.gps_update_interval)
                continue

            fix = self._gps.get_coord()
            if fix is None:
                self._log_gps_state()
                self._stop.wait(self._config.gps_update_interval)
                continue

            lat, lon, age = fix
            if age > self._config.gps_stale_threshold_sec:
                if not self._stale_announced:
                    self._voice.say_nav("GPS sinyali zayıf, konum güncellenemiyor.")
                    self._stale_announced = True
                self._stop.wait(self._config.gps_update_interval)
                continue
            self._stale_announced = False
            try:
                result = self._nav.update(Coord(lat, lon))
            except Exception:  # noqa: BLE001
                logger.exception("[Navigation] nav.update failed")
                self._stop.wait(self._config.gps_update_interval)
                continue

            self._announce(result)
            self._stop.wait(self._config.gps_update_interval)

        logger.info("[Navigation] Stopped.")

    # ── Announcement logic ───────────────────────────────────────

    def _announce(self, result: ProgressResult) -> None:
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
            dist <= self._config.approach_threshold_m
            and self._prewarned_step_id != step.step_id
        ):
            self._voice.say_nav(f"{int(dist)} metre sonra {step.text}")
            self._prewarned_step_id = step.step_id
            self._last_spoken = step.text
            return

        # (b) Long-stretch reminder — periodic distance ping
        if dist >= self._config.long_stretch_threshold_m:
            now = time.monotonic()
            if (now - self._last_progress_time) > self._config.progress_announce_interval:
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