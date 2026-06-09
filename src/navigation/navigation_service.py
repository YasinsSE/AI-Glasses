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

import collections
import logging
import threading
import time
from typing import Optional

from navigation.router.geo_utils import calculate_bearing, haversine_distance

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
        self._last_turn_at: float = -999.0   # last turn instruction time (progress-ping guard)
        self._prewarned_step_id: Optional[int] = None
        self._stale_announced: bool = False
        # Destination-progress (Faz 2): are we actually getting closer to the
        # final target? Robust to a mis-snapped OSM route geometry.
        self._dest = None
        self._dist_window: "collections.deque" = collections.deque()
        # Post-turn confirmation (B3): armed at each turn instruction with the
        # expected bearing; judged once the user has walked a few metres.
        self._turn_check: Optional[dict] = None

    # ── Thread entry point ───────────────────────────────────────

    def run(self) -> None:
        logger.info("[Navigation] Service started.")
        while not self._stop_event.is_set():
            if self._modes.mode != SystemMode.ACTIVE:
                self._stop_event.wait(self._config.gps.update_interval)
                continue

            # Record GPS health every poll — even with no fix — so the black box
            # shows satellite count / serial loss / dropouts during the walk.
            self._log_gps_health()

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

            self._announce(result, Coord(lat, lon))
            self._stop_event.wait(self._config.gps.update_interval)

        logger.info("[Navigation] Stopped.")

    # ── Announcement logic ───────────────────────────────────────

    def _log_gps_health(self) -> None:
        """Emit a GPS health snapshot to the recorder (visible even with no fix)."""
        try:
            h = self._gps.get_health()
        except Exception:  # noqa: BLE001 — mock GPS or transient read error
            return
        self._rec.log_gps_status(
            getattr(h, "status", None),
            getattr(h, "satellites", None),
            getattr(h, "hdop", None),
            getattr(h, "serial_ok", None),
            getattr(h, "fix_age_sec", None),
        )

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

    def _destination_progress(self, position):
        """Are we actually approaching the FINAL target? (Faz 2)

        Returns ('on_track' | 'wrong_way' | None, dist_to_target_m). Compares the
        straight-line distance to the destination over a sliding window, so it is
        robust to a route whose node geometry does not match the real sidewalk.
        """
        route = self._nav.get_route()
        if not route:
            return None, None
        dest = route[-1].location
        now = time.monotonic()
        d = haversine_distance(position.lat, position.lon, dest.lat, dest.lon)

        # Reset the window when the destination changes (a new route).
        if (self._dest is None
                or haversine_distance(dest.lat, dest.lon, self._dest.lat, self._dest.lon) > 5.0):
            self._dest = dest
            self._dist_window.clear()
        self._dist_window.append((now, d))
        cutoff = now - self._config.nav.progress_window_sec
        while len(self._dist_window) > 2 and self._dist_window[0][0] < cutoff:
            self._dist_window.popleft()
        if len(self._dist_window) < 2:
            return None, d

        delta = d - self._dist_window[0][1]   # +: moved away, −: got closer
        if delta <= -2.0:
            return "on_track", d
        if delta >= self._config.nav.wrong_way_gain_m:
            return "wrong_way", d
        return None, d

    def _update_crossing_expected(self, result: ProgressResult) -> None:
        """B2 fusion: tell perception when the route is about to cross a road.

        ``crossing_expected`` flips True while the upcoming step enters a
        'crossing' OSM segment within ``crossing_expect_m`` — perception then
        frames its road-ahead warning as crossing guidance instead of a flat
        "girmeyin".
        """
        step = result.current_step
        dist = result.distance_to_next
        self._nav.crossing_expected = bool(
            step is not None
            and getattr(step, "road_type", None) == "crossing"
            and dist is not None
            and dist <= self._config.nav.crossing_expect_m
        )

    def _announce(self, result: ProgressResult, position) -> None:
        self._update_crossing_expected(result)
        step = result.current_step
        self._rec.log_nav(
            result.status.value if hasattr(result.status, "value") else result.status,
            distance_to_next_m=getattr(result, "distance_to_next", None),
            step_text=(step.text if step is not None else None),
        )
        now = time.monotonic()

        if result.status == RouteStatus.WAYPOINT_HIT:
            # current_step is the REACHED step — the action to execute NOW. This
            # is the SECOND trigger of the two-stage turn: the approach already
            # said "X metre sonra sağa dönün"; at the node we say "Şimdi sağa
            # dönün". Start/continue steps just speak their text; arrival is owned
            # by FINISHED (never announced a segment early).
            if step is not None and step.action == "finish":
                self._prewarned_step_id = None
                return
            if step is not None and step.action in ("turn_left", "turn_right"):
                self._speak_event(f"Şimdi {step.text}")
                self._arm_turn_check(step, position, now)
            elif step is not None:
                self._speak_event(step.text)
            self._last_turn_at = now
            self._prewarned_step_id = None  # next step not yet pre-warned
            return

        if result.status == RouteStatus.FINISHED:
            self._speak_event("Hedefinize ulaştınız, iyi günler.")
            return

        # Judge the armed post-turn check (one verdict, then disarmed).
        self._maybe_confirm_turn(position, now)

        prog, dist_to_tgt = self._destination_progress(position)
        dist = result.distance_to_next

        # Wrong-way is the only hard warning; it applies whether the tracker says
        # OFF_ROUTE or PROGRESSING (the cross-track corridor makes a true
        # off-route rare, so we never go silent — we keep judging real progress).
        if prog == "wrong_way":
            self._speak_event("Yanlış yöne gidiyorsunuz, hedef geride kaldı.")
            return

        # (a) Approach pre-warning — once per step (turn instruction). Only while
        #     PROGRESSING and never for the finish step (arrival = FINISHED).
        if (
            result.status == RouteStatus.PROGRESSING
            and dist is not None and step is not None
            and step.action != "finish"
            and dist <= self._config.nav.approach_threshold_m
            and self._prewarned_step_id != step.step_id
        ):
            if step.action == "start":
                self._voice.say_nav(step.text)
            else:
                self._voice.say_nav(f"{self._dist_phrase(dist)} sonra {step.text}")
            self._prewarned_step_id = step.step_id
            self._last_spoken = step.text
            self._last_turn_at = now
            return

        # (b) Destination-progress ping. Works in OFF_ROUTE too (safety net), but
        #     is held back when a turn is imminent or was just announced, so the
        #     user never hears the turn distance and the target distance
        #     back-to-back ("26 m dönüş" then "62 m hedef" was confusing).
        near_turn = (
            dist is not None and step is not None and step.action != "finish"
            and dist <= self._config.nav.approach_threshold_m
        )
        just_turned = (
            (now - self._last_turn_at)
            < self._config.nav.progress_suppress_after_turn_sec
        )
        if near_turn or just_turned:
            return
        if (now - self._last_progress_time) > self._config.nav.progress_announce_interval:
            if prog == "on_track" and dist_to_tgt is not None:
                self._voice.say_progress(f"Doğru yoldasınız, hedefe {int(dist_to_tgt)} metre.")
                self._last_progress_time = now
            elif dist is not None and dist >= self._config.nav.long_stretch_threshold_m:
                self._voice.say_progress(f"Hedefe {int(dist)} metre.")
                self._last_progress_time = now

    # ── Post-turn confirmation (B3) ──────────────────────────────

    def _arm_turn_check(self, step, position, now: float) -> None:
        """Remember the bearing the route expects right after this turn."""
        if not self._config.nav.turn_confirm_enabled:
            return
        route = self._nav.get_route()
        nxt = next((s for s in route if s.step_id == step.step_id + 1), None)
        if nxt is None:
            return
        expected = calculate_bearing(
            step.location.lat, step.location.lon,
            nxt.location.lat, nxt.location.lon,
        )
        self._turn_check = {"expected": expected, "pos": position, "at": now}

    def _maybe_confirm_turn(self, position, now: float) -> None:
        """One verdict per turn: confirm, warn, or stay silent on ambiguity.

        Missing a turn is the costliest navigation error for a blind user —
        off-route detection only fires ~50 m later. Heading source is GPS
        course over ground when the module reports one, else the bearing of
        the user's own displacement since the turn (works on any GPS).
        """
        chk = self._turn_check
        if chk is None:
            return
        nav = self._config.nav
        if (now - chk["at"]) > nav.turn_confirm_timeout_sec:
            self._turn_check = None
            return
        moved = haversine_distance(position.lat, position.lon,
                                   chk["pos"].lat, chk["pos"].lon)
        if moved < nav.turn_confirm_min_move_m:
            return
        course = None
        get_course = getattr(self._gps, "get_course_deg", None)
        if get_course is not None:
            try:
                course = get_course()
            except Exception:  # noqa: BLE001
                course = None
        if course is None:
            course = calculate_bearing(chk["pos"].lat, chk["pos"].lon,
                                       position.lat, position.lon)
        diff = abs((course - chk["expected"] + 180.0) % 360.0 - 180.0)
        self._turn_check = None
        if diff <= nav.turn_confirm_tolerance_deg:
            self._voice.say_progress("Doğru yöne döndünüz.")
        elif diff >= nav.turn_wrong_threshold_deg:
            self._voice.say_nav("Yanlış yöne döndünüz, durun.")
        # else: GPS too ambiguous to judge — silence beats a wrong accusation.

    def _dist_phrase(self, dist_m: float) -> str:
        """Render a short distance for speech: steps when countable, else metres.

        Step counts are rounded to the nearest 5 — "23 adım" suggests a
        precision the GPS does not have, "yaklaşık 25 adım" does not.
        """
        nav = self._config.nav
        if nav.steps_phrasing and dist_m <= nav.steps_phrase_max_m:
            steps = max(5, int(round(dist_m / nav.step_length_m / 5.0)) * 5)
            return f"yaklaşık {steps} adım"
        return f"{int(dist_m)} metre"

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