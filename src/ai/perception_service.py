"""PerceptionService — camera capture + perception pipeline as a long-running thread.

Owns the camera lifecycle (open / read / release) and runs the
``PerceptionPipeline`` at a controlled FPS, then routes alerts through the
shared ``VoicePolicy``.

Three gates control whether the loop body runs:

    1. ``modes.mode == ACTIVE``        — skip everything in WARMUP / SLEEP.
    2. ``voice.is_speaking_priority()`` — skip while a nav announcement plays.
    3. ``voice.in_post_nav_silence()`` — also skip during the post-nav window
       (obstacle alerts would be muted anyway, so save the inference cost).

The cv2 lifecycle is intentionally inlined as private helpers rather than a
separate ``CameraSource`` class — there is one consumer and the helper is
twenty lines, so a stand-alone class would be premature abstraction.
"""

import logging
import threading
import time
from typing import Optional

from ai.geometry import CameraGeometry
from ai.perception import Alert, ClassID, PerceptionPipeline
from main.config import ALASConfig
from main.lifecycle import ModeManager, SystemMode
from tts_stt.voice_policy import VoicePolicy

logger = logging.getLogger("ALAS.perception_service")

# Average sustained FPS over this many frames before logging it.
_FPS_LOG_WINDOW = 30

# Classes that only make sense to announce while a navigation route is active.
_NAV_ONLY_CLASSES = {int(ClassID.CROSSWALK)}


class PerceptionService(threading.Thread):
    """Camera capture + perception inference loop."""

    def __init__(
        self,
        config: ALASConfig,
        voice: VoicePolicy,
        modes: ModeManager,
        stop_event: threading.Event,
        nav=None,
    ) -> None:
        super().__init__(name="PerceptionService", daemon=True)
        self._config = config
        self._voice = voice
        self._modes = modes
        self._stop = stop_event
        self._nav = nav  # NavigationSystem reference for crosswalk filtering.

        self._pipeline: Optional[PerceptionPipeline] = None
        self._cap = None  # cv2.VideoCapture, lazy import.
        # Dedupe identical TTS lines. (text, monotonic timestamp). The TTL
        # prevents the user from being silently denied a real warning when
        # they re-encounter the same hazard after walking away.
        self._last_spoken: Optional[tuple] = None
        self._last_path_guidance: Optional[tuple] = None
        self.model_ready = threading.Event()

    # ── Camera helpers (private) ─────────────────────────────────

    def _open_camera(self) -> bool:
        import cv2  # Local import keeps cv2 out of pure unit tests.

        cap = cv2.VideoCapture(self._config.camera_index)
        # Capture directly at the model's input resolution to skip one resize.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)
        # Buffersize=1 makes read() always return the freshest frame instead
        # of whatever queued up while TTS was speaking.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # MJPG fourcc relieves USB bandwidth pressure on Jetson Nano. Some
        # CSI/GStreamer pipelines reject it — best-effort.
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        if not cap.isOpened():
            logger.error("[Perception] Cannot open camera.")
            return False

        # USB cams silently downgrade fourcc / resolution. Log what we
        # actually got so a "slow FPS" report can be diagnosed without
        # plugging in a monitor.
        try:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(
                chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
            ).strip()
            logger.info(
                "[Perception] Camera negotiated: %dx%d @ fourcc=%r (requested %dx%d)",
                actual_w, actual_h, fourcc,
                self._config.camera_width, self._config.camera_height,
            )
        except Exception:
            logger.debug("[Perception] Could not read back camera settings.")

        self._cap = cap
        return True

    def _read_frame(self):
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("[Perception] Frame grab failed.")
            return None
        return frame

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.exception("[Perception] camera release failed")
            self._cap = None

    # ── Thread entry point ───────────────────────────────────────

    def run(self) -> None:
        # Model load is heavy; do it inside the thread so main() never blocks.
        try:
            self._pipeline = PerceptionPipeline(
                model_path=self._config.model_path,
                input_h=self._config.model_input_h,
                input_w=self._config.model_input_w,
                camera_geometry=CameraGeometry(
                    height_m=self._config.camera_height_m,
                    tilt_deg=self._config.camera_tilt_deg,
                    vfov_deg=self._config.camera_vfov_deg,
                ),
            )
        except Exception:
            logger.exception("[Perception] Model load failed")
            self._voice.emergency("Görüş sistemi başlatılamadı.")
            self.model_ready.set()  # Unblock await_ready so the user is not stuck.
            return

        if not self._open_camera():
            self._voice.emergency("Kamera açılamadı.")
            self.model_ready.set()
            return

        self.model_ready.set()
        logger.info(
            "[Perception] Pipeline ready — target ~%.1f FPS",
            self._config.perception_fps,
        )

        try:
            self._loop()
        finally:
            self._release_camera()
            logger.info("[Perception] Stopped.")

    # ── Main loop ────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._config.perception_fps
        frames_done = 0
        window_start = time.monotonic()

        while not self._stop.is_set():
            # Mode gate — skip in WARMUP / SLEEP / SHUTDOWN.
            if self._modes.mode != SystemMode.ACTIVE:
                self._stop.wait(0.2)
                continue

            # Active-utterance gate — wait on the event so we wake the instant
            # the priority utterance ends, instead of polling at 5 Hz.
            if self._voice.is_speaking_priority():
                self._voice.wait_until_idle(0.5)
                continue

            # Post-nav silence: obstacle alerts would be dropped anyway, so
            # skip inference entirely until the window passes.
            if self._voice.in_post_nav_silence():
                self._stop.wait(0.2)
                continue

            t0 = time.monotonic()
            frame = self._read_frame()
            if frame is None:
                self._stop.wait(0.2)
                continue

            try:
                result = self._pipeline.process(frame)
            except Exception:
                logger.exception("[Perception] pipeline.process failed")
                self._stop.wait(0.5)
                continue

            self._dispatch(result)

            # FPS health log — surfaces the case where inference itself is
            # already slower than the requested interval.
            frames_done += 1
            if frames_done >= _FPS_LOG_WINDOW:
                now = time.monotonic()
                achieved = frames_done / (now - window_start)
                logger.info(
                    "[Perception] sustained %.2f FPS (target %.1f)",
                    achieved, self._config.perception_fps,
                )
                if result.total_ms > interval * 1000.0:
                    logger.warning(
                        "[Perception] frame time %.0fms > interval %.0fms — "
                        "FPS cap is no longer the bottleneck.",
                        result.total_ms, interval * 1000.0,
                    )
                frames_done = 0
                window_start = now

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop.wait(timeout=sleep_time)

    # ── Alert dispatch ───────────────────────────────────────────

    def _dispatch(self, result) -> None:
        """Filter, dedupe, and forward perception output to the voice policy.

        Three independent decisions:
          1. Pick the top obstacle alert (filtered by class for nav-only
             classes like CROSSWALK).
          2. Decide whether to also emit path guidance — only when no nav
             route is active (the route already gives directional cues) and
             only when the guidance text actually changed (or its TTL elapsed).
          3. Combine, dedupe against the last-spoken line with a TTL, speak
             once, then stamp cooldown on the alert that was actually spoken.
        """
        nav_active = self._nav is not None and self._nav.is_active
        now = time.monotonic()

        top_alert: Optional[Alert] = None
        for alert in result.alerts:
            if alert.class_id in _NAV_ONLY_CLASSES and not nav_active:
                continue
            top_alert = alert
            break

        guidance_text = self._select_path_guidance(
            result.path_guidance, nav_active, now,
        )

        if guidance_text and top_alert is not None:
            message = "{} — {}".format(guidance_text, top_alert.text)
        elif guidance_text:
            message = guidance_text
        elif top_alert is not None:
            message = top_alert.text
        else:
            message = None

        if message and self._should_speak(message, now):
            self._voice.say_obstacle(message)
            self._last_spoken = (message, now)
            if top_alert is not None and self._pipeline is not None:
                # Cooldown is consumed only by speech that actually reached
                # the user, never by a candidate that lost to dedupe.
                self._pipeline.mark_alert_spoken(top_alert.class_id)

        if not result.scene.is_safe:
            logger.info(
                "[Perception] Hazard: %s | walkable: %.0f%% | "
                "inf: %.0fms total: %.0fms",
                result.scene.dominant_hazard,
                result.scene.walkable_ratio * 100.0,
                result.inference_ms, result.total_ms,
            )

    def _select_path_guidance(
        self,
        guidance_text: Optional[str],
        nav_active: bool,
        now: float,
    ) -> Optional[str]:
        """Decide whether path guidance should reach the user this frame.

        Suppressed entirely when navigation is active — the route's own
        turn-by-turn announcements already tell the user which way to go,
        and stacking ambient guidance on top creates conflicting cues.

        When navigation is idle, the same line is only re-emitted after a
        cooldown so the user is not bombarded with "Düz yürüyün" every
        single frame they stand still.
        """
        if guidance_text is None or nav_active:
            return None

        cooldown = self._config.path_guidance_cooldown_sec
        if self._last_path_guidance is None:
            self._last_path_guidance = (guidance_text, now)
            return guidance_text

        last_text, last_ts = self._last_path_guidance
        changed = guidance_text != last_text
        elapsed = now - last_ts
        if changed or elapsed >= cooldown:
            self._last_path_guidance = (guidance_text, now)
            return guidance_text
        return None

    def _should_speak(self, message: str, now: float) -> bool:
        """Dedupe identical consecutive utterances, but not forever.

        Without a TTL the user is silently denied a real warning when they
        re-encounter the same obstacle later in the walk. The TTL gives
        ``generate_alerts`` cooldowns enough headroom to do their job while
        still re-warning when the world has had time to change.
        """
        if self._last_spoken is None:
            return True
        last_text, last_ts = self._last_spoken
        if message != last_text:
            return True
        return (now - last_ts) >= self._config.obstacle_dedupe_ttl_sec
