"""
PerceptionService — camera + perception pipeline as a long-running thread.
==========================================================================
Owns the camera lifecycle (open / read / release) and runs the
``PerceptionPipeline`` at a controlled FPS. Speaks any alerts through the
shared ``VoicePolicy``.

Two gates control whether the loop body runs:

    1. ``modes.mode == ACTIVE`` — skip everything in WARMUP and SLEEP.
    2. ``voice.is_speaking_priority()`` — skip inference while a navigation
       announcement is being spoken (matches the original ``pause_event``).

The post-nav silence window is enforced *inside* ``voice.say_obstacle()`` —
perception still runs the pipeline so scene state stays fresh, the alerts
just get dropped silently.

The cv2 lifecycle is intentionally inlined as a private helper (``_open``,
``_read``, ``_release``) rather than a separate ``CameraSource`` class — there
is one consumer and the helper is fifteen lines, so a stand-alone class would
be premature abstraction.
"""

#from __future__ import annotations

import logging
import threading
import time
from typing import Optional
from typing import Dict
from ai.geometry import CameraGeometry
from ai.perception import PerceptionPipeline
from main.config import ALASConfig
from main.lifecycle import ModeManager, SystemMode
from tts_stt.voice_policy import VoicePolicy

logger = logging.getLogger("ALAS.perception_service")


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
        self._nav = nav  # NavigationSystem reference for crosswalk filtering

        self._pipeline: Optional[PerceptionPipeline] = None
        self._cap = None  # cv2.VideoCapture, lazy import
        self.model_ready = threading.Event()

    # ── Camera helper (private) ──────────────────────────────────

    def _open_camera(self) -> bool:
        import cv2  # local import keeps cv2 out of pure unit tests

        self._cap = cv2.VideoCapture(self._config.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)
        if not self._cap.isOpened():
            logger.error("[Perception] Cannot open camera.")
            return False
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
            except Exception:  # noqa: BLE001
                logger.exception("[Perception] camera release failed")
            self._cap = None

    # ── Thread entry point ───────────────────────────────────────

    def run(self) -> None:
        # Model load — heavy, done inside the thread so main() does not block.
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
        except Exception:  # noqa: BLE001
            logger.exception("[Perception] Model load failed")
            self._voice.emergency("Görüş sistemi başlatılamadı.")
            self.model_ready.set()  # unblock await_ready so user is not stuck
            return

        if not self._open_camera():
            self._voice.emergency("Kamera açılamadı.")
            self.model_ready.set()
            return

        self.model_ready.set()
        logger.info(
            f"[Perception] Pipeline ready — ~{self._config.perception_fps} FPS"
        )

        try:
            self._loop()
        finally:
            self._release_camera()
            logger.info("[Perception] Stopped.")

    # ── Main loop ────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._config.perception_fps

        while not self._stop.is_set():
            # Mode gate — skip in WARMUP / SLEEP / SHUTDOWN
            if self._modes.mode != SystemMode.ACTIVE:
                self._stop.wait(0.2)
                continue

            # Active-utterance gate — do not waste inference during nav speech
            if self._voice.is_speaking_priority():
                self._stop.wait(0.2)
                continue

            t0 = time.monotonic()
            frame = self._read_frame()
            if frame is None:
                self._stop.wait(0.2)
                continue

            try:
                result = self._pipeline.process(frame)
            except Exception:  # noqa: BLE001
                logger.exception("[Perception] pipeline.process failed")
                self._stop.wait(0.5)
                continue

            # Filter alerts: crosswalk only announced when navigation is active
            nav_active = self._nav is not None and self._nav.is_active
            filtered_alerts = [
                a for a in result.alerts
                if not ("geçidi" in a and not nav_active)
            ]
            top_hazard = filtered_alerts[0] if filtered_alerts else None

            if result.path_guidance:
                combined = (
                    f"{result.path_guidance} — {top_hazard}"
                    if top_hazard
                    else result.path_guidance
                )
                self._voice.say_obstacle(combined)
            elif top_hazard:
                self._voice.say_obstacle(top_hazard)

            if not result.scene.is_safe:
                logger.info(
                    f"[Perception] Hazard: {result.scene.dominant_hazard} | "
                    f"walkable: {result.scene.walkable_ratio:.0%} | "
                    f"inf: {result.inference_ms:.0f}ms total: {result.total_ms:.0f}ms"
                )

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop.wait(timeout=sleep_time)
