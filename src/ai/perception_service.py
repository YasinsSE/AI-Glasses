"""PerceptionService — camera capture and inference loop.

Runs PerceptionPipeline at a fixed FPS and routes alerts through VoicePolicy.
Loop is gated on system mode, active TTS, and post-nav silence window.
"""

import contextlib
import logging
import os
import threading
import time
from typing import Optional

from ai.geometry import CameraGeometry
from ai.perception import Alert, ClassID, PerceptionPipeline, SAFETY_SAFE, SAFETY_CAUTION, SAFETY_UNSAFE
from main.config import ALASConfig
from main.lifecycle import ModeManager, SystemMode
from navigation.local_planner import VFHPlanner
from tts_stt.voice_policy import VoicePolicy

logger = logging.getLogger("ALAS.perception_service")


@contextlib.contextmanager
def _silence_native_stdio():
    """Mute C-level stdout+stderr for the duration of the block.

    The NVIDIA Argus camera stack (``GST_ARGUS:`` / ``CONSUMER:`` sensor-mode
    dump) and OpenCV's GStreamer backend print verbose native messages straight
    to file descriptors 1/2 — bypassing Python logging — every time the CSI
    camera opens (and on every wake from STANDBY). We swallow only that short
    window so the journal stays in the single ALAS log format. Python-level
    open failures are still detected via ``cap.isOpened()`` and logged normally.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


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
        vfh: Optional[VFHPlanner] = None,
        recorder=None,
        monitor=None,
    ) -> None:
        super().__init__(name="PerceptionService", daemon=True)
        self._config = config
        self._voice = voice
        self._modes = modes
        self._stop_event = stop_event
        self._nav = nav  # NavigationSystem reference for crosswalk filtering.
        self._vfh = vfh  # Optional local planner; None disables VFH guidance.
        self._monitor = monitor  # ActivityMonitor (auto-STANDBY) or None.
        self._prev_gray = None    # last downscaled grayscale frame (visual motion).
        from main.session_recorder import NullRecorder
        self._rec = recorder or NullRecorder()  # field-test black-box recorder

        self._pipeline: Optional[PerceptionPipeline] = None
        self._cap = None  # cv2.VideoCapture, lazy import.
        self._last_path_guidance: Optional[tuple] = None
        self._last_vfh: Optional[tuple] = None  # (text, monotonic timestamp)
        # Global minimum gap between any two obstacle speaks.
        self._last_obstacle_speak_at: float = 0.0

        # ── Situation tracking (replaces string-based dedupe) ──────────────
        # Speech gating keys on a stable "situation signature"
        # (hazard_class, zone, safety_level) rather than the rendered string,
        # so flickering proximity/direction words no longer defeat dedupe.
        # _cur_sig tracks frame-to-frame continuity (escalation counting);
        # _last_spoken_sig + _last_obstacle_speak_at drive the speak cadence.
        self._cur_sig: Optional[tuple] = None
        self._last_spoken_sig: Optional[tuple] = None
        self._last_safety: int = -1          # safety_level of the last spoken sig
        self._in_path_frames: int = 0        # consecutive frames hazard blocks path
        self._sig_escalated: bool = False    # current sig already escalated to a dodge
        # Hysteresis state, keyed by class_id, so zone/proximity wording only
        # changes after the new value persists. (class_id -> ...).
        self._zone_state: dict = {}          # cid -> (stable, pending, pending_count)
        self._band_state: dict = {}          # cid -> proximity band string

        # Tracks safety-level transitions for "Yol açık" announcement.
        self._last_safe_announce_at: float = -999.0
        self._prev_safety_level: int = -1  # -1 = unknown (first frame)
        # Loop heartbeat for the stall watchdog (set each iteration).
        self._last_heartbeat: float = 0.0
        self.model_ready = threading.Event()

    # ── Camera helpers (private) ─────────────────────────────────

    def _build_csi_pipeline(self) -> str:
        """Argus/ISP pipeline for CSI sensors (IMX219).

        We capture at a native sensor mode and let nvvidconv do the downscale on
        the ISP, so the only data crossing into CPU/NumPy is already at model
        input resolution in BGR. drop=true + max-buffers=1 keeps read() returning
        the freshest frame — stale frames pile up while TTS blocks the loop.
        """
        ai = self._config.ai
        return (
            f"nvarguscamerasrc sensor-id={ai.camera_index} "
            f"sensor-mode={ai.csi_sensor_mode} ! "
            f"video/x-raw(memory:NVMM), width={ai.csi_capture_width}, "
            f"height={ai.csi_capture_height}, format=(string)NV12, "
            f"framerate=(fraction){ai.csi_framerate}/1 ! "
            f"nvvidconv flip-method={ai.csi_flip_method} ! "
            f"video/x-raw, width={ai.camera_width}, height={ai.camera_height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! "
            f"appsink drop=true max-buffers=1"
        )

    def _open_camera(self) -> bool:
        import cv2

        ai = self._config.ai

        # CSI sensors need the Argus pipeline; v4l2 (index) only serves USB UVC.
        if ai.use_csi_camera:
            pipeline = self._build_csi_pipeline()
            # Mute the native Argus/GStreamer sensor-mode dump during open.
            with _silence_native_stdio():
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                opened = cap.isOpened()
            if not opened:
                logger.error(
                    "[Perception] Cannot open CSI camera via GStreamer. "
                    "Check `gst-launch-1.0 nvarguscamerasrc` works standalone."
                )
                try:
                    cap.release()  # free the partial handle before a retry
                except Exception:
                    pass
                return False
            logger.info(
                "[Perception] CSI camera opened (sensor-id=%d) — capture %dx%d, "
                "delivering %dx%d BGR",
                ai.camera_index, ai.csi_capture_width, ai.csi_capture_height,
                ai.camera_width, ai.camera_height,
            )
            self._cap = cap
            return True

        # --- USB UVC path -------------------------------------------------
        with _silence_native_stdio():
            cap = cv2.VideoCapture(ai.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ai.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ai.camera_height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        if not cap.isOpened():
            logger.error("[Perception] Cannot open camera.")
            try:
                cap.release()  # free the partial handle before a retry
            except Exception:
                pass
            return False

        try:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(
                chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
            ).strip()
            logger.info(
                "[Perception] Camera negotiated: %dx%d @ fourcc=%r (requested %dx%d)",
                actual_w, actual_h, fourcc, ai.camera_width, ai.camera_height,
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

    def _acquire_camera(self) -> bool:
        """Open the camera with retries.

        After a deep-STANDBY wake the OS may not have fully released the camera
        device yet (true for both USB UVC and the CSI/Argus stack on Jetson), so
        the first ``VideoCapture`` can fail to lock it. We retry a few times with
        a short delay and a settle pause on success, so the PTT wake recovers
        gracefully instead of dying.
        """
        attempts = 4
        delay = 1.0
        for i in range(attempts):
            if self._stop_event.is_set():
                return False
            if self._open_camera():
                # Let the device settle before the first read().
                self._stop_event.wait(0.3)
                return True
            logger.warning(
                "[Perception] camera open attempt %d/%d failed; retrying in %.1fs.",
                i + 1, attempts, delay,
            )
            self._stop_event.wait(delay)
        return False

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.exception("[Perception] camera release failed")
            self._cap = None

    def _visual_motion_metric(self, frame) -> Optional[float]:
        """Mean absolute frame-diff (0..255) vs the previous frame.

        Cheap stillness signal for the auto-STANDBY monitor: downscale to
        64x48 grayscale and average the per-pixel delta. Returns None on the
        first frame (no baseline yet) or on error.
        """
        try:
            import cv2
            import numpy as np
            small = cv2.resize(frame, (64, 48))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            prev = self._prev_gray
            self._prev_gray = gray
            if prev is None or prev.shape != gray.shape:
                return None
            return float(np.mean(cv2.absdiff(gray, prev)))
        except Exception:  # noqa: BLE001
            return None

    # ── Thread entry point ───────────────────────────────────────

    def run(self) -> None:
        # Model load is heavy; do it inside the thread so main() never blocks.
        try:
            self._pipeline = PerceptionPipeline(
                model_path=self._config.ai.model_path,
                input_h=self._config.ai.model_input_h,
                input_w=self._config.ai.model_input_w,
                camera_geometry=CameraGeometry(
                    height_m=self._config.ai.camera_height_m,
                    tilt_deg=self._config.ai.camera_tilt_deg,
                    vfov_deg=self._config.ai.camera_vfov_deg,
                ),
            )
        except Exception:
            logger.exception("[Perception] Model load failed")
            self._voice.emergency("Görüş sistemi başlatılamadı.")
            self.model_ready.set()  # Unblock await_ready so the user is not stuck.
            return

        # The camera is opened lazily by the loop on the first ACTIVE iteration
        # (and re-opened after each deep-STANDBY wake), so model readiness no
        # longer blocks on it. This keeps a single camera-acquire code path.
        self.model_ready.set()
        logger.info(
            "[Perception] Pipeline ready — target ~%.1f FPS",
            self._config.ai.perception_fps,
        )

        try:
            self._loop()
        finally:
            self._release_camera()
            logger.info("[Perception] Stopped.")

    # ── Main loop ────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._config.ai.perception_fps
        frames_done = 0
        window_start = time.monotonic()
        self._last_heartbeat = time.monotonic()

        while not self._stop_event.is_set():
            # ── Stall watchdog ───────────────────────────────────────────
            # The thread runs every iteration in well under a second; a large
            # gap between heartbeats means the whole process was starved (e.g.
            # Chromium hogging the Jetson). On recovery we drop the stale
            # situation state so the user is warned about the CURRENT scene,
            # not the one from before the freeze.
            beat = time.monotonic()
            gap = beat - self._last_heartbeat
            self._last_heartbeat = beat
            if gap > self._config.ai.stall_warn_sec:
                logger.warning("[Perception] loop stalled %.1fs — resetting situation state.", gap)
                self._reset_situation()
                if self._pipeline is not None:
                    self._pipeline.reset_cooldowns()
                if gap > self._config.ai.stall_announce_sec:
                    self._voice.say_obstacle("Görüş gecikiyor.")

            # Mode gate — skip in WARMUP / SLEEP / SHUTDOWN.
            if self._modes.mode != SystemMode.ACTIVE:
                # Deep STANDBY: fully close the camera to save power. Inference
                # is already idle here; releasing the device is the extra win.
                if self._cap is not None:
                    logger.info("[Perception] STANDBY — releasing camera to save power.")
                    self._release_camera()
                    self._prev_gray = None
                self._stop_event.wait(0.2)
                continue

            # ACTIVE: ensure the camera is open. After a PTT wake from deep
            # STANDBY this re-acquires the device (with retries).
            if self._cap is None:
                logger.info("[Perception] Waking — acquiring camera.")
                if not self._acquire_camera():
                    self._voice.emergency("Kamera açılamadı, tekrar deneniyor.")
                    self._stop_event.wait(1.0)
                    continue
                self._reset_situation()
                if self._pipeline is not None:
                    self._pipeline.reset_cooldowns()
                self._prev_gray = None

            # Active-utterance gate — wait on the event so we wake the instant
            # the priority utterance ends, instead of polling at 5 Hz.
            if self._voice.is_speaking_priority():
                self._voice.wait_until_idle(0.5)
                continue

            # Post-nav silence: obstacle alerts would be dropped anyway, so
            # skip inference entirely until the window passes.
            if self._voice.in_post_nav_silence():
                self._stop_event.wait(0.2)
                continue

            t0 = time.monotonic()
            frame = self._read_frame()
            if frame is None:
                self._stop_event.wait(0.2)
                continue

            # Feed the auto-STANDBY monitor with a visual-stillness sample.
            if self._monitor is not None:
                metric = self._visual_motion_metric(frame)
                if metric is not None:
                    self._monitor.report_visual(metric)

            try:
                result = self._pipeline.process(frame)
            except Exception:
                logger.exception("[Perception] pipeline.process failed")
                self._stop_event.wait(0.5)
                continue

            self._dispatch(result, frame)

            # FPS health log — surfaces the case where inference itself is
            # already slower than the requested interval.
            frames_done += 1
            if frames_done >= _FPS_LOG_WINDOW:
                now = time.monotonic()
                achieved = frames_done / (now - window_start)
                logger.debug(
                    "[Perception] sustained %.2f FPS (target %.1f)",
                    achieved, self._config.ai.perception_fps,
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
                self._stop_event.wait(timeout=sleep_time)

    # ── Alert dispatch ───────────────────────────────────────────

    def _dispatch(self, result, frame=None) -> None:
        """Filter, track, and forward perception output to the voice policy.

        Speech is gated on a stable *situation signature* — (hazard class,
        stabilised zone, safety level) — instead of the rendered string, so a
        parked car whose proximity/direction wording flickers frame-to-frame
        is announced once and then held for ``min_obstacle_repeat_sec``.

        A hazard that persists in the forward path escalates from a calm "var"
        notice to an actionable VFH dodge ("...hafif sağa yönelin"); one that
        drifts out of the path is dropped silently.
        """
        nav_active = self._nav is not None and self._nav.is_active
        now = time.monotonic()
        scene = result.scene
        safety_level = getattr(scene, "safety_level", SAFETY_UNSAFE)

        # Top obstacle alert (respecting nav-only classes like CROSSWALK).
        top_alert: Optional[Alert] = None
        for alert in result.alerts:
            if alert.class_id in _NAV_ONLY_CLASSES and not nav_active:
                continue
            top_alert = alert
            break

        # ── SAFE: "Yol açık" on transition, then periodic reminder ──────────
        if safety_level == SAFETY_SAFE:
            just_became_safe = self._prev_safety_level not in (SAFETY_SAFE, -1)
            interval_expired = (
                now - self._last_safe_announce_at
                >= self._config.ai.safe_announce_interval_sec
            )
            safe_msg = "Yol açık, devam edebilirsiniz."
            spoke = False
            if just_became_safe or interval_expired:
                self._voice.say_obstacle(safe_msg)
                self._last_safe_announce_at = now
                spoke = True
            self._prev_safety_level = SAFETY_SAFE
            self._reset_situation()  # left the hazard — next one starts fresh
            self._rec.log_perception(result, chosen=safe_msg if spoke else None)
            if spoke and frame is not None:
                self._rec.maybe_save_overlay(frame, result.mask, "safe", info={
                    "walkable": scene.walkable_ratio,
                })
            return

        self._prev_safety_level = safety_level

        # Plan a VFH escape route once for this frame. No announce cooldown
        # here — the signature cadence below decides what reaches the user.
        vfh = self._vfh_plan(result, nav_active)

        # ── No obstacle alert: fall back to directional guidance ────────────
        if top_alert is None:
            guidance_text = self._vfh_standalone_text(vfh, now)
            if guidance_text is None:
                guidance_text = self._select_path_guidance(
                    result.path_guidance, nav_active, now,
                )
            spoke = False
            if (guidance_text
                    and (now - self._last_obstacle_speak_at)
                    >= self._config.ai.min_obstacle_interval_sec):
                self._voice.say_obstacle(guidance_text)
                self._last_obstacle_speak_at = now
                spoke = True
            self._rec.log_perception(result, chosen=guidance_text if spoke else None)
            return

        # ── Obstacle present: stabilise wording, build signature ────────────
        cid = top_alert.class_id
        zone = self._stabilize_zone(cid, top_alert.zone)
        band = self._proximity_band(cid, top_alert.distance_m, top_alert.pixel_ratio)
        sig = (cid, zone, safety_level)

        # Frame-to-frame continuity (drives escalation counting).
        if sig != self._cur_sig:
            self._cur_sig = sig
            self._sig_escalated = False
            self._in_path_frames = 0

        in_path = (zone == "center") and top_alert.blocks_path
        self._in_path_frames = self._in_path_frames + 1 if in_path else 0
        can_dodge = vfh is not None and bool(getattr(vfh, "text", None))
        escalate = (
            in_path
            and self._in_path_frames >= self._config.ai.escalation_frames
            and can_dodge
        )
        escalate_now = escalate and not self._sig_escalated

        vfh_text = vfh.text if (escalate and can_dodge) else None
        message = self._render_message(top_alert, zone, band, scene, vfh_text, escalate)

        # ── Cadence: short on change/escalation, long when unchanged ────────
        sig_changed = sig != self._last_spoken_sig
        safety_increased = safety_level > self._last_safety
        if sig_changed or escalate_now or safety_increased:
            min_interval = self._config.ai.min_obstacle_interval_sec
        else:
            min_interval = self._config.ai.min_obstacle_repeat_sec
        interval_ok = (now - self._last_obstacle_speak_at) >= min_interval

        spoke = False
        if interval_ok and message:
            self._voice.say_obstacle(message)
            self._last_obstacle_speak_at = now
            self._last_spoken_sig = sig
            self._last_safety = safety_level
            self._sig_escalated = escalate
            spoke = True
            if self._pipeline is not None:
                self._pipeline.mark_alert_spoken(cid)

        self._rec.log_perception(result, chosen=message if spoke else None)
        if spoke and frame is not None:
            tag = (scene.dominant_hazard or "alert").replace(" ", "_")
            self._rec.maybe_save_overlay(frame, result.mask, tag, info={
                "walkable": scene.walkable_ratio,
            })

        if safety_level >= SAFETY_CAUTION:
            logger.debug(
                "[Perception] sig=%s escalate=%s walkable=%.0f%% inf=%.0fms",
                sig, escalate, scene.walkable_ratio * 100.0, result.inference_ms,
            )

    # ── Situation helpers ────────────────────────────────────────

    def _reset_situation(self) -> None:
        """Forget all hazard tracking — used on SAFE and after a loop stall."""
        self._cur_sig = None
        self._last_spoken_sig = None
        self._last_safety = -1
        self._in_path_frames = 0
        self._sig_escalated = False
        self._zone_state.clear()
        self._band_state.clear()
        self._last_obstacle_speak_at = 0.0

    def _stabilize_zone(self, class_id: int, zone: str) -> str:
        """Hysteresis on the dominant zone: a new zone must persist two frames
        before it replaces the current one, so a single jittery frame does not
        flip "önünüzde" ↔ "solunuzda" and spawn a fresh signature."""
        stable, pending, count = self._zone_state.get(class_id, (zone, zone, 0))
        if zone == stable:
            self._zone_state[class_id] = (stable, stable, 0)
            return stable
        if zone == pending:
            count += 1
        else:
            pending, count = zone, 1
        if count >= 2:
            self._zone_state[class_id] = (zone, zone, 0)
            return zone
        self._zone_state[class_id] = (stable, pending, count)
        return stable

    def _proximity_band(
        self, class_id: int, distance_m: Optional[float], pixel_ratio: float,
    ) -> str:
        """Return ``'very_close' | 'near' | 'far'`` with hysteresis so the band
        does not dither across a threshold from frame to frame."""
        from ai.perception import VERY_CLOSE_M, NEARBY_M, VERY_CLOSE_RATIO, NEARBY_RATIO
        cur = self._band_state.get(class_id, "far")
        m = self._config.ai.proximity_hysteresis_m
        if distance_m is not None:
            if cur == "very_close":
                band = ("very_close" if distance_m < VERY_CLOSE_M + m
                        else "near" if distance_m < NEARBY_M + m else "far")
            elif cur == "near":
                band = ("very_close" if distance_m < VERY_CLOSE_M - m
                        else "near" if distance_m < NEARBY_M + m else "far")
            else:  # far
                band = ("very_close" if distance_m < VERY_CLOSE_M - m
                        else "near" if distance_m < NEARBY_M - m else "far")
        else:
            band = ("very_close" if pixel_ratio > VERY_CLOSE_RATIO
                    else "near" if pixel_ratio > NEARBY_RATIO else "far")
        self._band_state[class_id] = band
        return band

    _LOC_WORD = {"left": "Solunuzda", "right": "Sağınızda", "center": "Önünüzde"}
    _HAZARD_NOUN = {
        int(ClassID.VEHICLE): "araç",
        int(ClassID.COLLISION_OBSTACLE): "engel",
        int(ClassID.DYNAMIC_HAZARD): "hareketli nesne",
    }

    def _render_message(self, alert, zone, band, scene, vfh_text, escalate) -> str:
        """Compose a short, prioritised Turkish line from the structured alert.

        Order of preference: fully-blocked stop > actionable VFH dodge >
        centre-blocking stop > calm "var" notice. At most one modifier
        (proximity) is appended, and only when very close.
        """
        cid = alert.class_id

        # Non-positional hazards have fixed phrasing.
        if cid == int(ClassID.VEHICLE_ROAD):
            return "Dikkat, araç yolu, girmeyin"
        if cid == int(ClassID.CROSSWALK):
            return "Yaya geçidi, geçiş güvenli"
        if cid == int(ClassID.FALL_HAZARD):
            return "Zemin tehlikesi, dikkatli ilerleyin"

        noun = self._HAZARD_NOUN.get(cid, "engel")
        loc = self._LOC_WORD.get(zone, "Önünüzde")

        # No way through at all.
        if alert.blocks_path and scene.walkable_ratio < 0.03:
            return "Durun, yol kapalı"

        # Persisting in the path and we have a clear escape sector → dodge.
        if escalate and vfh_text:
            dodge = vfh_text[0].lower() + vfh_text[1:] if vfh_text else vfh_text
            return f"Önünüzde {noun} var, {dodge}"

        # In the path, blocking, but no dodge yet → cautious stop.
        if zone == "center" and alert.blocks_path:
            return f"Durun, önünüzde {noun}"

        # Calm informational notice.
        msg = f"{loc} {noun} var"
        if band == "very_close":
            msg += ", çok yakın"
        return msg

    def _vfh_plan(self, result, nav_active: bool):
        """Run the VFH local planner for this frame; return its guidance or None.

        Unlike the old path, this does NOT apply an announce cooldown — the
        signature cadence in ``_dispatch`` owns repeat suppression now.
        """
        if self._vfh is None:
            return None
        target_action = None
        if nav_active and self._nav is not None:
            step = getattr(self._nav, "current_step", None)
            if step is not None:
                target_action = getattr(step, "action", None)
        try:
            guidance = self._vfh.plan(
                result.mask, result.scene, target_action=target_action,
            )
        except Exception:
            logger.exception("[Perception] VFH plan failed")
            return None
        if guidance is not None:
            logger.debug(
                "[Perception] VFH: action=%s sector=%d",
                guidance.action.value, guidance.sector_index,
            )
        return guidance

    def _vfh_standalone_text(self, guidance, now: float) -> Optional[str]:
        """VFH line for the no-obstacle case, with its own announce cooldown."""
        if guidance is None or not getattr(guidance, "text", None):
            return None
        cooldown = self._config.vfh.announce_cooldown_sec
        if self._last_vfh is not None:
            last_text, last_ts = self._last_vfh
            if guidance.text == last_text and (now - last_ts) < cooldown:
                return None
        self._last_vfh = (guidance.text, now)
        return guidance.text

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

        cooldown = self._config.ai.path_guidance_cooldown_sec
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

