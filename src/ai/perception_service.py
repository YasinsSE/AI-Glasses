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
from ai.perception import (Alert, ClassID, PerceptionPipeline, VERY_CLOSE_RATIO,
                           SAFETY_SAFE, SAFETY_CAUTION, SAFETY_UNSAFE)
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
        collector=None,
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
        from main.dataset_collector import NullCollector
        self._collector = collector or NullCollector()  # raw-frame fine-tuning capture

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
        # Closing-threat tracking: frame-to-frame walkable / distance of the top
        # obstacle, so a rapidly approaching hazard re-warns even when its
        # situation signature is unchanged.
        self._prev_walkable: Optional[float] = None
        self._prev_top_distance: Optional[float] = None
        # Consecutive frames the closing signal has held (anti-noise gate before
        # a side obstacle can fire an imminent alarm).
        self._closing_frames: int = 0
        # Path-keeping (Faz 2): last guidance line + when it was spoken.
        self._last_path_speak_at: float = -999.0
        self._last_path_text: Optional[str] = None
        # Path-keeping smoothing + ambient awareness (Faz 3).
        self._offset_ema: Optional[float] = None
        self._drift_dir: str = "straight"      # "left" | "right" | "straight"
        self._drift_count: int = 0
        # Ambient awareness (Faz 3/4): per-(class,zone) last-announce time gives a
        # re-arm memory so a hazard that flickers out of detection and back is not
        # re-announced as "new"; _last_ambient_at is the global min-gap.
        self._ambient_last_at: dict = {}
        self._last_ambient_at: float = -999.0
        # Narrow-passage state (Faz 6): True while squeezing between obstacles.
        self._in_narrow: bool = False
        self._narrow_count: int = 0   # consecutive frames past the enter/exit threshold
        # Crossing / road-ahead (Faz 4): consecutive crossing-candidate frames and
        # the last time we voiced a road/crossing caution.
        self._crossing_frames: int = 0
        self._last_road_at: float = -999.0
        # Loop heartbeat for the stall watchdog (set each iteration).
        self._last_heartbeat: float = 0.0
        # Thermal guard: True while the loop runs at thermal_min_fps because the
        # SoC is hot. Checked every thermal_check_interval_sec from sysfs.
        self._thermal_throttled: bool = False
        self._last_thermal_check: float = 0.0
        # Low-light state: warn once when the scene goes dark (model output is
        # unreliable there), reassure once when light returns.
        self._low_light: bool = False
        self._low_light_count: int = 0
        self._last_low_light_at: float = -999.0
        # Adaptive-FPS calm counter: consecutive frames with a SAFE scene and a
        # still camera (drops the loop to fps_idle once it persists).
        self._calm_frames: int = 0
        # Fast-path collision tripwire (B5) state.
        self._fast_collision_frames: int = 0
        self._last_fast_collision_at: float = -999.0
        # Last captured BGR frame, for on-demand consumers (the "oku" OCR
        # command). cv2.read() allocates a fresh array per frame, so handing
        # out the reference is safe.
        self.last_frame = None
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
            f"videorate drop-only=true ! "
            f"video/x-raw, framerate=(fraction){ai.csi_delivery_fps}/1 ! "
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
        # Cap OpenCV's worker pool: resize/cvtColor on a 512x384 frame gains
        # nothing from 4 threads, and Vosk + the ONNX intent classifier need
        # the other Nano cores. Must run before the first cv2 operation.
        try:
            import cv2
            cv2.setNumThreads(2)
        except Exception:  # noqa: BLE001
            pass

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
            try:
                self._collector.close()
            except Exception:  # noqa: BLE001
                pass
            logger.info("[Perception] Stopped.")

    # ── Main loop ────────────────────────────────────────────────

    def _check_thermal(self, now: float) -> None:
        """Deliberate degrade instead of silent DVFS throttling.

        The Waveshare carrier's cooling lets the SoC creep toward the throttle
        point under sustained TensorRT load; once DVFS kicks in, inference time
        silently doubles. We read the thermal zones every few seconds and, past
        ``thermal_throttle_c``, drop the loop to ``thermal_min_fps`` ourselves —
        keeping per-frame latency predictable — then recover with hysteresis.
        """
        ai = self._config.ai
        if not ai.thermal_guard_enabled:
            return
        if (now - self._last_thermal_check) < ai.thermal_check_interval_sec:
            return
        self._last_thermal_check = now
        from main.session_recorder import read_soc_temps
        temps = read_soc_temps()
        if not temps:
            return  # not a Jetson (desktop/mock) — guard is a no-op
        hottest = max(temps.values())
        if not self._thermal_throttled and hottest >= ai.thermal_throttle_c:
            self._thermal_throttled = True
            logger.warning(
                "[Perception] SoC %.1fC >= %.1fC — throttling to %.1f FPS.",
                hottest, ai.thermal_throttle_c, ai.thermal_min_fps,
            )
        elif self._thermal_throttled and hottest <= ai.thermal_recover_c:
            self._thermal_throttled = False
            logger.info(
                "[Perception] SoC cooled to %.1fC — restoring %.1f FPS.",
                hottest, ai.perception_fps,
            )

    def _target_interval(self) -> float:
        """Seconds per frame for the CURRENT conditions.

        Priority order: thermal guard (never speed up while hot) > hazard
        boost (UNSAFE scene or a closing threat) > idle drop (sustained
        SAFE + still camera) > the configured base rate.
        """
        ai = self._config.ai
        if self._thermal_throttled:
            fps = ai.thermal_min_fps
        elif not ai.adaptive_fps_enabled:
            fps = ai.perception_fps
        elif self._prev_safety_level == SAFETY_UNSAFE or self._closing_frames > 0:
            fps = max(ai.fps_alert, ai.perception_fps)
        elif self._calm_frames >= ai.idle_safe_persist_frames:
            fps = min(ai.fps_idle, ai.perception_fps)
        else:
            fps = ai.perception_fps
        return 1.0 / max(fps, 0.1)

    def _loop(self) -> None:
        interval = self._target_interval()
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
            self._check_thermal(beat)
            interval = self._target_interval()
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

            # SAFETY: keep perceiving while a priority utterance plays or during
            # the post-nav silence window — but MUTED, so only an imminent
            # collision (which preempts the audio) reaches the user. Previously
            # the loop skipped inference entirely here, so a car approaching
            # during a turn instruction was invisible until the sentence ended.
            muted = (self._voice.is_speaking_priority()
                     or self._voice.in_post_nav_silence())

            t0 = time.monotonic()
            frame = self._read_frame()
            if frame is None:
                self._stop_event.wait(0.2)
                continue
            self.last_frame = frame

            # Visual-stillness sample: feeds the auto-STANDBY monitor AND the
            # adaptive-FPS calm counter (SAFE scene + still camera → fps_idle).
            metric = self._visual_motion_metric(frame)
            if self._monitor is not None and metric is not None:
                self._monitor.report_visual(metric)
            calm = (self._prev_safety_level == SAFETY_SAFE
                    and metric is not None
                    and metric < self._config.ai.idle_motion_threshold)
            self._calm_frames = self._calm_frames + 1 if calm else 0

            # Low-light honesty check: warn (once) when the scene is too dark
            # for the model to be trusted.
            self._check_low_light(frame, t0, muted=muted)

            try:
                result = self._pipeline.process(frame)
            except Exception:
                logger.exception("[Perception] pipeline.process failed")
                self._stop_event.wait(0.5)
                continue

            self._dispatch(result, frame, muted=muted)

            # Heartbeat overlay — guarantees a visual timeline regardless of what
            # _dispatch decided to say. (_dispatch may save an event-tagged frame
            # first; this is throttled by frame_min_interval_s so it only fills
            # the gaps — e.g. long path-keeping stretches that never hit the
            # obstacle branch, which previously left almost no frames saved.)
            self._rec.maybe_save_overlay(
                frame, result.mask, "scene",
                info={"walkable": result.scene.walkable_ratio},
            )

            # Raw-frame capture for offline fine-tuning (--capture-dataset). Saves
            # the CLEAN frame (no overlay) + optional predicted mask, throttled.
            self._collector.maybe_capture(frame, result.mask)

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

    def _check_low_light(self, frame, now: float, muted: bool = False) -> None:
        """Warn once when the scene is too dark for the model to be trusted.

        The model fails SILENTLY in the dark — masks still look plausible — so
        the honest move is to tell the user vision is degraded and let them
        weight the cane over the glasses. Mean brightness of a sparse pixel
        sample; persistence + hysteresis keep a shadow or underpass from
        flapping the message.
        """
        ai = self._config.ai
        try:
            brightness = float(frame[::16, ::16].mean())
        except Exception:  # noqa: BLE001
            return
        if not self._low_light:
            self._low_light_count = (self._low_light_count + 1
                                     if brightness < ai.low_light_enter else 0)
            if self._low_light_count >= ai.low_light_persist_frames:
                self._low_light = True
                self._low_light_count = 0
                if not muted and (now - self._last_low_light_at) >= ai.low_light_rearm_sec:
                    self._last_low_light_at = now
                    self._voice.say_obstacle(
                        "Ortam karanlık, görüşüm sınırlı. Dikkatli ilerleyin."
                    )
        else:
            self._low_light_count = (self._low_light_count + 1
                                     if brightness > ai.low_light_exit else 0)
            if self._low_light_count >= ai.low_light_persist_frames:
                self._low_light = False
                self._low_light_count = 0
                if not muted:
                    self._voice.say_obstacle("Görüş normale döndü.")

    # ── Alert dispatch ───────────────────────────────────────────

    def _dispatch(self, result, frame=None, muted: bool = False) -> None:
        """Filter, track, and forward perception output to the voice policy.

        Speech is gated on a stable *situation signature* — (hazard class,
        stabilised zone, safety level) — instead of the rendered string, so a
        parked car whose proximity/direction wording flickers frame-to-frame
        is announced once and then held for ``min_obstacle_repeat_sec``.

        A hazard that persists in the forward path escalates from a calm "var"
        notice to an actionable VFH dodge ("...hafif sağa yönelin"); one that
        drifts out of the path is dropped silently.

        ``muted=True`` (a priority utterance is playing or the post-nav silence
        window is open): situation state is still tracked and frames recorded,
        but ONLY an imminent collision is allowed to speak — and it preempts the
        playing audio. Everything else stays silent until the window passes.
        """
        nav_active = self._nav is not None and self._nav.is_active
        now = time.monotonic()
        scene = result.scene
        safety_level = getattr(scene, "safety_level", SAFETY_UNSAFE)

        # Fast-path tripwire FIRST: a near-centre region drowning in hazard
        # pixels speaks a preempting stop regardless of what the cadence and
        # signature logic below would decide. Independent by design — a bug in
        # the gating chain must not be able to swallow a wall-in-the-face.
        if self._fast_collision_check(result, now):
            self._rec.log_perception(result, chosen="Durun, önünüz kapalı")
            if frame is not None:
                self._rec.maybe_save_overlay(frame, result.mask, "fast_collision",
                                             info={"walkable": scene.walkable_ratio})
            return

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
            if (just_became_safe or interval_expired) and not muted:
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

        ai = self._config.ai
        cid = top_alert.class_id if top_alert is not None else None
        zone = self._stabilize_zone(cid, top_alert.zone) if top_alert is not None else "center"
        in_path = top_alert is not None and zone == "center" and top_alert.blocks_path

        # Frame-to-frame CLOSING signal (computed up front so it gates both the
        # imminent-side override and the in-path urgency). prev_* updated once.
        dist = top_alert.distance_m if top_alert is not None else None
        walk_drop = ((self._prev_walkable - scene.walkable_ratio)
                     if self._prev_walkable is not None else 0.0)
        dist_drop = ((self._prev_top_distance - dist)
                     if (self._prev_top_distance is not None and dist is not None) else 0.0)
        self._prev_walkable = scene.walkable_ratio
        self._prev_top_distance = dist
        closing = (walk_drop >= ai.walkable_drop_urgent
                   or dist_drop >= ai.closing_distance_urgent_m)
        # The closing signal must PERSIST across frames before it can fire a side
        # alarm: a single noisy distance jump (uncalibrated FOV / class flicker)
        # must not be read as "rushing at me".
        self._closing_frames = self._closing_frames + 1 if closing else 0
        closing_sustained = self._closing_frames >= ai.closing_persist_frames
        is_close = (top_alert is not None and cid != int(ClassID.CROSSWALK)
                    and ((dist is not None and dist < ai.imminent_distance_m)
                         or top_alert.pixel_ratio > VERY_CLOSE_RATIO))

        # Imminent SIDE override — ONLY a genuine collision course. A side
        # obstacle qualifies only when it actually intrudes the walking corridor
        # (it is moving into the path ahead) AND we are sustainedly closing on
        # it. A parked car we merely walk PAST sits beside the corridor
        # (low corridor_overlap) → never imminent. This is what stopped the
        # "araç çok yakın" spam on every parked car (keci field test).
        in_corridor = (top_alert is not None
                       and top_alert.corridor_overlap >= ai.imminent_corridor_min)
        imminent_side = (not in_path) and is_close and closing_sustained and in_corridor

        # ── ROAD STRAIGHT AHEAD (Faz 4) ─────────────────────────────────────
        # vehicle_road only becomes the top alert when no vehicle is relevant
        # (vehicle outranks it). A road ahead is informational/cautionary, not a
        # hard-stop loop — and may upgrade to a crossing notice. Handle it on its
        # own event cadence and return.
        if cid == int(ClassID.VEHICLE_ROAD):
            if muted:  # priority audio playing → only collisions may speak
                self._rec.log_perception(result, chosen=None)
                return
            self._emit_road_ahead(result, frame, now, nav_active)
            return

        # ── PATH-KEEPING (balanced UX) ──────────────────────────────────────
        # Nothing blocks the path AND nothing is imminently closing → keep the
        # user on the sidewalk + occasional obstacle awareness (Faz 2/3).
        if not (in_path or imminent_side):
            if muted:
                self._rec.log_perception(result, chosen=None)
                return
            self._emit_path_keeping(result, frame, now, nav_active, top_alert, zone)
            return

        # ── WARNING: centerline block OR imminent closing side obstacle ─────
        band = self._proximity_band(cid, top_alert.distance_m, top_alert.pixel_ratio)
        sig = (cid, zone, safety_level)
        if sig != self._cur_sig:
            self._cur_sig = sig
            self._sig_escalated = False
            self._in_path_frames = 0
        self._in_path_frames = self._in_path_frames + 1 if in_path else 0
        can_dodge = vfh is not None and bool(getattr(vfh, "text", None))
        escalate = (in_path and self._in_path_frames >= ai.escalation_frames and can_dodge)
        escalate_now = escalate and not self._sig_escalated

        if in_path:
            vfh_text = vfh.text if (escalate and can_dodge) else None
            message = self._render_message(top_alert, zone, band, scene, vfh_text, escalate)
            if closing and message and not (escalate and vfh_text):
                message = "Dikkat, çok yakın, durun"
        else:
            # Imminent obstacle intruding the corridor → tell the user WHERE to
            # go, not just that something is close (the field test wanted
            # guidance, not a bare alarm). Steer toward the open side.
            noun = self._HAZARD_NOUN.get(cid, "engel")
            steer = self._steer_away(zone, getattr(result, "corridor", None))
            if steer:
                message = "Dikkat, %s, %s yönelin" % (noun, steer)
            else:
                message = "Dikkat, %s, durun" % noun

        urgent = closing or imminent_side
        # A genuine collision course preempts whatever audio is playing (and
        # bypasses the post-nav silence window). This is the ONLY thing allowed
        # to speak while muted.
        preempt = imminent_side or (in_path and closing)

        # ── Cadence: urgent overrides the repeat; else change/escalation/repeat ──
        sig_changed = sig != self._last_spoken_sig
        safety_increased = safety_level > self._last_safety
        if urgent:
            min_interval = ai.urgent_interval_sec          # override the 20 s repeat
        elif sig_changed or escalate_now or safety_increased:
            min_interval = ai.min_obstacle_interval_sec
        else:
            min_interval = ai.min_obstacle_repeat_sec
        interval_ok = (now - self._last_obstacle_speak_at) >= min_interval

        spoke = False
        # While muted, suppress everything EXCEPT a preempting collision.
        if interval_ok and message and (preempt or not muted):
            # Closing / imminent threats are spoken faster + higher-pitched, and
            # a collision course cuts through the current utterance.
            self._voice.say_obstacle(message, urgent=urgent, preempt=preempt)
            self._last_obstacle_speak_at = now
            self._last_spoken_sig = sig
            self._last_safety = safety_level
            self._sig_escalated = escalate
            spoke = True
            if self._pipeline is not None:
                self._pipeline.mark_alert_spoken(cid)
        elif interval_ok and message and muted:
            # Calibration signal (B6): a due warning was eaten by the mute
            # window. The field-test summary counts these per reason — a high
            # "muted_unsafe" count means post_nav_silence_sec is too long.
            self._rec.log_speak(
                "obstacle", message, False,
                reason=("muted_unsafe" if safety_level == SAFETY_UNSAFE else "muted"),
            )

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

    def _emit_path_keeping(self, result, frame, now: float, nav_active: bool,
                           top_alert, zone) -> None:
        """Keep the user on the walkable corridor + obstacle awareness.

        Runs DURING navigation too (Faz 6 fix). The route's turn-by-turn gives
        the strategic direction but NOT the user's head orientation, so they
        still need the camera-relative "hafif sola/sağa, yolun ortasına"
        centering, squeeze warnings, and side-hazard awareness. Conflicts with a
        turn instruction are avoided by the post-nav silence window (obstacle
        speech is briefly muted right after each turn).
        """
        ai = self._config.ai

        # ── 1) Event-driven obstacle awareness ─────────────────────────────
        # A new side/front hazard is announced once. Re-arm memory: the SAME
        # (class, zone) is held silent for ambient_rearm_sec even if it briefly
        # drops out of detection and reappears (a car flickering behind a pole
        # must not read as "new"). We intentionally do NOT clear state when the
        # hazard is momentarily absent — that was the flicker bug.
        if top_alert is not None and top_alert.class_id in self._HAZARD_NOUN:
            amb_sig = (top_alert.class_id, zone)
            last = self._ambient_last_at.get(amb_sig, -1e9)
            if ((now - self._last_ambient_at) >= ai.ambient_min_gap_sec
                    and (now - last) >= ai.ambient_rearm_sec):
                self._ambient_last_at[amb_sig] = now
                self._last_ambient_at = now
                noun = self._HAZARD_NOUN.get(top_alert.class_id, "engel")
                text = "%s %s var" % (self._zone_word(zone), noun)
                self._voice.say_obstacle(text)
                self._rec.log_perception(result, chosen=text)
                return

        # ── 2) Narrow-passage awareness (hysteresis + persistence) ──────────
        # Squeezing between two obstacles: the corridor does not fully close but
        # the walkable share drops. The segmentation mask is noisy, so on top of
        # the enter/exit hysteresis we require the condition to PERSIST a few
        # frames before flipping — otherwise it chattered "daralıyor"/"açıldı"
        # ~60 times in one walk (keci_testv4).
        corridor = getattr(result, "corridor", None)
        free = corridor.free_ratio if (corridor is not None and corridor.valid) else 0.0
        if not self._in_narrow:
            self._narrow_count = (self._narrow_count + 1
                                  if free < ai.narrow_enter_ratio else 0)
            if self._narrow_count >= ai.narrow_persist_frames:
                self._in_narrow = True
                self._narrow_count = 0
                self._voice.say_obstacle("Yürünebilir alan daralıyor, dikkatli ilerleyin")
                self._rec.log_perception(result, chosen="narrowing")
                return
            # not yet confirmed narrow → fall through to drift guidance
        else:
            self._narrow_count = (self._narrow_count + 1
                                  if free > ai.narrow_exit_ratio else 0)
            if self._narrow_count >= ai.narrow_persist_frames:
                self._in_narrow = False
                self._narrow_count = 0
                self._voice.say_obstacle("Alan açıldı, yol temiz")
                self._rec.log_perception(result, chosen="cleared")
            else:
                self._rec.log_perception(result, chosen=None)  # still squeezing
            return

        # ── 3) Path-keeping guidance ────────────────────────────────────────

        # EMA-smooth the offset (anti-flicker), then hysteresis on direction.
        a = ai.offset_ema_alpha
        self._offset_ema = (corridor.offset if self._offset_ema is None
                            else a * corridor.offset + (1 - a) * self._offset_ema)
        off = self._offset_ema
        if off <= -ai.centerline_drift_warn:
            want = "left"
        elif off >= ai.centerline_drift_warn:
            want = "right"
        elif abs(off) <= ai.drift_clear_band:
            want = "straight"
        else:
            want = self._drift_dir   # inside the hysteresis band → hold
        if want != self._drift_dir:
            self._drift_count += 1
            if self._drift_count >= ai.drift_persist_frames:
                self._drift_dir, self._drift_count = want, 0
        else:
            self._drift_count = 0

        if self._drift_dir == "left":
            text, on_path = "Hafif sola, yolun ortasına", False
        elif self._drift_dir == "right":
            text, on_path = "Hafif sağa, yolun ortasına", False
        else:
            text, on_path = "Düz devam edin", True

        interval = (ai.path_confirm_interval_sec if on_path
                    else ai.path_correct_interval_sec)
        # "Düz devam" only on its (long) interval; a drift correction may also
        # speak when the direction actually changes.
        changed = text != self._last_path_text
        if not ((now - self._last_path_speak_at) >= interval or (changed and not on_path)):
            self._rec.log_perception(result, chosen=None)
            return
        self._last_path_speak_at = now
        self._last_path_text = text
        # Drift corrections go out as panned earcons (speech fallback inside);
        # the rare "Düz devam edin" confirmation stays spoken.
        if on_path:
            self._voice.say_obstacle(text)
        else:
            self._voice.say_drift(self._drift_dir, text)
        self._rec.log_perception(result, chosen=text)

    def _emit_road_ahead(self, result, frame, now: float, nav_active: bool) -> None:
        """Road straight ahead → a cautionary, event-driven notice (Faz 4).

        Upgrades to a crossing caution when a walkable sidewalk is CONFIRMED
        beyond the road over ``crossing_persist_frames``. By design this NEVER
        tells the user it is safe to cross: at this range the far-sidewalk
        evidence is exactly where the segmentation model is least reliable, so a
        false positive must degrade to "be careful", not "you may go". Spoken on
        a single event cadence (``ambient_min_gap_sec``) and — unlike side
        ambient notices — voiced even during navigation, because a road in the
        path is a safety hazard the route's turn-by-turn does not cover.
        """
        ai = self._config.ai
        corridor = getattr(result, "corridor", None)
        candidate = bool(corridor is not None and getattr(corridor, "crossing", False))
        if candidate and ai.crossing_detection_enabled:
            self._crossing_frames += 1
        else:
            self._crossing_frames = 0
        confirmed = self._crossing_frames >= ai.crossing_persist_frames

        if (now - self._last_road_at) < ai.ambient_min_gap_sec:
            self._rec.log_perception(result, chosen=None)
            return

        # B2 fusion: the route itself says we are about to cross here, so a
        # flat "girmeyin" would contradict the turn-by-turn guidance. Still
        # NEVER an assurance — only context plus "be careful".
        expected = bool(getattr(self._nav, "crossing_expected", False))
        if confirmed:
            # Crossing confirmed → caution only, never an assurance to cross.
            text = ("Önünüzde araç yolu, karşı tarafta kaldırım var, "
                    "geçişte dikkatli olun")
        elif expected:
            text = ("Rotanız yoldan karşıya geçiyor, geçidi göremiyorum, "
                    "çok dikkatli geçin")
        else:
            # No crossing confirmed → protective "do not enter". This is also the
            # safe fallback when the strict crossing shields miss a real crossing:
            # over-cautioning "girmeyin" beats waving the user into traffic.
            text = "Dikkat, önünüzde araç yolu, girmeyin"
        self._last_road_at = now
        self._voice.say_obstacle(text)
        self._rec.log_perception(result, chosen=text)
        if frame is not None:
            self._rec.maybe_save_overlay(frame, result.mask, "road", info={
                "walkable": result.scene.walkable_ratio,
            })

    # Classes that count toward the fast-path tripwire (anything solid enough
    # to walk into; road/crosswalk are surfaces, not collisions).
    _FAST_HAZARD_IDS = (
        int(ClassID.VEHICLE), int(ClassID.COLLISION_OBSTACLE),
        int(ClassID.DYNAMIC_HAZARD), int(ClassID.FALL_HAZARD),
    )

    def _fast_collision_check(self, result, now: float) -> bool:
        """Cheap, gating-independent stop tripwire. True when it spoke."""
        ai = self._config.ai
        mask = getattr(result, "mask", None)
        if not ai.fast_collision_enabled or mask is None:
            return False
        import numpy as np
        h, w = mask.shape
        region = mask[int(h * 0.6):, int(w * 0.3):int(w * 0.7)]
        if region.size == 0:
            return False
        ratio = float(np.isin(region, self._FAST_HAZARD_IDS).mean())
        if ratio < ai.fast_collision_ratio:
            self._fast_collision_frames = 0
            return False
        self._fast_collision_frames += 1
        if self._fast_collision_frames < ai.fast_collision_persist_frames:
            return False
        if (now - self._last_fast_collision_at) < ai.urgent_interval_sec:
            return False  # already screamed — let the normal flow continue
        self._last_fast_collision_at = now
        self._last_obstacle_speak_at = now
        self._voice.say_obstacle("Durun, önünüz kapalı", urgent=True, preempt=True)
        return True

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
        self._prev_walkable = None
        self._prev_top_distance = None
        self._closing_frames = 0
        self._last_path_text = None
        self._offset_ema = None
        self._drift_dir = "straight"
        self._drift_count = 0
        self._ambient_last_at.clear()
        self._last_ambient_at = -999.0
        self._crossing_frames = 0
        self._last_road_at = -999.0
        self._in_narrow = False
        self._narrow_count = 0

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
    # Sentence-leading form for ambient obstacle awareness (Faz 3).
    _SIDE_WORD = {"left": "Sol tarafınızda", "right": "Sağ tarafınızda", "center": "Önünüzde"}
    # Mid-sentence form for the directional imminent warning (Faz 3).
    _SIDE_WORD_MID = {"left": "solunuzda", "right": "sağınızda", "center": "önünüzde"}
    # Clock bearings for the 3 detection zones (blind-navigation convention).
    _CLOCK_WORD = {"left": "Saat on yönünde", "right": "Saat iki yönünde",
                   "center": "Tam önünüzde"}

    def _zone_word(self, zone: str) -> str:
        """Sentence-leading position word — clock bearing or side word."""
        if self._config.ai.clock_direction_enabled:
            return self._CLOCK_WORD.get(zone, "Tam önünüzde")
        return self._SIDE_WORD.get(zone, "Önünüzde")
    _HAZARD_NOUN = {
        int(ClassID.VEHICLE): "araç",
        int(ClassID.COLLISION_OBSTACLE): "engel",
        int(ClassID.FALL_HAZARD): "engel",
        int(ClassID.DYNAMIC_HAZARD): "hareketli nesne",
    }

    @staticmethod
    def _steer_away(zone, corridor) -> Optional[str]:
        """Pick the open side to steer toward, away from a corridor obstacle.

        Prefer the opposite of the obstacle's side ("on my left → go right").
        When the obstacle reads as centre, fall back to the corridor centroid
        (where the open walkable area actually is). Returns "sola"/"sağa"/None.
        """
        if zone == "left":
            return "sağa"
        if zone == "right":
            return "sola"
        if corridor is not None and getattr(corridor, "valid", False):
            if corridor.offset <= -0.2:
                return "sola"
            if corridor.offset >= 0.2:
                return "sağa"
        return None

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
            return "Yaya geçidi, dikkatli geçin"
        if cid == int(ClassID.FALL_HAZARD):
            return "Zemin tehlikesi, dikkatli ilerleyin"

        noun = self._HAZARD_NOUN.get(cid, "engel")
        loc = self._zone_word(zone)

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

