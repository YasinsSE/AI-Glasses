"""AI / perception module configuration.

Holds every tunable parameter owned by the perception subsystem: model I/O,
camera capture, camera mounting geometry (used for ground-plane distance
estimation in :mod:`ai.geometry`), and the TTS dispatch cadences enforced by
``PerceptionService``.

This dataclass is the per-module config for the ``ai`` package. It is composed
by :class:`main.config.ALASConfig`, which remains the single launch authority
and applies command-line overrides.
"""

from dataclasses import dataclass


@dataclass
class AIConfig:
    # Model I/O — input dimensions must match the exported TRT/ONNX engine.
    model_path: str = "models/segmentation/alas_engine.trt"
    model_input_h: int = 384
    model_input_w: int = 512

    # Camera capture. Capturing at the model's input resolution skips a CPU
    # resize on every frame.
    camera_index: int = 0
    camera_width: int = 512
    camera_height: int = 384

    # CSI cameras (IMX219 / Raspberry Pi Cam) on Jetson are NOT reachable through
    # the v4l2 path that cv2.VideoCapture(index) builds — they require the Argus
    # ISP via an nvarguscamerasrc GStreamer pipeline. USB UVC cams use the index.
    use_csi_camera: bool = True
    # Native sensor capture resolution before hardware downscale to the model
    # input. 1280x720 is a valid IMX219 sensor mode; nvvidconv scales it to
    # camera_width/height on the ISP, so the CPU never touches a full-res frame.
    csi_capture_width: int = 1280
    csi_capture_height: int = 720
    csi_framerate: int = 30
    # 0=none, 2=180°. Flip if the glasses-mounted module is physically inverted.
    csi_flip_method: int = 0
    csi_sensor_mode: int = 4

    # Pedestrian motion is slow, so consecutive frames carry little new
    # information. Capping inference at ~2 FPS keeps the Jetson Nano cool and
    # avoids overwhelming the user with TTS alerts.
    perception_fps: float = 2.0

    # Camera mounting geometry (consumed by ai/geometry.py).
    camera_height_m: float = 1.65   # Glasses-mounted camera height above ground.
    camera_tilt_deg: float = 5.0    # Positive = camera tilted downwards.
    camera_vfov_deg: float = 60.0   # Vertical field of view.

    # Perception dispatcher cadences.
    obstacle_dedupe_ttl_sec: float = 12.0    # Re-allow the same situation after N s.
    path_guidance_cooldown_sec: float = 8.0  # Min gap between identical path-guidance lines.
    # Minimum gap between obstacle speaks when the hazard SITUATION changed
    # (different hazard class, the dominant zone shifted, or safety escalated).
    min_obstacle_interval_sec: float = 4.0
    # Minimum gap between obstacle speaks when the situation is UNCHANGED
    # (same hazard class, same zone, same safety level). Prevents repeating
    # "Önünüzde araç var" every few seconds for a static parked car.
    min_obstacle_repeat_sec: float = 20.0
    # "Yol açık" repeat interval while the scene stays safe.
    safe_announce_interval_sec: float = 30.0

    # ── Closing-threat escalation (Faz 1) ────────────────────────────────────
    # A centre hazard that keeps the SAME signature (same class/zone, already
    # UNSAFE) would otherwise stay silent for ``min_obstacle_repeat_sec`` (20 s)
    # even as the user walks into it. When the gap is CLOSING fast we override
    # that and re-warn within ``urgent_interval_sec``.
    urgent_interval_sec: float = 2.0        # min gap between re-warns while closing
    imminent_distance_m: float = 2.5        # hazard nearer than this = imminent
    walkable_drop_urgent: float = 0.12      # walkable_ratio dropping ≥ this between
                                            # frames = rapidly closing
    closing_distance_urgent_m: float = 0.8  # distance shrinking ≥ this between frames

    # ── Situation tracking / escalation ──────────────────────────────────
    # A hazard must persist in the forward path for this many consecutive
    # frames before the calm "var" notice escalates to an actionable VFH
    # dodge ("...hafif sağa yönelin"). Stops a single noisy frame from
    # commanding the user to swerve.
    escalation_frames: int = 2
    # Hysteresis margin (metres) before the proximity band (uzak/yakın/çok
    # yakın) is allowed to flip. Without it the distance estimate dithers
    # across the threshold and re-renders the same warning every frame.
    proximity_hysteresis_m: float = 0.6
    # A hazard counts as "blocking the forward path" only when it sits in the
    # centre zone AND the walkable ratio is below this — otherwise the user
    # can simply walk past it and no dodge instruction is needed.
    block_walkable_ratio: float = 0.12

    # ── Perception-loop stall watchdog (Jetson under memory pressure) ─────
    # If a single frame takes longer than this, treat the loop as having
    # stalled: on recovery the dispatcher's cooldowns and situation state are
    # reset so we speak about the CURRENT scene, not the one from before the
    # freeze.
    stall_warn_sec: float = 8.0
    # Only voice "Görüş gecikiyor" when the stall exceeded this longer gap,
    # so brief hiccups stay silent.
    stall_announce_sec: float = 15.0
