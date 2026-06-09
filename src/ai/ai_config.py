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
    # Frames per second actually DELIVERED to the CPU. The sensor still runs at
    # csi_framerate (Argus needs a native mode), but videorate drops to this
    # before the CPU-side videoconvert — at 30 fps that conversion burned a
    # large share of a Nano core for frames the 2 FPS model never saw. Keep a
    # margin above perception_fps so freshest-frame drop still works.
    csi_delivery_fps: int = 5
    # 0=none, 2=180°. Flip if the glasses-mounted module is physically inverted.
    csi_flip_method: int = 0
    csi_sensor_mode: int = 4

    # Pedestrian motion is slow, so consecutive frames carry little new
    # information. Capping inference at ~2 FPS keeps the Jetson Nano cool and
    # avoids overwhelming the user with TTS alerts.
    perception_fps: float = 2.0

    # Camera mounting geometry (consumed by ai/geometry.py for distance estimation).
    # These are config variables — adjust once the final rig is fixed.
    # ⚠ Hardware is an IMX219-120 (120° lens) in csi_sensor_mode 4 (1280×720, a
    #   cropped readout), so the effective VERTICAL FOV is neither 120 nor a clean
    #   60. The value below is an estimate; for accurate distances CALIBRATE
    #   empirically (place an object at a known distance, tune vfov/tilt to match).
    camera_height_m: float = 1.65   # Glasses-mounted camera height above ground.
    camera_tilt_deg: float = 5.0    # Positive = camera tilted downwards.
    camera_vfov_deg: float = 60.0   # Vertical field of view (CALIBRATE — see note above).

    # Clock-direction wording ("saat iki yönünde araç") instead of side words
    # ("sağınızda araç"). This is the established orientation convention among
    # blind users — a clock bearing is actionable without knowing how far the
    # speaker's "right" extends. Maps the 3 detection zones to 10/12/2 o'clock.
    clock_direction_enabled: bool = True

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
    # "Yol açık" repeat interval while the scene stays safe (raised 30→45 to keep
    # the reassurance from being too frequent now that more frames read SAFE).
    safe_announce_interval_sec: float = 45.0

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
    # A side obstacle is a real COLLISION COURSE (imminent) only when it both
    # (a) intrudes the walking corridor by at least this share of its pixels —
    # a parked car beside the path has near-zero overlap — and (b) keeps closing
    # for ``closing_persist_frames`` consecutive frames. Together these stopped
    # the "araç çok yakın" spam on every parked car (Faz 5, keci field test).
    imminent_corridor_min: float = 0.20
    closing_persist_frames: int = 2

    # ── Path-keeping (Faz 2) ─────────────────────────────────────────────────
    # When nothing blocks the centerline ahead, keep the user on the walkable
    # corridor instead of announcing every side car.
    centerline_drift_warn: float = 0.40       # smoothed |offset| above this → "hafif sola/sağa"
    path_confirm_interval_sec: float = 25.0   # periodic "Düz devam edin" when centred (rare)
    path_correct_interval_sec: float = 8.0    # min gap between drift corrections
    # Faz 3 — quieter path-keeping + event-driven obstacle awareness.
    offset_ema_alpha: float = 0.4             # EMA smoothing of the corridor offset (anti-flicker)
    drift_clear_band: float = 0.25            # |offset| below this → back to "straight" (hysteresis)
    drift_persist_frames: int = 2             # a new drift direction must persist this many frames
    # Narrow-passage awareness (Faz 6) — state machine with hysteresis. When the
    # near-corridor walkable share drops below ``narrow_enter_ratio`` we warn once
    # ("daralıyor"); when it climbs back above ``narrow_exit_ratio`` we reassure
    # once ("Alan açıldı, yol temiz"). The gap between the two prevents chatter.
    narrow_enter_ratio: float = 0.22
    narrow_exit_ratio: float = 0.40
    narrow_persist_frames: int = 3            # squeeze must hold this many frames (anti-noise)
    ambient_min_gap_sec: float = 15.0         # min gap before a NEW hazard awareness notice
    # Per-(class,zone) re-arm: once a side hazard is announced, the SAME hazard is
    # not re-announced for this long even if it briefly drops out of detection and
    # reappears (a car flickering behind a pole must not read as "new"). Longer
    # than ambient_min_gap_sec on purpose.
    ambient_rearm_sec: float = 40.0

    # ── Crossing detection (Faz 4) ───────────────────────────────────────────
    # Road straight ahead with a walkable sidewalk beyond → an informational,
    # CAUTIONARY notice ("…geçişte dikkatli olun"); never an assurance to cross.
    # Geometric shields live in perception.py (CROSSING_*); these are the
    # service-side temporal guard + master switch (turn OFF in the field if it
    # ever false-fires toward traffic).
    crossing_detection_enabled: bool = True
    crossing_persist_frames: int = 3          # frames the candidate must persist before speaking

    # ── Fast-path collision tripwire (B5) ─────────────────────────────────
    # Independent of the situation-tracking/gating chain: when hazardous
    # classes fill this share of the near-centre region (bottom 40%, centre
    # 40% of width) for a couple of frames, speak a preempting "Durun" no
    # matter what the cadence logic decided. A safety net that guarantees the
    # gating layers can never swallow a wall-in-the-face case.
    fast_collision_enabled: bool = True
    fast_collision_ratio: float = 0.45
    fast_collision_persist_frames: int = 2

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

    # ── Adaptive perception FPS (context-aware) ───────────────────────────
    # A fixed 2 FPS is wrong in both directions: wasted heat while standing
    # still on a clear sidewalk, and slow reaction while a hazard is closing.
    # The loop picks its rate from context — calm+still → fps_idle, hazard
    # UNSAFE or closing → fps_alert, otherwise perception_fps. The thermal
    # guard below always wins (no boost while hot).
    adaptive_fps_enabled: bool = True
    fps_idle: float = 1.0
    fps_alert: float = 3.5
    # Visual motion metric (mean frame-diff, 0..255) below this == standing
    # still; the SAFE+still state must persist this many frames before the
    # rate drops, so a pause at a kerb does not flap the FPS.
    idle_motion_threshold: float = 1.5
    idle_safe_persist_frames: int = 10

    # ── Low-light reliability warning ─────────────────────────────────────
    # The segmentation model was trained on daylight scenes; in the dark it
    # fails SILENTLY (plausible-looking but wrong masks). Below
    # ``low_light_enter`` mean brightness (0-255) we tell the user once that
    # vision is degraded, so they fall back to the cane instead of trusting
    # the glasses. Hysteresis + persistence keep shadows/underpasses from
    # flapping the message.
    low_light_enter: float = 35.0
    low_light_exit: float = 55.0
    low_light_persist_frames: int = 4
    low_light_rearm_sec: float = 180.0

    # ── Thermal guard (Waveshare Nano cooling is marginal) ───────────────
    # The SoC throttles its clocks near ~97 C, but inference already slows
    # well before that. Instead of letting DVFS silently halve the FPS, we
    # degrade DELIBERATELY: above ``thermal_throttle_c`` the loop drops to
    # ``thermal_min_fps`` (announced once), and recovers below
    # ``thermal_recover_c``. The gap between the two is hysteresis.
    thermal_guard_enabled: bool = True
    thermal_check_interval_sec: float = 10.0
    thermal_throttle_c: float = 70.0
    thermal_recover_c: float = 62.0
    thermal_min_fps: float = 1.0

    # ── Perception-loop stall watchdog (Jetson under memory pressure) ─────
    # If a single frame takes longer than this, treat the loop as having
    # stalled: on recovery the dispatcher's cooldowns and situation state are
    # reset so we speak about the CURRENT scene, not the one from before the
    # freeze.
    stall_warn_sec: float = 8.0
    # Only voice "Görüş gecikiyor" when the stall exceeded this longer gap,
    # so brief hiccups stay silent.
    stall_announce_sec: float = 15.0
