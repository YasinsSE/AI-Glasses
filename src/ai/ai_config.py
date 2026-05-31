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
    obstacle_dedupe_ttl_sec: float = 12.0    # Re-allow an identical alert after N s.
    path_guidance_cooldown_sec: float = 8.0  # Min gap between identical path-guidance lines.
    # Global minimum gap between any two spoken obstacle alerts, regardless of
    # whether the combined guidance+alert text changed. Prevents the dispatcher
    # from speaking every frame when guidance oscillates left/right.
    min_obstacle_interval_sec: float = 4.0
    # "Yol açık, devam edebilirsiniz." is emitted this often while the scene is
    # safe (no hazard with priority ≥ 3). First emission fires at the transition
    # from unsafe → safe; subsequent emissions repeat after this interval.
    safe_announce_interval_sec: float = 15.0
