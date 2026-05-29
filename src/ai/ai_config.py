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
