"""
ALAS System Configuration
=========================
All tuneable parameters for the main loop in one place.
Adjust these values based on Jetson Nano performance and user comfort.
"""

from dataclasses import dataclass


@dataclass
class ALASConfig:
    # ── AI / Perception ──────────────────────────────────────────
    model_path: str = "models/segmentation/alas_engine.trt"
    model_input_h: int = 384       # must match the TRT engine input
    model_input_w: int = 512       # must match the TRT engine input
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    perception_fps: float = 2.0    # ~1 frame every 0.5s — saves Jetson resources
                                   # yaya yürüyor, ard arda frame'ler çok farklı olmaz

    # ── Navigation ───────────────────────────────────────────────
    osm_map_path: str = "src/navigation/router/map.osm"
    gps_port: str = "/dev/ttyTHS1"
    gps_baudrate: int = 9600
    gps_warmup_sec: float = 60.0
    gps_update_interval: float = 4.0           # GPS her 4 saniyede bir kontrol
    progress_announce_interval: float = 30.0   # "Hedefe X metre" her 30 sn'de bir

    # ── Voice (STT / TTS) ────────────────────────────────────────
    stt_listen_timeout: float = 5.0   # max dinleme süresi (saniye)
    stt_silence_sec: float = 1.5      # sessizlik sonrası tanıma bitiş

    # ── Button (GPIO) ────────────────────────────────────────────
    button_pin: int = 18              # BCM pin number on Jetson Nano
    button_debounce_ms: int = 300

    # ── General ──────────────────────────────────────────────────
    log_dir: str = "src/logs"
