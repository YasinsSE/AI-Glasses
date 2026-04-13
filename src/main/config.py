"""
ALAS System Configuration
=========================
All tuneable parameters for the main loop in one place. ``ALASConfig.from_cli``
parses command-line overrides so the main entry point stays a thin orchestrator.

The dataclass deliberately mixes "tunables" (frame rate, distances, …) with
"runtime modes" (``mock``, ``no_camera``). It is mildly impure, but it keeps
``main()`` reading from a single source of truth — every other approach
requires passing two objects around.
"""

#from __future__ import annotations

import argparse
from typing import Dict
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

    # ── Camera geometry (for distance estimation, see ai/geometry.py) ──
    camera_height_m: float = 1.65   # height of glasses-mounted camera above ground
    camera_tilt_deg: float = 5.0    # downward tilt — positive = looking down
    camera_vfov_deg: float = 60.0   # vertical field of view in degrees

    # ── Navigation ───────────────────────────────────────────────
    osm_map_path: str = "navigation/router/map.osm"
    gps_port: str = "/dev/ttyTHS1"
    gps_baudrate: int = 9600
    gps_warmup_sec: float = 60.0
    gps_update_interval: float = 4.0           # GPS her 4 saniyede bir kontrol
    progress_announce_interval: float = 30.0   # "Hedefe X metre" her 30 sn'de bir
    approach_threshold_m: float = 30.0         # pre-warn when distance to next < N
    long_stretch_threshold_m: float = 100.0    # > N → fall back to 30s reminder

    # ── Voice (STT / TTS) ────────────────────────────────────────
    stt_listen_timeout: float = 5.0   # max dinleme süresi (saniye)
    stt_silence_sec: float = 1.5      # sessizlik sonrası tanıma bitiş
    post_nav_silence_sec: float = 3.0 # obstacle alerts muted N s after a nav utterance

    # ── Button (GPIO) ────────────────────────────────────────────
    button_pin: int = 18              # BCM pin number on Jetson Nano
    button_debounce_ms: int = 300

    # ── Boot / warmup / sleep ────────────────────────────────────
    warmup_timeout_sec: float = 90.0      # await_ready max wait before forcing ACTIVE
    sleep_idle_timeout_sec: float = 0.0   # 0 = never auto-sleep; >0 enables idle-sleep

    # ── Runtime flags (set by --mock / --no-camera) ──────────────
    mock: bool = False
    no_camera: bool = False

    # ── General ──────────────────────────────────────────────────
    log_dir: str = "src/logs"

    # ------------------------------------------------------------------
    # CLI parsing
    # ------------------------------------------------------------------
    @classmethod
    def from_cli(cls, argv=None):
        """
        Build a config from command-line arguments. Any flag left unset
        falls back to the dataclass default.
        """
        parser = argparse.ArgumentParser(description="ALAS — AI Smart Glasses System")
        parser.add_argument("--model",     default=None, help="Path to .trt/.engine or .onnx model")
        parser.add_argument("--camera",    type=int, default=None, help="Camera device index")
        parser.add_argument("--fps",       type=float, default=None, help="Perception target FPS")
        parser.add_argument("--map",       default=None, help="Path to .osm map file")
        parser.add_argument("--gps-port",  default=None, help="GPS serial port")
        parser.add_argument("--mock",      action="store_true", help="Desktop test mode (no GPIO/GPS)")
        parser.add_argument("--no-camera", action="store_true", help="Disable perception thread")
        args = parser.parse_args(argv)

        config = cls()
        if args.model:
            config.model_path = args.model
        if args.camera is not None:
            config.camera_index = args.camera
        if args.fps:
            config.perception_fps = args.fps
        if args.map:
            config.osm_map_path = args.map
        if args.gps_port:
            config.gps_port = args.gps_port
        config.mock = args.mock
        config.no_camera = args.no_camera
        return config
