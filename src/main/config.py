"""ALAS system configuration.

All tuneable parameters for the main loop live here. ``ALASConfig.from_cli``
parses command-line overrides so the main entry point stays a thin orchestrator.

The dataclass deliberately mixes tunables (frame rate, distances, ...) with
runtime modes (``mock``, ``no_camera``). It is mildly impure, but it keeps
``main()`` reading from a single source of truth.
"""

import argparse
import os
from dataclasses import dataclass

# Resolve repo-relative paths against the src/ directory so the system can be
# launched from any working directory, not just from inside src/.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    """Make a relative path absolute against the src/ root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_SRC_ROOT, path))


@dataclass
class ALASConfig:
    # AI / Perception
    model_path: str = "models/segmentation/alas_engine.trt"
    model_input_h: int = 384       # Must match the TRT engine input.
    model_input_w: int = 512       # Must match the TRT engine input.
    camera_index: int = 0
    # Capture at the model's input resolution to skip a CPU resize each frame.
    camera_width: int = 512
    camera_height: int = 384
    # Pedestrian motion is slow, so consecutive frames carry little new info.
    # Capping inference at ~2 FPS keeps the Jetson Nano cool and avoids
    # bombarding the user with TTS alerts.
    perception_fps: float = 2.0

    # Camera geometry (used for distance estimation in ai/geometry.py).
    camera_height_m: float = 1.65   # Glasses-mounted camera height above ground.
    camera_tilt_deg: float = 5.0    # Positive = camera tilted downwards.
    camera_vfov_deg: float = 60.0   # Vertical field of view.

    # Navigation
    osm_map_path: str = "navigation/router/map.osm"
    gps_port: str = "/dev/ttyTHS1"
    gps_baudrate: int = 9600
    gps_warmup_sec: float = 60.0
    gps_update_interval: float = 4.0           # Poll GPS every N seconds.
    progress_announce_interval: float = 30.0   # "X metres to go" reminder period.
    approach_threshold_m: float = 30.0         # Pre-warn when distance to next < N.
    long_stretch_threshold_m: float = 100.0    # > N → fall back to 30s reminder.

    # Voice (STT / TTS)
    stt_listen_timeout: float = 5.0   # Maximum listening window in seconds.
    stt_silence_sec: float = 1.5      # End recognition after this much silence.
    post_nav_silence_sec: float = 3.0 # Mute obstacle alerts after a nav utterance.

    # Perception dispatcher cadence
    obstacle_dedupe_ttl_sec: float = 12.0   # Re-allow an identical alert after N s.
    path_guidance_cooldown_sec: float = 8.0 # Min gap between identical path guidance lines.
    gps_stale_threshold_sec: float = 10.0   # Treat a fix older than N s as stale.

    # Button (GPIO)
    button_pin: int = 18              # BCM pin number on Jetson Nano.
    button_debounce_ms: int = 300

    # Boot / warmup / sleep
    warmup_timeout_sec: float = 90.0      # await_ready max wait before forcing ACTIVE.
    sleep_idle_timeout_sec: float = 0.0   # 0 = never auto-sleep; >0 enables idle-sleep.

    # Runtime flags (set by --mock / --no-camera / --bypass-*)
    mock: bool = False
    no_camera: bool = False
    bypass_stt: bool = False           # Skip STT/microphone — typed input via stdin.
    bypass_warmup: bool = False        # Skip the entire warmup phase.
    bypass_gps_warmup: bool = False    # Skip only the GPS readiness wait (e.g. indoors).

    # General
    log_dir: str = "src/logs"

    @classmethod
    def from_cli(cls, argv=None):
        """Build a config from command-line arguments; defaults fill the rest."""
        parser = argparse.ArgumentParser(description="ALAS — AI Smart Glasses System")
        parser.add_argument("--model",     default=None, help="Path to .trt/.engine or .onnx model")
        parser.add_argument("--camera",    type=int, default=None, help="Camera device index")
        parser.add_argument("--fps",       type=float, default=None, help="Perception target FPS")
        parser.add_argument("--map",       default=None, help="Path to .osm map file")
        parser.add_argument("--gps-port",  default=None, help="GPS serial port")
        parser.add_argument("--mock",      action="store_true", help="Desktop test mode (no GPIO/GPS)")
        parser.add_argument("--no-camera", action="store_true", help="Disable perception thread")
        parser.add_argument("--bypass-stt", action="store_true",
                            help="Skip microphone — voice commands typed via keyboard")
        parser.add_argument("--bypass-warmup", action="store_true",
                            help="Skip GPS/model warmup — jump straight to ACTIVE mode")
        parser.add_argument("--bypass-gps-warmup", action="store_true",
                            help="Skip only the GPS readiness wait (e.g. indoor dev)")
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
        config.bypass_stt = args.bypass_stt
        config.bypass_warmup = args.bypass_warmup
        config.bypass_gps_warmup = args.bypass_gps_warmup

        # Anchor relative paths to src/ so the entry point can be invoked
        # from any working directory.
        config.osm_map_path = _resolve(config.osm_map_path)
        config.model_path = _resolve(config.model_path)
        config.log_dir = _resolve(config.log_dir)
        return config
