"""ALAS system configuration — composition root.

``ALASConfig`` is the single launch authority for the whole system. It does not
own module tunables itself; instead it composes one config dataclass per module
(``ai``, ``vfh``, ``gps``, ``nav``, ``voice``) and holds only cross-cutting
fields and runtime modes (``mock``, ``no_camera``, ``bypass_*``).

``ALASConfig.from_cli`` parses command-line overrides and writes them into the
relevant nested config, so the main entry point stays a thin orchestrator and
every module reads its parameters from its own typed config object
(e.g. ``config.ai.model_path``, ``config.gps.port``).
"""

import argparse
import os
from typing import Optional, Tuple
from dataclasses import dataclass, field

from ai.ai_config import AIConfig
from navigation.local_planner.planner_config import VFHConfig
from navigation.router.nav_config import NavConfig
from navigation.sensors.sensor_config import GPSConfig
from tts_stt.voice_config import VoiceConfig

# Resolve repo-relative paths against the src/ directory so the system can be
# launched from any working directory, not just from inside src/.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_SRC_ROOT)


def _resolve(path: str) -> str:
    """Make a relative path absolute against the src/ root (for code-relative paths)."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_SRC_ROOT, path))


def _resolve_repo(path: str) -> str:
    """Make a relative path absolute against the repository root.

    Used for assets that live beside src/ rather than inside it: ``models/``,
    ``outputs/``, and ``src/logs``. This is why ``--model models/...`` works no
    matter what directory the system is launched from.
    """
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


@dataclass
class IdleConfig:
    """Automatic power-saving STANDBY detection (see main/activity_monitor.py).

    The monitor fuses motion evidence from pluggable sources (visual frame-diff,
    GPS speed/displacement, and — in the future — an MPU9250 IMU) and, after
    sustained inactivity, drops the system into STANDBY (``SystemMode.SLEEP``) so
    the camera and U-Net inference shut down to save LiPo battery. Wake-up is via
    the PTT button.
    """
    enabled: bool = False                 # --auto-standby. Off until field-tested.
    idle_enter_sec: float = 90.0          # Sustained no-motion before STANDBY.
    poll_interval_sec: float = 1.0        # Monitor evaluation cadence.
    # Visual stillness: mean absolute frame-diff (0..255) below this == "still".
    visual_motion_threshold: float = 2.5
    # GPS motion: speed at/above this (km/h) OR displacement past the radius == moving.
    gps_moving_kmh: float = 1.5
    gps_stationary_radius_m: float = 8.0
    # Ignore a source reading older than this many seconds (treat as "no data").
    source_stale_sec: float = 5.0
    # Optional short WAV played on PTT wake (empty == spoken cue only).
    wake_cue_wav: str = ""


@dataclass
class RecorderConfig:
    """Tunables for the field-test black-box recorder (see session_recorder.py)."""
    overlay_jpeg_quality: int = 65       # cv2 JPEG quality for saved overlays (lower = smaller/faster).
    frame_min_interval_s: float = 2.0    # Minimum gap between saved overlay frames.
    telemetry_interval_s: float = 5.0    # SoC temperature / load sampling period.
    checkpoint_interval_s: float = 30.0  # Rolling summary_partial.md rewrite period.
    queue_maxsize: int = 200             # Bounded queue; full -> drop (OOM protection).
    min_free_mb: int = 500               # Refuse to record below this free disk space.
    recent_events_max: int = 5000        # In-RAM window for the rolling checkpoint
                                         # preview; final summary reads events.jsonl.


@dataclass
class ALASConfig:
    # ── Composed module configs ──────────────────────────────────
    ai: AIConfig = field(default_factory=AIConfig)
    vfh: VFHConfig = field(default_factory=VFHConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)

    # ── Cross-cutting paths ──────────────────────────────────────
    osm_map_path: str = "navigation/router/map.osm"

    # ── Mic-less navigation ──────────────────────────────────────
    # --auto-nav <category>: with no microphone, automatically route to the
    # nearest POI of this category once the system is ACTIVE with a GPS fix,
    # and let a PTT press re-trigger the same route. "" = disabled.
    auto_nav_category: str = ""
    # --auto-nav-coord "LAT,LON": route to an EXACT map-picked coordinate instead
    # of the nearest category. Best for testing turn-by-turn (pick a spot that
    # needs turns). Overrides auto_nav_category when set. None = disabled.
    auto_nav_coord: Optional[Tuple[float, float]] = None
    # Settle delay AFTER "SYSTEM READY" before the auto-route is issued, so the
    # boot announcements finish and the camera/GPS pipeline is steady first.
    auto_nav_delay_sec: float = 10.0

    # ── Field-test recorder ──────────────────────────────────────
    rec: RecorderConfig = field(default_factory=RecorderConfig)
    record: bool = False               # --record: enable the black-box recorder.
    live: bool = False                 # --live: one-line stdout dashboard.
    record_dir: str = "outputs/field_tests"

    # ── Boot / warmup / sleep ────────────────────────────────────
    warmup_timeout_sec: float = 60.0      # await_ready max wait before forcing ACTIVE.
    sleep_idle_timeout_sec: float = 0.0   # 0 = never auto-sleep; >0 enables idle-sleep.

    # ── Runtime flags (set by --mock / --no-camera / --bypass-*) ──
    mock: bool = False
    mock_button: bool = False          # --mock-button: use Enter key instead of GPIO, GPS stays real.
    no_camera: bool = False
    bypass_stt: bool = False           # Skip STT/microphone — typed input via stdin.
    bypass_warmup: bool = False        # Skip the entire warmup phase.
    bypass_gps_warmup: bool = False    # Skip only the GPS readiness wait (e.g. indoors).

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
        parser.add_argument("--mock-button", action="store_true",
                            help="Use Enter key instead of GPIO button; GPS and camera stay real")
        parser.add_argument("--no-camera", action="store_true", help="Disable perception thread")
        parser.add_argument("--bypass-stt", action="store_true",
                            help="Skip microphone — voice commands typed via keyboard")
        parser.add_argument("--bypass-warmup", action="store_true",
                            help="Skip GPS/model warmup — jump straight to ACTIVE mode")
        parser.add_argument("--bypass-gps-warmup", action="store_true",
                            help="Skip only the GPS readiness wait (e.g. indoor dev)")
        parser.add_argument("--record", action="store_true",
                            help="Enable the field-test black-box recorder")
        parser.add_argument("--live", action="store_true",
                            help="Print a one-line live status dashboard to stdout")
        parser.add_argument("--auto-standby", action="store_true",
                            help="Enable automatic power-saving STANDBY on sustained inactivity")
        parser.add_argument("--auto-nav", metavar="CATEGORY", default=None,
                            help="No-mic mode: auto-route to nearest CATEGORY (e.g. eczane) "
                                 "on startup; a PTT press re-triggers it")
        parser.add_argument("--auto-nav-coord", metavar="LAT,LON", default=None,
                            help="No-mic mode: auto-route to this exact coordinate "
                                 "(e.g. 39.9245,32.8465); overrides --auto-nav. Best for "
                                 "testing turn-by-turn — pick a destination that needs turns")
        args = parser.parse_args(argv)

        config = cls()

        # Apply CLI overrides into the relevant nested config.
        if args.model:
            config.ai.model_path = args.model
        if args.camera is not None:
            config.ai.camera_index = args.camera
        if args.fps:
            config.ai.perception_fps = args.fps
        if args.map:
            config.osm_map_path = args.map
        if args.gps_port:
            config.gps.port = args.gps_port
        config.mock = args.mock
        config.mock_button = args.mock_button
        config.no_camera = args.no_camera
        config.bypass_stt = args.bypass_stt
        config.bypass_warmup = args.bypass_warmup
        config.bypass_gps_warmup = args.bypass_gps_warmup
        config.record = args.record
        config.live = args.live
        config.idle.enabled = args.auto_standby
        config.auto_nav_category = (args.auto_nav or "").strip().lower()
        if args.auto_nav_coord:
            try:
                _lat_s, _lon_s = args.auto_nav_coord.split(",")
                config.auto_nav_coord = (float(_lat_s), float(_lon_s))
            except (ValueError, AttributeError):
                parser.error("--auto-nav-coord must be 'LAT,LON', e.g. 39.9245,32.8465")

        # Anchor relative paths so the entry point can be invoked from any
        # working directory.
        config.osm_map_path = _resolve(config.osm_map_path)
        config.ai.model_path = _resolve_repo(config.ai.model_path)
        config.record_dir = _resolve_repo(config.record_dir)
        # nav.log_dir defaults to src/navigation/router/ via NavConfig — no override needed.
        return config
