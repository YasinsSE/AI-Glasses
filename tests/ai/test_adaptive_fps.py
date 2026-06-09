"""Unit tests for the context-aware perception rate (B1) + thermal guard (A1).

Run via pytest:
    python3 -m pytest tests/ai/test_adaptive_fps.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.ai_config import AIConfig
from ai.perception import SAFETY_SAFE, SAFETY_UNSAFE
from ai.perception_service import PerceptionService


def _service():
    cfg = SimpleNamespace(ai=AIConfig(), vfh=SimpleNamespace(enabled=False))
    return PerceptionService(
        cfg, SimpleNamespace(), SimpleNamespace(), threading.Event(),
    )


def test_base_rate_by_default():
    svc = _service()
    assert abs(svc._target_interval() - 1.0 / svc._config.ai.perception_fps) < 1e-9


def test_unsafe_scene_boosts_rate():
    svc = _service()
    svc._prev_safety_level = SAFETY_UNSAFE
    assert abs(svc._target_interval() - 1.0 / svc._config.ai.fps_alert) < 1e-9


def test_closing_threat_boosts_rate():
    svc = _service()
    svc._closing_frames = 1
    assert abs(svc._target_interval() - 1.0 / svc._config.ai.fps_alert) < 1e-9


def test_sustained_calm_drops_rate():
    svc = _service()
    svc._prev_safety_level = SAFETY_SAFE
    svc._calm_frames = svc._config.ai.idle_safe_persist_frames
    assert abs(svc._target_interval() - 1.0 / svc._config.ai.fps_idle) < 1e-9


def test_thermal_guard_beats_hazard_boost():
    """Never speed up while the SoC is hot, even with a closing hazard."""
    svc = _service()
    svc._thermal_throttled = True
    svc._prev_safety_level = SAFETY_UNSAFE
    svc._closing_frames = 5
    assert abs(svc._target_interval() - 1.0 / svc._config.ai.thermal_min_fps) < 1e-9


def test_adaptive_disabled_pins_base_rate():
    svc = _service()
    svc._config.ai.adaptive_fps_enabled = False
    svc._prev_safety_level = SAFETY_UNSAFE
    assert abs(svc._target_interval() - 1.0 / svc._config.ai.perception_fps) < 1e-9
