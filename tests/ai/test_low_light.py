"""Unit tests for the low-light reliability warning (C6).

The model fails silently in the dark, so the service must warn once when the
scene goes dark (with persistence, so a single shadowed frame is ignored) and
reassure once when light returns.

Run via pytest:
    python3 -m pytest tests/ai/test_low_light.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.ai_config import AIConfig
from ai.perception_service import PerceptionService


class FakeVoice:
    def __init__(self):
        self.said = []

    def say_obstacle(self, text, urgent=False, preempt=False):
        self.said.append(text)


def _service():
    cfg = SimpleNamespace(ai=AIConfig(), vfh=SimpleNamespace(enabled=False))
    voice = FakeVoice()
    svc = PerceptionService(cfg, voice, SimpleNamespace(), threading.Event())
    return svc, voice


def _frame(brightness):
    return np.full((48, 64, 3), brightness, dtype=np.uint8)


def test_dark_scene_warns_once_after_persistence():
    svc, voice = _service()
    n = svc._config.ai.low_light_persist_frames
    dark = _frame(10)
    for i in range(n - 1):
        svc._check_low_light(dark, now=100.0 + i)
    assert voice.said == []  # not persistent yet
    svc._check_low_light(dark, now=100.0 + n)
    assert any("karanlık" in s for s in voice.said), voice.said
    # Staying dark must NOT repeat the warning.
    before = len(voice.said)
    for i in range(3 * n):
        svc._check_low_light(dark, now=110.0 + i)
    assert len(voice.said) == before


def test_recovery_reassures_once():
    svc, voice = _service()
    n = svc._config.ai.low_light_persist_frames
    for i in range(n):
        svc._check_low_light(_frame(10), now=100.0 + i)
    for i in range(n):
        svc._check_low_light(_frame(120), now=200.0 + i)
    assert any("normale döndü" in s for s in voice.said), voice.said


def test_single_shadow_frame_ignored():
    svc, voice = _service()
    svc._check_low_light(_frame(10), now=100.0)   # one shadowed frame
    svc._check_low_light(_frame(120), now=101.0)  # back to daylight
    assert voice.said == []


def test_muted_dark_transition_stays_silent():
    svc, voice = _service()
    n = svc._config.ai.low_light_persist_frames
    for i in range(n):
        svc._check_low_light(_frame(10), now=100.0 + i, muted=True)
    assert voice.said == []
    assert svc._low_light is True  # state still tracked while muted
