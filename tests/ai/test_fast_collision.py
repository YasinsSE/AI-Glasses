"""Unit tests for the fast-path collision tripwire (B5).

The tripwire must fire on a hazard-flooded near-centre region regardless of
the gating chain, persist-filter single noisy frames, and respect its own
urgent cooldown.

Run via pytest:
    python3 -m pytest tests/ai/test_fast_collision.py
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
from ai.perception import ClassID
from ai.perception_service import PerceptionService


class FakeVoice:
    def __init__(self):
        self.said = []  # (text, urgent, preempt)

    def say_obstacle(self, text, urgent=False, preempt=False):
        self.said.append((text, urgent, preempt))


def _service():
    cfg = SimpleNamespace(ai=AIConfig(), vfh=SimpleNamespace(enabled=False))
    voice = FakeVoice()
    svc = PerceptionService(cfg, voice, SimpleNamespace(), threading.Event())
    return svc, voice


def _blocked_result():
    """Mask whose near-centre tripwire region is 100% vehicle."""
    mask = np.zeros((384, 512), dtype=np.uint8)
    mask[230:, 153:359] = int(ClassID.VEHICLE)
    return SimpleNamespace(
        mask=mask,
        scene=SimpleNamespace(safety_level=2, is_safe=False,
                              walkable_ratio=0.0, dominant_hazard="vehicle"),
        alerts=[],
    )


def test_tripwire_fires_after_persistence_and_preempts():
    svc, voice = _service()
    res = _blocked_result()
    assert svc._fast_collision_check(res, now=100.0) is False  # frame 1: arming
    assert svc._fast_collision_check(res, now=100.5) is True   # frame 2: fires
    assert voice.said == [("Durun, önünüz kapalı", True, True)]


def test_tripwire_respects_urgent_cooldown():
    svc, voice = _service()
    res = _blocked_result()
    svc._fast_collision_check(res, now=100.0)
    svc._fast_collision_check(res, now=100.5)
    # Still blocked moments later → no second scream inside the cooldown.
    assert svc._fast_collision_check(res, now=101.0) is False
    assert len(voice.said) == 1
    # After the cooldown it may re-warn.
    assert svc._fast_collision_check(res, now=103.0) is True
    assert len(voice.said) == 2


def test_single_noisy_frame_does_not_fire():
    svc, voice = _service()
    clear = SimpleNamespace(mask=np.zeros((384, 512), dtype=np.uint8),
                            scene=None, alerts=[])
    svc._fast_collision_check(_blocked_result(), now=100.0)
    svc._fast_collision_check(clear, now=100.5)  # counter resets
    assert svc._fast_collision_check(_blocked_result(), now=101.0) is False
    assert voice.said == []


def test_disabled_flag_turns_tripwire_off():
    svc, voice = _service()
    svc._config.ai.fast_collision_enabled = False
    res = _blocked_result()
    for i in range(5):
        assert svc._fast_collision_check(res, now=100.0 + i) is False
    assert voice.said == []
