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


def _result(mask):
    return SimpleNamespace(
        mask=mask,
        scene=SimpleNamespace(safety_level=2, is_safe=False,
                              walkable_ratio=0.0, dominant_hazard="vehicle"),
        alerts=[],
    )


def _blocked_result():
    """Truly boxed in: the ENTIRE bottom band is vehicle — no escape zone.

    (np.zeros means class 0 = WALKABLE_SURFACE, so anything left at zero in
    the bottom band would count as an exit for the escape check.)
    """
    mask = np.zeros((384, 512), dtype=np.uint8)
    mask[230:, :] = int(ClassID.VEHICLE)
    return _result(mask)


def _corridor_result():
    """Centre flooded by vehicles BUT a green corridor on the left — the
    field-test 10-06 'parked car row' case. Must steer, not stop."""
    mask = np.zeros((384, 512), dtype=np.uint8)
    mask[230:, :] = int(ClassID.VEHICLE)
    mask[230:, :160] = int(ClassID.WALKABLE_SURFACE)  # left walkway
    return _result(mask)


def test_tripwire_fires_after_persistence_and_preempts():
    svc, voice = _service()
    res = _blocked_result()
    assert svc._fast_collision_check(res, now=100.0) is False  # frame 1: arming
    assert svc._fast_collision_check(res, now=100.5) is True   # frame 2: fires
    assert voice.said == [("Durun, önünüz kapalı", True, True)]


def test_walkable_corridor_stands_down():
    """Escape on the side → no 'Durun'; the normal flow steers via VFH."""
    svc, voice = _service()
    res = _corridor_result()
    for i in range(6):
        assert svc._fast_collision_check(res, now=100.0 + i) is False
    assert voice.said == []


def test_walkable_dead_ahead_stands_down():
    """Centre region keeps enough walkable → cannot be 'önünüz kapalı'."""
    svc, voice = _service()
    mask = np.zeros((384, 512), dtype=np.uint8)
    mask[230:, :] = int(ClassID.VEHICLE)
    # Re-open a walkable stripe inside the centre tripwire region (>25%).
    mask[230:, 200:260] = int(ClassID.WALKABLE_SURFACE)
    res = _result(mask)
    for i in range(6):
        assert svc._fast_collision_check(res, now=100.0 + i) is False
    assert voice.said == []


def test_standing_blockage_repeats_slowly_without_preempt():
    """Same standing wall: repeat at fast_collision_repeat_sec (8 s), and the
    repeats must NOT preempt (only the fresh trigger does)."""
    svc, voice = _service()
    res = _blocked_result()
    svc._fast_collision_check(res, now=100.0)
    assert svc._fast_collision_check(res, now=100.5) is True   # fresh: preempt
    assert svc._fast_collision_check(res, now=103.0) is False  # 2.5 s later: silent now
    assert svc._fast_collision_check(res, now=106.0) is False  # still inside 8 s
    assert svc._fast_collision_check(res, now=109.0) is True   # 8.5 s: re-warn
    assert voice.said == [("Durun, önünüz kapalı", True, True),
                          ("Durun, önünüz kapalı", True, False)]


def test_cleared_blockage_rearms_fresh_trigger():
    """Blockage clears, then a NEW wall appears → preempting warn again."""
    svc, voice = _service()
    clear = _result(np.zeros((384, 512), dtype=np.uint8))
    svc._fast_collision_check(_blocked_result(), now=100.0)
    svc._fast_collision_check(_blocked_result(), now=100.5)   # spoke (fresh)
    svc._fast_collision_check(clear, now=101.0)               # cleared → re-arm
    svc._fast_collision_check(_blocked_result(), now=103.0)   # arming
    assert svc._fast_collision_check(_blocked_result(), now=103.5) is True
    assert voice.said[-1] == ("Durun, önünüz kapalı", True, True)  # preempts again


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
