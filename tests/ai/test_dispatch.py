"""Unit tests for PerceptionService._dispatch speech gating.

These lock in the field-test fixes: the same hazard situation must not repeat
just because a proximity/direction word flickered (the ECZANE_TEST bug where
identical warnings fired 5 s apart), a hazard drifting out of the path must go
silent, and a hazard persisting in the forward path must escalate to a VFH
dodge.

Run standalone or via pytest:
    python3 tests/ai/test_dispatch.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.ai_config import AIConfig
from ai.perception import (
    Alert, ClassID, SceneAnalysis,
    SAFETY_SAFE, SAFETY_CAUTION, SAFETY_UNSAFE,
)
from ai.perception_service import PerceptionService


class FakeVoice:
    def __init__(self):
        self.said = []

    def say_obstacle(self, text):
        self.said.append(text)


class FakeVFH:
    """Always returns a left-dodge escape route."""
    def plan(self, mask, scene, target_action=None):
        return SimpleNamespace(
            text="Hafif sağa yönelin",
            action=SimpleNamespace(value="right"),
            sector_index=4,
            histogram=[0.1, 0.2],
        )


def _service(vfh=None):
    cfg = SimpleNamespace(
        ai=AIConfig(),
        vfh=SimpleNamespace(announce_cooldown_sec=6.0, enabled=bool(vfh)),
    )
    voice = FakeVoice()
    svc = PerceptionService(
        cfg, voice, SimpleNamespace(), threading.Event(),
        nav=None, vfh=vfh, recorder=None,
    )
    return svc, voice


def _result(class_id, zone, walkable, safety, blocks_path,
            distance_m=4.0, pixel_ratio=0.1):
    alert = Alert(
        class_id=int(class_id), text="x", priority=5,
        zone=zone, distance_m=distance_m, pixel_ratio=pixel_ratio,
        blocks_path=blocks_path,
    )
    scene = SceneAnalysis(
        walkable_ratio=walkable, zones=[], is_safe=(safety == SAFETY_SAFE),
        dominant_hazard="vehicle", safety_level=safety,
    )
    return SimpleNamespace(
        alerts=[alert], scene=scene, mask=None,
        inference_ms=100.0, total_ms=150.0, path_guidance=None,
    )


def test_same_situation_not_repeated_on_walkable_flicker():
    """The ECZANE bug: same hazard, walkable 23%->36% must NOT re-speak."""
    svc, voice = _service()
    t = [1000.0]
    svc_now = lambda: t[0]
    import ai.perception_service as ps
    orig = ps.time.monotonic
    ps.time.monotonic = svc_now
    try:
        # First detection: vehicle on the left, UNSAFE.
        svc._dispatch(_result(ClassID.VEHICLE, "left", 0.234, SAFETY_UNSAFE, False))
        assert len(voice.said) == 1, voice.said
        # 5 s later: identical situation, only walkable changed 23%->36%.
        t[0] += 5.0
        svc._dispatch(_result(ClassID.VEHICLE, "left", 0.364, SAFETY_UNSAFE, False))
        assert len(voice.said) == 1, ("repeated within repeat window", voice.said)
        # After the long repeat interval it may speak again.
        t[0] += svc._config.ai.min_obstacle_repeat_sec + 0.1
        svc._dispatch(_result(ClassID.VEHICLE, "left", 0.30, SAFETY_UNSAFE, False))
        assert len(voice.said) == 2, voice.said
    finally:
        ps.time.monotonic = orig


def test_zone_change_speaks_again():
    """Hazard moving left -> center is a new situation; should speak."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [2000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        svc._dispatch(_result(ClassID.VEHICLE, "left", 0.2, SAFETY_UNSAFE, False))
        assert len(voice.said) == 1
        t[0] += 4.5  # past short interval, under repeat interval
        # Zone needs two frames to flip (hysteresis), then a new signature.
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.2, SAFETY_UNSAFE, True))
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.2, SAFETY_UNSAFE, True))
        assert len(voice.said) == 2, voice.said
        assert "Önünüzde" in voice.said[1] or "Durun" in voice.said[1]
    finally:
        ps.time.monotonic = orig


def test_persisting_center_hazard_escalates_to_vfh():
    """A vehicle blocking the path for >=escalation_frames escalates to a dodge."""
    svc, voice = _service(vfh=FakeVFH())
    import ai.perception_service as ps
    t = [3000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        # Centre, blocking path (low walkable). First frame: cautious "Durun".
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.08, SAFETY_UNSAFE, True))
        assert "Durun" in voice.said[-1], voice.said
        # Still blocking a few seconds later: escalation_frames reached and a
        # VFH escape route exists → actionable dodge instruction.
        t[0] += 4.5
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.08, SAFETY_UNSAFE, True))
        assert any("yönelin" in m for m in voice.said), voice.said
    finally:
        ps.time.monotonic = orig


def test_safe_resets_and_announces():
    svc, voice = _service()
    import ai.perception_service as ps
    t = [4000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.05, SAFETY_UNSAFE, True))
        n = len(voice.said)
        # Transition to SAFE → "Yol açık".
        svc._dispatch(_result(ClassID.WALKABLE_SURFACE, "center", 0.9, SAFETY_SAFE, False))
        assert any("Yol açık" in m for m in voice.said[n:]), voice.said
        # Situation state cleared.
        assert svc._cur_sig is None and svc._last_spoken_sig is None
    finally:
        ps.time.monotonic = orig


if __name__ == "__main__":
    test_same_situation_not_repeated_on_walkable_flicker()
    test_zone_change_speaks_again()
    test_persisting_center_hazard_escalates_to_vfh()
    test_safe_resets_and_announces()
    print("all dispatch tests passed")
