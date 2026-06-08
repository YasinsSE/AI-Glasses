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
    Alert, ClassID, CorridorInfo, SceneAnalysis,
    SAFETY_SAFE, SAFETY_CAUTION, SAFETY_UNSAFE,
)
from ai.perception_service import PerceptionService


class FakeVoice:
    def __init__(self):
        self.said = []
        self.preempted = []

    def say_obstacle(self, text, urgent=False, preempt=False):
        self.said.append(text)
        if preempt:
            self.preempted.append(text)


class FakeVFH:
    """Always returns a left-dodge escape route."""
    def plan(self, mask, scene, target_action=None):
        return SimpleNamespace(
            text="Hafif sağa yönelin",
            action=SimpleNamespace(value="right"),
            sector_index=4,
            histogram=[0.1, 0.2],
        )


def _service(vfh=None, nav=None):
    cfg = SimpleNamespace(
        ai=AIConfig(),
        vfh=SimpleNamespace(announce_cooldown_sec=6.0, enabled=bool(vfh)),
    )
    voice = FakeVoice()
    svc = PerceptionService(
        cfg, voice, SimpleNamespace(), threading.Event(),
        nav=nav, vfh=vfh, recorder=None,
    )
    return svc, voice


def _result(class_id, zone, walkable, safety, blocks_path,
            distance_m=4.0, pixel_ratio=0.1, crossing=False, corridor_overlap=0.0):
    alert = Alert(
        class_id=int(class_id), text="x", priority=5,
        zone=zone, distance_m=distance_m, pixel_ratio=pixel_ratio,
        blocks_path=blocks_path, corridor_overlap=corridor_overlap,
    )
    scene = SceneAnalysis(
        walkable_ratio=walkable, zones=[], is_safe=(safety == SAFETY_SAFE),
        dominant_hazard="vehicle", safety_level=safety,
    )
    return SimpleNamespace(
        alerts=[alert], scene=scene, mask=None,
        inference_ms=100.0, total_ms=150.0, path_guidance=None,
        corridor=CorridorInfo(valid=True, offset=0.0, free_ratio=0.5,
                              crossing=crossing),
    )


def _result_corridor(free_ratio, safety=SAFETY_CAUTION):
    """A no-hazard result with a given corridor free_ratio (for path-keeping)."""
    scene = SceneAnalysis(
        walkable_ratio=0.5, zones=[], is_safe=(safety == SAFETY_SAFE),
        dominant_hazard=None, safety_level=safety,
    )
    return SimpleNamespace(
        alerts=[], scene=scene, mask=None,
        inference_ms=100.0, total_ms=150.0, path_guidance=None,
        corridor=CorridorInfo(valid=True, offset=0.0, free_ratio=free_ratio),
    )


def _no_unsafe_crossing_wording(said):
    """A road/crossing notice must NEVER imply it is safe to cross."""
    banned = ("geçebilir", "güvenli", "geçebilirsiniz")
    return not any(b in s.lower() for s in said for b in banned)


def test_road_ahead_is_cautionary_not_permissive():
    """Road straight ahead (no crossing) → a caution, never a 'safe to cross'."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [5000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.3,
                              SAFETY_CAUTION, False, crossing=False))
        assert any("araç yolu" in s for s in voice.said), voice.said
        assert _no_unsafe_crossing_wording(voice.said), voice.said
    finally:
        ps.time.monotonic = orig


def test_crossing_announced_only_after_persistence():
    """A crossing notice fires only after the candidate persists, and is a caution."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [6000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        # First frame: candidate seen but not yet confirmed → plain road caution.
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.3,
                              SAFETY_CAUTION, False, crossing=True))
        assert any("araç yolu" in s for s in voice.said), voice.said
        assert not any("kaldırım" in s for s in voice.said), voice.said
        # Hold the candidate; the road-event gap suppresses speech meanwhile,
        # but the persistence counter keeps climbing.
        t[0] += 0.5
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.3,
                              SAFETY_CAUTION, False, crossing=True))
        t[0] += 0.5
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.3,
                              SAFETY_CAUTION, False, crossing=True))
        # After the event gap, confirmed crossing upgrades the wording.
        t[0] += svc._config.ai.ambient_min_gap_sec + 0.1
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.3,
                              SAFETY_CAUTION, False, crossing=True))
        assert any("karşı tarafta kaldırım" in s for s in voice.said), voice.said
        assert _no_unsafe_crossing_wording(voice.said), voice.said
    finally:
        ps.time.monotonic = orig


def test_same_situation_not_repeated_on_walkable_flicker():
    """The ECZANE bug: same hazard, walkable 23%->36% must NOT re-speak."""
    svc, voice = _service()
    t = [1000.0]
    svc_now = lambda: t[0]
    import ai.perception_service as ps
    orig = ps.time.monotonic
    ps.time.monotonic = svc_now
    try:
        # First detection: vehicle blocking the centerline, UNSAFE.
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.10, SAFETY_UNSAFE, True))
        assert len(voice.said) == 1, voice.said
        # 5 s later: identical situation, only walkable flickered (no closing).
        t[0] += 5.0
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.11, SAFETY_UNSAFE, True))
        assert len(voice.said) == 1, ("repeated within repeat window", voice.said)
        # After the long repeat interval it may speak again.
        t[0] += svc._config.ai.min_obstacle_repeat_sec + 0.1
        svc._dispatch(_result(ClassID.VEHICLE, "center", 0.10, SAFETY_UNSAFE, True))
        assert len(voice.said) == 2, voice.said
    finally:
        ps.time.monotonic = orig


def test_ambient_side_obstacle_announced_once():
    """A side hazard is announced once (Faz 3 awareness), not repeated unchanged."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [2000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        # Side car: not blocking centerline, not imminent → ambient notice, once.
        svc._dispatch(_result(ClassID.VEHICLE, "right", 0.3, SAFETY_UNSAFE, False,
                              distance_m=5.0, pixel_ratio=0.05))
        assert any("Sağ taraf" in s for s in voice.said), voice.said
        t[0] += 3.0  # same hazard shortly after → must NOT repeat the notice
        svc._dispatch(_result(ClassID.VEHICLE, "right", 0.3, SAFETY_UNSAFE, False,
                              distance_m=5.0, pixel_ratio=0.05))
        assert sum("Sağ taraf" in s for s in voice.said) == 1, voice.said
    finally:
        ps.time.monotonic = orig


def test_parked_side_car_no_imminent_spam():
    """A parked car we walk PAST (beside the corridor) must NOT fire imminent (Faz 5).

    The car is close and the distance shrinks frame-to-frame, but its corridor
    overlap is ~0 (it is off to the side), so it is not a collision course. This
    is the keci-test regression: "araç çok yakın" on every parked car.
    """
    svc, voice = _service()
    import ai.perception_service as ps
    t = [3000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        for d in (5.0, 4.0, 2.0):  # closing fast, but off to the side
            t[0] += 0.5
            svc._dispatch(_result(ClassID.VEHICLE, "right", 0.3, SAFETY_UNSAFE, False,
                                  distance_m=d, pixel_ratio=0.12, corridor_overlap=0.0))
        assert not any("yönelin" in s or "çok yakın" in s for s in voice.said), voice.said
    finally:
        ps.time.monotonic = orig


def test_corridor_intrusion_warns_with_guidance():
    """An obstacle that intrudes the corridor AND keeps closing warns WITH a
    steer direction, not a bare alarm (Faz 5)."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [3000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        # Frame 1 baseline, then sustained closing (5→4→2 m) with high corridor
        # overlap (the car is moving into the path ahead, on our right).
        for d in (5.0, 4.0, 2.0):
            t[0] += 0.5
            svc._dispatch(_result(ClassID.VEHICLE, "right", 0.3, SAFETY_UNSAFE, False,
                                  distance_m=d, pixel_ratio=0.12, corridor_overlap=0.5))
        assert any("yönelin" in s for s in voice.said), voice.said
        # Obstacle on the right → steer to the open left.
        assert any("sola" in s for s in voice.said), voice.said
        # A collision course PREEMPTS the current audio (Faz 6).
        assert any("yönelin" in s for s in voice.preempted), voice.preempted
    finally:
        ps.time.monotonic = orig


def test_narrow_passage_enter_and_clear():
    """Squeezing between obstacles warns once on entering and reassures once on
    opening back up (Faz 6 narrow-passage state machine)."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [3000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        # The squeeze must persist narrow_persist_frames before it announces
        # (anti-noise); a single low frame must NOT fire.
        t[0] += 0.5
        svc._dispatch(_result_corridor(free_ratio=0.10))
        assert not any("daralıyor" in s for s in voice.said), voice.said
        for _ in range(2):
            t[0] += 0.5
            svc._dispatch(_result_corridor(free_ratio=0.10))   # now confirmed narrow
        assert any("daralıyor" in s for s in voice.said), voice.said
        assert not any("Alan açıldı" in s for s in voice.said), voice.said
        # Opening up also has to persist before "Alan açıldı".
        for _ in range(3):
            t[0] += 0.5
            svc._dispatch(_result_corridor(free_ratio=0.55))
        assert any("Alan açıldı" in s for s in voice.said), voice.said
    finally:
        ps.time.monotonic = orig


def test_path_keeping_speaks_during_navigation():
    """Camera guidance must be spoken even while navigation is active (Faz 6
    regression fix: it used to be fully suppressed during nav, so a nav test
    produced no centering/awareness guidance at all)."""
    nav = SimpleNamespace(is_active=True, current_step=None)
    svc, voice = _service(nav=nav)
    import ai.perception_service as ps
    t = [3000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        svc._dispatch(_result_corridor(free_ratio=0.6))  # nav active, open path
        assert voice.said, "guidance was suppressed during navigation"
    finally:
        ps.time.monotonic = orig


def test_muted_suppresses_non_collision_but_collision_preempts():
    """While a priority utterance plays (muted), only a collision speaks."""
    svc, voice = _service()
    import ai.perception_service as ps
    t = [3000.0]
    orig = ps.time.monotonic
    ps.time.monotonic = lambda: t[0]
    try:
        # Muted + narrow passage (non-collision) → stays silent.
        svc._dispatch(_result_corridor(free_ratio=0.10), muted=True)
        assert voice.said == [], voice.said
        # Muted + collision course → speaks AND preempts.
        for d in (5.0, 4.0, 2.0):
            t[0] += 0.5
            svc._dispatch(_result(ClassID.VEHICLE, "right", 0.3, SAFETY_UNSAFE, False,
                                  distance_m=d, pixel_ratio=0.12, corridor_overlap=0.5),
                          muted=True)
        assert voice.preempted, voice.preempted
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
