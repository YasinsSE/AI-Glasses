"""Regression tests for the 11-06 school field-test fixes.

Reproduces the exact field failures:
  - school1 t+103-115: a standing close blocker stayed SILENT for 12 s behind
    the 20 s same-situation gate while the user walked into it,
  - school2 t+142: "girmeyin" fired although walkable area clearly continued
    (and the user was already walking on a road-dominant alley),
  - both sessions: "SoC 50 °C" pinned by the fake PMIC-Die zone.

Run via pytest:
    python3 -m pytest tests/ai/test_field_1106_fixes.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import ai.perception_service as ps
from ai.ai_config import AIConfig
from ai.perception import Alert, ClassID, CorridorInfo, SceneAnalysis, SAFETY_CAUTION
from ai.perception_service import PerceptionService
from main.session_recorder import max_real_temp


class FakeVoice:
    def __init__(self):
        self.said = []

    def say_obstacle(self, text, urgent=False, preempt=False):
        self.said.append((text, urgent))

    def say_drift(self, direction, text):
        self.said.append((text, False))


def _service():
    cfg = SimpleNamespace(ai=AIConfig(),
                          vfh=SimpleNamespace(announce_cooldown_sec=6.0, enabled=False))
    voice = FakeVoice()
    svc = PerceptionService(cfg, voice, SimpleNamespace(), threading.Event(),
                            nav=None, vfh=None, recorder=None)
    return svc, voice


def _result(class_id, zone, walkable, blocks_path, distance_m=4.0,
            pixel_ratio=0.1, mask=None):
    alert = Alert(class_id=int(class_id), text="x", priority=5, zone=zone,
                  distance_m=distance_m, pixel_ratio=pixel_ratio,
                  blocks_path=blocks_path, corridor_overlap=0.0)
    scene = SceneAnalysis(walkable_ratio=walkable, zones=[], is_safe=False,
                          dominant_hazard="vehicle", safety_level=SAFETY_CAUTION)
    return SimpleNamespace(alerts=[alert], scene=scene, mask=mask,
                           inference_ms=100.0, total_ms=150.0, path_guidance=None,
                           corridor=CorridorInfo(valid=True, offset=0.0,
                                                 free_ratio=0.5, crossing=False))


class _Clock:
    def __init__(self, start=5000.0):
        self.t = start

    def __enter__(self):
        self._orig = ps.time.monotonic
        ps.time.monotonic = lambda: self.t
        return self

    def __exit__(self, *a):
        ps.time.monotonic = self._orig


def test_standing_close_blocker_rewarns_within_seconds():
    """school1 t+103: same situation, user walking INTO it — must not hide
    behind the 20 s repeat gate; a close in-path blocker re-warns at the
    4 s change-cadence."""
    svc, voice = _service()
    with _Clock() as clk:
        walk = 0.28
        for i in range(24):  # 12 s at 2 FPS, walkable eroding 28% → 5%
            walk = max(0.05, walk - 0.01)
            svc._dispatch(_result(ClassID.VEHICLE, "center", walk,
                                  blocks_path=True, distance_m=1.5,
                                  pixel_ratio=0.2))
            clk.t += 0.5
    # Old behaviour: 1 utterance then 20 s of silence. New: re-warns ~every 4 s.
    assert len(voice.said) >= 3, voice.said


def test_gradual_approach_turns_urgent():
    """The windowed detector must mark a slow walking approach as CLOSING
    (urgent tone) even though per-frame walkable drops are tiny."""
    svc, voice = _service()
    with _Clock() as clk:
        walk = 0.40
        for i in range(20):  # 10 s, ~2%/frame erosion → well past the window threshold
            walk = max(0.04, walk - 0.02)
            svc._dispatch(_result(ClassID.VEHICLE, "center", walk,
                                  blocks_path=True, distance_m=1.5,
                                  pixel_ratio=0.2))
            clk.t += 0.5
    assert any(urgent for _, urgent in voice.said), voice.said


def test_road_with_walkable_continuation_is_soft():
    """school2 t+142: walkable 0.33 in view → no 'girmeyin', soft caution."""
    svc, voice = _service()
    with _Clock():
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.33,
                              blocks_path=False))
    texts = [t for t, _ in voice.said]
    assert any("dikkatli ilerleyin" in t for t in texts), texts
    assert not any("girmeyin" in t for t in texts), texts


def test_road_without_walkable_keeps_protective_girmeyin():
    svc, voice = _service()
    with _Clock():
        svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.05,
                              blocks_path=False))
    assert any("girmeyin" in t for t, _ in voice.said), voice.said


def test_already_on_road_says_so_once_then_quiet():
    """Walking ON a road-dominant alley: one 'yolda yürüyorsunuz', no
    repeated entry warnings."""
    svc, voice = _service()
    mask = np.full((384, 512), int(ClassID.VEHICLE_ROAD), dtype=np.uint8)
    with _Clock() as clk:
        for i in range(12):  # 6 s on the road
            svc._dispatch(_result(ClassID.VEHICLE_ROAD, "center", 0.10,
                                  blocks_path=False, mask=mask))
            clk.t += 0.5
    texts = [t for t, _ in voice.said]
    assert texts.count("Araç yolunda yürüyorsunuz, dikkatli ilerleyin") == 1, texts
    # An entry warning right at the boundary is fine; once "on the road" is
    # recognised there must be NO further entry warnings.
    on_road_idx = texts.index("Araç yolunda yürüyorsunuz, dikkatli ilerleyin")
    assert not any("girmeyin" in t for t in texts[on_road_idx + 1:]), texts


def test_max_real_temp_ignores_pmic_die():
    """11-06 sessions: PMIC-Die pinned every 'SoC peak' to a flat 50.0."""
    temps = {"CPU-therm": 33.0, "GPU-therm": 31.0, "PMIC-Die": 50.0, "AO-therm": 44.0}
    assert max_real_temp(temps) == 44.0
    assert max_real_temp({"PMIC-Die": 50.0}) is None
    assert max_real_temp({}) is None
