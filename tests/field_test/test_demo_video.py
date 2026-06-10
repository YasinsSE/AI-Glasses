"""Unit tests for the demo-video scheduling/lookup logic (no cv2/PIL needed).

Run via pytest:
    python3 -m pytest tests/field_test/test_demo_video.py
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "eval" / "field_test"))

from make_demo_video import build_schedule, latest_before, status_at, subtitle_at  # noqa: E402


def _frames():
    return [
        {"t": 10.0, "file": "frames/a.jpg"},
        {"t": 10.5, "file": "frames/b.jpg"},
        {"t": 30.5, "file": "frames/c.jpg"},  # 20 s idle gap before this
    ]


def test_schedule_holds_each_frame_for_its_real_gap():
    sched = build_schedule(_frames(), fps=10, speed=1.0, max_hold_s=2.0)
    a = [s for s in sched if s[0] == "frames/a.jpg"]
    assert len(a) == 5  # 0.5 s gap at 10 fps


def test_long_gap_is_compressed_but_real_time_sweeps_it():
    sched = build_schedule(_frames(), fps=10, speed=1.0, max_hold_s=2.0)
    b = [s for s in sched if s[0] == "frames/b.jpg"]
    assert len(b) == 20  # 20 s real gap → only 2 s of video
    # ...but the real_t values still sweep the whole 20 s pause.
    assert b[0][1] == 10.5 and b[-1][1] > 28.0


def test_speed_halves_video_frames():
    n1 = len(build_schedule(_frames(), fps=10, speed=1.0))
    n2 = len(build_schedule(_frames(), fps=10, speed=2.0))
    assert n2 < n1 and abs(n2 - n1 / 2) <= 3


def test_empty_input():
    assert build_schedule([], fps=10) == []


def test_subtitle_window_and_suppressed_hidden():
    speaks = [
        {"t": 5.0, "method": "obstacle", "text": "Engel", "spoken": True},
        {"t": 6.0, "method": "obstacle", "text": "Gizli", "spoken": False},
    ]
    assert subtitle_at(5.1, speaks) == ("Engel", False)
    assert subtitle_at(8.4, speaks) == ("Engel", False)   # within 3.5 s window
    assert subtitle_at(9.0, speaks) is None               # window expired
    # Suppressed lines never show, even inside their window.
    assert subtitle_at(6.1, speaks) == ("Engel", False)


def test_subtitle_urgent_flag():
    speaks = [{"t": 1.0, "method": "obstacle", "text": "Durun, önünüz kapalı",
               "spoken": True}]
    assert subtitle_at(1.5, speaks) == ("Durun, önünüz kapalı", True)


def test_latest_before_and_status():
    percs = [{"t": 1.0, "safety_level": 0, "walkable": 0.5, "total_ms": 200.0},
             {"t": 2.0, "safety_level": 2, "walkable": 0.1, "total_ms": 250.0}]
    telem = [{"t": 1.5, "temps_c": {"cpu": 40.0, "gpu": 55.0}}]
    assert latest_before(1.9, percs)["safety_level"] == 0
    st = status_at(2.5, percs, telem, [], [])
    assert st["safety"] == "TEHLİKE"
    assert st["temp"] == 55.0
    assert abs(st["fps"] - 4.0) < 0.01
    # Before any data: graceful placeholders.
    st0 = status_at(0.5, percs, telem, [], [])
    assert st0["safety"] == "—" and st0["temp"] is None
