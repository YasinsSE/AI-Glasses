"""Unit tests for post-turn direction confirmation (B3).

After a turn instruction the service must confirm the user actually went the
expected way (GPS course or displacement bearing), warn early when they went
the opposite way, and stay silent when the heading is ambiguous.

Run via pytest:
    python3 -m pytest tests/navigation/test_turn_confirm.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.navigation_service import NavigationService
from navigation.router.models import Coord, RouteStep
from navigation.router.nav_config import NavConfig


class FakeVoice:
    def __init__(self):
        self.nav = []
        self.progress = []

    def say_nav(self, text):
        self.nav.append(text)

    def say_progress(self, text):
        self.progress.append(text)


# Turn node at step 1; the route then heads EAST (bearing ~90°) to step 2.
_TURN = RouteStep(1, "sağa dönün", Coord(39.9200, 32.8500), "turn_right", 100)
_NEXT = RouteStep(2, "düz devam edin", Coord(39.9200, 32.8520), "continue", 170)


class FakeNav:
    def get_route(self):
        return [_TURN, _NEXT]


def _service(course=None):
    cfg = SimpleNamespace(nav=NavConfig(), gps=SimpleNamespace(update_interval=1.0))
    gps = SimpleNamespace(get_course_deg=lambda: course)
    voice = FakeVoice()
    svc = NavigationService(
        cfg, FakeNav(), gps, voice, SimpleNamespace(), threading.Event(),
    )
    return svc, voice


def test_correct_turn_confirmed():
    svc, voice = _service(course=92.0)  # walking east, as expected
    svc._arm_turn_check(_TURN, Coord(39.9200, 32.8500), now=100.0)
    # Moved ~17 m east of the turn node.
    svc._maybe_confirm_turn(Coord(39.9200, 32.8502), now=110.0)
    assert any("Doğru yöne" in s for s in voice.progress), voice.progress
    assert svc._turn_check is None  # one verdict, then disarmed


def test_wrong_turn_warned_early():
    svc, voice = _service(course=270.0)  # walking WEST — opposite way
    svc._arm_turn_check(_TURN, Coord(39.9200, 32.8500), now=100.0)
    svc._maybe_confirm_turn(Coord(39.9200, 32.8498), now=110.0)
    assert any("Yanlış yöne" in s for s in voice.nav), voice.nav


def test_no_verdict_before_enough_movement():
    svc, voice = _service(course=270.0)
    svc._arm_turn_check(_TURN, Coord(39.9200, 32.8500), now=100.0)
    # Only ~2 m moved — too early to judge; check stays armed.
    svc._maybe_confirm_turn(Coord(39.92, 32.850023), now=103.0)
    assert voice.nav == [] and voice.progress == []
    assert svc._turn_check is not None


def test_ambiguous_heading_stays_silent():
    svc, voice = _service(course=170.0)  # ~80° off: between tolerance and wrong
    svc._arm_turn_check(_TURN, Coord(39.9200, 32.8500), now=100.0)
    svc._maybe_confirm_turn(Coord(39.9199, 32.8501), now=110.0)
    assert voice.nav == [] and voice.progress == []
    assert svc._turn_check is None  # judged once, silently


def test_displacement_bearing_fallback_when_no_cog():
    """Without get_course_deg support, the user's own displacement decides."""
    svc, voice = _service(course=None)
    svc._arm_turn_check(_TURN, Coord(39.9200, 32.8500), now=100.0)
    # Displacement is due east (~90°) = the expected direction.
    svc._maybe_confirm_turn(Coord(39.9200, 32.8502), now=110.0)
    assert any("Doğru yöne" in s for s in voice.progress), voice.progress


def test_timeout_disarms_without_speaking():
    svc, voice = _service(course=270.0)
    svc._arm_turn_check(_TURN, Coord(39.9200, 32.8500), now=100.0)
    svc._maybe_confirm_turn(Coord(39.9200, 32.8498), now=140.0)  # past 30 s
    assert voice.nav == [] and voice.progress == []
    assert svc._turn_check is None
