"""Unit tests for NavigationService._announce (Faz 5 turn / arrival fixes).

Locks in: the arrival message is NOT spoken a segment early (the ghost
"Hedefinize ulaştınız" + post-arrival progressing bug), FINISHED owns the
arrival announcement, and the turn pre-warning fires at the earlier 45 m
threshold and arms the post-turn progress suppression.

Run standalone or via pytest:
    python3 tests/navigation/test_navigation_announce.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.navigation_service import NavigationService
from navigation.router.models import Coord, RouteStep, RouteStatus, ProgressResult
from navigation.router.nav_config import NavConfig


class FakeVoice:
    def __init__(self):
        self.nav = []
        self.progress = []

    def say_nav(self, text):
        self.nav.append(text)

    def say_progress(self, text):
        self.progress.append(text)


class FakeNav:
    def __init__(self, route):
        self._route = route

    def get_route(self):
        return self._route


_ROUTE = [
    RouteStep(0, "Rota başlıyor", Coord(39.92, 32.85), "start", 0),
    RouteStep(1, "sağa dönün", Coord(39.92, 32.86), "turn_right", 850),
    RouteStep(2, "Hedefinize ulaştınız", Coord(39.921, 32.86), "finish", 100),
]


def _service():
    cfg = SimpleNamespace(nav=NavConfig(), gps=SimpleNamespace(update_interval=1.0))
    voice = FakeVoice()
    svc = NavigationService(
        cfg, FakeNav(_ROUTE), SimpleNamespace(), voice,
        SimpleNamespace(), threading.Event(),
    )
    return svc, voice


def test_finish_step_not_announced_at_waypoint_hit():
    """Hitting the previous node must NOT speak arrival; FINISHED owns it."""
    svc, voice = _service()
    finish = _ROUTE[2]
    svc._announce(
        ProgressResult(RouteStatus.WAYPOINT_HIT, "Hedefinize ulaştınız",
                       distance_to_next=0.0, current_step=finish),
        Coord(39.92, 32.86))
    assert not any("ulaştınız" in s for s in voice.nav), voice.nav
    # The real arrival, at the destination node:
    svc._announce(ProgressResult(RouteStatus.FINISHED, "done"), Coord(39.921, 32.86))
    assert any("Hedefinize ulaştınız" in s for s in voice.nav), voice.nav


def test_turn_prewarn_fires_at_45m_and_arms_suppression():
    """A turn within the (raised) 45 m approach is pre-warned, and the post-turn
    progress suppression timer is armed."""
    svc, voice = _service()
    turn = _ROUTE[1]
    svc._announce(
        ProgressResult(RouteStatus.PROGRESSING, "sağa dönün",
                       distance_to_next=40.0, current_step=turn),
        Coord(39.92, 32.8559))
    assert any("sağa dönün" in s for s in voice.nav), voice.nav
    assert any(s.startswith("40 metre sonra") for s in voice.nav), voice.nav
    assert svc._last_turn_at != -999.0  # suppression armed


def test_turn_now_announced_at_waypoint():
    """Reaching the turn node speaks the second trigger 'Şimdi sağa dönün' (Faz 6)."""
    svc, voice = _service()
    turn = _ROUTE[1]  # action turn_right, text "sağa dönün"
    svc._announce(
        ProgressResult(RouteStatus.WAYPOINT_HIT, turn.text,
                       distance_to_next=0.0, current_step=turn),
        Coord(39.92, 32.86))
    assert any(s == "Şimdi sağa dönün" for s in voice.nav), voice.nav


def test_off_route_still_reports_progress_not_silence():
    """OFF_ROUTE no longer goes silent — it keeps reporting progress (the field
    test went silent for ~200 s during a false off-route)."""
    svc, voice = _service()
    svc._announce(
        ProgressResult(RouteStatus.OFF_ROUTE, "off",
                       distance_to_next=120.0, current_step=_ROUTE[1]),
        Coord(39.92, 32.855))
    assert any("metre" in s for s in voice.progress), (voice.nav, voice.progress)


if __name__ == "__main__":
    test_finish_step_not_announced_at_waypoint_hit()
    test_turn_prewarn_fires_at_45m_and_arms_suppression()
    test_off_route_still_reports_progress_not_silence()
    print("all navigation_announce tests passed")
