"""Unit tests for the Faz 5 navigation fixes.

Locks in the off-route regression (a long straight OSM segment must not read as
off-route in its middle) and the cross-track distance helper.

Run standalone or via pytest:
    python3 tests/navigation/test_route_tracker.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.router.geo_utils import cross_track_distance, haversine_distance
from navigation.router.models import Coord, RouteStep, RouteStatus
from navigation.router.nav_config import NavConfig
from navigation.router.route_tracker import RouteTracker

# East–west segment at lat 39.92: 0.01° lon ≈ 850 m (a long straight stretch).
A = Coord(39.92, 32.85)
B = Coord(39.92, 32.86)


def _route():
    return [
        RouteStep(0, "Rota başlıyor", A, "start", 0),
        RouteStep(1, "sağa dönün", B, "turn_right", 850),
        RouteStep(2, "Hedefinize ulaştınız", Coord(39.921, 32.86), "finish", 100),
    ]


def test_cross_track_zero_on_the_line():
    """A point on the segment (far from both nodes) has ~0 perpendicular distance."""
    mid = Coord(39.92, 32.855)  # ~425 m from each node, but on the line
    assert haversine_distance(mid.lat, mid.lon, A.lat, A.lon) > 400  # far from node A
    xt = cross_track_distance(mid.lat, mid.lon, A.lat, A.lon, B.lat, B.lon)
    assert xt < 1.0, xt


def test_cross_track_perpendicular_offset():
    """A point offset north of the line reports that perpendicular distance."""
    off = Coord(39.9206, 32.855)  # ~67 m north of the east–west line
    xt = cross_track_distance(off.lat, off.lon, A.lat, A.lon, B.lat, B.lon)
    assert 55 < xt < 80, xt


def test_clamp_beyond_segment_end():
    """Past the segment end, distance is measured to the nearest endpoint."""
    beyond = Coord(39.92, 32.87)  # ~850 m east of B, still on the line
    xt = cross_track_distance(beyond.lat, beyond.lon, A.lat, A.lon, B.lat, B.lon)
    near_b = haversine_distance(beyond.lat, beyond.lon, B.lat, B.lon)
    assert abs(xt - near_b) < 1.0, (xt, near_b)


def test_mid_long_segment_is_not_off_route():
    """The keci regression: middle of a long segment must stay PROGRESSING."""
    tr = RouteTracker(NavConfig())
    tr.load_route(_route())
    mid = Coord(39.92, 32.855)  # 425 m from both nodes, on the line
    res = tr.check_progress(mid)
    assert res.status == RouteStatus.PROGRESSING, res.status


def test_truly_off_line_is_off_route():
    """A position well off the route polyline IS off-route."""
    tr = RouteTracker(NavConfig())
    tr.load_route(_route())
    off = Coord(39.9215, 32.855)  # ~167 m north of the line → beyond corridor
    res = tr.check_progress(off)
    assert res.status == RouteStatus.OFF_ROUTE, res.status


def test_waypoint_hit_returns_reached_turn_step():
    """WAYPOINT_HIT reports the REACHED step (the turn to execute now), so the
    service can say 'Şimdi sağa dönün' (Faz 6)."""
    tr = RouteTracker(NavConfig())
    tr.load_route(_route())
    tr.check_progress(A)                  # reach start node → advance to turn
    res = tr.check_progress(B)            # reach the turn node
    assert res.status == RouteStatus.WAYPOINT_HIT, res.status
    assert res.current_step.action == "turn_right", res.current_step
    assert res.current_step.text == "sağa dönün", res.current_step


if __name__ == "__main__":
    test_cross_track_zero_on_the_line()
    test_cross_track_perpendicular_offset()
    test_clamp_beyond_segment_end()
    test_mid_long_segment_is_not_off_route()
    test_truly_off_line_is_off_route()
    print("all route_tracker tests passed")
