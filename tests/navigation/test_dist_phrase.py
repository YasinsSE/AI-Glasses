"""Unit tests for NavigationService._dist_phrase (step-based wording, B4).

Run via pytest:
    python3 -m pytest tests/navigation/test_dist_phrase.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.navigation_service import NavigationService
from navigation.router.nav_config import NavConfig


def _service(**nav_overrides):
    nav_cfg = NavConfig(**nav_overrides)
    cfg = SimpleNamespace(nav=nav_cfg, gps=SimpleNamespace(update_interval=1.0))
    return NavigationService(
        cfg, SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
        SimpleNamespace(), threading.Event(),
    )


def test_short_distance_spoken_in_steps():
    svc = _service()
    # 14 m / 0.7 m per step = 20 steps, already a multiple of 5.
    assert svc._dist_phrase(14.0) == "yaklaşık 20 adım"


def test_steps_rounded_to_nearest_five_with_floor():
    svc = _service()
    # 10 m → 14.3 steps → rounds to 15.
    assert svc._dist_phrase(10.0) == "yaklaşık 15 adım"
    # Very short distances never claim fewer than 5 steps.
    assert svc._dist_phrase(1.0) == "yaklaşık 5 adım"


def test_long_distance_stays_metric():
    svc = _service()
    assert svc._dist_phrase(40.0) == "40 metre"


def test_steps_phrasing_can_be_disabled():
    svc = _service(steps_phrasing=False)
    assert svc._dist_phrase(14.0) == "14 metre"
