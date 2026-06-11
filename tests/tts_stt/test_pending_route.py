"""Unit tests for deferred routing when GPS has no fix (Kızılay case).

A navigation command arriving before the first GPS fix must be queued with a
spoken acknowledgement and start automatically when a fix appears; "iptal"
cancels the queued destination; a timeout gives up audibly.

Run via pytest:
    python3 -m pytest tests/tts_stt/test_pending_route.py
"""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tts_stt.voice_commands import VoiceCommandHandler


class FakeVoice:
    def __init__(self):
        self.said = []

    def say_prompt(self, text):
        self.said.append(text)


class FakeGPS:
    def __init__(self):
        self.fix = None

    def get_coord(self):
        return self.fix


class FakeNav:
    is_active = False

    def __init__(self):
        self.routed_category = None
        self.origin = None

    def navigate_to_nearest(self, coord, category):
        self.routed_category = category
        self.origin = (coord.lat, coord.lon)
        poi = SimpleNamespace(distance_m=80.0,
                              coord=SimpleNamespace(lat=39.92, lon=32.85))
        return True, "ok", poi

    def get_route(self):
        return []


def _handler(tmp_path, timeout=5.0, fallback_origin=None):
    cfg = SimpleNamespace(
        saved_places_path=str(tmp_path / "places.json"),
        bypass_stt=True,
        fallback_origin=fallback_origin,
        voice=SimpleNamespace(
            stt_listen_timeout=5, stt_silence_sec=1.5,
            nav_confirm_enabled=True, nav_confirm_min_words=3,
            confirm_listen_timeout=6.0, confirm_silence_sec=1.2,
            pending_route_timeout_sec=timeout,
        ),
    )
    nav, gps, voice = FakeNav(), FakeGPS(), FakeVoice()
    handler = VoiceCommandHandler(
        cfg, nav, gps, None, voice, SimpleNamespace(), threading.Event(),
    )
    handler.PENDING_POLL_SEC = 0.02  # fast polling for the test
    return handler, nav, gps, voice


def _wait_until(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_no_fix_queues_and_starts_when_fix_arrives(tmp_path):
    handler, nav, gps, voice = _handler(tmp_path)
    assert handler.route_to("eczane") is False
    assert any("kaydedildi" in s for s in voice.said), voice.said
    assert nav.routed_category is None
    gps.fix = (39.92, 32.85, 0.5)  # fix arrives mid-walk
    assert _wait_until(lambda: nav.routed_category == "eczane")
    assert any("GPS sinyali bulundu" in s for s in voice.said), voice.said


def test_iptal_cancels_pending_destination(tmp_path):
    handler, nav, gps, voice = _handler(tmp_path)
    handler.route_to("eczane")
    handler._handle_system_command("iptal")
    assert any("Bekleyen hedef iptal edildi" in s for s in voice.said), voice.said
    gps.fix = (39.92, 32.85, 0.5)
    time.sleep(0.15)  # waiter must NOT fire after cancellation
    assert nav.routed_category is None


def test_new_command_replaces_pending(tmp_path):
    handler, nav, gps, voice = _handler(tmp_path)
    handler.route_to("eczane")
    handler.route_to("hastane")  # newer request replaces the queued one
    gps.fix = (39.92, 32.85, 0.5)
    assert _wait_until(lambda: nav.routed_category is not None)
    time.sleep(0.1)  # give a stale waiter the chance to misfire
    assert nav.routed_category == "hastane"


def test_timeout_gives_up_audibly(tmp_path):
    handler, nav, gps, voice = _handler(tmp_path, timeout=0.05)
    handler.route_to("eczane")
    assert _wait_until(lambda: any("hâlâ yok" in s for s in voice.said))
    assert nav.routed_category is None


def test_fallback_origin_routes_immediately_without_fix(tmp_path):
    """Kızılay demo: no fix + --fallback-origin → route NOW from the known
    test-start coordinate instead of queueing."""
    handler, nav, gps, voice = _handler(
        tmp_path, fallback_origin=(39.924377, 32.845707))
    assert handler.route_to("eczane") is True
    assert nav.routed_category == "eczane"
    assert nav.origin == (39.924377, 32.845707)
    assert any("kayıtlı başlangıç konumu" in s for s in voice.said), voice.said


def test_real_fix_beats_fallback_origin(tmp_path):
    handler, nav, gps, voice = _handler(
        tmp_path, fallback_origin=(39.924377, 32.845707))
    gps.fix = (39.9300, 32.8500, 0.5)  # live GPS available
    handler.route_to("eczane")
    assert nav.origin == (39.9300, 32.8500)
    assert not any("kayıtlı başlangıç" in s for s in voice.said)
