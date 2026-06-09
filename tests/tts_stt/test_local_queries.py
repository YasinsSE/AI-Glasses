"""Unit tests for VoiceCommandHandler local queries (C1/C2/C5).

Covers "neredeyim", "evi kaydet" / "eve git", and the spoken status summary —
all keyword-matched BEFORE the SLM intent classifier so they behave the same
in --bypass-stt mode.

Run via pytest:
    python3 -m pytest tests/tts_stt/test_local_queries.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.router.models import Coord, RouteStep
from tts_stt.voice_commands import VoiceCommandHandler


class FakeVoice:
    def __init__(self):
        self.said = []

    def say_prompt(self, text):
        self.said.append(text)


class FakeGPS:
    def __init__(self, fix=(39.92, 32.85, 0.5)):
        self.fix = fix

    def get_coord(self):
        return self.fix

    def get_health(self):
        return SimpleNamespace(satellites=8)


class FakeNav:
    def __init__(self, road="Atatürk Caddesi", active=False):
        self.road = road
        self.is_active = active
        self.routed_to = None

    def where_am_i(self, position, max_dist_m=150.0):
        return self.road

    def get_route(self):
        return [RouteStep(0, "x", Coord(39.921, 32.851), "finish", 0)]

    def stop_navigation(self):
        self.is_active = False

    def start_navigation(self, origin, dest):
        self.routed_to = (dest.lat, dest.lon)
        return True, "ok"


def _handler(tmp_path, nav=None, gps=None):
    cfg = SimpleNamespace(
        saved_places_path=str(tmp_path / "places.json"),
        bypass_stt=True,
        voice=SimpleNamespace(stt_listen_timeout=5, stt_silence_sec=1),
    )
    voice = FakeVoice()
    handler = VoiceCommandHandler(
        cfg, nav or FakeNav(), gps or FakeGPS(), None, voice,
        SimpleNamespace(), threading.Event(),
    )
    return handler, voice


def test_where_am_i_speaks_road_name(tmp_path):
    handler, voice = _handler(tmp_path)
    assert handler._handle_local_query("neredeyim") is True
    assert any("Atatürk Caddesi üzerindesiniz" in s for s in voice.said), voice.said


def test_where_am_i_without_gps(tmp_path):
    handler, voice = _handler(tmp_path, gps=FakeGPS(fix=None))
    assert handler._handle_local_query("neredeyim") is True
    assert any("GPS sinyali yok" in s for s in voice.said), voice.said


def test_save_then_goto_home(tmp_path):
    nav = FakeNav()
    handler, voice = _handler(tmp_path, nav=nav)
    assert handler._handle_local_query("evi kaydet") is True
    assert any("kaydedildi" in s for s in voice.said), voice.said

    assert handler._handle_local_query("eve git") is True
    assert nav.routed_to == (39.92, 32.85)


def test_goto_unsaved_place_explains_how_to_save(tmp_path):
    handler, voice = _handler(tmp_path)
    assert handler._handle_local_query("eve git") is True
    assert any("Kayıtlı ev konumu yok" in s for s in voice.said), voice.said


def test_evet_is_not_a_place_command(tmp_path):
    handler, _ = _handler(tmp_path)
    # "evet" must not token-match the "ev" bookmark.
    assert handler._handle_local_query("evet git") is False


def test_status_summary(tmp_path):
    handler, voice = _handler(tmp_path)
    assert handler._handle_local_query("durum") is True
    said = " ".join(voice.said)
    assert "GPS iyi" in said and "aktif rota yok" in said, voice.said
