"""Unit tests for the spoken navigation confirmation flow.

A destination inferred from a long utterance ("ilaç almak istiyorum eczaneye
gitmem lazım") must be confirmed ("...rota başlatılsın mı?") before routing;
terse direct commands ("eczane") start immediately. Unclear answers default
to NO — never start a route the user may not have asked for.

Run via pytest:
    python3 -m pytest tests/tts_stt/test_nav_confirm.py
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main.lifecycle import SystemMode
from tts_stt.voice_commands import VoiceCommandHandler

EDGE_UTTERANCE = "ilaç almak istiyorum eczaneye gitmem lazım"


class FakeVoice:
    def __init__(self):
        self.said = []

    def say_prompt(self, text):
        self.said.append(text)


class FakeSTT:
    """Returns queued utterances; records the grammar of each listen call."""

    def __init__(self, answers, intent="navigation"):
        self.answers = list(answers)
        self.grammars = []
        self._slm = SimpleNamespace(predict=lambda text: (intent, 1.0))

    def listen(self, timeout_sec=5.0, silence_sec=1.5, grammar=None):
        self.grammars.append(grammar)
        return self.answers.pop(0) if self.answers else ""


class FakeGPS:
    def get_coord(self):
        return (39.92, 32.85, 0.5)


class FakeNav:
    is_active = False

    def __init__(self):
        self.routed_category = None

    def navigate_to_nearest(self, coord, category):
        self.routed_category = category
        poi = SimpleNamespace(distance_m=120.0,
                              coord=SimpleNamespace(lat=39.921, lon=32.851))
        return True, "ok", poi

    def get_route(self):
        return []


def _handler(tmp_path, stt):
    cfg = SimpleNamespace(
        saved_places_path=str(tmp_path / "places.json"),
        bypass_stt=False,
        voice=SimpleNamespace(
            stt_listen_timeout=5, stt_silence_sec=1.5,
            nav_confirm_enabled=True, nav_confirm_min_words=3,
            confirm_listen_timeout=6.0, confirm_silence_sec=1.2,
        ),
    )
    nav, voice = FakeNav(), FakeVoice()
    handler = VoiceCommandHandler(
        cfg, nav, FakeGPS(), stt, voice,
        SimpleNamespace(mode=SystemMode.ACTIVE), threading.Event(),
    )
    return handler, nav, voice


def test_edge_utterance_confirms_then_routes_on_evet(tmp_path):
    stt = FakeSTT(["evet"])
    handler, nav, voice = _handler(tmp_path, stt)
    handler._handle_navigation(EDGE_UTTERANCE)
    assert any("rota başlatılsın mı" in s for s in voice.said), voice.said
    assert nav.routed_category == "eczane"
    # The yes/no listen ran with the constrained grammar.
    assert stt.grammars and "evet" in stt.grammars[0]


def test_hayir_cancels(tmp_path):
    stt = FakeSTT(["hayır"])
    handler, nav, voice = _handler(tmp_path, stt)
    handler._handle_navigation(EDGE_UTTERANCE)
    assert nav.routed_category is None
    assert any("İptal" in s for s in voice.said), voice.said


def test_unclear_twice_defaults_to_no(tmp_path):
    stt = FakeSTT(["şey", "hmm belki"])
    handler, nav, voice = _handler(tmp_path, stt)
    handler._handle_navigation(EDGE_UTTERANCE)
    assert nav.routed_category is None
    assert any("Evet mi, hayır mı" in s for s in voice.said), voice.said


def test_terse_command_skips_confirmation(tmp_path):
    stt = FakeSTT([])  # no confirmation listen must happen
    handler, nav, voice = _handler(tmp_path, stt)
    handler._handle_navigation("eczaneye git")
    assert nav.routed_category == "eczane"
    assert stt.grammars == []
    assert not any("başlatılsın mı" in s for s in voice.said)


def test_general_intent_with_poi_keyword_recovers(tmp_path):
    """SLM misclassifies as 'general' → keyword fallback still navigates,
    guarded by the confirmation question."""
    stt = FakeSTT([EDGE_UTTERANCE, "evet"], intent="general")
    handler, nav, voice = _handler(tmp_path, stt)
    handler.handle_press()
    assert nav.routed_category == "eczane"
    assert any("rota başlatılsın mı" in s for s in voice.said), voice.said


def test_bypass_typed_command_never_confirms(tmp_path):
    handler, nav, voice = _handler(tmp_path, stt=FakeSTT([]))
    handler._stt = None  # typed --bypass-stt path
    handler._handle_navigation(EDGE_UTTERANCE)
    assert nav.routed_category == "eczane"
    assert not any("başlatılsın mı" in s for s in voice.said)
