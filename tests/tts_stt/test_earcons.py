"""Unit tests for earcon resolution (C4).

Run via pytest:
    python3 -m pytest tests/tts_stt/test_earcons.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tts_stt.voice_policy import VoicePolicy


def _policy(enabled=True, earcon_dir=""):
    cfg = SimpleNamespace(voice=SimpleNamespace(
        earcons_enabled=enabled, earcon_dir=earcon_dir,
        post_nav_silence_sec=3.0,
    ))
    return VoicePolicy(cfg)


def test_default_dir_resolves_generated_wavs():
    vp = _policy()
    for direction in ("left", "right", "straight"):
        path = vp._earcon_path(direction)
        assert path is not None and path.endswith(f"drift_{direction}.wav"), path


def test_disabled_returns_none():
    assert _policy(enabled=False)._earcon_path("left") is None


def test_missing_file_returns_none(tmp_path):
    assert _policy(earcon_dir=str(tmp_path))._earcon_path("left") is None
