"""Unit tests for OCR text cleaning (C3) — no tesseract needed.

Run via pytest:
    python3 -m pytest tests/ai/test_ocr_reader.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.ocr_reader import _clean


def test_keeps_words_drops_noise():
    raw = "ECZANE\n| .. ~  x\nAtatürk   Cad.\n42\n"
    assert _clean(raw) == "ECZANE Atatürk Cad 42"


def test_turkish_characters_count_as_letters():
    assert _clean("ŞEHİR İÇİ") == "ŞEHİR İÇİ"


def test_empty_and_pure_noise():
    assert _clean("") == ""
    assert _clean("~ | . , ! ?") == ""
