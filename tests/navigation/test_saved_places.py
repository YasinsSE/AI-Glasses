"""Unit tests for SavedPlaces (named GPS bookmarks, "evi kaydet" / "eve git").

Run standalone or via pytest:
    python3 -m pytest tests/navigation/test_saved_places.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.saved_places import SavedPlaces


def test_save_and_reload_roundtrip(tmp_path):
    path = str(tmp_path / "places.json")
    places = SavedPlaces(path)
    places.save("Ev", 39.924501, 32.846502)

    # A fresh instance must read the same bookmark back (name lowercased).
    reloaded = SavedPlaces(path)
    assert reloaded.get("ev") == (39.924501, 32.846502)
    assert reloaded.get("EV") == (39.924501, 32.846502)
    assert reloaded.names() == ["ev"]


def test_missing_place_and_missing_file(tmp_path):
    places = SavedPlaces(str(tmp_path / "nope.json"))
    assert places.get("ev") is None
    assert places.names() == []


def test_corrupt_file_starts_empty(tmp_path):
    path = tmp_path / "places.json"
    path.write_text("{not json", encoding="utf-8")
    places = SavedPlaces(str(path))
    assert places.get("ev") is None
    # And saving over the corrupt file recovers it.
    places.save("ev", 1.0, 2.0)
    assert SavedPlaces(str(path)).get("ev") == (1.0, 2.0)
