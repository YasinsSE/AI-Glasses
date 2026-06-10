"""Unit tests for the field-test viewer generator (Viewer 2.0).

Builds the HTML from a synthetic event list and checks structure — no
browser required. The embedded JSON must parse and carry the prepared data.

Run via pytest:
    python3 -m pytest tests/field_test/test_viewer.py
"""

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "eval" / "field_test"))

from viewer import build_html  # noqa: E402


def _events():
    return [
        {"type": "system", "event": "session_start", "t": 0.0,
         "wall": "2026-06-10T10:00:00+03:00"},
        {"type": "perception", "t": 1.0, "walkable": 0.42, "safety_level": 0,
         "hazard": None, "inf_ms": 150.0, "total_ms": 200.0, "chosen": None},
        {"type": "frame", "t": 1.1, "file": "frames/f_00001_scene.jpg", "tag": "scene",
         "wall": "2026-06-10T10:00:01+03:00"},
        {"type": "speak", "t": 1.5, "method": "obstacle",
         "text": "Saat iki yönünde araç var", "spoken": True, "reason": None},
        {"type": "speak", "t": 2.0, "method": "obstacle",
         "text": "Önünüzde engel", "spoken": False, "reason": "muted_unsafe"},
        {"type": "nav", "t": 3.0, "status": "waypoint_hit",
         "distance_to_next_m": 12.0, "step_text": "sağa dönün"},
        {"type": "telemetry", "t": 5.0, "temps_c": {"CPU-therm": 48.0, "GPU-therm": 51.5},
         "gpu_pct": 40.0, "ram": {"used_pct": 47.0}},
        {"type": "gps", "t": 6.0, "lat": 39.9200, "lon": 32.8500},
        {"type": "gps", "t": 7.0, "lat": 39.9201, "lon": 32.8501},
        {"type": "system", "t": 60.0, "error": "low_disk"},
    ]


def _embedded_data(html):
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "embedded data block missing"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_build_html_structure_and_data():
    html = build_html(Path("/tmp/test_session"), _events())
    # All five tabs present.
    for token in ("Özet", "Zaman Çizelgesi", "Harita", "Grafikler", "Konuşmalar"):
        assert token in html, token
    data = _embedded_data(html)
    s = data["summary"]
    assert s["session"] == "test_session"
    assert s["n_spoken"] == 1 and s["n_suppressed"] == 1
    assert s["reasons"] == {"muted_unsafe": 1}
    assert s["temp_peak"] == 51.5
    assert s["walked_m"] is not None and 5 <= s["walked_m"] <= 25
    assert data["frames"][0]["file"] == "frames/f_00001_scene.jpg"
    assert len(data["gps"]) == 2
    # Anomaly markers for charts: only the error system event, not session_start.
    assert [m["label"] for m in data["sysMarks"]] == ["low_disk"]


def test_turkish_text_survives_into_json():
    html = build_html(Path("/tmp/s"), _events())
    data = _embedded_data(html)
    assert data["speaks"][0]["text"] == "Saat iki yönünde araç var"


def test_empty_session_still_builds():
    html = build_html(Path("/tmp/bos"), [])
    assert "ALAS" in html
    data = _embedded_data(html)
    assert data["summary"]["n_events"] == 0
    assert data["frames"] == [] and data["gps"] == []


def test_script_close_tag_in_text_is_escaped():
    """A '</script>' inside spoken text must not break the data block."""
    evs = _events() + [{"type": "speak", "t": 9.0, "method": "prompt",
                        "text": "tehlikeli </script> metin", "spoken": True}]
    html = build_html(Path("/tmp/s"), evs)
    data = _embedded_data(html)
    assert any("tehlikeli" in s["text"] for s in data["speaks"])
