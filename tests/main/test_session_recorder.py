"""Unit tests for the field-test black-box recorder (src/main/session_recorder.py).

Covers the field-readiness guarantees with no hardware:
  * valid JSONL + summary generation,
  * frame-save throttling,
  * bounded-queue drop instead of blocking/raising,
  * pre-flight disk check disabling recording + warning,
  * crash-tolerant report parsing (truncated final line) and GPS clock rebasing.

How to run (from the repository root):
    python tests/main/test_session_recorder.py
    pytest tests/main/test_session_recorder.py
"""

import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

# Make src/ importable whether run via pytest or as a standalone script.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main.config import ALASConfig
from main import session_recorder as sr

OUTPUT_DIR = _REPO_ROOT / "outputs" / "tests" / "main"


def _config(tmp: Path) -> ALASConfig:
    cfg = ALASConfig()
    cfg.record = True
    cfg.record_dir = str(tmp)
    cfg.rec.telemetry_interval_s = 0.1
    cfg.rec.checkpoint_interval_s = 0.1
    cfg.rec.frame_min_interval_s = 1.5
    return cfg


def _stop_threads(rec: sr.SessionRecorder) -> None:
    rec._stop.set()
    try:
        rec._writer.join(timeout=2)
    except Exception:
        pass


def test_writes_valid_jsonl_and_summary():
    tmp = OUTPUT_DIR / "session_valid"
    rec = sr.SessionRecorder(_config(tmp))
    rec.log_mode("warmup", "active")
    rec.log_speak("obstacle", "Önünüzde engel var", spoken=True)
    rec.log_speak("obstacle", "Önünüzde engel var", spoken=False, reason="dedupe")
    rec.log_gps(39.92, 32.84, 1.0, 9, 1.1, "ok")
    rec.finalize()

    lines = (rec.session_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # Every line must be valid JSON with a type + monotonic timestamp.
    for line in lines:
        ev = json.loads(line)
        assert "type" in ev and "t" in ev
    assert (rec.session_dir / "summary.md").exists()
    assert (rec.session_dir / "gps_track.gpx").exists()
    summary = (rec.session_dir / "summary.md").read_text(encoding="utf-8")
    assert "Suppressed: **1**" in summary
    assert "dedupe" in summary


def test_frame_save_throttle():
    tmp = OUTPUT_DIR / "session_throttle"
    rec = sr.SessionRecorder(_config(tmp))
    _stop_threads(rec)  # freeze the writer so frames just queue
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    rec.maybe_save_overlay(frame, mask, "alert")
    rec.maybe_save_overlay(frame, mask, "alert")  # within the throttle window
    assert rec._frame_seq == 1  # second call throttled out


def test_bounded_queue_drops_instead_of_raising():
    tmp = OUTPUT_DIR / "session_drop"
    cfg = _config(tmp)
    cfg.rec.queue_maxsize = 1
    rec = sr.SessionRecorder(cfg)
    _stop_threads(rec)  # stop draining so the queue can fill
    # Fill the queue to capacity, then flood — must drop, never raise.
    try:
        while True:
            rec._q.put_nowait(("event", {"type": "filler"}))
    except queue.Full:
        pass
    for _ in range(50):
        rec.log_system(event="spam")  # would block/raise without the guard
    assert rec._dropped > 0


def test_preflight_disk_check_disables_and_warns():
    tmp = OUTPUT_DIR / "session_disk"
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = _config(tmp)
    cfg.rec.min_free_mb = 10 ** 12  # require more space than any disk has

    class _Voice:
        def __init__(self):
            self.warned = None

        def emergency(self, text):
            self.warned = text

    voice = _Voice()
    rec = sr.build_recorder(cfg, voice)
    assert isinstance(rec, sr.NullRecorder)        # recording disabled
    assert voice.warned and "depolama" in voice.warned  # spoken storage warning


def test_report_is_crash_tolerant_and_rebases_clock():
    tmp = OUTPUT_DIR / "session_crash"
    tmp.mkdir(parents=True, exist_ok=True)
    events_path = tmp / "events.jsonl"
    # A clock_sync anchor, a frame, and a deliberately truncated final line.
    lines = [
        json.dumps({"type": "clock_sync", "t": 1.0, "gps_utc": "2026-05-30T12:00:00+00:00"}),
        json.dumps({"type": "frame", "file": "frames/f_00001_alert.jpg", "tag": "alert", "t": 6.0}),
        json.dumps({"type": "speak", "method": "obstacle", "text": "x", "spoken": True, "t": 6.1}),
        '{"type": "perception", "walkable": 0.5, "is_saf',  # truncated, no newline
    ]
    events_path.write_text("\n".join(lines), encoding="utf-8")

    # report.py is in eval/; import it via path.
    eval_dir = _REPO_ROOT / "eval" / "field_test"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import report

    events, skipped = report.load_events(events_path)
    assert skipped == 1                 # the truncated line was skipped, not fatal
    assert len(events) == 3

    summary = sr.build_summary(events, skipped_lines=skipped)
    assert "Wall-clock anchor" in summary          # rebasing available
    # Frame at t=6 is 5 s after the t=1 anchor -> 12:00:05 absolute.
    assert "12:00:05" in summary

    rc = report.main([str(tmp)])
    assert rc == 0 and (tmp / "summary.md").exists()


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}  {exc}")
    print()
    print("OK — all recorder tests passed." if failures == 0 else f"FAILED — {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
