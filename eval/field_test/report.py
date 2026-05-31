"""Offline report generator for an ALAS field-test session.

Reads a session folder's ``events.jsonl`` and (re)builds ``summary.md``. This is
the recovery path when the device lost power mid-walk and ``finalize()`` never
ran: it tolerates a truncated/garbage final line and produces the best report it
can from whatever parsed.

How to run (from the repository root):
    python eval/field_test/report.py outputs/field_tests/<timestamp>/
    python eval/field_test/report.py outputs/field_tests/<timestamp>/ --out /tmp/report.md
"""

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable whether run via pytest or as a standalone script.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main.session_recorder import build_summary, write_gpx, _write_viewer


def load_events(events_path: Path):
    """Parse events.jsonl defensively. Returns (events, skipped_line_count).

    Each line is parsed independently; malformed or truncated lines (typical
    after a power-cut) are skipped rather than aborting the whole report.
    """
    events = []
    skipped = 0
    if not events_path.exists():
        return events, skipped
    with open(events_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                skipped += 1
    return events, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild an ALAS field-test summary")
    parser.add_argument("session_dir", help="Path to a outputs/field_tests/<timestamp>/ folder")
    parser.add_argument("--out", default=None, help="Output path (default: <session_dir>/summary.md)")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        print(f"ERROR: no events.jsonl in {session_dir}", file=sys.stderr)
        return 1

    events, skipped = load_events(events_path)
    if skipped:
        print(f"[report] Skipped {skipped} unreadable line(s) (likely a power-cut truncation).")

    summary = build_summary(events, title="ALAS Field Test Report", skipped_lines=skipped)
    out_path = Path(args.out) if args.out else session_dir / "summary.md"
    out_path.write_text(summary, encoding="utf-8")

    # Best-effort GPX and viewer regeneration (harmless on error).
    try:
        write_gpx(events, session_dir / "gps_track.gpx")
    except Exception:
        pass
    try:
        _write_viewer(events, session_dir)
        print(f"[report] viewer.html written to {session_dir}")
    except Exception:
        pass

    print(f"[report] {len(events)} events -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
