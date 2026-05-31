"""Interactive HTML viewer for an ALAS field-test session.

Generates a self-contained ``viewer.html`` inside the session directory.
For each saved overlay frame, the viewer shows:
  - the annotated image
  - the exact local wall time (from events.jsonl)
  - all spoken feedback within ±3 s of the frame
  - the perception result at that moment (walkable %, hazard, walkable safe?)
  - all navigation events within ±5 s

Also generates a chronological "speak timeline" at the top so the reviewer
can see how often the system was talking and jump to any spoken moment.

How to run (from the repository root):
    python eval/field_test/viewer.py outputs/field_tests/<timestamp>/
    python eval/field_test/viewer.py outputs/field_tests/<timestamp>/ --out /tmp/viewer.html
"""

import argparse
import json
import sys
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_events(events_path: Path):
    events = []
    if not events_path.exists():
        return events
    with open(events_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return events


def _wall(ev: dict) -> str:
    """Return the wall timestamp string, prefer local-time isoformat."""
    w = ev.get("wall", "")
    if w:
        # Remove sub-second precision for display brevity
        return w[:19].replace("T", " ")
    return f"t+{ev.get('t', 0):.1f}s"


def _nearby(events, t_center, before=3.0, after=3.0):
    return [e for e in events if abs(e.get("t", -999) - t_center) <= max(before, after)
            and e.get("t", -999) >= t_center - before
            and e.get("t", -999) <= t_center + after]


def _esc(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ── HTML builder ─────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #111; color: #eee; }
h1 { padding: 18px 20px 6px; font-size: 1.4rem; color: #fff; }
.meta { padding: 4px 20px 14px; color: #888; font-size: .85rem; }

/* ── Timeline bar ── */
.tl-section { padding: 12px 20px; background: #1a1a1a; border-bottom: 1px solid #333; }
.tl-section h2 { font-size: 1rem; color: #aaa; margin-bottom: 8px; }
.tl { display: flex; flex-wrap: wrap; gap: 4px; }
.tl-chip { padding: 2px 7px; border-radius: 3px; font-size: .72rem;
           cursor: pointer; white-space: nowrap; }
.tl-chip:hover { opacity: .8; }
.tl-obstacle { background: #a33; }
.tl-nav      { background: #36a; }
.tl-announce { background: #484; }
.tl-safe     { background: #383; }
.tl-emergency{ background: #a63; }

/* ── Frame cards ── */
.frames { display: flex; flex-wrap: wrap; gap: 18px; padding: 18px 20px; }
.card { background: #1e1e1e; border: 1px solid #333; border-radius: 6px;
        width: 580px; overflow: hidden; }
.card img { width: 100%; display: block; }
.card-body { padding: 10px 12px; }
.card-time { font-size: .78rem; color: #888; margin-bottom: 6px; }
.card-time strong { color: #ccc; }
.card-perc { font-size: .8rem; margin-bottom: 8px; }
.safe-badge  { display: inline-block; background: #2a5; color:#fff; border-radius:3px;
               padding: 1px 6px; font-size:.72rem; margin-right:6px; }
.unsafe-badge{ display: inline-block; background: #a33; color:#fff; border-radius:3px;
               padding: 1px 6px; font-size:.72rem; margin-right:6px; }
.section-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
                 color: #666; margin: 8px 0 4px; }
.speak-row { font-size: .82rem; padding: 3px 0; border-left: 3px solid #555;
             padding-left: 7px; margin-bottom: 3px; }
.speak-row.obstacle { border-color: #a33; }
.speak-row.nav      { border-color: #36a; }
.speak-row.safe     { border-color: #2a5; }
.speak-row.announce { border-color: #484; }
.speak-time { color: #888; font-size: .72rem; margin-right: 5px; }
.nav-row { font-size: .78rem; color: #8cf; padding: 2px 0; }
.empty { color: #555; font-size: .8rem; font-style: italic; }
"""

_CARD_TMPL = """
<div class="card" id="f{seq}">
  <img src="{img_path}" loading="lazy" alt="frame {seq}">
  <div class="card-body">
    <div class="card-time">
      Frame #{seq} &nbsp;·&nbsp; <strong>{wall}</strong> &nbsp;·&nbsp; tag: {tag}
    </div>
    {perc_block}
    {speaks_block}
    {nav_block}
  </div>
</div>
"""


def _perc_block(perc_ev):
    if not perc_ev:
        return ""
    w = perc_ev.get("walkable", 0)
    safe = perc_ev.get("is_safe", True)
    hazard = perc_ev.get("hazard") or "—"
    badge = '<span class="safe-badge">SAFE</span>' if safe else '<span class="unsafe-badge">UNSAFE</span>'
    return (f'<div class="card-perc">{badge} '
            f'walkable: <strong>{w:.1%}</strong> &nbsp; hazard: <strong>{_esc(hazard)}</strong></div>')


def _speaks_block(speaks):
    if not speaks:
        return '<div class="empty">No speech near this frame.</div>'
    html = '<div class="section-label">Spoken feedback</div>'
    for s in speaks:
        method = s.get("method", "obstacle")
        cls = {"obstacle": "obstacle", "nav": "nav", "announce": "announce",
               "safe": "safe", "progress": "nav", "emergency": "obstacle"}.get(method, "obstacle")
        spoken_flag = "" if s.get("spoken") else " <em style='color:#666'>(suppressed)</em>"
        html += (f'<div class="speak-row {cls}">'
                 f'<span class="speak-time">{_wall(s)}</span>'
                 f'{_esc(s.get("text", ""))} [{method}]{spoken_flag}</div>')
    return html


def _nav_block(navs):
    if not navs:
        return ""
    html = '<div class="section-label">Navigation events</div>'
    for n in navs:
        dist = (f", {n['distance_to_next_m']:.0f}m" if n.get("distance_to_next_m") is not None else "")
        step = (f" — {_esc(n['step_text'])}" if n.get("step_text") else "")
        html += (f'<div class="nav-row">'
                 f'<span class="speak-time">{_wall(n)}</span>'
                 f'{_esc(n.get("status", ""))}{dist}{step}</div>')
    return html


def _tl_chip(ev):
    method = ev.get("method", "obstacle")
    cls = {"obstacle": "obstacle", "nav": "nav", "announce": "announce",
           "safe": "safe", "progress": "nav", "emergency": "emergency"}.get(method, "obstacle")
    t = ev.get("t", 0)
    label = _esc(ev.get("text", "")[:40])
    tip = _esc(ev.get("text", ""))
    return (f'<span class="tl-chip tl-{cls}" '
            f'title="{_wall(ev)} — {tip}">'
            f't+{t:.0f}s: {label}</span>')


def build_html(session_dir: Path, events: list) -> str:
    by_type: dict = {}
    for ev in events:
        by_type.setdefault(ev.get("type"), []).append(ev)

    speaks = by_type.get("speak", [])
    navs = by_type.get("nav", [])
    frames = by_type.get("frame", [])
    perceptions = by_type.get("perception", [])

    spoken = [s for s in speaks if s.get("spoken")]
    session_start = by_type.get("system", [{}])[0].get("wall", "")

    # Build per-t lookup for closest perception event
    def _closest_perc(t_target):
        best = None
        for p in perceptions:
            if best is None or abs(p.get("t", 0) - t_target) < abs(best.get("t", 0) - t_target):
                best = p
        return best if best and abs(best.get("t", 0) - t_target) <= 2.0 else None

    # ── Timeline chips ──────────────────────────────────────────
    tl_html = "".join(_tl_chip(s) for s in spoken)

    # ── Frame cards ─────────────────────────────────────────────
    cards_html = ""
    for fr in frames:
        t_fr = fr.get("t", 0)
        img_rel = fr.get("file", "")
        tag = fr.get("tag", "")
        seq_str = img_rel.split("_")[1] if "_" in img_rel else "?"

        nearby_speaks = _nearby(speaks, t_fr, before=2.0, after=0.5)
        nearby_navs   = _nearby(navs,   t_fr, before=2.0, after=2.0)
        perc_ev       = _closest_perc(t_fr)

        cards_html += _CARD_TMPL.format(
            seq=seq_str,
            img_path=_esc(img_rel),
            wall=_wall(fr),
            tag=_esc(tag),
            perc_block=_perc_block(perc_ev),
            speaks_block=_speaks_block(nearby_speaks),
            nav_block=_nav_block(nearby_navs),
        )

    n_frames = len(frames)
    n_spoken = len(spoken)
    ts = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    duration = (max(ts) - min(ts)) if ts else 0.0

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALAS Field Test Viewer — {session_dir.name}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>ALAS Field Test Viewer</h1>
<div class="meta">
  Session: <strong>{_esc(str(session_dir.name))}</strong> &nbsp;·&nbsp;
  Started: <strong>{_esc(session_start[:19].replace("T"," "))}</strong> &nbsp;·&nbsp;
  Duration: <strong>{duration:.0f}s ({duration/60:.1f}min)</strong> &nbsp;·&nbsp;
  Frames: <strong>{n_frames}</strong> &nbsp;·&nbsp;
  Utterances: <strong>{n_spoken}</strong>
</div>

<div class="tl-section">
  <h2>Spoken timeline — click to jump to frame</h2>
  <div class="tl">{tl_html}</div>
</div>

<div class="frames">
{cards_html}
</div>

<script>
document.querySelectorAll('.tl-chip').forEach(chip => {{
  chip.addEventListener('click', () => {{
    // find the nearest frame card by t value from title
    const title = chip.getAttribute('title') || '';
    // scroll to first card (rough — chips link to nearest frame)
    const cards = document.querySelectorAll('.card');
    if (cards.length) cards[0].scrollIntoView({{behavior:'smooth'}});
  }});
}});
</script>
</body>
</html>
"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML viewer for an ALAS field-test session")
    parser.add_argument("session_dir", help="Path to outputs/field_tests/<timestamp>/")
    parser.add_argument("--out", default=None, help="Output HTML path (default: <session_dir>/viewer.html)")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        print(f"ERROR: no events.jsonl in {session_dir}", file=sys.stderr)
        return 1

    events = _load_events(events_path)
    html = build_html(session_dir, events)

    out_path = Path(args.out) if args.out else session_dir / "viewer.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[viewer] {len(events)} events → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
