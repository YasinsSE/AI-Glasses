"""Compose a presentation/demo MP4 from a recorded ALAS session (jury video).

Run on the analysis machine (Mac), AFTER a walk recorded with ``--demo``
(dense ~0.5 s frames; plain ``--record`` works too but looks choppier):

    python3 eval/field_test/make_demo_video.py outputs/field_tests/<oturum>/
    python3 eval/field_test/make_demo_video.py outputs/field_tests/<oturum>/ --speed 2

Output: ``<oturum>/demo.mp4`` — 1280x720:
    - left  (960x720): the annotated camera frame (mask overlay + legend,
      exactly what the perception system saw)
    - right (320 px):  live status panel — mode, safety, walkable %, GPS,
      distance to target, perception FPS, SoC temperature
    - bottom strip:    the sentence ALAS is speaking at that moment, as a
      subtitle (urgent/emergency lines highlighted red)

Turkish text is drawn with Pillow (cv2.putText cannot render it), so this
tool needs ``pip3 install pillow`` — on the analysis machine only, never on
the Jetson. Timing maps saved-frame timestamps to real duration; idle gaps
longer than ``--max-hold`` seconds are compressed, ``--speed`` accelerates
playback uniformly.
"""

import argparse
import json
import sys
from pathlib import Path

# ── Event loading (same defensive parser as the viewer) ─────────────────────

def load_events(events_path: Path):
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


# ── Pure scheduling / lookup logic (unit-tested, no cv2/PIL needed) ─────────

def build_schedule(frame_events, fps=10, speed=1.0, max_hold_s=2.0):
    """Map saved frames to video frames.

    Returns a list of ``(file, real_t)`` — one entry per OUTPUT video frame.
    Each saved frame is held until the next one; a real gap longer than
    ``max_hold_s`` plays as only ``max_hold_s`` of video, but the per-frame
    ``real_t`` values still sweep the FULL real gap, so subtitles and status
    that happened inside a compressed pause are not lost — just sped up.
    """
    frames = [f for f in frame_events if f.get("file")]
    if not frames:
        return []
    out = []
    for cur, nxt in zip(frames, frames[1:] + [None]):
        t0 = float(cur.get("t", 0.0))
        real_gap = (float(nxt.get("t", t0)) - t0) if nxt else 1.0
        real_gap = max(real_gap, 1.0 / fps)
        vid_dur = min(real_gap, max_hold_s) / max(speed, 0.1)
        n = max(1, int(round(vid_dur * fps)))
        for j in range(n):
            out.append((cur["file"], t0 + real_gap * j / n))
    return out


def subtitle_at(real_t, speaks, show_sec=3.5):
    """The utterance to display at ``real_t``: latest spoken line whose
    window [t, t+show_sec] covers the moment. Suppressed lines never show.
    Returns (text, urgent) or None."""
    best = None
    for s in speaks:
        if not s.get("spoken"):
            continue
        t = s.get("t", -1e9)
        if t <= real_t <= t + show_sec:
            if best is None or t >= best.get("t", -1e9):
                best = s
    if best is None:
        return None
    urgent = best.get("method") in ("emergency",) or "Durun" in (best.get("text") or "")
    return best.get("text", ""), urgent


def latest_before(real_t, rows, key=None):
    """Latest event at or before ``real_t`` (rows must be t-ascending-ish)."""
    best = None
    for r in rows:
        t = r.get("t", -1e9)
        if t <= real_t and (best is None or t >= best.get("t", -1e9)):
            best = r
    return best if (key is None or best is None) else best.get(key)


def status_at(real_t, percs, telem, navs, gpses):
    """Panel fields for the given moment (latest known value of each)."""
    p = latest_before(real_t, percs)
    tl = latest_before(real_t, telem)
    nv = latest_before(real_t, navs)
    gp = latest_before(real_t, gpses)
    safety = {0: "GÜVENLİ", 1: "DİKKAT", 2: "TEHLİKE"}.get(
        p.get("safety_level") if p else None, "—")
    temps = (tl or {}).get("temps_c") or {}
    # PMIC-Die is a fake constant zone (always 50.0 on the Nano) — exclude.
    real_temps = [v for k, v in temps.items() if k != "PMIC-Die"]
    return {
        "safety": safety,
        "walkable": (p or {}).get("walkable"),
        "fps": (1000.0 / p["total_ms"]) if p and p.get("total_ms") else None,
        "temp": max(real_temps) if real_temps else None,
        "nav_status": (nv or {}).get("status"),
        "nav_dist": (nv or {}).get("distance_to_next_m"),
        "nav_step": (nv or {}).get("step_text"),
        "gps": (gp.get("lat"), gp.get("lon")) if gp else None,
        "sats": (gp or {}).get("sats"),
    }


# ── Rendering (cv2 + PIL imported lazily so the logic stays testable) ───────

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",        # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Linux
]


def _load_fonts():
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            return (ImageFont.truetype(path, 22), ImageFont.truetype(path, 17),
                    ImageFont.truetype(path, 26))
        except OSError:
            continue
    f = ImageFont.load_default()
    return f, f, f


_SAFETY_COLOR = {"GÜVENLİ": (47, 168, 108), "DİKKAT": (217, 154, 38),
                 "TEHLİKE": (224, 82, 82), "—": (90, 100, 120)}


def _render_canvas(frame_bgr, st, sub, t_label, fonts):
    """Compose one 1280x720 video frame (returns BGR ndarray)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:, :, :] = (22, 17, 14)  # BGR of #0e1116
    canvas[:, :960] = cv2.resize(frame_bgr, (960, 720), interpolation=cv2.INTER_LINEAR)

    # Right panel + subtitle via PIL (Turkish glyphs).
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    f_big, f_small, f_sub = fonts
    x = 980

    d.text((x, 26), "ALAS", font=f_big, fill=(255, 255, 255))
    d.text((x + 72, 26), "Saha Demosu", font=f_big, fill=(77, 163, 255))
    d.text((x, 58), t_label, font=f_small, fill=(133, 147, 168))

    # Safety badge.
    color = _SAFETY_COLOR.get(st["safety"], (90, 100, 120))
    d.rounded_rectangle((x, 92, x + 280, 130), 8, fill=color + (255,))
    d.text((x + 14, 99), st["safety"], font=f_big, fill=(255, 255, 255))

    rows = []
    if st["walkable"] is not None:
        rows.append(("Yürünebilir alan", f"%{st['walkable'] * 100:.0f}"))
    if st["fps"] is not None:
        rows.append(("Algı hızı", f"{st['fps']:.1f} FPS"))
    if st["temp"] is not None:
        rows.append(("SoC sıcaklık", f"{st['temp']:.0f} °C"))
    if st["sats"] is not None:
        rows.append(("GPS uydu", str(st["sats"])))
    elif st["gps"] is not None:
        rows.append(("GPS", "fix var"))
    if st["nav_dist"] is not None:
        rows.append(("Sonraki adım", f"{st['nav_dist']:.0f} m"))
    if st["nav_step"]:
        rows.append(("Talimat", str(st["nav_step"])[:24]))
    y = 152
    for label, val in rows:
        d.text((x, y), label, font=f_small, fill=(133, 147, 168))
        d.text((x, y + 20), val, font=f_big, fill=(219, 226, 238))
        y += 58

    # Subtitle strip. Accent bar instead of a speaker emoji — the system
    # fonts used here have no emoji glyphs (renders as an empty box).
    if sub is not None:
        text, urgent = sub
        bg = (160, 40, 40, 215) if urgent else (10, 12, 18, 200)
        accent = (255, 255, 255, 255) if urgent else (77, 163, 255, 255)
        d.rectangle((0, 645, 1280, 720), fill=bg)
        d.rectangle((24, 663, 30, 702), fill=accent)
        d.text((44, 668), text, font=f_sub, fill=(255, 255, 255))

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def make_video(session_dir: Path, out_path: Path, fps=10, speed=1.0,
               max_hold_s=2.0, show_sec=3.5) -> int:
    try:
        import cv2  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: {exc}. Bu araç analiz makinesinde çalışır: "
              "pip3 install pillow opencv-python", file=sys.stderr)
        return 1
    import cv2

    events = load_events(session_dir / "events.jsonl")
    by_type: dict = {}
    for ev in events:
        by_type.setdefault(ev.get("type"), []).append(ev)
    frames = by_type.get("frame", [])
    if not frames:
        print("ERROR: oturumda kayıtlı kare yok (frames/). --demo veya --record "
              "ile kaydedilmiş bir oturum verin.", file=sys.stderr)
        return 1

    schedule = build_schedule(frames, fps=fps, speed=speed, max_hold_s=max_hold_s)
    speaks = by_type.get("speak", [])
    percs = by_type.get("perception", [])
    telem = by_type.get("telemetry", [])
    navs = by_type.get("nav", [])
    gpses = by_type.get("gps", [])
    fonts = _load_fonts()
    t0 = min((e["t"] for e in events if isinstance(e.get("t"), (int, float))), default=0.0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (1280, 720))
    if not writer.isOpened():
        print("ERROR: VideoWriter açılamadı (mp4v codec).", file=sys.stderr)
        return 1

    cache_file, cache_img = None, None
    written = 0
    for file_rel, real_t in schedule:
        if file_rel != cache_file:
            img = cv2.imread(str(session_dir / file_rel))
            if img is None:
                continue  # missing frame on disk — skip its slots
            cache_file, cache_img = file_rel, img
        if cache_img is None:
            continue
        rel = real_t - t0
        t_label = f"t+{int(rel // 60)}:{int(rel % 60):02d}"
        st = status_at(real_t, percs, telem, navs, gpses)
        sub = subtitle_at(real_t, speaks, show_sec=show_sec)
        writer.write(_render_canvas(cache_img, st, sub, t_label, fonts))
        written += 1
        if written % 200 == 0:
            print(f"  {written}/{len(schedule)} kare...")
    writer.release()
    print(f"[demo] {written} video karesi ({written / fps:.0f} sn) → {out_path}")
    return 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ALAS oturum kaydından sunum videosu üret")
    ap.add_argument("session_dir", help="outputs/field_tests/<oturum>/ yolu")
    ap.add_argument("--out", default=None, help="Çıktı MP4 yolu (varsayılan: <oturum>/demo.mp4)")
    ap.add_argument("--fps", type=int, default=10, help="Video FPS (varsayılan 10)")
    ap.add_argument("--speed", type=float, default=1.0, help="Hızlandırma çarpanı (örn. 2)")
    ap.add_argument("--max-hold", type=float, default=2.0,
                    help="Kareler arası boşluğun videoda oynayacağı azami süre (sn)")
    ap.add_argument("--subtitle-sec", type=float, default=3.5,
                    help="Her cümlenin altyazıda kalma süresi (sn)")
    args = ap.parse_args(argv)

    session_dir = Path(args.session_dir)
    if not (session_dir / "events.jsonl").exists():
        print(f"ERROR: events.jsonl yok: {session_dir}", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else session_dir / "demo.mp4"
    return make_video(session_dir, out_path, fps=args.fps, speed=args.speed,
                      max_hold_s=args.max_hold, show_sec=args.subtitle_sec)


if __name__ == "__main__":
    sys.exit(main())
