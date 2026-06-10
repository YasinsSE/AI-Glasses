"""Interactive HTML viewer for an ALAS field-test session (Viewer 2.0).

Generates a single, self-contained ``viewer.html`` inside the session
directory — double-click to open, no server, shareable as a file. The event
data is embedded as JSON; only the frame JPEGs are loaded relatively from
``frames/``, so the HTML must stay next to that directory.

Tabs:
    Özet           — session stats: duration, FPS, p95 inference, peak SoC
                     temp, walked distance, utterance/suppression breakdown.
    Zaman Çizelgesi — large frame display + scrubber (speak/nav ticks),
                     ←/→ steps frames, Space plays at real-time pacing,
                     side panel shows events around the current moment.
    Harita         — Leaflet (CDN) with the GPS track + speak/nav markers.
                     Hidden when the session has no fixes; degrades to a
                     plain-text note when offline.
    Grafikler      — dependency-free inline SVG: walkable %, inference ms,
                     SoC temp + GPU load over a shared time axis.
    Konuşmalar     — filterable table of every utterance (incl. suppressed),
                     row click jumps to that moment on the timeline.

How to run (from the repository root):
    python3 eval/field_test/viewer.py outputs/field_tests/<timestamp>/
    python3 eval/field_test/viewer.py outputs/field_tests/<timestamp>/ --out /tmp/viewer.html

``build_html(session_dir, events)`` keeps its original signature — it is
called by SessionRecorder.finalize() and eval/field_test/report.py.
"""

import argparse
import json
import math
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


def _esc(text) -> str:
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _downsample(rows, max_points=2000):
    """Stride-downsample chart rows so a multi-hour log stays light."""
    if len(rows) <= max_points:
        return rows
    stride = len(rows) // max_points + 1
    return rows[::stride]


# ── Data extraction (Python side does the math; JS only renders) ────────────

def _prepare(session_dir: Path, events: list) -> dict:
    by_type: dict = {}
    for ev in events:
        by_type.setdefault(ev.get("type"), []).append(ev)

    ts = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    t0 = min(ts) if ts else 0.0
    duration = (max(ts) - t0) if ts else 0.0

    def rel(ev):
        t = ev.get("t")
        return round(t - t0, 2) if isinstance(t, (int, float)) else 0.0

    frames = [{"t": rel(e), "file": e.get("file", ""), "tag": e.get("tag", ""),
               "wall": (e.get("wall") or "")[11:19]}
              for e in by_type.get("frame", []) if e.get("file")]

    speaks = [{"t": rel(e), "method": e.get("method", "?"),
               "text": e.get("text", ""), "spoken": bool(e.get("spoken")),
               "reason": e.get("reason"), "wall": (e.get("wall") or "")[11:19]}
              for e in by_type.get("speak", [])]

    navs = [{"t": rel(e), "status": str(e.get("status", "")),
             "dist": e.get("distance_to_next_m"), "step": e.get("step_text")}
            for e in by_type.get("nav", [])]

    percs = [{"t": rel(e), "walkable": e.get("walkable"),
              "safety": e.get("safety_level"), "hazard": e.get("hazard"),
              "inf": e.get("inf_ms"), "total": e.get("total_ms"),
              "chosen": e.get("chosen")}
             for e in by_type.get("perception", [])]

    telem = []
    for e in by_type.get("telemetry", []):
        temps = e.get("temps_c") or {}
        telem.append({"t": rel(e),
                      "temp": max(temps.values()) if temps else None,
                      "gpu": e.get("gpu_pct"),
                      "ram": (e.get("ram") or {}).get("used_pct")})

    gps = [{"t": rel(e), "lat": e["lat"], "lon": e["lon"]}
           for e in by_type.get("gps", [])
           if isinstance(e.get("lat"), (int, float)) and isinstance(e.get("lon"), (int, float))]

    sys_marks = [{"t": rel(e), "label": e.get("error") or e.get("event") or "system"}
                 for e in by_type.get("system", [])
                 if e.get("error")]  # only anomalies become chart markers

    # Walked distance: consecutive fixes, jumps > 50 m discarded as GPS noise.
    walked = 0.0
    for a, b in zip(gps, gps[1:]):
        d = _haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
        if d <= 50.0:
            walked += d

    spoken = [s for s in speaks if s["spoken"]]
    suppressed = [s for s in speaks if not s["spoken"]]
    reasons: dict = {}
    for s in suppressed:
        key = s["reason"] or "unspecified"
        reasons[key] = reasons.get(key, 0) + 1
    methods: dict = {}
    for s in spoken:
        methods[s["method"]] = methods.get(s["method"], 0) + 1

    totals = [p["total"] for p in percs if p.get("total")]
    infs = [p["inf"] for p in percs if p.get("inf")]
    fps = [1000.0 / t for t in totals if t]
    temps_all = [t["temp"] for t in telem if t.get("temp") is not None]

    start_wall = ""
    for e in by_type.get("system", []):
        if e.get("event") == "session_start":
            start_wall = (e.get("wall") or "")[:19].replace("T", " ")
            break

    summary = {
        "session": session_dir.name,
        "start": start_wall,
        "duration_s": round(duration, 1),
        "n_events": len(events),
        "n_frames_processed": len(percs),
        "n_frames_saved": len(frames),
        "fps_avg": round(_avg(fps), 2) if fps else None,
        "fps_min": round(min(fps), 2) if fps else None,
        "inf_avg": round(_avg(infs)) if infs else None,
        "inf_p95": round(_pct(infs, 95)) if infs else None,
        "temp_peak": round(max(temps_all), 1) if temps_all else None,
        "walked_m": round(walked) if gps else None,
        "n_spoken": len(spoken),
        "n_suppressed": len(suppressed),
        "methods": methods,
        "reasons": reasons,
        "safety_counts": {
            "safe": sum(1 for p in percs if p.get("safety") == 0),
            "caution": sum(1 for p in percs if p.get("safety") == 1),
            "unsafe": sum(1 for p in percs if p.get("safety") == 2),
        },
    }

    return {
        "summary": summary,
        "frames": frames,
        "speaks": speaks,
        "navs": navs,
        "percs": _downsample(percs),
        "telem": _downsample(telem),
        "gps": _downsample(gps, 4000),
        "sysMarks": sys_marks,
        "duration": round(duration, 2),
    }


# ── HTML template ────────────────────────────────────────────────────────────
# Token-substituted (no str.format) so the embedded JS/CSS braces stay as-is.

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALAS — __SESSION__</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#0e1116; --panel:#161b24; --panel2:#1d2430; --line:#2a3342;
  --txt:#dbe2ee; --dim:#8593a8; --accent:#4da3ff;
  --safe:#2fa86c; --caution:#d99a26; --unsafe:#e05252; --nav:#4da3ff;
}
body { font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
       background: var(--bg); color: var(--txt); }
header { display:flex; align-items:baseline; gap:14px; padding:14px 22px;
         background:linear-gradient(90deg,#121826,#0e1116); border-bottom:1px solid var(--line); }
header h1 { font-size:1.15rem; letter-spacing:.04em; }
header h1 span { color:var(--accent); }
header .sub { color:var(--dim); font-size:.8rem; }
nav.tabs { display:flex; gap:4px; padding:8px 22px 0; background:var(--bg);
           border-bottom:1px solid var(--line); }
nav.tabs button { background:none; border:1px solid transparent; border-bottom:none;
  color:var(--dim); padding:8px 16px; font-size:.86rem; cursor:pointer;
  border-radius:8px 8px 0 0; }
nav.tabs button.on { background:var(--panel); color:var(--txt);
  border-color:var(--line); }
main { padding:18px 22px; }
section.tab { display:none; }
section.tab.on { display:block; }

/* Özet */
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.kpi .v { font-size:1.5rem; font-weight:600; margin-top:4px; }
.kpi .l { color:var(--dim); font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; }
.kpi .s { color:var(--dim); font-size:.72rem; margin-top:4px; }
.breakdown { margin-top:18px; display:flex; gap:12px; flex-wrap:wrap; }
.breakdown .box { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 16px; min-width:220px; }
.breakdown h3 { font-size:.8rem; color:var(--dim); margin-bottom:8px; text-transform:uppercase; letter-spacing:.06em; }
.breakdown li { list-style:none; font-size:.84rem; padding:2px 0; display:flex; justify-content:space-between; gap:18px; }
.breakdown li b { color:var(--accent); }

/* Zaman çizelgesi */
.tl-wrap { display:grid; grid-template-columns: minmax(0,1fr) 330px; gap:16px; }
.viewer { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.viewer img { width:100%; display:block; min-height:200px; background:#000; }
.viewer .bar { display:flex; align-items:center; gap:10px; padding:8px 12px; font-size:.8rem; color:var(--dim); }
.viewer .bar .t { color:var(--txt); font-variant-numeric:tabular-nums; }
.viewer .bar button { background:var(--panel2); color:var(--txt); border:1px solid var(--line);
  border-radius:6px; padding:3px 12px; cursor:pointer; font-size:.8rem; }
.viewer .bar select { background:var(--panel2); color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:2px 6px; }
.scrub { position:relative; height:46px; background:var(--panel2); cursor:pointer; }
.scrub .tick { position:absolute; bottom:4px; width:2px; height:10px; background:#48536a; }
.scrub .tick.cur { background:#fff; height:16px; }
.scrub .ev { position:absolute; top:5px; width:4px; height:9px; border-radius:2px; }
.scrub .ev.obstacle,.scrub .ev.emergency { background:var(--unsafe); }
.scrub .ev.nav,.scrub .ev.progress { background:var(--nav); }
.scrub .ev.announce,.scrub .ev.prompt { background:var(--safe); }
.scrub .ev.earcon { background:#9a7bd0; }
.scrub .playhead { position:absolute; top:0; bottom:0; width:2px; background:var(--accent); }
.sidefeed { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px; max-height:560px; overflow-y:auto; }
.sidefeed h3 { font-size:.78rem; color:var(--dim); text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px; }
.evrow { font-size:.82rem; padding:5px 8px; border-left:3px solid #555; margin-bottom:5px;
  background:var(--panel2); border-radius:0 6px 6px 0; }
.evrow .et { color:var(--dim); font-size:.7rem; margin-right:6px; font-variant-numeric:tabular-nums; }
.evrow.obstacle,.evrow.emergency { border-color:var(--unsafe); }
.evrow.nav,.evrow.progress { border-color:var(--nav); }
.evrow.announce,.evrow.prompt { border-color:var(--safe); }
.evrow.sup { opacity:.45; }
.badge { display:inline-block; border-radius:4px; padding:1px 7px; font-size:.7rem; font-weight:600; color:#fff; }
.badge.safe { background:var(--safe); } .badge.caution { background:var(--caution); }
.badge.unsafe { background:var(--unsafe); }
.empty { color:var(--dim); font-style:italic; font-size:.82rem; }

/* Harita */
#map { height:540px; border-radius:10px; border:1px solid var(--line); }
.map-note { color:var(--dim); font-size:.85rem; margin-top:8px; }

/* Grafikler */
.chart { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; margin-bottom:14px; }
.chart h3 { font-size:.82rem; color:var(--dim); margin-bottom:6px; }
.chart svg { width:100%; height:130px; display:block; }
.chart .tip { font-size:.74rem; color:var(--accent); min-height:1em; }

/* Konuşmalar */
.filters { display:flex; gap:12px; margin-bottom:12px; align-items:center; font-size:.84rem; color:var(--dim); }
.filters select { background:var(--panel2); color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:3px 8px; }
table.speaks { width:100%; border-collapse:collapse; font-size:.84rem; }
table.speaks th { text-align:left; color:var(--dim); font-size:.72rem; text-transform:uppercase;
  letter-spacing:.06em; padding:6px 10px; border-bottom:1px solid var(--line); }
table.speaks td { padding:6px 10px; border-bottom:1px solid #1c2330; }
table.speaks tr { cursor:pointer; }
table.speaks tr:hover td { background:var(--panel2); }
table.speaks tr.sup td { opacity:.45; }
.mchip { display:inline-block; border-radius:4px; padding:1px 8px; font-size:.7rem; color:#fff; background:#555; }
.mchip.obstacle,.mchip.emergency { background:var(--unsafe); }
.mchip.nav,.mchip.progress { background:var(--nav); }
.mchip.announce,.mchip.prompt { background:var(--safe); }
kbd { background:var(--panel2); border:1px solid var(--line); border-radius:4px;
  padding:0 6px; font-size:.74rem; }
</style>
</head>
<body>
<header>
  <h1>ALAS <span>Saha Testi</span></h1>
  <div class="sub">__SESSION__ · __START__ · __DURATION__</div>
</header>
<nav class="tabs" id="tabs">
  <button data-tab="ozet" class="on">Özet</button>
  <button data-tab="zaman">Zaman Çizelgesi</button>
  <button data-tab="harita" id="tab-harita">Harita</button>
  <button data-tab="grafik">Grafikler</button>
  <button data-tab="konusma">Konuşmalar</button>
</nav>
<main>
  <section class="tab on" id="ozet"><div class="cards" id="kpis"></div><div class="breakdown" id="breakdown"></div></section>
  <section class="tab" id="zaman">
    <div class="tl-wrap">
      <div>
        <div class="viewer">
          <img id="frameImg" alt="kare">
          <div class="scrub" id="scrub"></div>
          <div class="bar">
            <button id="btnPlay">▶</button>
            <select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select>
            <span class="t" id="timeLabel">–</span>
            <span style="margin-left:auto">gezinme: <kbd>←</kbd><kbd>→</kbd> kare, <kbd>boşluk</kbd> oynat</span>
          </div>
        </div>
      </div>
      <div class="sidefeed"><h3>Bu Anda Olanlar (±4 sn)</h3><div id="feed"></div></div>
    </div>
  </section>
  <section class="tab" id="harita"><div id="map"></div><div class="map-note" id="mapNote"></div></section>
  <section class="tab" id="grafik" ></section>
  <section class="tab" id="konusma">
    <div class="filters">
      Tür: <select id="fMethod"><option value="">hepsi</option></select>
      <label><input type="checkbox" id="fSup" checked> bastırılanları göster</label>
    </div>
    <table class="speaks"><thead><tr><th>t</th><th>saat</th><th>tür</th><th>metin</th><th>durum</th></tr></thead>
    <tbody id="speakRows"></tbody></table>
  </section>
</main>

<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const S = D.summary;
const fmt = (t) => { const m = Math.floor(t/60), s = Math.floor(t%60); return m+':'+String(s).padStart(2,'0'); };
const esc = (x) => String(x ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

/* ── Tabs ── */
const tabs = document.getElementById('tabs');
tabs.addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  tabs.querySelectorAll('button').forEach(x => x.classList.toggle('on', x===b));
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('on', s.id===b.dataset.tab));
  if (b.dataset.tab === 'harita') initMap();
});

/* ── Özet ── */
function kpi(l, v, s) { return `<div class="kpi"><div class="l">${l}</div><div class="v">${v ?? '–'}</div>${s?`<div class="s">${s}</div>`:''}</div>`; }
document.getElementById('kpis').innerHTML =
  kpi('Süre', fmt(S.duration_s), S.duration_s.toFixed(0)+' sn') +
  kpi('İşlenen Kare', S.n_frames_processed, S.n_frames_saved + ' kayıtlı görüntü') +
  kpi('Algı FPS', S.fps_avg ?? '–', S.fps_min!=null ? 'min '+S.fps_min : '') +
  kpi('Inference', S.inf_avg!=null ? S.inf_avg+' ms' : '–', S.inf_p95!=null ? 'p95 '+S.inf_p95+' ms' : '') +
  kpi('Tepe Sıcaklık', S.temp_peak!=null ? S.temp_peak+' °C' : '–', S.temp_peak>=70?'⚠ termal eşik üstü':'') +
  kpi('Yürünen Mesafe', S.walked_m!=null ? S.walked_m+' m' : '–', S.walked_m==null?'GPS verisi yok':'') +
  kpi('Konuşma', S.n_spoken, (S.duration_s>0? (S.n_spoken/(S.duration_s/60)).toFixed(1)+' / dk':'')) +
  kpi('Bastırılan', S.n_suppressed, (S.reasons.muted_unsafe? '⚠ muted_unsafe: '+S.reasons.muted_unsafe : ''));
function listBox(title, obj) {
  const items = Object.entries(obj).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<li>${esc(k)} <b>${v}</b></li>`).join('');
  return `<div class="box"><h3>${title}</h3><ul>${items || '<li class="empty">yok</li>'}</ul></div>`;
}
document.getElementById('breakdown').innerHTML =
  listBox('Konuşma türleri', S.methods) + listBox('Bastırma sebepleri', S.reasons) +
  listBox('Güvenlik dağılımı (kare)', S.safety_counts);

/* ── Zaman çizelgesi ── */
const frames = D.frames; let idx = 0, playing = false, playTimer = null;
const img = document.getElementById('frameImg'), scrub = document.getElementById('scrub');
const timeLabel = document.getElementById('timeLabel'), feed = document.getElementById('feed');
function buildScrub() {
  if (!frames.length) return;
  let html = '';
  frames.forEach((f,i) => { html += `<div class="tick" data-i="${i}" style="left:${100*f.t/D.duration}%"></div>`; });
  D.speaks.filter(s=>s.spoken).forEach(s => { html += `<div class="ev ${esc(s.method)}" title="${esc(s.text)}" style="left:${100*s.t/D.duration}%"></div>`; });
  html += '<div class="playhead" id="playhead"></div>';
  scrub.innerHTML = html;
}
function seek(i, stopPlay=true) {
  if (!frames.length) return;
  idx = Math.max(0, Math.min(frames.length-1, i));
  if (stopPlay) stop();
  render();
}
function nearestFrame(t) {
  let best = 0;
  frames.forEach((f,i) => { if (Math.abs(f.t-t) < Math.abs(frames[best].t-t)) best = i; });
  return best;
}
function render() {
  if (!frames.length) { img.alt = 'Bu oturumda kayıtlı kare yok'; return; }
  const f = frames[idx];
  img.src = f.file;
  timeLabel.textContent = `t+${fmt(f.t)}  ·  ${f.wall || ''}  ·  kare ${idx+1}/${frames.length} [${f.tag}]`;
  document.querySelectorAll('.scrub .tick').forEach((el,i)=>el.classList.toggle('cur', i===idx));
  const ph = document.getElementById('playhead');
  if (ph) ph.style.left = (100*f.t/D.duration) + '%';
  /* side feed */
  const t = f.t, W = 4.0;
  const sp = D.speaks.filter(s => Math.abs(s.t-t) <= W);
  const nv = D.navs.filter(n => Math.abs(n.t-t) <= W);
  let pc = null; D.percs.forEach(p => { if (pc===null || Math.abs(p.t-t) < Math.abs(pc.t-t)) pc = p; });
  let html = '';
  if (pc && Math.abs(pc.t-t) <= 2.5) {
    const b = pc.safety===0?'safe':pc.safety===1?'caution':'unsafe';
    html += `<div class="evrow"><span class="badge ${b}">${b.toUpperCase()}</span>
      yürünebilir %${((pc.walkable||0)*100).toFixed(0)}${pc.hazard?' · '+esc(pc.hazard):''}</div>`;
  }
  sp.forEach(s => { html += `<div class="evrow ${esc(s.method)}${s.spoken?'':' sup'}">
    <span class="et">t+${fmt(s.t)}</span>${esc(s.text)}${s.spoken?'':' <i>(bastırıldı: '+esc(s.reason||'?')+')</i>'}</div>`; });
  nv.forEach(n => { html += `<div class="evrow nav"><span class="et">t+${fmt(n.t)}</span>
    ${esc(n.status)}${n.dist!=null?' · '+Math.round(n.dist)+' m':''}${n.step?' · '+esc(n.step):''}</div>`; });
  feed.innerHTML = html || '<div class="empty">Bu anın yakınında olay yok.</div>';
}
function play() {
  if (!frames.length || idx >= frames.length-1) return;
  playing = true; document.getElementById('btnPlay').textContent = '⏸';
  const step = () => {
    if (!playing) return;
    if (idx >= frames.length-1) { stop(); return; }
    const speed = parseFloat(document.getElementById('speed').value);
    const dt = Math.min(2.0, Math.max(0.15, frames[idx+1].t - frames[idx].t)) / speed;
    playTimer = setTimeout(() => { seek(idx+1, false); step(); }, dt*1000);
  };
  step();
}
function stop() { playing = false; clearTimeout(playTimer); document.getElementById('btnPlay').textContent = '▶'; }
document.getElementById('btnPlay').addEventListener('click', () => playing ? stop() : play());
scrub.addEventListener('click', (e) => {
  const r = scrub.getBoundingClientRect();
  seek(nearestFrame(D.duration * (e.clientX - r.left) / r.width));
});
document.addEventListener('keydown', (e) => {
  if (!document.getElementById('zaman').classList.contains('on')) return;
  if (e.key === 'ArrowRight') { seek(idx+1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { seek(idx-1); e.preventDefault(); }
  else if (e.key === ' ') { playing ? stop() : play(); e.preventDefault(); }
});
buildScrub(); render();

/* ── Harita ── */
let mapInited = false;
function initMap() {
  if (mapInited) return; mapInited = true;
  const note = document.getElementById('mapNote');
  if (!D.gps.length) { note.textContent = 'Bu oturumda GPS fix yok.'; return; }
  const css = document.createElement('link'); css.rel = 'stylesheet';
  css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
  document.head.appendChild(css);
  const js = document.createElement('script');
  js.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  js.onload = () => {
    const pts = D.gps.map(g => [g.lat, g.lon]);
    const map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution: '© OpenStreetMap' }).addTo(map);
    const line = L.polyline(pts, { color: '#4da3ff', weight: 4 }).addTo(map);
    map.fitBounds(line.getBounds(), { padding: [30,30] });
    L.circleMarker(pts[0], {radius:7, color:'#2fa86c'}).addTo(map).bindPopup('Başlangıç');
    L.circleMarker(pts[pts.length-1], {radius:7, color:'#e05252'}).addTo(map).bindPopup('Bitiş');
    const nearGps = (t) => { let b = D.gps[0];
      D.gps.forEach(g => { if (Math.abs(g.t-t) < Math.abs(b.t-t)) b = g; }); return b; };
    D.speaks.filter(s => s.spoken && (s.method==='obstacle'||s.method==='emergency')).forEach(s => {
      const g = nearGps(s.t);
      if (Math.abs(g.t - s.t) <= 5)
        L.circleMarker([g.lat,g.lon], {radius:4, color:'#e05252', fillOpacity:.7})
          .addTo(map).bindPopup(`t+${fmt(s.t)}: ${esc(s.text)}`);
    });
    D.navs.filter(n => /waypoint|finish/i.test(n.status)).forEach(n => {
      const g = nearGps(n.t);
      if (Math.abs(g.t - n.t) <= 5)
        L.circleMarker([g.lat,g.lon], {radius:4, color:'#4da3ff', fillOpacity:.7})
          .addTo(map).bindPopup(`t+${fmt(n.t)}: ${esc(n.step||n.status)}`);
    });
  };
  js.onerror = () => {
    note.innerHTML = 'Harita için internet bağlantısı gerekli (Leaflet CDN yüklenemedi). ' +
      `İz: ${D.gps.length} fix, ilk nokta ${D.gps[0].lat.toFixed(5)}, ${D.gps[0].lon.toFixed(5)}.`;
  };
  document.head.appendChild(js);
}
if (!D.gps.length) document.getElementById('tab-harita').style.display = 'none';

/* ── Grafikler ── */
function chart(title, series, unit) {
  // series: [{label, color, pts:[[t,v],...]}]
  const all = series.flatMap(s => s.pts);
  if (!all.length) return '';
  const W = 1000, H = 130, PAD = 34;
  const vmax = Math.max(...all.map(p => p[1])), vmin = Math.min(...all.map(p => p[1]));
  const span = (vmax - vmin) || 1;
  const X = t => PAD + (W-2*PAD) * t / (D.duration || 1);
  const Y = v => (H-18) - (H-30) * (v - vmin) / span;
  let paths = series.map(s =>
    `<path fill="none" stroke="${s.color}" stroke-width="1.6" d="M` +
    s.pts.map(p => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' L ') + '"/>').join('');
  let marks = D.sysMarks.map(m =>
    `<line x1="${X(m.t)}" y1="10" x2="${X(m.t)}" y2="${H-18}" stroke="#e05252" stroke-dasharray="3,3"/>` +
    `<title>${esc(m.label)}</title>`).join('');
  const legend = series.map(s => `<tspan fill="${s.color}">● ${s.label}</tspan>  `).join('');
  return `<div class="chart"><h3>${title}</h3>
    <svg viewBox="0 0 ${W} ${H}" data-unit="${unit}">
      <text x="${PAD}" y="12" font-size="11" fill="#8593a8">${legend}</text>
      <text x="2" y="${Y(vmax)+4}" font-size="10" fill="#8593a8">${vmax.toFixed(0)}</text>
      <text x="2" y="${Y(vmin)+4}" font-size="10" fill="#8593a8">${vmin.toFixed(0)}</text>
      ${marks}${paths}
    </svg><div class="tip"></div></div>`;
}
const pcs = D.percs, tlm = D.telem;
document.getElementById('grafik').innerHTML =
  chart('Yürünebilir alan (%)',
    [{label:'walkable %', color:'#2fa86c', pts: pcs.filter(p=>p.walkable!=null).map(p=>[p.t, p.walkable*100])}], '%') +
  chart('Inference süresi (ms)',
    [{label:'inference ms', color:'#4da3ff', pts: pcs.filter(p=>p.inf!=null).map(p=>[p.t, p.inf])},
     {label:'toplam ms', color:'#9a7bd0', pts: pcs.filter(p=>p.total!=null).map(p=>[p.t, p.total])}], 'ms') +
  chart('Termal ve GPU',
    [{label:'SoC °C', color:'#d99a26', pts: tlm.filter(t=>t.temp!=null).map(t=>[t.t, t.temp])},
     {label:'GPU %', color:'#e05252', pts: tlm.filter(t=>t.gpu!=null).map(t=>[t.t, t.gpu])},
     {label:'RAM %', color:'#8593a8', pts: tlm.filter(t=>t.ram!=null).map(t=>[t.t, t.ram])}], '');

/* ── Konuşmalar ── */
const fM = document.getElementById('fMethod'), fS = document.getElementById('fSup');
[...new Set(D.speaks.map(s=>s.method))].sort().forEach(m => {
  const o = document.createElement('option'); o.value = m; o.textContent = m; fM.appendChild(o);
});
function renderSpeaks() {
  const rows = D.speaks
    .filter(s => !fM.value || s.method === fM.value)
    .filter(s => fS.checked || s.spoken)
    .map((s) => `<tr class="${s.spoken?'':'sup'}" data-t="${s.t}">
      <td>t+${fmt(s.t)}</td><td>${s.wall||''}</td>
      <td><span class="mchip ${esc(s.method)}">${esc(s.method)}</span></td>
      <td>${esc(s.text)}</td>
      <td>${s.spoken ? 'konuşuldu' : 'bastırıldı'+(s.reason?' ('+esc(s.reason)+')':'')}</td></tr>`)
    .join('');
  document.getElementById('speakRows').innerHTML = rows;
}
fM.addEventListener('change', renderSpeaks);
fS.addEventListener('change', renderSpeaks);
document.getElementById('speakRows').addEventListener('click', (e) => {
  const tr = e.target.closest('tr'); if (!tr) return;
  tabs.querySelector('[data-tab="zaman"]').click();
  seek(nearestFrame(parseFloat(tr.dataset.t)));
});
renderSpeaks();
</script>
</body>
</html>
"""


def build_html(session_dir: Path, events: list) -> str:
    """Build the self-contained viewer HTML (signature kept for the recorder)."""
    data = _prepare(session_dir, events)
    s = data["summary"]
    dur = s["duration_s"]
    duration_str = f"{int(dur // 60)} dk {int(dur % 60)} sn"
    # </script> inside the embedded JSON would terminate the data block early.
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__SESSION__", _esc(session_dir.name))
            .replace("__START__", _esc(s["start"] or "saat bilinmiyor"))
            .replace("__DURATION__", duration_str)
            .replace("__DATA_JSON__", data_json))


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
