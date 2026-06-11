"""Session recorder for ALAS field tests (--record flag).

Writes to outputs/field_tests/<timestamp>/:
    events.jsonl        — append-only event log (one JSON per line)
    frames/             — annotated overlay JPEGs on notable events
    session.json        — run metadata
    summary_partial.md  — rolling checkpoint
    summary.md          — final report (on finalize)
    gps_track.gpx       — GPS track
    viewer.html         — per-frame HTML viewer

All disk I/O runs on a background writer thread; real-time threads only
enqueue events. Queue is bounded — drops on overflow instead of OOM.
Without --record, build_recorder() returns NullRecorder (no-op).
"""

import collections
import glob
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ALAS.recorder")


# ── Telemetry helpers (Jetson sysfs; graceful no-op elsewhere) ───────────────

def read_soc_temps() -> dict:
    """Read all thermal-zone temperatures in °C. Empty dict when unavailable."""
    temps = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(zone, "temp")) as f:
                milli = int(f.read().strip())
            type_path = os.path.join(zone, "type")
            name = (
                open(type_path).read().strip()
                if os.path.exists(type_path)
                else os.path.basename(zone)
            )
            temps[name] = round(milli / 1000.0, 1)
        except Exception:
            continue
    return temps


# Jetson Nano quirk: the PMIC-Die zone is not a real sensor — it reports a
# CONSTANT (50.0 °C on this board, 100 °C on some others). Including it in a
# max() pins every "SoC temperature" reading to that constant: the 11-06
# school sessions reported a flat 50 °C peak while the real zones (CPU/GPU/AO)
# ranged 22-46 °C. Every consumer must take the max through this filter.
FAKE_THERMAL_ZONES = ("PMIC-Die",)


def max_real_temp(temps: dict):
    """Hottest REAL thermal zone in °C, or None. Filters fake constant zones."""
    real = [v for k, v in temps.items() if k not in FAKE_THERMAL_ZONES]
    return max(real) if real else None


def _load_avg() -> Optional[float]:
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


# ── GPU / RAM / power telemetry (Jetson sysfs; graceful no-op elsewhere) ─────

def _read_int_file(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _call_with_timeout(fn, timeout: float = 1.0):
    """Run ``fn()`` on a short-lived daemon thread; return its result or None if
    it does not finish within ``timeout``.

    The INA3221 power sensor sits on the I2C bus, so a momentarily busy bus could
    stall a sysfs read. Bounding it here guarantees the telemetry loop never
    blocks (worst case = one skipped power sample).
    """
    result = [None]

    def _runner():
        try:
            result[0] = fn()
        except Exception:
            result[0] = None

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout)
    return result[0]


def read_gpu_load_pct() -> Optional[float]:
    """Jetson GPU load as a percentage (0–100), or None when unavailable.

    The Tegra sysfs value is per-mille (0–1000), so divide by 10 — otherwise a
    full load reads as '850%' instead of '85%'.
    """
    for path in ("/sys/devices/gpu.0/load",
                 "/sys/devices/platform/gpu.0/load",
                 "/sys/devices/57000000.gpu/load"):
        raw = _read_int_file(path)
        if raw is not None:
            return round(raw / 10.0, 1)
    return None


def read_mem() -> dict:
    """Used RAM from /proc/meminfo: {used_mb, total_mb, used_pct}. Empty on error."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.strip().split()
                if parts:
                    info[key.strip()] = int(parts[0])  # value in kB
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if not total or avail is None:
            return {}
        used_kb = total - avail
        return {"used_mb": round(used_kb / 1024.0, 1),
                "total_mb": round(total / 1024.0, 1),
                "used_pct": round(100.0 * used_kb / total, 1)}
    except Exception:
        return {}


# JetPack 4.6.x Nano fallback: INA3221 on I2C bus 6, addr 0x40.
#   in_power0 → VDD_IN (total)   in_power1 → VDD_GPU   in_power2 → VDD_CPU
_INA3221_FALLBACK_DIR = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0"


def _read_power_rails() -> dict:
    """Read INA3221 rail power in mW → {rail_label: mW, ..., 'total': mW}.

    Iterates the iio:device dirs and reads ONLY ``in_power<N>_input`` for N in
    0..2 — deliberately NOT a ``in_power*_input`` glob, which also matches
    ``in_power0_trigger_input`` (not a power value). Rail labels come from
    ``rail_name_<N>`` (e.g. POM_5V_IN / POM_5V_GPU / POM_5V_CPU on the Nano).
    """
    dirs = glob.glob("/sys/bus/i2c/devices/*/iio:device*")
    dirs += glob.glob("/sys/bus/i2c/drivers/ina3221*/*/iio:device*")
    dirs.append(_INA3221_FALLBACK_DIR)

    rails: dict = {}
    seen: set = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        real = os.path.realpath(d)  # devices/ and drivers/ symlink to the same node
        if real in seen:
            continue
        seen.add(real)
        for ch in (0, 1, 2):
            path = os.path.join(d, f"in_power{ch}_input")
            if not os.path.exists(path):
                continue
            mw = _read_int_file(path)
            if mw is None:
                continue
            label = (_read_text_file(os.path.join(d, f"rail_name_{ch}"))
                     or _read_text_file(os.path.join(d, f"in_power{ch}_label"))
                     or f"rail{ch}")
            rails[label] = mw

    if rails and "total" not in rails:
        in_rail = next((v for k, v in rails.items() if "IN" in k.upper()), None)
        rails["total"] = in_rail if in_rail is not None else sum(rails.values())
    return rails


def read_power_mw() -> dict:
    """INA3221 rail power (mW), I2C-timeout-guarded. Empty dict when unavailable."""
    return _call_with_timeout(_read_power_rails, timeout=1.0) or {}


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ── No-op recorder (recording disabled) ─────────────────────────────────────

class NullRecorder:
    """Drop-in no-op so callers never branch on ``if recorder``."""

    enabled = False

    def log_system(self, **kw) -> None: ...
    def log_mode(self, frm, to) -> None: ...
    def log_perception(self, result, chosen=None) -> None: ...
    def log_speak(self, method, text, spoken, reason=None) -> None: ...
    def log_nav(self, status, distance_to_next_m=None, step_text=None, gps=None) -> None: ...
    def log_route(self, steps, origin=None, destination=None) -> None: ...
    def log_gps(self, lat, lon, age_s, sats, hdop, status, utc=None) -> None: ...
    def log_gps_status(self, status, sats, hdop, serial_ok, fix_age_s=None) -> None: ...
    def log_command(self, text, intent=None, confidence=None, action=None) -> None: ...
    def note_gps_utc(self, utc_dt, captured_mono) -> None: ...
    def maybe_save_overlay(self, frame_bgr, mask, tag, info=None) -> None: ...
    def set_mode(self, mode) -> None: ...
    def finalize(self) -> None: ...


# ── The recorder ────────────────────────────────────────────────────────────

class SessionRecorder:
    """Captures a single field-test session to disk (see module docstring)."""

    enabled = True

    def __init__(self, config) -> None:
        self._cfg = config
        self._rc = config.rec
        self._t0 = time.monotonic()

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(config.record_dir) / ts
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._events_path = self.session_dir / "events.jsonl"
        self._events_file = open(self._events_path, "a", buffering=1)  # line-buffered
        # Bounded in-memory window for the rolling checkpoint preview only. The
        # authoritative final summary is rebuilt from events.jsonl on disk at
        # finalize(), so memory stays flat over a multi-hour walk.
        self._recent = collections.deque(maxlen=self._rc.recent_events_max)

        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue(maxsize=self._rc.queue_maxsize)
        self._dropped = 0
        self._dropped_reported = 0

        self._frame_seq = 0
        self._last_frame_t = 0.0
        self._frames_disabled = False           # set if disk runs low mid-walk
        self._low_disk_announced = False

        self._clock_synced = False
        self._status: dict = {"mode": "?", "hazard": None, "utterance": None,
                              "gps": None, "temp": None, "gpu": None, "power": None}
        self._stop = threading.Event()
        self._last_checkpoint = time.monotonic()

        self._write_meta()

        self._writer = threading.Thread(target=self._writer_loop, name="RecorderWriter", daemon=True)
        self._writer.start()
        self._telemetry = threading.Thread(target=self._telemetry_loop, name="RecorderTelemetry", daemon=True)
        self._telemetry.start()
        if config.live:
            threading.Thread(target=self._live_loop, name="RecorderLive", daemon=True).start()

        logger.info("[Recorder] Recording to %s", self.session_dir)
        self.log_system(event="session_start", session=str(self.session_dir))

    # ── Time / emit ─────────────────────────────────────────────

    def _rel(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def _emit(self, ev: dict) -> None:
        ev.setdefault("t", self._rel())
        # Store local time (with tz offset) so the wall field is human-readable
        # without a UTC conversion step. Jetson's local timezone is respected as
        # long as the OS clock/TZ is correct.
        ev.setdefault("wall", datetime.now().astimezone().isoformat())
        try:
            self._q.put_nowait(("event", ev))
        except queue.Full:
            with self._lock:
                self._dropped += 1

    # ── Public logging API ──────────────────────────────────────

    def log_system(self, **kw) -> None:
        self._emit({"type": "system", **kw})

    def log_mode(self, frm, to) -> None:
        self._status["mode"] = str(to)
        self._emit({"type": "mode", "from": str(frm), "to": str(to)})

    def set_mode(self, mode) -> None:
        self._status["mode"] = str(mode)

    def log_perception(self, result, chosen=None) -> None:
        scene = result.scene
        top = result.alerts[0] if getattr(result, "alerts", None) else None
        self._status["hazard"] = scene.dominant_hazard
        if result.total_ms:
            self._status["fps"] = round(1000.0 / result.total_ms, 1)
        self._emit({
            "type": "perception",
            "walkable": round(scene.walkable_ratio, 3),
            "is_safe": scene.is_safe,
            "safety_level": getattr(scene, "safety_level", None),
            "hazard": scene.dominant_hazard,
            "alert": ({"class": int(top.class_id), "text": top.text, "priority": top.priority}
                      if top else None),
            "guidance": result.path_guidance,
            "chosen": chosen,
            "inf_ms": round(result.inference_ms, 1),
            "total_ms": round(result.total_ms, 1),
        })

    def log_speak(self, method, text, spoken, reason=None) -> None:
        if spoken:
            self._status["utterance"] = text
        self._emit({"type": "speak", "method": method, "text": text,
                    "spoken": bool(spoken), "reason": reason})

    def log_nav(self, status, distance_to_next_m=None, step_text=None, gps=None) -> None:
        self._emit({"type": "nav", "status": str(status),
                    "distance_to_next_m": distance_to_next_m,
                    "step_text": step_text, "gps": gps})

    def log_route(self, steps, origin=None, destination=None) -> None:
        """Record the full planned route (every step + coords) when created."""
        try:
            step_dicts = [s.to_dict() for s in (steps or [])]
        except Exception:  # noqa: BLE001
            step_dicts = []
        self._emit({"type": "route", "n_steps": len(step_dicts),
                    "origin": origin, "destination": destination,
                    "steps": step_dicts})

    def log_gps(self, lat, lon, age_s, sats, hdop, status, utc=None) -> None:
        self._status["gps"] = f"{lat:.5f},{lon:.5f}"
        self._emit({"type": "gps", "lat": lat, "lon": lon, "age_s": age_s,
                    "sats": sats, "hdop": hdop, "status": str(status), "utc": utc})

    def log_gps_status(self, status, sats, hdop, serial_ok, fix_age_s=None) -> None:
        """GPS health every poll — visible even with NO fix (dropouts, serial loss)."""
        self._emit({"type": "gps_status", "status": str(status), "sats": sats,
                    "hdop": hdop, "serial_ok": bool(serial_ok), "fix_age_s": fix_age_s})

    def log_command(self, text, intent=None, confidence=None, action=None) -> None:
        self._emit({"type": "command", "text": text, "intent": intent,
                    "confidence": confidence, "action": action})

    def note_gps_utc(self, utc_dt, captured_mono) -> None:
        """Anchor absolute time once, from satellite UTC (no-RTC safe)."""
        if self._clock_synced or utc_dt is None:
            return
        self._clock_synced = True
        rel = round(captured_mono - self._t0, 3)
        self._emit({"type": "clock_sync", "t": rel, "gps_utc": utc_dt.isoformat()})
        logger.info("[Recorder] Clock anchored to GPS UTC %s", utc_dt.isoformat())

    def maybe_save_overlay(self, frame_bgr, mask, tag, info=None) -> None:
        """Queue an annotated overlay for saving, throttled and disk-aware.

        ``info`` is an optional dict forwarded to ``render_overlay`` for
        on-frame annotations: walkable, hazard, spoken, t.
        """
        if frame_bgr is None or mask is None or self._frames_disabled:
            return
        now = time.monotonic()
        if now - self._last_frame_t < self._rc.frame_min_interval_s:
            return
        self._last_frame_t = now
        self._frame_seq += 1
        seq = self._frame_seq
        rel_name = f"frames/f_{seq:05d}_{tag}.jpg"
        if info is not None:
            info.setdefault("t", self._rel())
        self._emit({"type": "frame", "file": rel_name, "tag": tag})
        try:
            self._q.put_nowait(("frame", (seq, tag, frame_bgr, mask, info)))
        except queue.Full:
            with self._lock:
                self._dropped += 1

    # ── Writer thread ───────────────────────────────────────────

    def _writer_loop(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                kind, payload = self._q.get(timeout=0.5)
            except queue.Empty:
                self._periodic()
                continue
            try:
                if kind == "event":
                    self._events_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    with self._lock:
                        self._recent.append(payload)
                elif kind == "frame":
                    self._write_frame(payload)
            except Exception:
                logger.exception("[Recorder] writer error")
            self._periodic()

    def _write_frame(self, payload) -> None:
        import cv2  # local import: event-only sessions never need OpenCV
        from ai.perception import render_overlay
        seq, tag, frame_bgr, mask, info = payload
        overlay = render_overlay(frame_bgr, mask, info=info)
        path = self.frames_dir / f"f_{seq:05d}_{tag}.jpg"
        cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, self._rc.overlay_jpeg_quality])

    def _periodic(self) -> None:
        self._report_drops()
        now = time.monotonic()
        if now - self._last_checkpoint >= self._rc.checkpoint_interval_s:
            self._last_checkpoint = now
            self._write_checkpoint()

    def _report_drops(self) -> None:
        with self._lock:
            dropped, reported = self._dropped, self._dropped_reported
        if dropped > reported:
            # Written directly (we are on the writer thread) to avoid re-queueing.
            ev = {"t": self._rel(), "wall": datetime.now(timezone.utc).isoformat(),
                  "type": "system", "error": "queue_overflow", "dropped": dropped}
            try:
                self._events_file.write(json.dumps(ev, ensure_ascii=False) + "\n")
                with self._lock:
                    self._recent.append(ev)
                    self._dropped_reported = dropped
            except Exception:
                logger.exception("[Recorder] drop-report write failed")

    def _write_checkpoint(self) -> None:
        try:
            with self._lock:
                events = list(self._recent)  # recent window — a live preview
            (self.session_dir / "summary_partial.md").write_text(
                build_summary(events, title="ALAS Field Test (in-progress checkpoint)"),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("[Recorder] checkpoint write failed")

    # ── Telemetry thread ────────────────────────────────────────

    def _telemetry_loop(self) -> None:
        while not self._stop.wait(self._rc.telemetry_interval_s):
            temps = read_soc_temps()  # full dict logged; consumers filter fakes
            real_max = max_real_temp(temps)
            if real_max is not None:
                self._status["temp"] = real_max
            gpu = read_gpu_load_pct()
            ram = read_mem()
            power = read_power_mw()
            if gpu is not None:
                self._status["gpu"] = gpu
            if power.get("total") is not None:
                self._status["power"] = power["total"]
            self._emit({"type": "telemetry", "temps_c": temps, "load1": _load_avg(),
                        "gpu_pct": gpu, "ram": ram or None, "power_mw": power or None})
            self._check_disk()

    def _check_disk(self) -> None:
        try:
            free_mb = shutil.disk_usage(self.session_dir).free / (1024 * 1024)
        except Exception:
            return
        if free_mb < self._rc.min_free_mb and not self._frames_disabled:
            self._frames_disabled = True
            self._emit({"type": "system", "error": "low_disk",
                        "free_mb": round(free_mb), "action": "frame_saving_disabled"})
            logger.warning("[Recorder] Low disk (%.0f MB) — frame saving disabled.", free_mb)

    # ── Live dashboard thread ───────────────────────────────────

    def _live_loop(self) -> None:
        import sys
        while not self._stop.wait(1.0):
            s = self._status
            line = (f"\r[LIVE] mode={s.get('mode')} fps={s.get('fps','?')} "
                    f"gpu={s.get('gpu') if s.get('gpu') is not None else '?'}% "
                    f"temp={s.get('temp','?')}C "
                    f"pow={s.get('power') if s.get('power') is not None else '?'}mW "
                    f"hazard={s.get('hazard')} "
                    f"say={(s.get('utterance') or '')[:20]!r} gps={s.get('gps')}   ")
            sys.stdout.write(line)
            sys.stdout.flush()

    # ── Metadata + finalize ─────────────────────────────────────

    def _write_meta(self) -> None:
        meta = {
            "started": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "model_path": getattr(self._cfg.ai, "model_path", None),
            "perception_fps": getattr(self._cfg.ai, "perception_fps", None),
            "mock": getattr(self._cfg, "mock", None),
        }
        try:
            (self.session_dir / "session.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.exception("[Recorder] session.json write failed")

    def _read_events_file(self) -> list:
        """Re-read events.jsonl from disk (authoritative, memory-safe).

        Tolerant of a truncated last line (power-cut). Used by finalize so the
        final summary is complete regardless of the bounded in-RAM window.
        """
        events: list = []
        try:
            with open(self._events_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:  # noqa: BLE001 — skip a truncated/corrupt line
                        pass
        except Exception:  # noqa: BLE001
            logger.exception("[Recorder] re-reading events.jsonl failed")
        return events

    def finalize(self) -> None:
        """Flush, stop threads, and write the final summary + GPX track."""
        self.log_system(event="session_end")
        self._stop.set()
        try:
            self._writer.join(timeout=5.0)
        except Exception:
            pass
        try:
            self._events_file.flush()
        except Exception:
            pass
        # Authoritative, complete summary from disk (not the bounded RAM window).
        events = self._read_events_file()
        try:
            (self.session_dir / "summary.md").write_text(
                build_summary(events, title="ALAS Field Test Report"), encoding="utf-8")
            write_gpx(events, self.session_dir / "gps_track.gpx")
            _write_viewer(events, self.session_dir)
        except Exception:
            logger.exception("[Recorder] finalize summary failed")
        try:
            self._events_file.flush()
            self._events_file.close()
        except Exception:
            pass
        # Force the page cache to the SD card. The status LED goes dark right
        # after finalize as the "safe to cut LiPo power" signal — without this
        # sync the files may still live only in RAM and a power cut would
        # truncate them despite the LED's promise.
        try:
            os.sync()
        except Exception:
            pass
        logger.info("[Recorder] Session finalized: %s", self.session_dir)


# ── Factory (disk pre-flight + TTS warning) ──────────────────────────────────

def build_recorder(config, voice=None):
    """Return a SessionRecorder, or a NullRecorder when recording is off/unsafe.

    Refuses to record (and warns over TTS) when free disk space is below
    ``min_free_mb`` — the assistive system still runs, only the black box is
    disabled, so a forgotten-to-clean SD card never silently fills mid-walk.
    """
    if not getattr(config, "record", False):
        return NullRecorder()
    base = config.record_dir
    try:
        os.makedirs(base, exist_ok=True)
        free_mb = shutil.disk_usage(base).free / (1024 * 1024)
    except Exception:
        free_mb = float("inf")
    if free_mb < config.rec.min_free_mb:
        logger.warning("[Recorder] Low disk (%.0f MB < %d MB) — recording disabled.",
                       free_mb, config.rec.min_free_mb)
        if voice is not None:
            try:
                voice.emergency("Dikkat, depolama alanı yetersiz, kayıt yapılmayacak.")
            except Exception:
                logger.exception("[Recorder] low-disk announce failed")
        return NullRecorder()
    return SessionRecorder(config)


# ── Viewer helper (calls eval/field_test/viewer.py) ─────────────────────────

def _write_viewer(events: list, session_dir) -> None:
    """Generate viewer.html alongside the summary. Best-effort; never raises."""
    try:
        import importlib.util, sys as _sys
        _repo = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "field_test_viewer", _repo / "eval" / "field_test" / "viewer.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        html = mod.build_html(Path(session_dir), events)
        (Path(session_dir) / "viewer.html").write_text(html, encoding="utf-8")
        logger.info("[Recorder] viewer.html written to %s", session_dir)
    except Exception:
        logger.debug("[Recorder] viewer.html generation skipped", exc_info=True)


# ── Report building (shared with eval/field_test/report.py) ──────────────────

def _abs_time(ev: dict, sync: Optional[dict]) -> Optional[str]:
    """Absolute UTC ISO time for an event, rebased from the GPS clock_sync anchor."""
    if sync is None:
        return None
    try:
        anchor = datetime.fromisoformat(sync["gps_utc"])
        from datetime import timedelta
        return (anchor + timedelta(seconds=ev["t"] - sync["t"])).isoformat()
    except Exception:
        return None


def build_summary(events: list, title: str = "ALAS Field Test Report",
                  skipped_lines: int = 0) -> str:
    """Build a human-readable Markdown summary from a list of event dicts.

    Tolerant of partial/odd data so it works on both clean and crash-truncated
    sessions. Shared by the live recorder's finalize() and the offline report.py.
    """
    by_type: dict = {}
    for ev in events:
        by_type.setdefault(ev.get("type"), []).append(ev)

    perception = by_type.get("perception", [])
    speaks = by_type.get("speak", [])
    navs = by_type.get("nav", [])
    gpses = by_type.get("gps", [])
    telem = by_type.get("telemetry", [])
    commands = by_type.get("command", [])
    systems = by_type.get("system", [])
    sync = (by_type.get("clock_sync") or [None])[0]

    ts = [ev["t"] for ev in events if isinstance(ev.get("t"), (int, float))]
    duration = (max(ts) - min(ts)) if ts else 0.0

    lines = [f"# {title}", ""]

    # Session
    lines.append("## Session")
    lines.append(f"- Duration: **{duration:.0f} s** ({duration/60:.1f} min)")
    lines.append(f"- Events recorded: **{len(events)}**")
    if skipped_lines:
        lines.append(f"- ⚠ Skipped {skipped_lines} unreadable line(s) (likely a power-cut truncation).")
    if sync:
        lines.append(f"- Wall-clock anchor (GPS UTC): `{sync.get('gps_utc')}` — wall times reconstructed.")
    else:
        lines.append("- ⚠ No GPS time anchor — wall times unavailable; using relative time.")
    lines.append("")

    # Performance
    if perception:
        infs = [e.get("inf_ms", 0) for e in perception if e.get("inf_ms") is not None]
        totals = [e.get("total_ms", 0) for e in perception if e.get("total_ms") is not None]
        fps = [1000.0 / t for t in totals if t]
        lines.append("## Perception performance")
        lines.append(f"- Frames processed: **{len(perception)}**")
        if fps:
            lines.append(f"- FPS: avg **{_avg(fps):.1f}**, min **{min(fps):.1f}**")
        if infs:
            lines.append(f"- Inference ms: avg **{_avg(infs):.0f}**, p95 **{_pct(infs, 95):.0f}**")
        n_safe    = sum(1 for e in perception if e.get("safety_level", None) == 0
                        or (e.get("safety_level") is None and e.get("is_safe") is True))
        n_caution = sum(1 for e in perception if e.get("safety_level") == 1)
        n_unsafe  = sum(1 for e in perception if e.get("safety_level", 2) == 2
                        and e.get("is_safe") is not True)
        lines.append(f"- Safety: safe **{n_safe}** | caution **{n_caution}** | unsafe **{n_unsafe}**")
        lines.append("")

    # Thermal / power
    if telem:
        lines.append("## Thermal / power")
        all_temps = [max(e["temps_c"].values()) for e in telem if e.get("temps_c")]
        if all_temps:
            tmax = max(all_temps)
            lines.append(f"- Peak SoC temp: **{tmax:.0f} °C**")
            hot = sum(1 for t in all_temps if t >= 80)
            if hot:
                lines.append(f"- ⚠ {hot} sample(s) ≥ 80 °C — possible thermal throttling "
                             f"(correlate with FPS dips in the timeline).")
        loads = [e.get("load1") for e in telem if e.get("load1") is not None]
        if loads:
            lines.append(f"- CPU load (1 min): avg **{_avg(loads):.2f}**, max **{max(loads):.2f}**")
        gpus = [e.get("gpu_pct") for e in telem if e.get("gpu_pct") is not None]
        if gpus:
            lines.append(f"- GPU load: avg **{_avg(gpus):.0f}%**, peak **{max(gpus):.0f}%**")
        rams = [e["ram"]["used_mb"] for e in telem
                if e.get("ram") and e["ram"].get("used_mb") is not None]
        if rams:
            rpct = [e["ram"]["used_pct"] for e in telem
                    if e.get("ram") and e["ram"].get("used_pct") is not None]
            lines.append(f"- RAM used: avg **{_avg(rams):.0f} MB**, peak **{max(rams):.0f} MB**"
                         + (f" ({max(rpct):.0f}% peak)" if rpct else ""))
        # Power draw + integrated energy (mWh) → battery consumption estimate.
        totals = [(e["t"], e["power_mw"]["total"]) for e in telem
                  if e.get("power_mw") and e["power_mw"].get("total") is not None
                  and isinstance(e.get("t"), (int, float))]
        if totals:
            pvals = [p for _, p in totals]
            lines.append(f"- Power draw (VDD_IN): avg **{_avg(pvals):.0f} mW**, "
                         f"peak **{max(pvals):.0f} mW**")
            energy_mws = 0.0  # trapezoidal integral of power over time (mW·s)
            for (t0, p0), (t1, p1) in zip(totals, totals[1:]):
                energy_mws += (p0 + p1) / 2.0 * (t1 - t0)
            mwh = energy_mws / 3600.0
            lines.append(f"- Energy this session: **{mwh:.1f} mWh** "
                         f"(≈ {mwh / 1000.0:.2f} Wh over {duration / 60:.1f} min)")
        elif not gpus and not rams:
            lines.append("- (GPU/RAM/power sensors not readable — desktop run or "
                         "INA3221 sysfs path/JetPack mismatch.)")
        lines.append("")

    # Voice quality
    lines.append("## Voice output")
    spoken = [s for s in speaks if s.get("spoken")]
    suppressed = [s for s in speaks if not s.get("spoken")]
    lines.append(f"- Utterances spoken: **{len(spoken)}**"
                 + (f" (~{len(spoken)/(duration/60):.1f}/min)" if duration > 0 else ""))
    lines.append(f"- Suppressed: **{len(suppressed)}**")
    reasons: dict = {}
    for s in suppressed:
        reasons[s.get("reason") or "unspecified"] = reasons.get(s.get("reason") or "unspecified", 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"    - {reason}: {n}")
    by_method: dict = {}
    for s in spoken:
        by_method[s.get("method")] = by_method.get(s.get("method"), 0) + 1
    if by_method:
        lines.append("- Spoken by type: "
                     + ", ".join(f"{m}={n}" for m, n in sorted(by_method.items(), key=lambda x: -x[1])))
    # Repeat pressure: how often the SAME line was chosen by the dispatcher.
    # High counts here mean a cadence (min_obstacle_repeat_sec etc.) is too
    # short for real walks — the calibration signal for tuning B6.
    chosen_counts: dict = {}
    for e in perception:
        c = e.get("chosen")
        if c:
            chosen_counts[c] = chosen_counts.get(c, 0) + 1
    repeats = sorted(((t, n) for t, n in chosen_counts.items() if n >= 5),
                     key=lambda x: -x[1])[:5]
    if repeats:
        lines.append("- Most repeated announcements:")
        for text, n in repeats:
            lines.append(f"    - ×{n}: {text}")
    lines.append("")

    # Navigation
    if navs:
        statuses: dict = {}
        for n in navs:
            statuses[n.get("status")] = statuses.get(n.get("status"), 0) + 1
        lines.append("## Navigation")
        lines.append("- Status counts: "
                     + ", ".join(f"{s}={c}" for s, c in sorted(statuses.items(), key=lambda x: -x[1])))
        # Show waypoint-hit and step-change events with timestamps
        notable = [n for n in navs if n.get("status") in (
            "WAYPOINT_HIT", "OFF_ROUTE", "FINISHED", "STARTED", "REROUTING"
        ) or n.get("step_text")]
        if notable:
            lines.append("- Notable events:")
            for n in notable[:30]:
                at = _abs_time(n, sync) or f"t+{n.get('t', 0):.0f}s"
                dist = (f", {n['distance_to_next_m']:.0f}m to next"
                        if n.get("distance_to_next_m") is not None else "")
                step = (f" — {n['step_text']}"
                        if n.get("step_text") else "")
                lines.append(f"    - [{at}] {n.get('status')}{dist}{step}")
        lines.append("")

    # Voice input
    if commands:
        lines.append("## Voice commands")
        lines.append(f"- Commands handled: **{len(commands)}**")
        for c in commands[:15]:
            lines.append(f"    - \"{c.get('text')}\" → {c.get('intent')} "
                         f"(conf {c.get('confidence')}) → {c.get('action')}")
        lines.append("")

    # Health flags
    errors = [s for s in systems if s.get("error")]
    if errors:
        lines.append("## ⚠ Health flags")
        seen: dict = {}
        for e in errors:
            seen[e.get("error")] = seen.get(e.get("error"), 0) + 1
        for err, n in seen.items():
            lines.append(f"- `{err}` ×{n}")
        lines.append("")

    # Saved frames
    frames = by_type.get("frame", [])
    if frames:
        lines.append("## Saved overlays")
        lines.append(f"- {len(frames)} overlay image(s) in `frames/` (saved on notable events).")
        for f in frames[:20]:
            at = _abs_time(f, sync) or f"t+{f.get('t'):.0f}s"
            lines.append(f"    - `{f.get('file')}` ({f.get('tag')}, {at})")
        lines.append("")

    lines.append(f"- GPS fixes logged: **{len(gpses)}**")
    return "\n".join(lines) + "\n"


def write_gpx(events: list, path) -> None:
    """Write GPS fixes to a minimal GPX file for mapping the walk."""
    pts = [e for e in events if e.get("type") == "gps"
           and e.get("lat") is not None and e.get("lon") is not None]
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="ALAS"><trk><name>ALAS field test</name><trkseg>']
    for e in pts:
        out.append(f'<trkpt lat="{e["lat"]}" lon="{e["lon"]}"></trkpt>')
    out.append("</trkseg></trk></gpx>")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]
