# ALAS Field-Test Recorder & Report

Tools for recording and reviewing a real outdoor walk on the wearable (no screen).

## Record a session (on the device)

Run the system with `--record` (optionally `--live` for a one-line stdout
dashboard if you tether a laptop over local SSH/serial — no internet needed):

```
cd src
python -m main.alas_main --record --model models/segmentation/alas_engine.trt
python -m main.alas_main --record --live --model models/segmentation/alas_engine.trt
```

Each run writes a session folder under `outputs/field_tests/<timestamp>/`:

| File | Contents |
|---|---|
| `events.jsonl` | Structured timeline — perception, every spoken/suppressed utterance (+reason), nav, GPS, telemetry, voice commands, mode changes. |
| `frames/` | Annotated overlay JPEGs, saved only on notable events (alert spoken, nav announcement, off-route). |
| `session.json` | Run metadata (start time, git commit, model). |
| `summary.md` | Human-readable report (written at clean shutdown). |
| `summary_partial.md` | Rolling checkpoint (rewritten every ~30 s — survives a power cut). |
| `gps_track.gpx` | The walk, for mapping. |

### Field-readiness behavior

- **No screen needed** — everything is captured for review at home.
- **Bounded memory** — a full SD card / throttle drops events (logged as
  `queue_overflow`) instead of OOM-killing the system.
- **No real-time clock** — absolute time is anchored once from GPS satellite UTC
  (`clock_sync`); `report.py` reconstructs true wall-clock times from it.
- **Low disk** — if free space is below `min_free_mb`, recording is disabled with
  a spoken warning, but the assistive system keeps running.
- **Power cut** — `events.jsonl` is flushed per line; rebuild the summary at home
  with `report.py` even from an unclosed session.

## Review a session (at home)

```
python eval/field_test/report.py outputs/field_tests/<timestamp>/
```

Regenerates `summary.md` (and `gps_track.gpx`) from `events.jsonl`, tolerating a
truncated final line from an unclean shutdown. The summary reports: duration &
distance, FPS/latency, **peak SoC temperature and throttling flags**, the
**alert breakdown (spoken vs suppressed and why, utterances/min)**, navigation
status counts, voice-command outcomes, and links to the saved overlays.

Tuning lives in `RecorderConfig` (`src/main/config.py`): `overlay_jpeg_quality`,
`frame_min_interval_s`, `telemetry_interval_s`, `queue_maxsize`, `min_free_mb`.
