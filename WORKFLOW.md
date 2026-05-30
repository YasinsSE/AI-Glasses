# ALAS System Workflow & Architecture

## System Overview

ALAS (AI-Based Smart Glasses) is an assistive wearable running on NVIDIA Jetson Nano that helps visually impaired users navigate and detect obstacles through real-time semantic segmentation and GPS-based turn-by-turn navigation.

---

## System State Machine

```
                      (process start)
                            |
                            v
                      +-----------+
                      |  WARMUP   |   GPS warming, model loading
                      +-----+-----+   "Hazirlaniyorum" announced
                            |
          await_ready() or --bypass-warmup
                            |
                            v
                      +-----------+  ---- voice "uyu" ---->  +-------+
                      |  ACTIVE   |                          | SLEEP |
                      +-----+-----+  <--- button press ---- +---+---+
                            |
          SIGINT/SIGTERM    |
          voice "kapat"     |
                            v
                      +----------+
                      | SHUTDOWN |   ordered shutdown
                      +----------+
```

### Mode Behavior Table

| Mode     | Perception | Navigation | Button | TTS                    |
|----------|-----------|------------|--------|------------------------|
| WARMUP   | skipped   | skipped    | active | announcements only     |
| ACTIVE   | running   | running    | active | full                   |
| SLEEP    | skipped   | skipped    | active | sleep/wake only        |
| SHUTDOWN | exiting   | exiting    | exiting| flushing then shutdown |

---

## Data Flow Diagram

```
+-------------------+
|   Physical World  |
+--------+----------+
         |
    +----+----+          +----------+
    | Camera  |          | GPS NEO  |
    | (cv2)   |          | -7M UART |
    +----+----+          +-----+----+
         |                     |
         v                     v
+------------------+    +------------------+
| PerceptionService|    | NavigationService|
| (Thread)         |    | (Thread)         |
|                  |    |                  |
| frame -> preproc |    | GPS fix ->       |
| -> TensorRT inf  |    | nav.update()     |
| -> postprocess   |    | -> 3-tier        |
| -> analyse_scene |    |    announcement  |
| -> generate_alert|    |                  |
+--------+---------+    +--------+---------+
         |                       |
         | say_obstacle()        | say_nav() / say_progress()
         v                       v
    +----+------------------------+----+
    |          VoicePolicy             |
    |  (central TTS gate)              |
    |                                  |
    |  - active-utterance gate         |
    |  - post-nav silence window (3s)  |
    |  - priority speak (blocking)     |
    |  - obstacle speak (non-blocking) |
    +----+-----------------------------+
         |
         v
    +----+----+
    | TTS     |
    | (pyttsx3|
    |  queue) |
    +---------+


+------------------+     +-------------------+
| ButtonListener   | --> | VoiceCommandHandler|
| (GPIO or stdin)  |     |                   |
| press detected   |     | STT or keyboard   |
|                  |     | -> intent classify |
+------------------+     | -> navigation      |
                          | -> system_command  |
                          | -> sleep/wake      |
                          +-------------------+
```

---

## Perception Pipeline (Detail)

```
Camera Frame (BGR 640x480)
        |
        v
   preprocess()
   BGR -> RGB -> resize(512x384) -> float32 [0,1] -> (1,H,W,3)
        |
        v
   TensorRT Engine (.trt)
   or ONNX Runtime (.onnx)
        |
        v
   postprocess()
   logits -> argmax -> class-ID mask (384x512) uint8
        |
        v
   analyse_scene()
   +-- per-class pixel ratio
   +-- dominant zone (left/center/right)
   +-- walkable overlap (dilated, static obstacles only)
   +-- distance estimation (ground-plane projection)
        |
        v
   generate_alerts()
   +-- cooldown check (per-class, e.g. vehicle 1.5s, obstacle 3s)
   +-- walkable gate (COLLISION_OBSTACLE, FALL_HAZARD only)
   +-- lateral suppression (DYNAMIC_HAZARD on side = silent)
   +-- proximity wording (<2m "cok yakin", <5m "yakin")
   +-- direction wording ("solunuzda", "saginizda")
        |
        v
   VoicePolicy.say_obstacle(alert_text)
```

### 7 Semantic Classes

| ID | Class              | Alert Priority | Cooldown |
|----|--------------------|---------------|----------|
| 0  | WALKABLE_SURFACE   | 0 (silent)    | -        |
| 1  | CROSSWALK          | 1             | 8.0s     |
| 2  | VEHICLE_ROAD       | 4             | 5.0s     |
| 3  | COLLISION_OBSTACLE | 3             | 3.0s     |
| 4  | FALL_HAZARD        | 3             | 3.0s     |
| 5  | DYNAMIC_HAZARD     | 4             | 4.0s     |
| 6  | VEHICLE            | 5             | 1.5s     |

### Distance Estimation (Ground-Plane Projection)

```
Given: camera height (h), tilt angle (t), vertical FOV (vfov)
For obstacle bottom pixel at row y in image of height H:

  theta_pix   = ((y + 0.5) / H - 0.5) * vfov
  theta_total = tilt + theta_pix
  distance    = h / tan(theta_total)

Defaults: h=1.65m, tilt=5 deg, vfov=60 deg
Calibration: adjust camera_tilt_deg until estimates match reality
```

---

## Navigation Flow (Detail)

```
Button Press
     |
     v
VoiceCommandHandler.handle_press()
     |
     +-- (SLEEP mode?) -> wake up, no STT
     |
     +-- (ACTIVE mode)
           |
           v
     STT listen (or typed input in bypass mode)
           |
           v
     Intent Classification (SLM or keyword fallback)
           |
           +-- "navigation" -> extract POI keyword
           |     -> nav.navigate_to_nearest(position, category)
           |     -> "En yakin eczane 250 metre uzakta. Rota hazir."
           |
           +-- "system_command"
           |     -> "uyu" = SLEEP mode
           |     -> "kapat" = SHUTDOWN
           |     -> "rota iptal" = stop navigation
           |
           +-- "general" -> "Anlasildi."


NavigationService (GPS loop, 4s interval):
     |
     v
   GPS fix -> nav.update(Coord(lat, lon)) -> ProgressResult
     |
     +-- WAYPOINT_HIT:  "saga donun" (speak immediately)
     +-- OFF_ROUTE:     "Rotadan ciktiniz"
     +-- FINISHED:      "Hedefinize ulastiniz"
     +-- PROGRESSING:
           |
           +-- dist < 30m  -> pre-warn ONCE: "20 metre sonra saga donun"
           +-- dist > 100m -> periodic 30s: "Hedefe 150 metre"
           +-- 30-100m     -> SILENT (already announced)
```

---

## TTS Discipline (VoicePolicy)

Two independent mechanisms prevent TTS spam:

1. **Active-Utterance Gate** (`is_speaking_priority()`)
   - True only DURING a priority utterance (nav instruction, announcement)
   - PerceptionService polls this and skips inference entirely
   - Prevents stale obstacle alerts queueing behind live nav speech

2. **Post-Nav Silence Window** (`_suppress_obstacles_until`)
   - After a nav utterance finishes, obstacle alerts are dropped for 3 seconds
   - Perception still runs (scene state stays fresh), alerts just get silenced
   - Configurable via `post_nav_silence_sec`

```
Priority levels:
  announce_*()  -> blocking, no silence window
  emergency()   -> blocking, no silence window
  say_nav()     -> blocking, SETS silence window
  say_prompt()  -> blocking, no silence window
  say_progress()-> non-blocking, no gates
  say_obstacle()-> non-blocking, CHECKED against silence window
```

---

## File Structure

```
src/
+-- main/
|   +-- alas_main.py            THIN orchestrator (~130 lines)
|   +-- config.py               ALASConfig dataclass + from_cli()
|   +-- lifecycle.py            SystemMode, ModeManager, shutdown
|   +-- __init__.py
|
+-- ai/
|   +-- perception.py           PerceptionPipeline + scene analysis
|   +-- perception_service.py   PerceptionService(Thread) + camera
|   +-- geometry.py             CameraGeometry + distance estimation
|   +-- jetson_inference.py     (legacy standalone inference script)
|
+-- navigation/
|   +-- router/
|   |   +-- navigator.py        NavigationSystem facade
|   |   +-- models.py           Coord, RouteStep, RouteStatus, ProgressResult
|   |   +-- route_calculator.py A* routing on OSM graph
|   |   +-- route_tracker.py    GPS position -> route progress
|   |   +-- poi_finder.py       POI search by category
|   |   +-- osm_parser.py       .osm file -> routing graph
|   |   +-- geo_utils.py        Haversine, bearing calculations
|   |   +-- nav_config.py       NavConfig dataclass
|   |   +-- nav_logger.py       Route event logging
|   |   +-- map.osm             OpenStreetMap data
|   |   +-- __init__.py
|   +-- navigation_service.py   NavigationService(Thread) + 3-tier announce
|   +-- sensors/
|       +-- gps_reader.py       GPSReader (NEO-7M UART)
|       +-- mock_gps.py         MockGPSReader (fixed position)
|       +-- gps_filter.py       Median filter for GPS fixes
|       +-- __init__.py
|
+-- tts_stt/
|   +-- tts.py                  speak() / wait_until_done() / shutdown_tts()
|   +-- stt.py                  STTEngine (Vosk + MLX intent classifier)
|   +-- voice_policy.py         VoicePolicy (central speech gate)
|   +-- voice_commands.py       VoiceCommandHandler + NAV_KEYWORDS
|   +-- button_listener.py      ButtonListener (GPIO + mock stdin)
|   +-- slm_classifier.py       SLM intent classifier (scikit-learn)
```

---

## Running the System

### Full hardware (Jetson Nano)
```bash
cd src && python -m main.alas_main --model models/segmentation/alas_engine.trt
```

### Desktop test (no hardware at all)
```bash
cd src && python -m main.alas_main --mock --no-camera --bypass-stt --bypass-warmup
```

### Partial test (with webcam, no GPS/microphone)
```bash
cd src && python -m main.alas_main --mock --camera 0 --bypass-stt --bypass-warmup \
    --model models/segmentation/alas_engine.onnx
```

### CLI Flags

| Flag              | Effect                                               |
|-------------------|------------------------------------------------------|
| `--mock`          | MockGPS instead of real GPS, keyboard instead of GPIO|
| `--no-camera`     | Disable perception thread entirely                   |
| `--bypass-stt`    | Type commands via keyboard instead of microphone      |
| `--bypass-warmup` | Skip GPS/model warmup, jump straight to ACTIVE       |
| `--model PATH`    | Path to .trt/.engine/.onnx model file                |
| `--camera N`      | Camera device index (default 0)                      |
| `--fps N`         | Perception target FPS (default 2.0)                  |
| `--map PATH`      | Path to .osm map file                                |
| `--gps-port PORT` | GPS serial port (default /dev/ttyTHS1)               |

---

## Shutdown Sequence

Ordered shutdown is critical. The sequence:

```
1. button.join(1s)        Stop accepting new button presses
2. nav.stop_navigation()  Stop generating new nav messages
3. services.join(3s each) Wait for perception + navigation threads
4. voice.announce_shutdown + voice.flush()  Speak goodbye, drain TTS
5. voice.shutdown()       Kill TTS worker
6. gps.stop()             Close serial port
```

Reordering can cause: half-spoken alerts, dropped goodbye message,
blocked serial port, or zombie threads.
