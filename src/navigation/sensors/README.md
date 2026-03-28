# GPS Sensor Module

A threaded GPS reader for the NEO-7M module over UART. Reads NMEA sentences in the background, filters outliers, and provides a clean coordinate with health diagnostics. Designed for safety-critical use in the ALAS assistive navigation system.

---

## Project Structure

```
sensors/
├── gps_reader.py      # Threaded UART reader — public API (start here)
├── gps_filter.py      # Median-based outlier filter — used internally
├── test_gps.py        # Hardware test script with JSON logging
├── __init__.py        # Package exports
└── README.md          # This file
```

---

## Quick Start

```python
from navigation.sensors.gps_reader import GPSReader, GPSStatus
from navigation.router.models import Coord

gps = GPSReader(port="/dev/ttyTHS1", warmup_sec=60.0)
gps.start()

# In your main loop (every 2–3 seconds):
result = gps.get_coord()
if result:
    lat, lon, age = result
    if age < 3.0:
        nav.update(Coord(lat, lon))

# Check module health:
health = gps.get_health()
if health.status == GPSStatus.SERIAL_ERROR:
    tts.speak("GPS bağlantısı kesildi")

# Shutdown:
gps.stop()
```

Context manager is also supported:

```python
with GPSReader(port="/dev/ttyTHS1") as gps:
    coord = gps.get_coord()
```

---

## Core Modules & Key Functions

### `gps_reader.py` — `GPSReader`

The main class. Runs a background thread that continuously reads NMEA from serial, validates and buffers fixes, and exposes a thread-safe API.

| Method / Property | Returns | Description |
|---|---|---|
| `GPSReader(port, baudrate, ...)` | — | Create reader instance |
| `start()` | `None` | Open serial, start background thread |
| `stop()` | `None` | Stop thread, close serial |
| `get_coord()` | `(lat, lon, age_sec)` or `None` | Filtered coordinate + fix age |
| `get_health()` | `GPSHealth` | Full diagnostic report |
| `fix_count` | `int` | Active fixes in current window |
| `satellites` | `int` | Visible satellite count (from GGA) |
| `hdop` | `float` | Horizontal dilution of precision |

**Constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `port` | `/dev/ttyTHS1` | Serial port (Jetson Nano UART) |
| `baudrate` | `9600` | NEO-7M default baud rate |
| `timeout` | `2.0` | Serial read timeout (seconds) |
| `window` | `5.0` | Fix accumulation window (seconds) |
| `min_sats` | `4` | Minimum satellites to accept a fix |
| `max_hdop` | `5.0` | Maximum HDOP to accept a fix |
| `max_speed_kmh` | `15.0` | Speed limit — above is treated as GPS glitch |
| `warmup_sec` | `60.0` | Cold start stabilization period (seconds) |

---

### `GPSStatus` — Status Enum

| Value | Meaning | Recommended Action |
|---|---|---|
| `WARMING_UP` | Cold start in progress | TTS: "GPS hazırlanıyor, lütfen bekleyin" |
| `NO_FIX` | Warmup done, no satellite lock | TTS: "GPS sinyali aranıyor" |
| `LOW_ACCURACY` | Fix exists but low quality | TTS: "GPS sinyali zayıf, dikkatli olun" |
| `OK` | Normal operation | Continue navigation |
| `SERIAL_ERROR` | UART connection lost | TTS: "GPS bağlantısı kesildi" |

---

### `GPSHealth` — Diagnostic Dataclass

| Field | Type | Description |
|---|---|---|
| `status` | `GPSStatus` | Current module state |
| `satellites` | `int` | Visible satellite count |
| `hdop` | `float` | Horizontal precision (lower = better, <2.0 ideal) |
| `fix_age_sec` | `float` | Seconds since last valid fix (`inf` = never) |
| `fix_count` | `int` | Fixes in current window |
| `serial_ok` | `bool` | Serial port accessible |

---

### `gps_filter.py` — `filter_fixes()`

Pure function, no state. Used internally by `GPSReader.get_coord()`. Can also be used standalone.

```python
from navigation.sensors.gps_filter import filter_fixes

raw = [(39.9240, 32.8453), (39.9241, 32.8454), (39.9290, 32.8500)]
result = filter_fixes(raw, max_deviation_m=15.0)
# (39.92405, 32.84535) — outlier at index 2 is rejected
```

**Algorithm:**
1. Less than 3 fixes → simple average
2. Find median center point
3. Compute haversine distance from each fix to median
4. Reject fixes beyond `max_deviation_m` (default 15m)
5. Return average of remaining fixes

---

## Integration with NavigationSystem

```python
from navigation.sensors.gps_reader import GPSReader, GPSStatus
from navigation.router.models import Coord, RouteStatus
from navigation.router.navigator import NavigationSystem

gps = GPSReader(port="/dev/ttyTHS1")
gps.start()

nav = NavigationSystem("map.osm")
nav.start_navigation(Coord(39.924, 32.845), Coord(39.921, 32.853))

while True:
    health = gps.get_health()

    if health.status == GPSStatus.WARMING_UP:
        pass  # wait silently

    elif health.status == GPSStatus.SERIAL_ERROR:
        tts.speak("GPS bağlantısı kesildi")
        break

    elif health.status == GPSStatus.NO_FIX:
        tts.speak("GPS sinyali aranıyor")

    else:
        coord = gps.get_coord()
        if coord:
            lat, lon, age = coord
            if age < 3.0:
                result = nav.update(Coord(lat, lon))

                if result.status == RouteStatus.WAYPOINT_HIT:
                    tts.speak(result.message)
                elif result.status == RouteStatus.OFF_ROUTE:
                    tts.speak("Rotadan çıktınız")
                elif result.status == RouteStatus.FINISHED:
                    tts.speak("Hedefe ulaştınız")
                    break

    time.sleep(2)

gps.stop()
```

---

## Hardware Test

```bash
# Run from project src/ directory:
cd ~/ALAS_PROJECT/AI-Glasses/src

# Quick test (10s warmup, 60s duration):
python3 -m navigation.sensors.test_gps --warmup 10 --duration 60

# Custom port:
python3 -m navigation.sensors.test_gps --port /dev/ttyUSB0

# Skip raw NMEA check:
python3 -m navigation.sensors.test_gps --skip-raw
```

Test script runs in two phases:
1. **Raw NMEA test** — verifies serial connection and data flow
2. **GPSReader test** — reads filtered coordinates and logs to `gps_log.json`

---

## Safety Notes

- **Cold start:** NEO-7M can emit false "valid" fixes in the first 30–60 seconds. The `warmup_sec` parameter gates all output until stabilization. Do not reduce below 30s in production.
- **Fix age:** `age_sec > 3.0` means the user may have walked 5–7 meters since the last fix. Navigation should not trust stale coordinates.
- **Serial reconnect:** If the UART cable disconnects, the reader attempts automatic reconnection with exponential backoff. `GPSHealth.serial_ok` reports the current state.
- **Outlier filtering:** A single multipath spike (common near buildings) can shift the position by 50–100m. The median filter with 15m threshold catches these.

