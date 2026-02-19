# OSM Navigation System

A lightweight, modular walking navigation system built in Python using OpenStreetMap data. It calculates routes with A\*, tracks GPS progress in real time, and finds nearby points of interest (POI) from a local `.osm` file.

---

## Project Structure

```
├── main.py               # Entry point / GPS simulation loop
├── navigator.py          # Public-facing API (start here)
├── route_calculator.py   # A* pathfinding → step-by-step directions
├── route_tracker.py      # GPS progress tracking state machine
├── poi_finder.py         # Find nearest POI (pharmacy, ATM, etc.)
├── nav_logger.py         # Save/load routes and session logs as JSON
├── osm_parser.py         # Parse .osm file → in-memory routing graph
├── geo_utils.py          # Haversine distance, bearing, turn instructions
├── models.py             # Shared data classes (Coord, RouteStep, etc.)
└── nav_config.py         # All tunable settings in one place
```

---

## Quick Start

```python
from models import Coord
from nav_config import NavConfig
from navigator import NavigationSystem

config = NavConfig(
    waypoint_threshold_m=15.0,
    off_route_threshold_m=40.0,
    log_dir="logs",
)

nav = NavigationSystem("map.osm", config=config)

origin      = Coord(39.924, 32.845)
destination = Coord(39.921, 32.853)

# Option 1 — Navigate to a specific coordinate
success, msg = nav.start_navigation(origin, destination)

# Option 2 — Navigate to nearest POI
success, msg, poi = nav.navigate_to_nearest(origin, "pharmacy")

# GPS loop
if success:
    result = nav.update(current_gps_coord)
    print(result.status, result.message)
```

---

## Core Modules & Key Functions

### `navigator.py` — `NavigationSystem`
The only class you need to interact with. Everything else is handled internally.

| Method | Description |
|---|---|
| `NavigationSystem(osm_path, config)` | Load map and initialize all modules |
| `start_navigation(origin, destination)` | Calculate route and begin tracking |
| `navigate_to_nearest(position, category)` | Find nearest POI and navigate to it |
| `find_nearby(position, category, radius_m)` | List nearby POIs without starting navigation |
| `update(position)` | Feed a GPS coordinate, returns `ProgressResult` |
| `stop_navigation()` | End the current session |
| `list_poi_categories()` | List all supported POI category names |

---

### `models.py` — Shared Data Types

| Class | Fields |
|---|---|
| `Coord` | `lat`, `lon` |
| `RouteStep` | `step_id`, `text`, `location`, `action`, `distance_meters`, `road_name` |
| `ProgressResult` | `status`, `message`, `distance_to_next`, `current_step` |
| `RouteStatus` | `INACTIVE`, `PROGRESSING`, `WAYPOINT_HIT`, `OFF_ROUTE`, `FINISHED` |

---

### `nav_config.py` — `NavConfig`
All settings in one place. Pass this to `NavigationSystem`.

| Setting | Default | Description |
|---|---|---|
| `walking_speed_kmh` | `5.0` | Used for ETA and A* heuristic |
| `waypoint_threshold_m` | `15.0` | Distance to mark a waypoint as reached |
| `off_route_threshold_m` | `40.0` | Distance before off-route is triggered |
| `log_dir` | `"."` | Directory for JSON log files |
| `route_filename` | `"active_route.json"` | Saved route file name |

---

### `poi_finder.py` — `POIFinder`
Used internally by `NavigationSystem`. Can also be used standalone.

```python
from poi_finder import POIFinder
from models import Coord

finder = POIFinder("map.osm")
result = finder.find_nearest(Coord(39.924, 32.845), category="pharmacy")
all_results = finder.find_all(Coord(39.924, 32.845), category="atm", radius_m=500)
```

**Supported POI categories:**

`atm`, `bakkal`, `banka`, `benzin`, `cafe`, `eczane`, `fuel`, `hastane`, `hospital`, `itfaiye`, `kafe`, `klinik`, `market`, `okul`, `otopark`, `park`, `pharmacy`, `polis`, `restaurant`, `restoran`, `supermarket`

---

### `geo_utils.py` — Utility Functions
Pure math, no dependencies. Can be imported anywhere.

| Function | Description |
|---|---|
| `haversine_distance(lat1, lon1, lat2, lon2)` | Great-circle distance in metres |
| `calculate_bearing(lat1, lon1, lat2, lon2)` | Forward azimuth in degrees [0, 360) |
| `get_turn_instruction(bearing_diff)` | Returns e.g. `"Turn right"`, `"Go straight"` |

---

## GPS Loop Pattern

```python
for gps_position in your_gps_source:
    result = nav.update(gps_position)

    if result.status == RouteStatus.WAYPOINT_HIT:
        print(result.message)            # next instruction

    elif result.status == RouteStatus.OFF_ROUTE:
        print("Off route — recalculate") # trigger reroute if needed

    elif result.status == RouteStatus.FINISHED:
        print("Destination reached")
        break
```

---

## Requirements

- Python 3.8+
- No third-party dependencies — standard library only (`xml.sax`, `heapq`, `math`, `json`)
- A valid `.osm` map file (export from [openstreetmap.org](https://www.openstreetmap.org))
