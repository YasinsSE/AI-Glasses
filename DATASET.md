# ALAS – AI-Based Assistive Glasses for Visually Impaired Navigation

ALAS is a wearable edge-AI system designed to assist visually impaired individuals during outdoor pedestrian navigation. It delivers continuous environmental awareness and audio-guided routing through fully local, real-time processing — no internet connection required at runtime.

The system integrates an RGB camera, lightweight semantic perception via YOLOv8, depth-based obstacle analysis, OSM-based offline navigation, and a voice I/O interface, all running on an NVIDIA Jetson Nano embedded compute unit.

---

## Problem Definition

Traditional mobility aids — white canes, guide dogs, or smartphone navigation apps — lack real-time obstacle detection and provide no fine-grained understanding of the immediate environment. These limitations significantly reduce safety and independence, particularly in unfamiliar or dense urban settings.

ALAS addresses this by running the full perception, navigation, and audio interaction stack directly on wearable embedded hardware, enabling safe pedestrian mobility through fully offline, low-latency processing.

---

## System Architecture

The system is structured into three tightly coupled layers.

### 1. Hardware & Edge Computing Layer

The physical foundation of the device, centered on the **NVIDIA Jetson Nano** for CUDA-accelerated AI inference.

| Component | Role |
|---|---|
| NVIDIA Jetson Nano | Main compute unit — CUDA inference, sensor orchestration |
| RGB Camera (IMX219) | Synchronized color and depth frame acquisition |
| NEO-7M GPS Module | Outdoor localization via UART/NMEA |
| IMU (6-DOF) | Motion estimation and orientation tracking |
| Microphone + Speaker | Primary human-machine interface (STT/TTS) |
| Push Button | Controlled STT activation |
| Battery + Power Module | Portable, self-contained operation |

### 2. AI & Perception Layer

Transforms raw sensor data into actionable environmental intelligence.

- **Image Preprocessing:** RGB frame resizing, normalization, and noise filtering targeting ≤20 ms preprocessing latency
- **Semantic Perception:** YOLOv8-based object detection and scene segmentation, exported to ONNX and executed via ONNX Runtime with CUDA acceleration on the Jetson Nano
- **Obstacle Classification:** Detected objects are mapped to five functional risk groups — walkable surface, vehicle road, collision obstacle, fall hazard, dynamic hazard — each triggering a different audio response
- **Depth Fusion:** Depth channel from the RGB-D camera is fused with segmentation output to estimate obstacle distance and severity; warnings are prioritized by proximity (critical threshold: ≤2 m)
- **Sensor Fusion:** GPS and IMU data are combined to maintain robust outdoor localization

### 3. Interaction & Navigation Layer

Converts perception outputs into safe, understandable guidance.

- **Offline Navigation:** Pre-processed OpenStreetMap (OSM) data loaded from local storage; route planning via A\* pathfinding on pedestrian graph networks
- **GPS Localization:** User position is continuously mapped to the nearest OSM node; route deviation is detected and triggers immediate recalculation
- **TTS (Text-to-Speech):** All navigation instructions, obstacle warnings, and system events are delivered as speech — fully on-device, no cloud dependency
- **STT (Speech-to-Text):** Button-activated voice command recognition for commands such as `"nearest pharmacy"`, `"cancel route"`, `"status"` — processed locally
- **Operating Modes:** System transitions between Standby, Environment Awareness, Navigation, Obstacle Avoidance, and Error/Recovery modes based on sensor state and user input

---

## Software Modules

```
src/
├── ai/
│   ├── dataset_yolo_format.py   # Dataset preparation and class mapping
│   └── ...                      # Model training, ONNX export
├── navigation/
│   └── router/
│       ├── navigator.py         # Public navigation API (NavigationSystem)
│       ├── route_calculator.py  # A* pathfinding → step-by-step directions
│       ├── route_tracker.py     # GPS progress tracking state machine
│       ├── poi_finder.py        # Nearest POI search (pharmacy, ATM, etc.)
│       ├── osm_parser.py        # .osm file → in-memory routing graph
│       ├── geo_utils.py         # Haversine distance, bearing, turn logic
│       ├── models.py            # Shared data types (Coord, RouteStep, etc.)
│       └── nav_config.py        # Tunable navigation parameters
├── sensors/
│   ├── gps_reader.py            # UART NMEA reader with reconnect & health API
│   └── gps_filter.py            # Median-based GPS fix filtering
└── ...
```

### Key Design Decisions

**U-Net over YOLOv8   segmentation:** YOLOv8 delivers higher frame rate and lower end-to-end latency on the Jetson Nano compared to U-Net architectures. For a safety-critical navigation system, responsiveness outweighs pixel-level mask precision.

**A\* over Dijkstra for routing:** Navigation module uses A\* with a haversine heuristic for faster route convergence on pedestrian OSM graphs. Routes are cached in memory after initial computation to avoid redundant CPU cycles.

**Offline-only operation:** All models, OSM graph data, and voice assets are stored locally. There is no network dependency at runtime — ensuring availability, privacy, and deterministic behavior.

**Median-based GPS filtering:** IQR-based filtering is statistically unreliable at small fix windows (~5 samples) and collapses to zero displacement when stationary. A median-based fixed-threshold filter is used instead for robustness in both static and dynamic conditions.

---

## Obstacle Classification Schema

Detected scene elements are mapped to functional risk groups that drive audio feedback:

| Group | Label | User Alert |
|---|---|---|
| 0 | Walkable surface | No alert (safe path confirmed) |
| 1 | Vehicle road | "Stay back — vehicle road ahead" |
| 2 | Collision obstacle | "Obstacle ahead" + direction (left/center/right) |
| 3 | Fall hazard | "Caution — ground hazard ahead" |
| 4 | Dynamic hazard | "Moving obstacle detected" |

---

## Operational Constraints

- Target environment: outdoor pedestrian scenarios (sidewalks, crosswalks, open urban areas)
- Complex indoor environments (malls, multi-floor buildings) are out of scope for the current prototype
- Single-user, single-device design; no cloud services, remote monitoring, or multi-user coordination
- Voice interface supports a defined command set in one primary language; multilingual support is deferred
- OSM data must be pre-processed and stored on-device before field use
- GPS cold-start warmup period (~60 s) is enforced before fixes are trusted for navigation

---

## Hardware Notes

- Jetson Nano serial device may enumerate as `/dev/ttyAMA10` depending on OS configuration — verify before GPS initialization
- IMX219 camera requires CUDA-capable CSI configuration on Jetson; use `nvgstcapture` or `gstreamer` pipelines for frame acquisition
- Power budget must account for sustained Jetson Nano GPU load; thermal management (fan) is required for continuous outdoor operation
