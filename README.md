# ALAS – AI-Based Assistive Glasses for Visually Impaired Navigation

ALAS is a wearable edge-AI system designed to assist visually impaired individuals during outdoor pedestrian navigation. It delivers continuous environmental awareness and audio-guided routing through fully local, real-time processing — no internet connection required at runtime.

The system integrates an RGB camera, lightweight semantic perception via U-Net, OSM-based offline navigation, and a voice I/O interface, all running on an NVIDIA Jetson Nano embedded compute unit.

>*📌 **Note on Active Development:** ALAS is a dynamic, continuously evolving project. Because our architecture and hardware are frequently optimized, some details below may occasionally contain legacy information while we sync the documentation.*

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
| RGB Camera (IMX219) | Real-time visual frame acquisition |
| NEO-7M GPS Module | Outdoor localization via UART/NMEA |
| IMU (6-DOF) | Motion estimation and orientation tracking |
| Microphone + Speaker | Primary human-machine interface (STT/TTS) |
| Push Button | Controlled STT activation |
| Battery + Power Module | Portable, self-contained LiPo operation |

### 2. AI & Perception Layer

Transforms raw sensor data into actionable environmental intelligence.

- **Image Preprocessing:** RGB frame resizing, normalization, and noise filtering targeting ≤20 ms preprocessing latency
- **Semantic Perception:** U-Net-based object detection and scene segmentation, exported to ONNX and executed via ONNX Runtime with CUDA acceleration on the Jetson Nano
- **Obstacle Classification:** Detected objects are mapped to five functional risk groups — walkable surface, vehicle road, collision obstacle, fall hazard, dynamic hazard — each triggering a different audio response
- **Inverse Perspective Mapping (IPM):** Instead of relying on heavy depth sensors, the system uses the camera's fixed mounting geometry (height, tilt, FOV) to dynamically estimate obstacle distances directly from the 2D segmentation mask (`ai.geometry`)
- **Sensor Fusion:** GPS and IMU data are combined to maintain robust outdoor localization

### 3. Interaction & Navigation Layer

Converts perception outputs into safe, understandable guidance.

- **Offline Navigation:** Pre-processed OpenStreetMap (OSM) data loaded from local storage; route planning via A\* pathfinding on pedestrian graph networks
- **GPS Localization:** User position is continuously mapped to the nearest OSM node; route deviation is detected and triggers immediate recalculation
- **Local Avoidance (VFH):** A Vector Field Histogram logic translates the immediate segmented scene into angular avoidance sectors, guiding the user around localized blockages
- **TTS (Text-to-Speech):** All navigation instructions, obstacle warnings, and system events are delivered as speech — fully on-device, no cloud dependency
- **STT (Speech-to-Text):** Button-activated voice command recognition for commands such as `"nearest pharmacy"`, `"cancel route"`, `"status"` — processed locally
- **Operating Modes:** System transitions between Standby, Environment Awareness, Navigation, Obstacle Avoidance, and Error/Recovery modes based on sensor state and user input

The final objective is a portable, locally operating assistive device capable of perceiving the environment, interpreting scene structure, and guiding the user safely through audio feedback.

## Field-Test Data System (Black-Box Recorder)

To support robust outdoor testing on a screenless edge device, ALAS includes a built-in mission data recorder.
- **Zero-Overhead Opt-In:** Disabled by default for end-users. When launched with `--record`, a background thread silently logs the session.
- **System Telemetry:** Logs FPS, thermal throttling (SoC temps), GPS accuracy, and all suppressed/spoken voice decisions.
- **Crash-Resilient:** Uses append-only JSONL, bounded queues for OOM protection, and pre-flight SD card space checks.

---

## SLM Model Setup

The base model is large and is not committed to the repository. On first setup,
run the following (from the repository root):

```
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct model.safetensors --local-dir src/tts_stt/my_custom_slm/
```
---

## Project Structure

The repository follows one consistent layout, using the AI segmentation module
as the template: each module is a self-contained package with its own config
dataclass; unit tests live in a central `tests/` tree mirroring `src/`; demos
and benchmarks live in a parallel `eval/` tree.

```
src/
├── main/                        # Composition root / orchestration
│   ├── alas_main.py             # Thin main loop
│   ├── lifecycle.py             # Modes, signals, graceful shutdown
│   └── config.py                # ALASConfig — composes the per-module configs + CLI
├── ai/                          # Template module
│   ├── ai_config.py             # AIConfig (model, camera, geometry, dispatch cadences)
│   ├── perception.py            # Segmentation pipeline + scene analysis
│   ├── perception_service.py    # Camera + inference thread
│   ├── preprocessing.py, geometry.py, jetson_inference.py
│   └── dataset/                 # Dataset preparation scripts
├── navigation/
│   ├── navigation_service.py
│   ├── router/                  # Offline OSM routing (navigator, A*, POI, …)
│   │   └── nav_config.py        # NavConfig
│   ├── sensors/
│   │   ├── gps_reader.py, gps_filter.py
│   │   └── sensor_config.py     # GPSConfig + CANDIDATE_PORTS
│   └── local_planner/
│       ├── vfh.py               # Vector Field Histogram planner
│       └── planner_config.py    # VFHConfig
└── tts_stt/
    ├── stt.py, tts.py, voice_commands.py, voice_policy.py, button_listener.py
    ├── slm_classifier.py        # Hybrid intent classifier
    └── voice_config.py          # VoiceConfig

tests/        # Unit tests, mirrors src/ (see tests/README.md)
eval/         # Demos & benchmarks, mirrors src/ (see eval/README.md)
outputs/      # Artifacts: outputs/tests/<module>/ and outputs/eval/<module>/
```

Per-module config dataclasses (`AIConfig`, `VFHConfig`, `GPSConfig`, `NavConfig`,
`VoiceConfig`) hold each module's tunables; `ALASConfig` composes them and is the
single launch authority, so modules read `config.ai.model_path`,
`config.gps.port`, and so on.

### Key Design Decisions

**U-Net segmentation over YOLOv8:** U-Net delivers higher frame rate and lower end-to-end latency on the Jetson Nano compared to YOLOv8 architectures. For a safety-critical navigation system, responsiveness outweighs pixel-level mask precision.

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

---

## Jetson Performance Setup

The Waveshare Nano carrier cools worse than the NVIDIA devkit; without pinned
clocks the SoC thermal-throttles under sustained TensorRT load and inference
time silently doubles. Run once per boot (or install the systemd unit):

```bash
sudo bash scripts/jetson_setup.sh          # nvpmodel MAXN + jetson_clocks + fan + zram
sudo systemctl set-default multi-user.target   # one-time: headless boot (~600 MB RAM back)
bash scripts/build_trt_engine.sh check     # verify the engine is FP16-fast (~150-180 ms)
```

Two in-app guards complement this (see `src/ai/ai_config.py`):

- **Thermal guard** — above `thermal_throttle_c` the perception loop degrades
  deliberately to `thermal_min_fps` instead of letting DVFS halve the FPS silently.
- **Adaptive FPS** — calm + still → 1 FPS, normal walking → 2 FPS, closing
  hazard → `fps_alert`; the thermal guard always wins.

Camera FOV/tilt must be calibrated once on the final rig for accurate distance
words: `python3 eval/ai/calibrate_camera.py --height 1.65 --points "1.0:R1,3.0:R2,5.0:R3"`.

## Voice Commands (local queries)

Besides POI navigation ("en yakın eczane"), the PTT button understands:

| Command | Action |
|---|---|
| "neredeyim" | Nearest named road from the offline OSM graph + remaining route distance |
| "evi kaydet" / "eve git" | Bookmark the current GPS fix / route back to it (`outputs/saved_places.json`) |
| "durum" | Spoken health summary: GPS quality, SoC temperature, route state |
| "oku" / "tabela" | OCR the camera frame and read signs aloud (needs `tesseract-ocr-tur`) |

Path-keeping corrections ("hafif sola/sağa") play as **panned stereo earcons**
(left ear = step left) instead of repeated speech; regenerate the tones with
`python3 scripts/generate_earcons.py`. Obstacle directions use the blind-navigation
clock convention ("saat iki yönünde araç") and short distances are spoken in steps.
