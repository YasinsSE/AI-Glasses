# ALAS Architecture & Personal Study Guide

**Core Concept - Concurrency:** ALAS runs three long-lived threads (`perception`, `navigation`, `button/voice`) that never call each other directly. They coordinate through shared objects: `ModeManager` (what mode are we in?), `stop_event` (time to quit?), and `VoicePolicy` (who's allowed to talk?).

---

## 0. Core Architecture

| File Path | Key Concepts & Study Notes |
| :--- | :--- |
| [**src/main/alas_main.py**](./src/main/alas_main.py) | **The whole system on one page.** It's a "recipe": build config → voice → gps → nav → services → start threads → wait → shutdown. |
| [**src/main/lifecycle.py**](./src/main/lifecycle.py) | **The state machine.** `SystemMode`: WARMUP / ACTIVE / SLEEP / SHUTDOWN. Handles signal handling, `await_ready`, and the ordered shutdown. This is where "modes" live. |
| [**src/main/config.py**](./src/main/config.py) | **The composition root.** See how one `ALASConfig` composes per-module configs (ai, vfh, gps, nav, voice) and parses CLI. |
| [**src/data_models.py**](./src/data_models.py) | **Common currency.** The `Frame` data type passed around the threads. |

---

## 1. Computer Vision & AI (Perception)

| File Path | Key Concepts & Study Notes |
| :--- | :--- |
| [**src/ai/preprocessing.py**](./src/ai/preprocessing.py) | Resize → normalize [0,1] → HWC to CHW → add batch dim. **Concept:** What a neural net actually expects as input (tensor layout, normalization). |
| [**src/ai/geometry.py**](./src/ai/geometry.py) | `pixel_to_ground_distance()` turns a pixel row into "N metres away." **Concept:** Inverse perspective mapping / ground-plane projection. Uses trigonometry (camera height, tilt, vertical FOV) to get distance from a 2D mask with no depth sensor. *Master the tan(θ) derivation; it's a classic interview trick.* |
| [**src/ai/jetson_inference.py**](./src/ai/jetson_inference.py) | **The inference backend** (TensorRT/ONNX). **Concept:** How a trained model is deployed (engine loading, I/O binding) vs trained. This is the edge-AI differentiator. |
| [**src/ai/perception.py**](./src/ai/perception.py) | **The big one (~700 lines).** Read in sections: `ClassID` enum → `analyse_scene()` (per-class pixel ratios) → walkable-overlap gating → `generate_alerts()` (priority + cooldown) → `generate_path_guidance()`. **Concept:** Turning a raw segmentation mask into semantic decisions. |
| [**src/ai/perception_service.py**](./src/ai/perception_service.py) | **The camera loop** that ties 1–4 together. Open camera → read frame → `pipeline.process()` → `_dispatch()` (filter/dedupe/speak). **Concept:** Real-time loop with FPS capping and the three "gates" that decide whether to even run inference. |

* **Hands-on:** `eval/ai/image_seg_demo.py` runs the pipeline on a single image and saves the overlay. Run it, open `perception.py`, and match each printed number to the code.

---

## 2. Navigation (Global A* + Local VFH)

| File Path | Key Concepts & Study Notes |
| :--- | :--- |
| [**src/navigation/router/geo_utils.py**](./src/navigation/router/geo_utils.py) | Haversine distance + bearing. **Concept:** Great-circle math on lat/lon. |
| [**src/navigation/router/osm_parser.py**](./src/navigation/router/osm_parser.py) | Turns an `.osm` XML file into a routable graph (nodes + edges, walkable road types). **Concept:** Building a graph from real map data. |
| [**src/navigation/router/route_calculator.py**](./src/navigation/router/route_calculator.py) | A* pathfinding. **Concept:** `f = g + h` where `h` is the haversine heuristic. Why that heuristic is *admissible* (never overestimates), ensuring A* finds the optimal path. The algorithmic heart. |
| [**src/navigation/router/route_tracker.py**](./src/navigation/router/route_tracker.py) | The progress state machine (WAYPOINT_HIT / OFF_ROUTE / FINISHED). **Concept:** Matching live GPS to the planned route, detecting deviation. |
| [**src/navigation/router/poi_finder.py**](./src/navigation/router/poi_finder.py) | "Nearest pharmacy" search. **Concept:** SAX-parsing OSM for amenities. |
| [**src/navigation/router/navigator.py**](./src/navigation/router/navigator.py) | The public `NavigationSystem` API that wraps it all. Read this to see the clean interface the system uses. |
| [**src/navigation/local_planner/vfh.py**](./src/navigation/local_planner/vfh.py) | **Concept:** Vector Field Histogram adapted to a segmentation mask. Folds the near-field mask into angular sectors, weights by cost × inverse-distance, builds a polar histogram, picks the lowest-cost open sector. Study `build_cost_grid()`, `_build_histogram()`, and `should_activate()`. |
| [**src/navigation/navigation_service.py**](./src/navigation/navigation_service.py) | **The Glue:** Thread that polls GPS, feeds `NavigationSystem`, and decides when to announce (three-tier approach / long-stretch / silence logic). |

* **Hands-on:** `eval/navigation/local_planner/vfh_demo.py` visualizes VFH on an image. Run it, watch which sector it picks.

---

## 3. Embedded / Peripherals / PIN Configuration

| File Path | Key Concepts & Study Notes |
| :--- | :--- |
| [**src/navigation/sensors/gps_reader.py**](./src/navigation/sensors/gps_reader.py) | **UART / Serial:** Background thread reading NMEA sentences over `serial.Serial`. Study: `_checksum_ok()` (XOR checksum), `_parse_gga`/`_parse_rmc`, reconnect logic, and `_to_decimal`. *This is real-world serial protocol parsing.* |
| [**src/navigation/sensors/gps_filter.py**](./src/navigation/sensors/gps_filter.py) | **Sensor filtering:** Median-based outlier rejection. **Concept:** Why raw GPS is noisy and how to clean it (median beats IQR at small sample sizes). |
| [**src/tts_stt/button_listener.py**](./src/tts_stt/button_listener.py) | **GPIO / PIN config:** `Jetson.GPIO`, `setmode(BCM)`, `setup(pin, IN, pull_up_down=PUD_UP)`, active-low reading, and software debounce. **Concepts:** Pull-up resistors, active-low logic, contact bounce. |
| [**src/navigation/sensors/sensor_config.py**](./src/navigation/sensors/sensor_config.py) | Where the serial port/baud and candidate ports live. |

* **Bridge to STM32 Knowledge:** On Jetson you write `GPIO.setup(18, IN, PUD_UP)`; on STM32 the same concept is configuring `GPIOx->MODER`, `GPIOx->PUPDR`.

---

## 4. AI in the Voice Layer (STT / TTS / Intent)

| File Path | Key Concepts & Study Notes |
| :--- | :--- |
| [**src/tts_stt/stt.py**](./src/tts_stt/stt.py) | Speech-to-text via Vosk (offline ASR). Audio capture format (sample rate, chunks), recognizer usage. |
| [**src/tts_stt/slm_classifier.py**](./src/tts_stt/slm_classifier.py) | **ML intent classification:** TF-IDF + Logistic Regression (`sklearn`) mapping a spoken command → intent (nav / system / general), with a confidence threshold. **Concept:** Lightweight classical-ML text classifier. |
| [**src/tts_stt/tts.py**](./src/tts_stt/tts.py) | Text-to-speech via `pyttsx3`, run in a subprocess + worker thread so it never blocks. |
| [**src/tts_stt/voice_policy.py**](./src/tts_stt/voice_policy.py) | **The single most elegant file in the project.** The one chokepoint all speech passes through. **Concept:** Centralizing a shared resource (the speaker) to enforce priority and post-nav silence windows, preventing overlapping audio. |
| [**src/tts_stt/voice_commands.py**](./src/tts_stt/voice_commands.py) | Orchestrates: Button → STT → Classify → Act. |

---

## 5. The Three End-to-End Traces (Mastery Level)

1. **A camera frame → a spoken warning:**
   `perception_service._loop` (reads frame) → `pipeline.process()` (`perception.py`) → `analyse_scene` → `generate_alerts` + `geometry.pixel_to_ground_distance` → `_dispatch` → `voice_policy.say_obstacle` → `tts.speak`.
2. **A GPS fix → a turn instruction:**
   `gps_reader._read_loop` (parses NMEA) → `navigation_service.run` (polls `get_coord`) → `NavigationSystem.update` (`navigator.py`) → `route_tracker` status → `_announce` → `voice_policy.say_nav`.
3. **A button press → a route started:**
   `button_listener._gpio_loop` (GPIO pin) → `voice_commands.handle_press` → `stt.listen` → `slm_classifier.predict` → `_handle_navigation` → `poi_finder` + `route_calculator` (A*) → navigation begins.