#!/usr/bin/env python3
"""
ALAS — AI-Based Smart Glasses Main Loop
========================================
Orchestrates three subsystems running concurrently:

  1. Perception  : Camera → preprocess → TensorRT inference → scene analysis → TTS alerts
  2. Navigation  : GPS tracking → route progress → TTS turn instructions
  3. Voice Input : Button press → STT → intent classification → action

Architecture:
  - Each subsystem runs in its own thread
  - TTS queue (from tts.py) serialises all audio output — no overlapping speech
  - A shared threading.Event coordinates graceful shutdown
  - Perception uses PerceptionPipeline (preprocess → TRT → postprocess → scene → alerts)

Run on Jetson Nano:
    cd src && python -m main.alas_main --model models/segmentation/alas_engine.trt

For desktop testing (no GPIO / GPS / camera):
    cd src && python -m main.alas_main --mock --no-camera
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time

# Ensure `src/` is on the path so all sub-packages resolve
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from main.config import ALASConfig
from ai.perception import PerceptionPipeline
from navigation.router import (
    NavigationSystem,
    Coord,
    NavConfig,
    RouteStatus,
)
from navigation.sensors import GPSReader, GPSStatus
from tts_stt.tts import speak, shutdown_tts, wait_until_done
from tts_stt.stt import STTEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ALAS")


# ═══════════════════════════════════════════════════════════════════
#  PERCEPTION THREAD
#  Camera → PerceptionPipeline → TTS obstacle alerts
# ═══════════════════════════════════════════════════════════════════

class PerceptionThread(threading.Thread):
    """
    Captures frames at a controlled rate, runs the full perception pipeline,
    and queues TTS alerts for any detected hazards.

    The pipeline internally handles:
      preprocess → TensorRT/ONNX inference → postprocess → scene analysis → alert generation
    """

    def __init__(self, config: ALASConfig, stop_event: threading.Event):
        super().__init__(name="Perception", daemon=True)
        self.config = config
        self._stop = stop_event
        self._pipeline: PerceptionPipeline | None = None
        # Pause perception while TTS is speaking a nav instruction
        # (avoids "obstacle ahead" interrupting "turn right in 50m")
        self.pause_event = threading.Event()

    def run(self) -> None:
        import cv2

        # ── Load the model (heavy — done inside thread to not block main) ──
        logger.info("[Perception] Loading AI model...")
        self._pipeline = PerceptionPipeline(
            model_path=self.config.model_path,
            input_h=self.config.model_input_h,
            input_w=self.config.model_input_w,
        )

        # ── Open camera ──
        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        if not cap.isOpened():
            logger.error("[Perception] Cannot open camera — thread exiting.")
            speak("Kamera açılamadı.")
            return

        frame_interval = 1.0 / self.config.perception_fps
        logger.info(
            f"[Perception] Pipeline ready — "
            f"~{self.config.perception_fps} FPS (1 frame every {frame_interval:.1f}s)"
        )

        # ── Main capture-process loop ──
        while not self._stop.is_set():
            # If paused (e.g. during navigation speech), wait
            if self.pause_event.is_set():
                self._stop.wait(timeout=0.5)
                continue

            t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                logger.warning("[Perception] Frame grab failed — skipping.")
                self._stop.wait(timeout=0.2)
                continue

            # ── Run full pipeline ──
            result = self._pipeline.process(frame)

            # ── Speak alerts ──
            for alert in result.alerts:
                speak(alert)

            # Log scene summary periodically
            if not result.scene.is_safe:
                logger.info(
                    f"[Perception] Hazard: {result.scene.dominant_hazard} | "
                    f"walkable: {result.scene.walkable_ratio:.0%} | "
                    f"inf: {result.inference_ms:.0f}ms total: {result.total_ms:.0f}ms"
                )

            # ── Throttle to target FPS ──
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                self._stop.wait(timeout=sleep_time)

        cap.release()
        logger.info("[Perception] Stopped.")


# ═══════════════════════════════════════════════════════════════════
#  NAVIGATION THREAD
#  GPS polling → route progress → TTS turn instructions
# ═══════════════════════════════════════════════════════════════════

class NavigationThread(threading.Thread):
    """
    Reads GPS at a fixed interval, feeds positions to the NavigationSystem,
    and speaks route instructions when waypoints are hit or the user goes off-route.
    """

    def __init__(
        self,
        config: ALASConfig,
        nav: NavigationSystem,
        gps: GPSReader,
        stop_event: threading.Event,
        perception_pause: threading.Event | None = None,
    ):
        super().__init__(name="Navigation", daemon=True)
        self.config = config
        self.nav = nav
        self.gps = gps
        self._stop = stop_event
        self._perception_pause = perception_pause

        # Avoid repeating the same instruction
        self._last_spoken: str = ""
        # Throttle PROGRESSING messages (don't say "120m to next" every 4 seconds)
        self._last_progress_time: float = 0.0

    def _speak_nav(self, text: str) -> None:
        """Speak a navigation message, briefly pausing perception alerts."""
        if self._perception_pause:
            self._perception_pause.set()
        speak(text)
        wait_until_done()
        if self._perception_pause:
            self._perception_pause.clear()

    def run(self) -> None:
        logger.info("[Navigation] GPS update loop started.")

        while not self._stop.is_set():
            # No active route → sleep and check again
            if not self.nav.is_active:
                self._stop.wait(timeout=self.config.gps_update_interval)
                continue

            fix = self.gps.get_coord()
            if fix is None:
                health = self.gps.get_health()
                if health.status == GPSStatus.WARMING_UP:
                    logger.debug("[Navigation] GPS warming up...")
                elif health.status == GPSStatus.NO_FIX:
                    logger.debug("[Navigation] No GPS fix yet.")
                self._stop.wait(timeout=self.config.gps_update_interval)
                continue

            lat, lon, _age = fix
            result = self.nav.update(Coord(lat, lon))
            now = time.monotonic()

            # ── Speak navigation events ──
            if result.message and result.message != self._last_spoken:

                if result.status == RouteStatus.WAYPOINT_HIT:
                    self._speak_nav(result.message)
                    self._last_spoken = result.message

                elif result.status == RouteStatus.OFF_ROUTE:
                    self._speak_nav("Rotadan çıktınız. Lütfen geri dönün.")
                    self._last_spoken = result.message

                elif result.status == RouteStatus.FINISHED:
                    self._speak_nav("Hedefinize ulaştınız, iyi günler.")
                    self._last_spoken = result.message

                elif result.status == RouteStatus.PROGRESSING:
                    # Only announce distance every progress_announce_interval seconds
                    if (now - self._last_progress_time) > self.config.progress_announce_interval:
                        if result.distance_to_next is not None:
                            dist = int(result.distance_to_next)
                            speak(f"Hedefe {dist} metre.")
                        self._last_progress_time = now

            self._stop.wait(timeout=self.config.gps_update_interval)

        logger.info("[Navigation] Stopped.")


# ═══════════════════════════════════════════════════════════════════
#  VOICE COMMAND HANDLER
#  Button press → STT listen → intent classification → action
# ═══════════════════════════════════════════════════════════════════

class VoiceCommandHandler:
    """
    Handles a single voice command session triggered by a button press:
      1. Play "listening" prompt
      2. STT captures speech → text
      3. MLX intent classifier → navigation / system_command / general
      4. Execute the appropriate action
    """

    # Spoken keyword → POI category mapping
    NAV_KEYWORDS: dict[str, str] = {
        "metro":    "metro",
        "metroy":   "metro",
        "eczane":   "eczane",
        "hastane":  "hastane",
        "hospital": "hastane",
        "market":   "market",
        "bakkal":   "bakkal",
        "restoran": "restoran",
        "kafe":     "kafe",
        "cafe":     "cafe",
        "park":     "park",
        "atm":      "atm",
        "banka":    "banka",
        "okul":     "okul",
        "benzin":   "benzin",
    }

    def __init__(
        self,
        config: ALASConfig,
        nav: NavigationSystem,
        gps: GPSReader,
        stt: STTEngine,
        stop_event: threading.Event,
    ):
        self.config = config
        self.nav = nav
        self.gps = gps
        self.stt = stt
        self._stop = stop_event

    def handle_button_press(self) -> None:
        """Called when the physical button is pressed (or Enter in mock mode)."""
        if self._stop.is_set():
            return

        speak("Sizi dinliyorum.")
        wait_until_done()

        text = self.stt.listen(
            timeout_sec=self.config.stt_listen_timeout,
            silence_sec=self.config.stt_silence_sec,
        )

        if not text:
            speak("Anlayamadım, tekrar deneyin.")
            return

        # Intent classification via fine-tuned SLM
        intent, conf = self.stt._slm.predict(text)
        logger.info(f'[Voice] "{text}" → intent={intent} conf={conf}')

        if intent == "navigation":
            self._handle_navigation(text)
        elif intent == "system_command":
            self._handle_system_command(text)
        else:
            speak("Anlaşıldı.")

    # ── Navigation intent ────────────────────────────────────────

    def _handle_navigation(self, text: str) -> None:
        """Extract destination from spoken text and start route navigation."""
        # If a route is already active, ask if they want to cancel
        if self.nav.is_active:
            speak("Mevcut rota iptal ediliyor, yeni rota hesaplanıyor.")
            self.nav.stop_navigation()

        fix = self.gps.get_coord()
        if fix is None:
            speak("GPS sinyali bulunamadı. Lütfen açık alanda deneyin.")
            return

        lat, lon, _ = fix
        position = Coord(lat, lon)

        category = self._extract_category(text)
        if not category:
            speak("Nereye gitmek istediğinizi anlayamadım. Lütfen tekrar söyleyin.")
            return

        speak(f"En yakın {category} aranıyor.")
        wait_until_done()

        success, msg, poi = self.nav.navigate_to_nearest(position, category)

        if success and poi:
            speak(
                f"En yakın {category} {int(poi.distance_m)} metre uzakta. "
                f"Rota hazır, yönlendirme başlıyor."
            )
        else:
            speak(f"Yakınlarda {category} bulunamadı.")

    # ── System command intent ────────────────────────────────────

    def _handle_system_command(self, text: str) -> None:
        """Handle system-level voice commands."""
        shutdown_words = ["kapat", "durdur", "bitir", "kapa"]
        stop_nav_words = ["rota", "navigasyon", "iptal"]

        text_lower = text.lower()

        # Check for navigation stop first
        if any(w in text_lower for w in stop_nav_words):
            if self.nav.is_active:
                self.nav.stop_navigation()
                speak("Navigasyon durduruldu.")
            else:
                speak("Aktif bir rota yok.")
            return

        # System shutdown
        if any(w in text_lower for w in shutdown_words):
            speak("Sistem kapatılıyor, iyi günler.")
            wait_until_done()
            self._stop.set()
        else:
            speak("Sistem komutu alındı.")

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_category(self, text: str) -> str | None:
        """Find a POI category keyword in spoken text."""
        text_lower = text.lower()
        for keyword, category in self.NAV_KEYWORDS.items():
            if keyword in text_lower:
                return category
        return None


# ═══════════════════════════════════════════════════════════════════
#  BUTTON LISTENER — GPIO or keyboard fallback
# ═══════════════════════════════════════════════════════════════════

def start_button_listener(
    config: ALASConfig,
    handler: VoiceCommandHandler,
    stop_event: threading.Event,
    mock: bool = False,
) -> threading.Thread:
    """
    Start a daemon thread that listens for button presses.
    - Jetson Nano: Jetson.GPIO, active-low with pull-up
    - Mock mode: keyboard Enter key
    """

    def _gpio_loop() -> None:
        import Jetson.GPIO as GPIO

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        logger.info(f"[Button] GPIO pin {config.button_pin} ready (active-low).")

        try:
            while not stop_event.is_set():
                if GPIO.input(config.button_pin) == GPIO.LOW:
                    handler.handle_button_press()
                    # Debounce: wait for release + cooldown
                    time.sleep(config.button_debounce_ms / 1000.0)
                    while GPIO.input(config.button_pin) == GPIO.LOW and not stop_event.is_set():
                        time.sleep(0.05)
                time.sleep(0.05)
        finally:
            GPIO.cleanup(config.button_pin)

    def _keyboard_loop() -> None:
        logger.info("[Button] Mock mode — press ENTER to simulate button press.")
        while not stop_event.is_set():
            try:
                input()
                if not stop_event.is_set():
                    handler.handle_button_press()
            except EOFError:
                break

    target = _keyboard_loop if mock else _gpio_loop
    t = threading.Thread(target=target, name="ButtonListener", daemon=True)
    t.start()
    return t


# ═══════════════════════════════════════════════════════════════════
#  MOCK GPS (for desktop testing without hardware)
# ═══════════════════════════════════════════════════════════════════

class MockGPSReader:
    """Returns a fixed coordinate — use for testing without a real GPS module."""

    def __init__(self, lat: float = 39.9245, lon: float = 32.8465):
        self._lat = lat
        self._lon = lon

    def start(self) -> None:
        logger.info(f"[MockGPS] Fixed position: ({self._lat}, {self._lon})")

    def stop(self) -> None:
        pass

    def get_coord(self):
        return (self._lat, self._lon, 0.0)

    def get_health(self):
        from navigation.sensors import GPSHealth, GPSStatus
        return GPSHealth(
            status=GPSStatus.OK, satellites=10, hdop=1.0,
            fix_age_sec=0.0, fix_count=5, serial_ok=True,
        )


# ═══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ALAS — AI Smart Glasses System")
    parser.add_argument("--model",    default=None, help="Path to .trt/.engine or .onnx model")
    parser.add_argument("--camera",   type=int, default=0, help="Camera device index")
    parser.add_argument("--fps",      type=float, default=None, help="Perception target FPS")
    parser.add_argument("--map",      default=None, help="Path to .osm map file")
    parser.add_argument("--gps-port", default=None, help="GPS serial port")
    parser.add_argument("--mock",     action="store_true", help="Desktop test mode (no GPIO/GPS)")
    parser.add_argument("--no-camera", action="store_true", help="Disable perception thread")
    args = parser.parse_args()

    config = ALASConfig()

    # CLI overrides
    if args.model:
        config.model_path = args.model
    if args.camera is not None:
        config.camera_index = args.camera
    if args.fps:
        config.perception_fps = args.fps
    if args.map:
        config.osm_map_path = args.map
    if args.gps_port:
        config.gps_port = args.gps_port

    # ── Global stop signal ───────────────────────────────────────
    stop_event = threading.Event()

    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received.")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── 1. TTS — greet the user ─────────────────────────────────
    speak("ALAS sistemi başlatılıyor.")

    # ── 2. GPS ───────────────────────────────────────────────────
    if args.mock:
        gps = MockGPSReader()
    else:
        gps = GPSReader(
            port=config.gps_port,
            baudrate=config.gps_baudrate,
            warmup_sec=config.gps_warmup_sec,
        )
    gps.start()
    logger.info("[Main] GPS started.")

    # ── 3. Navigation ────────────────────────────────────────────
    nav_config = NavConfig(log_dir=config.log_dir)
    nav = NavigationSystem(config.osm_map_path, nav_config)
    logger.info("[Main] Navigation system ready.")

    # ── 4. STT + intent classifier ───────────────────────────────
    stt = STTEngine()
    logger.info("[Main] STT engine ready.")

    # ── 5. Perception thread ─────────────────────────────────────
    perception = None
    if not args.no_camera:
        perception = PerceptionThread(config, stop_event)
        perception.start()
        logger.info("[Main] Perception thread started.")

    # ── 6. Navigation thread ─────────────────────────────────────
    nav_thread = NavigationThread(
        config, nav, gps, stop_event,
        perception_pause=perception.pause_event if perception else None,
    )
    nav_thread.start()
    logger.info("[Main] Navigation thread started.")

    # ── 7. Voice + button ────────────────────────────────────────
    voice_handler = VoiceCommandHandler(config, nav, gps, stt, stop_event)
    button_thread = start_button_listener(config, voice_handler, stop_event, mock=args.mock)
    logger.info("[Main] Button listener started.")

    # ── 8. Ready ─────────────────────────────────────────────────
    speak("Sistem hazır. Butona basarak komut verebilirsiniz.")
    logger.info("[Main] ════════ ALAS SYSTEM READY ════════")

    # ── Main thread idles — waits for shutdown ───────────────────
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()

    # ── Graceful shutdown ────────────────────────────────────────
    logger.info("[Main] Shutting down...")

    if nav.is_active:
        nav.stop_navigation()
    gps.stop()

    wait_until_done()
    shutdown_tts()

    if perception and perception.is_alive():
        perception.join(timeout=3)
    nav_thread.join(timeout=3)

    logger.info("[Main] ════════ ALAS SYSTEM STOPPED ════════")


if __name__ == "__main__":
    main()
