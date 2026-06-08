#!/usr/bin/env python3
"""ALAS — AI-Based Smart Glasses Main Loop.

Every step is one factory or constructor call followed by ``start()``
Each subsystem owns its own implementation:

    Perception      -> ai/perception_service.py
    Navigation      -> navigation/navigation_service.py
    Voice commands  -> tts_stt/voice_commands.py
    Button input    -> tts_stt/button_listener.py
    TTS gating      -> tts_stt/voice_policy.py
    Modes/shutdown  -> main/lifecycle.py
    Logging         -> main/logging_config.py
    GPS factory     -> navigation/sensors/__init__.py (build_gps)
    STT factory     -> tts_stt/__init__.py           (load_stt)

Run on Jetson Nano:
    python3 -m main.alas_main --model models/segmentation/alas_engine.trt
    python3 -m main.alas_main --model models/segmentation/alas_engine.trt --record

Field test
    python3 -m main.alas_main --model models/segmentation/alas_engine.trt --bypass-stt --record

Desktop test (no GPIO / GPS / camera / microphone):
    python3 -m main.alas_main --mock --no-camera --bypass-stt --bypass-warmup
    python3 -m main.alas_main --mock --no-camera --bypass-stt --bypass-warmup --record
"""

import os
import sys
import threading

# Make sub-packages importable when invoked as ``python3 -m main.alas_main``.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Quieten GStreamer/OpenCV native chatter so the journal stays in one ALAS log
# format. MUST be set before the camera stack (cv2 / nvarguscamerasrc) loads, so
# it lives here above the imports. (The verbose GST_ARGUS sensor-mode dump is a
# separate native print, muted around the camera open in perception_service.)
os.environ.setdefault("GST_DEBUG", "0")

from main import lifecycle
from main.activity_monitor import ActivityMonitor
from main.config import ALASConfig
from main.status_led import StatusLED
from main.lifecycle import SystemMode
from main.logging_config import configure_logging
from main.session_recorder import build_recorder
from ai.perception_service import PerceptionService
from navigation.local_planner import VFHPlanner
from navigation.navigation_service import NavigationService
from navigation.router import NavigationSystem
from navigation.sensors import build_gps
from tts_stt import load_stt
from tts_stt.button_listener import ButtonListener
from tts_stt.voice_commands import VoiceCommandHandler
from tts_stt.voice_policy import VoicePolicy


def main():
    # 1. Config, logging and lifecycle
    config = ALASConfig.from_cli()
    logger = configure_logging()
    stop_event = lifecycle.install_signal_handlers()
    modes = lifecycle.ModeManager(initial=SystemMode.WARMUP)

    # 2. Voice policy owns every TTS utterance and its priority/silence gates.
    voice = VoicePolicy(config)
    voice.announce_boot()

    # 2b. Field-test black-box recorder (--record). NullRecorder when disabled or
    #     when disk space is too low (it warns over TTS but keeps the system running).
    recorder = build_recorder(config, voice)
    voice.set_recorder(recorder)
    modes.set_recorder(recorder)

    # 3. GPS sensor — real UART or deterministic mock, already started.
    gps = build_gps(config)

    # 4. Navigation core — pure domain object, no threads of its own.
    nav = NavigationSystem(config.osm_map_path, config.nav)

    # 5. Speech recognition engine — loaded on a background thread so the
    #    boot announcement (and nav warmup)
    # VFH local planner — image-space escape routing over the segmentation
    # mask. None disables the override and PerceptionService falls back to
    # plain path_guidance.
    vfh = VFHPlanner(config) if config.vfh.enabled else None

    # Auto-STANDBY power saver. Disabled unless --auto-standby; when disabled its
    # report_* hooks are cheap no-ops and start() does nothing.
    monitor = ActivityMonitor(config, modes, voice, stop_event)
    monitor.set_nav(nav)

    # Optional raw-frame capture for offline model fine-tuning (--capture-dataset).
    from main.dataset_collector import build_collector
    collector = build_collector(config)

    perception = (
        None if config.no_camera
        else PerceptionService(config, voice, modes, stop_event, nav=nav, vfh=vfh,
                               recorder=recorder, monitor=monitor, collector=collector)
    )
    navigation = NavigationService(config, nav, gps, voice, modes, stop_event,
                                   recorder=recorder, monitor=monitor)
    commands = VoiceCommandHandler(config, nav, gps, None, voice, modes, stop_event,
                                   recorder=recorder, monitor=monitor)

    def _load_and_attach_stt():
        try:
            stt = load_stt(config)
            commands.set_stt(stt)
            logger.info("[Main] STT engine attached.")
        except Exception:
            logger.exception("[Main] Background STT load failed.")

    threading.Thread(target=_load_and_attach_stt, name="STTLoader", daemon=True).start()

    # 6. Background services — perception, navigation, button, voice commands.
    button = ButtonListener(
        config,
        on_press=commands.handle_press,
        modes=modes,
        stop_event=stop_event,
        mock=config.mock or config.mock_button,
    )

    # Mode-indicator LED (BCM 24 / pin 18): blink=WARMUP, solid=ACTIVE,
    # heartbeat=STANDBY, off=stopping. Driven here because alas_main owns the mode.
    status_led = StatusLED(modes, stop_event, mock=config.mock)

    if perception is not None:
        perception.start()
    navigation.start()
    button.start()
    monitor.start()  # no-op unless --auto-standby
    status_led.start()

    # 7. Wait for sensors/model warmup before transitioning to ACTIVE.
    logger.info("[Main] ======== WARMUP STARTING ========")
    if config.bypass_warmup:
        logger.info("[Main] --bypass-warmup active: skipping sensor readiness check.")
        modes.transition_to(SystemMode.ACTIVE)
    else:
        lifecycle.await_ready(
            modes, gps, perception, voice,
            timeout_sec=config.warmup_timeout_sec,
            bypass_gps=config.bypass_gps_warmup,
            stop_event=stop_event,
        )

    # If the user pressed the launch button during warmup, stop_event is now
    # set: skip the "ready" announcement and fall straight through to an
    # orderly shutdown instead of going ACTIVE for a split second.
    if stop_event.is_set():
        logger.info("[Main] Warmup aborted — shutting down without going ACTIVE.")
    else:
        voice.announce_ready()
        logger.info("[Main] ======== ALAS SYSTEM READY (ACTIVE) ========")

    # 7b. Mic-less auto-navigation: once ACTIVE with a GPS fix, route to the
    #     default destination so the field test gets turn-by-turn guidance
    #     without a microphone. An exact --auto-nav-coord wins over the nearest
    #     --auto-nav category. A PTT press re-triggers it later.
    if not stop_event.is_set() and (config.auto_nav_coord or config.auto_nav_category):
        def _auto_nav():
            # Wait for a GPS fix (already present once ACTIVE unless bypassed).
            for _ in range(int(config.warmup_timeout_sec) or 90):
                if stop_event.is_set():
                    return
                if gps.get_coord() is not None:
                    break
                stop_event.wait(1.0)
            # Settle delay so boot announcements finish and the pipeline steadies.
            if config.auto_nav_delay_sec > 0:
                stop_event.wait(config.auto_nav_delay_sec)
            if stop_event.is_set():
                return
            if config.auto_nav_coord:
                lat, lon = config.auto_nav_coord
                logger.info("[Main] Auto-nav: routing to coordinate %.6f,%.6f.", lat, lon)
                commands.route_to_coord(lat, lon)
            else:
                logger.info("[Main] Auto-nav: routing to nearest '%s'.", config.auto_nav_category)
                commands.route_to(config.auto_nav_category)
        threading.Thread(target=_auto_nav, name="AutoNav", daemon=True).start()

    # 8. Idle until SIGINT / SIGTERM.
    lifecycle.wait_for_shutdown(
        stop_event,
        services=[perception, navigation, button],
        voice=voice,
    )

    # 9. Graceful shutdown — stop services, release sensors, drain TTS.
    lifecycle.shutdown(
        button=button,
        services=[perception, navigation, monitor, status_led],
        nav=nav,
        gps=gps,
        voice=voice,
        modes=modes,
        recorder=recorder,
    )
    logger.info("[Main] ======== ALAS SYSTEM STOPPED ========")


if __name__ == "__main__":
    main()
