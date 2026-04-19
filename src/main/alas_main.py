#!/usr/bin/env python3
"""ALAS — AI-Based Smart Glasses Main Loop.

Thin orchestrator. Reads as a recipe; every step is one factory or
constructor call followed by ``start()``. Each subsystem owns its own
implementation:

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
    cd src && python -m main.alas_main --model models/segmentation/alas_engine.trt

Desktop test (no GPIO / GPS / camera / microphone):
    cd src && python -m main.alas_main --mock --no-camera --bypass-stt --bypass-warmup
"""

import os
import sys
import threading

# Make sub-packages importable when invoked as ``python -m main.alas_main``.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from main import lifecycle
from main.config import ALASConfig
from main.lifecycle import SystemMode
from main.logging_config import configure_logging
from ai.perception_service import PerceptionService
from navigation.navigation_service import NavigationService
from navigation.router import NavigationSystem, NavConfig
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

    # 3. GPS sensor — real UART or deterministic mock, already started.
    gps = build_gps(config)

    # 4. Navigation core — pure domain object, no threads of its own.
    nav = NavigationSystem(config.osm_map_path, NavConfig(log_dir=config.log_dir))

    # 5. Speech recognition engine — loaded on a background thread so the
    #    boot announcement (and nav warmup) are not blocked by Vosk model load.
    perception = (
        None if config.no_camera
        else PerceptionService(config, voice, modes, stop_event, nav=nav)
    )
    navigation = NavigationService(config, nav, gps, voice, modes, stop_event)
    commands = VoiceCommandHandler(config, nav, gps, None, voice, modes, stop_event)

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
        mock=config.mock,
    )

    if perception is not None:
        perception.start()
    navigation.start()
    button.start()

    # 7. Wait for sensors/model warmup before transitioning to ACTIVE.
    if config.bypass_warmup:
        logger.info("[Main] --bypass-warmup active: skipping sensor readiness check.")
        modes.transition_to(SystemMode.ACTIVE)
    else:
        lifecycle.await_ready(
            modes, gps, perception, voice,
            timeout_sec=config.warmup_timeout_sec,
            bypass_gps=config.bypass_gps_warmup,
        )
    voice.announce_ready()
    logger.info("[Main] ======== ALAS SYSTEM READY ========")

    # 8. Idle until SIGINT / SIGTERM.
    lifecycle.wait_for_shutdown(
        stop_event,
        services=[perception, navigation, button],
        voice=voice,
    )

    # 9. Graceful shutdown — stop services, release sensors, drain TTS.
    lifecycle.shutdown(
        button=button,
        services=[perception, navigation],
        nav=nav,
        gps=gps,
        voice=voice,
        modes=modes,
    )
    logger.info("[Main] ======== ALAS SYSTEM STOPPED ========")


if __name__ == "__main__":
    main()
