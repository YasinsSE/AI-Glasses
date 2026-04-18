#!/usr/bin/env python3
"""
ALAS — AI-Based Smart Glasses Main Loop
========================================
Thin orchestrator. Reads as a recipe; every step is one constructor call
followed by ``start()``. The how-to of each subsystem lives in its module:

    Perception      -> ai/perception_service.py
    Navigation      -> navigation/navigation_service.py
    Voice commands  -> tts_stt/voice_commands.py
    Button input    -> tts_stt/button_listener.py
    TTS gating      -> tts_stt/voice_policy.py
    Modes/shutdown  -> main/lifecycle.py

Run on Jetson Nano:
    cd src && python -m main.alas_main --model models/segmentation/alas_engine.trt

Desktop test (no GPIO / GPS / camera / microphone):
    cd src && python -m main.alas_main --mock --no-camera --bypass-stt --bypass-warmup
"""

import logging
import os
import sys

# Make sub-packages importable when invoked as ``python -m main.alas_main``
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from main import lifecycle
from main.config import ALASConfig
from main.lifecycle import SystemMode
from ai.perception_service import PerceptionService
from navigation.navigation_service import NavigationService
from navigation.router import NavigationSystem, NavConfig
from navigation.sensors import GPSReader, MockGPSReader
from tts_stt.button_listener import ButtonListener
from tts_stt.voice_commands import VoiceCommandHandler
from tts_stt.voice_policy import VoicePolicy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ALAS")


def _load_stt(config):
    """Load STTEngine only when not bypassed. Returns None when bypassed."""
    if config.bypass_stt:
        logger.info("[Main] --bypass-stt active: microphone disabled, typed input only.")
        return None
    try:
        from tts_stt.stt import STTEngine
        return STTEngine()
    except Exception:
        logger.exception("[Main] STTEngine could not be loaded — falling back to bypass mode.")
        return None


def main():
    # -- 1. Config & lifecycle ------------------------------------------------
    config = ALASConfig.from_cli()
    stop_event = lifecycle.install_signal_handlers()
    modes = lifecycle.ModeManager(initial=SystemMode.WARMUP)

    # -- 2. Voice policy (owns all TTS) ---------------------------------------
    voice = VoicePolicy(config)
    voice.announce_boot()

    # -- 3. GPS sensor --------------------------------------------------------
    gps = (
        MockGPSReader()
        if config.mock
        else GPSReader(
            port=config.gps_port,
            baudrate=config.gps_baudrate,
            warmup_sec=config.gps_warmup_sec,
        )
    )
    gps.start()

    # -- 4. Navigation core (pure domain object, no threads) ------------------
    nav = NavigationSystem(config.osm_map_path, NavConfig(log_dir=config.log_dir))

    # -- 5. Speech recognition engine (optional) ------------------------------
    stt = _load_stt(config)

    # -- 6. Background services -----------------------------------------------
    perception = (
        None if config.no_camera
        else PerceptionService(config, voice, modes, stop_event, nav=nav)
    )
    navigation = NavigationService(config, nav, gps, voice, modes, stop_event)
    commands = VoiceCommandHandler(config, nav, gps, stt, voice, modes, stop_event)
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

    # -- 7. Wait for warmup -> ACTIVE -----------------------------------------
    if config.bypass_warmup:
        logger.info("[Main] --bypass-warmup active: skipping sensor readiness check.")
        modes.transition_to(SystemMode.ACTIVE)
    else:
        lifecycle.await_ready(
            modes, gps, perception, voice,
            timeout_sec=config.warmup_timeout_sec,
        )
    voice.announce_ready()
    logger.info("[Main] ======== ALAS SYSTEM READY ========")

    # -- 8. Idle until shutdown -----------------------------------------------
    lifecycle.wait_for_shutdown(stop_event)

    # -- 9. Graceful shutdown -------------------------------------------------
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
