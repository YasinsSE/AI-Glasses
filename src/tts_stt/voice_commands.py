"""
VoiceCommandHandler -- button -> STT -> intent -> action.
=========================================================
Owns the spoken-command session triggered by a button press:

    1. If we are in SLEEP mode, the press is consumed as a wake event,
       no STT session is started.
    2. Otherwise prompt -> listen -> classify intent -> execute.

When ``--bypass-stt`` is active (stt is None), the handler prompts the user
to type their command via stdin instead of speaking. The typed text goes
through the same keyword-based intent classification as spoken text, so
navigation and system commands work identically.

Intents handled:
    - **navigation**     : extract a POI category and route to the nearest one.
    - **system_command** : sleep / shutdown / cancel-route.
    - anything else      : speak "Anlasildi." and return.

All speech goes through ``VoicePolicy`` so TTS gating stays centralised.
"""

import logging
import sys
import threading
from typing import Optional

from main.config import ALASConfig
from main.lifecycle import ModeManager, SystemMode
from navigation.router import Coord, NavigationSystem
from tts_stt.voice_policy import VoicePolicy

logger = logging.getLogger("ALAS.voice_commands")


class VoiceCommandHandler:
    """Handles a single voice-command session per button press."""

    # Spoken keyword -> POI category mapping
    NAV_KEYWORDS = {
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

    SLEEP_WORDS = ["uyu", "uyku", "uyut", "kapat ekran"]
    SHUTDOWN_WORDS = ["kapat", "durdur", "bitir", "kapa"]
    STOP_NAV_WORDS = ["rota", "navigasyon", "iptal"]

    def __init__(
        self,
        config,          # ALASConfig
        nav,             # NavigationSystem
        gps,             # GPSReader or MockGPSReader
        stt,             # STTEngine or None (bypass mode)
        voice,           # VoicePolicy
        modes,           # ModeManager
        stop_event,      # threading.Event
    ):
        self._config = config
        self._nav = nav
        self._gps = gps
        self._stt = stt
        self._voice = voice
        self._modes = modes
        self._stop = stop_event

    def set_stt(self, stt):
        # Single-reference attribute write; safe without a lock in CPython.
        self._stt = stt

    # -- Button-press entry point -------------------------------------------

    def handle_press(self):
        """Called by ButtonListener on button press / Enter in mock mode."""
        if self._stop.is_set():
            return

        # SLEEP wake -- consume the press, no STT session.
        if self._modes.mode == SystemMode.SLEEP:
            self._modes.transition_to(SystemMode.ACTIVE)
            self._voice.announce_wake()
            return

        # Only run STT in ACTIVE mode (skip during WARMUP).
        if self._modes.mode != SystemMode.ACTIVE:
            return

        # STT may still be loading on a background thread; refuse the press
        # rather than silently falling into the keyboard-bypass path.
        if self._stt is None and not self._config.bypass_stt:
            self._voice.say_prompt("Konuşma motoru hâlâ yükleniyor.")
            return

        # -- Get text: either via microphone (STT) or keyboard (bypass) -----
        text = self._get_text_input()
        if not text:
            self._voice.say_prompt("Anlayamadim, tekrar deneyin.")
            return

        # -- Classify intent ------------------------------------------------
        intent = self._classify_intent(text)
        logger.info('[Voice] "%s" -> intent=%s', text, intent)

        if intent == "navigation":
            self._handle_navigation(text)
        elif intent == "system_command":
            self._handle_system_command(text)
        else:
            self._voice.say_prompt("Anlasildi.")

    # -- Text input (STT vs keyboard bypass) --------------------------------

    def _get_text_input(self):
        """
        When STT is available, use the microphone. When STT is None
        (--bypass-stt), prompt the user to type their command.
        Returns the text string, or empty string on failure.
        """
        if self._stt is not None:
            # Normal microphone path
            self._voice.say_prompt("Sizi dinliyorum.")
            return self._stt.listen(
                timeout_sec=self._config.stt_listen_timeout,
                silence_sec=self._config.stt_silence_sec,
            )

        # Bypass mode: typed input from stdin
        try:
            sys.stdout.write("[Bypass STT] Komutunuzu yazin: ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                return ""
            return line.strip()
        except (EOFError, OSError):
            return ""

    def _classify_intent(self, text):
        """
        Use the SLM intent classifier if available; otherwise fall back to
        simple keyword matching so --bypass-stt mode still works.
        """
        # Try SLM classifier if STT (and its _slm) is loaded
        if self._stt is not None:
            try:
                intent, conf = self._stt._slm.predict(text)
                return intent
            except Exception:
                logger.exception("[Voice] SLM intent classification failed")

        # Keyword-based fallback (always used in bypass mode)
        text_lower = text.lower()
        if self._extract_category(text) is not None:
            return "navigation"
        for w in self.SLEEP_WORDS + self.SHUTDOWN_WORDS + self.STOP_NAV_WORDS:
            if w in text_lower:
                return "system_command"
        return "general"

    # -- Intent handlers ----------------------------------------------------

    def _handle_navigation(self, text):
        """Extract destination from spoken text and start route navigation."""
        if self._nav.is_active:
            self._voice.say_prompt("Mevcut rota iptal ediliyor, yeni rota hesaplaniyor.")
            self._nav.stop_navigation()

        fix = self._gps.get_coord()
        if fix is None:
            self._voice.say_prompt("GPS sinyali bulunamadi. Lutfen acik alanda deneyin.")
            return

        lat, lon, _ = fix
        position = Coord(lat, lon)

        category = self._extract_category(text)
        if not category:
            self._voice.say_prompt("Nereye gitmek istediginizi anlayamadim. Lutfen tekrar soyleyin.")
            return

        self._voice.say_prompt("En yakin %s araniyor." % category)

        success, _msg, poi = self._nav.navigate_to_nearest(position, category)

        if success and poi:
            self._voice.say_prompt(
                "En yakin %s %d metre uzakta. Rota hazir, yonlendirme basliyor."
                % (category, int(poi.distance_m))
            )
        else:
            self._voice.say_prompt("Yakinlarda %s bulunamadi." % category)

    def _handle_system_command(self, text):
        """Sleep / cancel-route / shutdown."""
        text_lower = text.lower()

        # Sleep
        if any(w in text_lower for w in self.SLEEP_WORDS):
            self._voice.announce_sleep()
            self._modes.transition_to(SystemMode.SLEEP)
            return

        # Cancel current navigation
        if any(w in text_lower for w in self.STOP_NAV_WORDS):
            if self._nav.is_active:
                self._nav.stop_navigation()
                self._voice.say_prompt("Navigasyon durduruldu.")
            else:
                self._voice.say_prompt("Aktif bir rota yok.")
            return

        # Shutdown
        if any(w in text_lower for w in self.SHUTDOWN_WORDS):
            self._voice.say_prompt("Sistem kapatiliyor, iyi gunler.")
            self._stop.set()
            return

        self._voice.say_prompt("Sistem komutu alindi.")

    # -- Helpers ------------------------------------------------------------

    def _extract_category(self, text):
        text_lower = text.lower()
        for keyword, category in self.NAV_KEYWORDS.items():
            if keyword in text_lower:
                return category
        return None
