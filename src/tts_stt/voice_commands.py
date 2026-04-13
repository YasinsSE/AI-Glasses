"""
VoiceCommandHandler — button → STT → intent → action.
======================================================
Owns the spoken-command session triggered by a button press:

    1. If we are in SLEEP mode, the press is consumed as a wake event,
       no STT session is started.
    2. Otherwise prompt → listen → classify intent → execute.

Intents handled:
    - **navigation** : extract a POI category from the spoken text and
                           ask NavigationSystem to route to the nearest one.
    - **system_command** : sleep / shutdown / cancel-route.
    - anything else      : speak "Anlaşıldı." and return.

All speech goes through ``VoicePolicy`` so TTS gating stays centralised.
"""

import logging
import threading
from typing import Optional

from main.config import ALASConfig
from main.lifecycle import ModeManager, SystemMode
from navigation.router import Coord, NavigationSystem
# from tts_stt.stt import STTEngine  <-- PYAUDIO HATASI VERMEMESI ICIN IPTAL EDILDI
from tts_stt.voice_policy import VoicePolicy

logger = logging.getLogger("ALAS.voice_commands")


class VoiceCommandHandler:
    """Handles a single voice-command session per button press."""

    # Spoken keyword → POI category mapping
    NAV_KEYWORDS: dict = {
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
        config: ALASConfig,
        nav: NavigationSystem,
        gps,
        stt,  # <--- STTEngine TIP BELIRTECI SILINDI
        voice: VoicePolicy,
        modes: ModeManager,
        stop_event: threading.Event,
    ) -> None:
        self._config = config
        self._nav = nav
        self._gps = gps
        self._stt = stt
        self._voice = voice
        self._modes = modes
        self._stop = stop_event

    # ── Button-press entry point ─────────────────────────────────

    def handle_press(self) -> None:
        """Called by ButtonListener on button press / Enter in mock mode."""
        if self._stop.is_set():
            return

        # SLEEP wake — consume the press, no STT session.
        if self._modes.mode == SystemMode.SLEEP:
            self._modes.transition_to(SystemMode.ACTIVE)
            self._voice.announce_wake()
            return

        # Only run STT in ACTIVE mode (skip during WARMUP).
        if self._modes.mode != SystemMode.ACTIVE:
            return

        self._voice.say_prompt("Sizi dinliyorum.")

        # EGER TESTTE MIKROFON YOKSA VE STT NONE ISE BURASI HATA VEREBILIR
        if self._stt is None:
            logger.warning("[Voice] STT modülü devre dışı bırakıldığı için sesli komut alınamıyor.")
            self._voice.say_prompt("Mikrofon kapalı, komut alamıyorum.")
            return

        text = self._stt.listen(
            timeout_sec=self._config.stt_listen_timeout,
            silence_sec=self._config.stt_silence_sec,
        )

        if not text:
            self._voice.say_prompt("Anlayamadım, tekrar deneyin.")
            return

        try:
            intent, conf = self._stt._slm.predict(text)
        except Exception:  # noqa: BLE001
            logger.exception("[Voice] intent classification failed")
            self._voice.say_prompt("Anlaşıldı.")
            return

        logger.info(f'[Voice] "{text}" → intent={intent} conf={conf}')

        if intent == "navigation":
            self._handle_navigation(text)
        elif intent == "system_command":
            self._handle_system_command(text)
        else:
            self._voice.say_prompt("Anlaşıldı.")

    # ── Intent handlers ──────────────────────────────────────────

    def _handle_navigation(self, text: str) -> None:
        """Extract destination from spoken text and start route navigation."""
        if self._nav.is_active:
            self._voice.say_prompt("Mevcut rota iptal ediliyor, yeni rota hesaplanıyor.")
            self._nav.stop_navigation()

        fix = self._gps.get_coord()
        if fix is None:
            self._voice.say_prompt("GPS sinyali bulunamadı. Lütfen açık alanda deneyin.")
            return

        lat, lon, _ = fix
        position = Coord(lat, lon)

        category = self._extract_category(text)
        if not category:
            self._voice.say_prompt("Nereye gitmek istediğinizi anlayamadım. Lütfen tekrar söyleyin.")
            return

        self._voice.say_prompt(f"En yakın {category} aranıyor.")

        success, _msg, poi = self._nav.navigate_to_nearest(position, category)

        if success and poi:
            self._voice.say_prompt(
                f"En yakın {category} {int(poi.distance_m)} metre uzakta. "
                f"Rota hazır, yönlendirme başlıyor."
            )
        else:
            self._voice.say_prompt(f"Yakınlarda {category} bulunamadı.")

    def _handle_system_command(self, text: str) -> None:
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
            self._voice.say_prompt("Sistem kapatılıyor, iyi günler.")
            self._stop.set()
            return

        self._voice.say_prompt("Sistem komutu alındı.")

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_category(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for keyword, category in self.NAV_KEYWORDS.items():
            if keyword in text_lower:
                return category
        return None