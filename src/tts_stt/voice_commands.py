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

    # Yes/no vocabulary for the spoken navigation confirmation. Token-matched;
    # the same list doubles as the Vosk grammar so the confirmation listen is
    # constrained decoding (far more robust than open vocabulary).
    YES_WORDS = ["evet", "olur", "tamam", "başlat", "tabii", "tabi",
                 "istiyorum", "onaylıyorum", "hadi", "başla"]
    NO_WORDS = ["hayır", "hayir", "iptal", "istemiyorum", "vazgeç",
                "vazgeçtim", "yok", "dur", "olmaz"]

    # Local queries — matched on keywords BEFORE the SLM intent classifier, so
    # they work identically in --bypass-stt mode and cannot be misrouted.
    WHERE_WORDS = ["neredeyim", "nerdeyim", "neredeyiz", "konumum"]
    STATUS_WORDS = ["durum", "pil", "batarya"]
    # Token → canonical bookmark name ("evi kaydet" / "eve git" → "ev").
    # Token-matched (not substring) so "evet" never reads as "ev".
    PLACE_ALIASES = {
        "ev": "ev", "evi": "ev", "eve": "ev", "evim": "ev", "evimi": "ev",
        "iş": "iş", "işi": "iş", "işe": "iş", "işyeri": "iş", "işyerine": "iş",
    }

    def __init__(
        self,
        config,          # ALASConfig
        nav,             # NavigationSystem
        gps,             # GPSReader or MockGPSReader
        stt,             # STTEngine or None (bypass mode)
        voice,           # VoicePolicy
        modes,           # ModeManager
        stop_event,      # threading.Event
        recorder=None,   # SessionRecorder or None
        monitor=None,    # ActivityMonitor (auto-STANDBY) or None
        perception=None, # PerceptionService or None (for the "oku" OCR command)
    ):
        self._config = config
        self._nav = nav
        self._gps = gps
        self._stt = stt
        self._voice = voice
        self._modes = modes
        self._stop = stop_event
        self._monitor = monitor
        self._perception = perception
        from main.session_recorder import NullRecorder
        self._rec = recorder or NullRecorder()  # field-test black-box recorder
        from navigation.saved_places import SavedPlaces
        self._places = SavedPlaces(config.saved_places_path)
        # Deferred-route state (urban canyon: command arrives before a GPS fix).
        self._pending_cancel: Optional[threading.Event] = None

    # Poll cadence of the deferred-route waiter; a class attribute so tests
    # can shrink it without monkeypatching time.
    PENDING_POLL_SEC = 3.0

    def set_stt(self, stt):
        # Single-reference attribute write; safe without a lock in CPython.
        self._stt = stt

    # -- Button-press entry point -------------------------------------------

    def handle_press(self):
        """Called by ButtonListener on button press / Enter in mock mode."""
        if self._stop.is_set():
            return

        # STANDBY (SLEEP) wake -- consume the press as a wake event, NOT an STT
        # session. The perception loop re-acquires the camera on the ACTIVE
        # transition; announce_wake() plays the wake-up cue.
        if self._modes.mode == SystemMode.SLEEP:
            self._modes.transition_to(SystemMode.ACTIVE)
            if self._monitor is not None:
                self._monitor.notify_wake()  # reset idle timer so we don't re-sleep
            self._voice.announce_wake()
            return

        # Only run STT in ACTIVE mode (skip during WARMUP).
        if self._modes.mode != SystemMode.ACTIVE:
            print(f"[Button] Ignored — system not ready yet (mode={self._modes.mode}). "
                  "Wait for 'Sistem hazir' then press Enter again.", flush=True)
            return

        # STT may still be loading on a background thread; refuse the press
        # rather than silently falling into the keyboard-bypass path.
        if self._stt is None and not self._config.bypass_stt:
            self._voice.say_prompt("Konuşma motoru hâlâ yükleniyor.")
            return

        # -- Mic-less (--bypass-stt) PTT behaviour --------------------------
        # With no microphone, a PTT press (re)starts guidance to the default
        # destination instead of prompting for stdin, which a headless service
        # cannot read. Falls through to the typed-stdin path only on an
        # interactive terminal with no default set (dev convenience).
        if self._stt is None and self._config.bypass_stt:
            coord = getattr(self._config, "auto_nav_coord", None)
            cat = getattr(self._config, "auto_nav_category", "")
            if coord:
                self.route_to_coord(coord[0], coord[1])
                self._rec.log_command("%s,%s" % coord, intent="navigation", action="ptt_auto_nav_coord")
                return
            if cat:
                self.route_to(cat)
                self._rec.log_command(cat, intent="navigation", action="ptt_auto_nav")
                return
            if not sys.stdin.isatty():
                self._voice.say_prompt("Mikrofon yok, komut verilemiyor.")
                return

        # -- Get text: either via microphone (STT) or keyboard (bypass) -----
        text = self._get_text_input()
        if not text:
            self._voice.say_prompt("Anlayamadim, tekrar deneyin.")
            self._rec.log_command("", intent=None, action="not_recognized")
            return

        # -- Local queries first (keyword-matched, SLM-independent) ----------
        if self._handle_local_query(text):
            self._rec.log_command(text, intent="local_query", action="local_query")
            return

        # -- Classify intent ------------------------------------------------
        intent = self._classify_intent(text)
        logger.info('[Voice] "%s" -> intent=%s', text, intent)

        if intent == "navigation":
            self._handle_navigation(text)
        elif intent == "system_command":
            self._handle_system_command(text)
        elif self._extract_category(text) is not None:
            # SLM said "general" but the utterance names a known destination
            # ("ilaç almak istiyorum eczaneye gitmem lazım"). Recover through
            # the navigation path — its confirmation question guards against
            # the classifier being right and us being wrong.
            self._handle_navigation(text)
        else:
            self._voice.say_prompt("Anlasildi.")

        self._rec.log_command(text, intent=intent, action=intent)

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
                timeout_sec=self._config.voice.stt_listen_timeout,
                silence_sec=self._config.voice.stt_silence_sec,
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
        """Extract destination from spoken/typed text and start route navigation.

        A destination INFERRED from a longer utterance ("ilaç almak istiyorum
        eczaneye gitmem lazım" → eczane) is confirmed before routing: STT
        mishears, and sending a blind user toward the wrong place costs far
        more than one extra question. Terse direct commands ("eczane",
        "eczaneye git") skip the question, as do typed --bypass-stt commands.
        """
        category = self._extract_category(text)
        if not category:
            self._voice.say_prompt("Nereye gitmek istediginizi anlayamadim. Lutfen tekrar soyleyin.")
            return
        if self._needs_confirmation(text):
            answer = self._confirm_yes_no(
                "En yakın %s için rota başlatılsın mı? Evet ya da hayır deyin."
                % category
            )
            if answer is not True:
                self._voice.say_prompt(
                    "İptal edildi. Tekrar denemek için butona basın.")
                self._rec.log_command(text, intent="navigation",
                                      action="nav_confirm_declined")
                return
        self.route_to(category)

    def _needs_confirmation(self, text) -> bool:
        v = self._config.voice
        if not getattr(v, "nav_confirm_enabled", True) or self._stt is None:
            return False  # typed commands are explicit; nothing to confirm
        return len(text.split()) >= getattr(v, "nav_confirm_min_words", 3)

    def _confirm_yes_no(self, question):
        """Ask, listen (grammar-constrained), parse. True / False / None.

        One retry on an unclear answer; anything still unclear counts as NO —
        for a navigation system the safe default is to not start a route the
        user may not have asked for.
        """
        v = self._config.voice
        vocab = self.YES_WORDS + self.NO_WORDS
        for attempt in range(2):
            self._voice.say_prompt(
                question if attempt == 0 else "Anlayamadım. Evet mi, hayır mı?")
            try:
                heard = self._stt.listen(
                    timeout_sec=getattr(v, "confirm_listen_timeout", 6.0),
                    silence_sec=getattr(v, "confirm_silence_sec", 1.2),
                    grammar=vocab,
                )
            except TypeError:  # an STT double without grammar support
                heard = self._stt.listen(
                    timeout_sec=getattr(v, "confirm_listen_timeout", 6.0),
                    silence_sec=getattr(v, "confirm_silence_sec", 1.2),
                )
            tokens = (heard or "").lower().replace(",", " ").split()
            if any(w in tokens for w in self.NO_WORDS):
                return False
            if any(w in tokens for w in self.YES_WORDS):
                return True
        return None

    def route_to(self, category):
        """Start navigation to the nearest POI of ``category``.

        Shared by spoken/typed commands, the mic-less PTT button, and the
        ``--auto-nav`` startup route. Returns True on success.
        """
        if self._nav.is_active:
            self._voice.say_prompt("Mevcut rota iptal ediliyor, yeni rota hesaplaniyor.")
            self._nav.stop_navigation()

        fix = self._origin_fix()
        if fix is None:
            self._defer_route(lambda: self.route_to(category), category)
            return False

        lat, lon, _ = fix
        self._voice.say_prompt("En yakin %s araniyor." % category)
        success, _msg, poi = self._nav.navigate_to_nearest(Coord(lat, lon), category)

        if success and poi:
            self._rec.log_route(self._nav.get_route(), origin=(lat, lon),
                                destination=(poi.coord.lat, poi.coord.lon))
            self._voice.say_prompt(
                "En yakin %s %d metre uzakta. Rota hazir, yonlendirme basliyor."
                % (category, int(poi.distance_m))
            )
            return True
        self._voice.say_prompt("Yakinlarda %s bulunamadi." % category)
        return False

    def route_to_coord(self, lat, lon):
        """Start navigation to an exact coordinate (test / map-picked destination).

        Uses the current GPS fix as origin and routes to ``(lat, lon)`` via the
        existing ``NavigationSystem.start_navigation``. Returns True on success.
        """
        if self._nav.is_active:
            self._voice.say_prompt("Mevcut rota iptal ediliyor, yeni rota hesaplaniyor.")
            self._nav.stop_navigation()

        fix = self._origin_fix()
        if fix is None:
            self._defer_route(lambda: self.route_to_coord(lat, lon), "seçilen hedef")
            return False

        flat, flon, _ = fix
        self._voice.say_prompt("Secilen hedefe rota hesaplaniyor.")
        success, _msg = self._nav.start_navigation(Coord(flat, flon), Coord(lat, lon))
        if success:
            self._rec.log_route(self._nav.get_route(), origin=(flat, flon),
                                destination=(lat, lon))
            self._voice.say_prompt("Rota hazir, yonlendirme basliyor.")
            return True
        self._voice.say_prompt("Hedefe rota bulunamadi.")
        return False

    def _origin_fix(self):
        """Route origin: real GPS fix first; the configured --fallback-origin
        as a demo crutch when there is none; None → caller defers the route.

        The DESTINATION never needs GPS (POIs live in the offline OSM map) —
        only the starting point does, which is why a known test-start
        coordinate is enough to keep a no-fix demo moving. The route tracker
        switches to real fixes automatically once GPS comes alive.
        """
        fix = self._gps.get_coord()
        if fix is not None:
            return fix
        fb = getattr(self._config, "fallback_origin", None)
        if fb:
            self._voice.say_prompt(
                "GPS sinyali yok, kayıtlı başlangıç konumu kullanılıyor.")
            self._rec.log_command("%s,%s" % fb, intent="navigation",
                                  action="nav_fallback_origin")
            return (fb[0], fb[1], 0.0)
        return None

    # -- Deferred routing (no GPS fix yet) -----------------------------------

    def _defer_route(self, retry, label):
        """Queue a navigation request until GPS produces a fix.

        Urban canyons (e.g. Kızılay) often deny a fix exactly when the user
        issues a command. Refusing the command forces them to keep re-asking;
        instead we acknowledge it, watch for a fix on a small daemon thread,
        and start routing automatically — with an announcement — the moment
        one arrives. A newer command (or "iptal") replaces/cancels the wait.
        """
        # Replace any previous pending request.
        if self._pending_cancel is not None:
            self._pending_cancel.set()
        cancel = threading.Event()
        self._pending_cancel = cancel
        timeout = getattr(self._config.voice, "pending_route_timeout_sec", 180.0)
        self._voice.say_prompt(
            "GPS sinyali henüz yok. %s hedefiniz kaydedildi; sinyal bulununca "
            "yönlendirme otomatik başlayacak." % label.capitalize()
        )
        self._rec.log_command(label, intent="navigation", action="nav_deferred_no_gps")

        def _wait_for_fix():
            import time as _time
            deadline = _time.monotonic() + timeout
            while not (cancel.is_set() or self._stop.is_set()):
                if self._gps.get_coord() is not None:
                    self._voice.say_prompt("GPS sinyali bulundu.")
                    retry()
                    return
                if _time.monotonic() >= deadline:
                    self._voice.say_prompt(
                        "GPS sinyali hâlâ yok, bekleyen hedef iptal edildi. "
                        "Açık alanda tekrar deneyin."
                    )
                    return
                cancel.wait(self.PENDING_POLL_SEC)

        threading.Thread(target=_wait_for_fix, name="PendingRoute",
                         daemon=True).start()

    def _cancel_pending_route(self) -> bool:
        """Cancel a queued no-GPS destination. True if one was pending."""
        if self._pending_cancel is not None and not self._pending_cancel.is_set():
            self._pending_cancel.set()
            return True
        return False

    def _handle_system_command(self, text):
        """Sleep / cancel-route / shutdown."""
        text_lower = text.lower()

        # Sleep
        if any(w in text_lower for w in self.SLEEP_WORDS):
            self._voice.announce_sleep()
            self._modes.transition_to(SystemMode.SLEEP)
            return

        # Cancel current navigation (or a destination queued waiting for GPS)
        if any(w in text_lower for w in self.STOP_NAV_WORDS):
            if self._cancel_pending_route():
                self._voice.say_prompt("Bekleyen hedef iptal edildi.")
            elif self._nav.is_active:
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

    # -- Local queries (neredeyim / durum / kayıtlı yerler) -------------------

    def _handle_local_query(self, text):
        """Keyword-matched commands answered from local state. True if handled."""
        t = text.lower()
        tokens = t.replace(",", " ").split()
        place = next(
            (self.PLACE_ALIASES[w] for w in tokens if w in self.PLACE_ALIASES),
            None,
        )

        if place and "kaydet" in t:
            self._save_place(place)
            return True
        if place and any(w in tokens for w in ("git", "gidelim", "götür", "dön")):
            self._goto_place(place)
            return True
        if any(w in t for w in self.WHERE_WORDS):
            self._handle_where()
            return True
        if any(w in tokens for w in self.STATUS_WORDS):
            self._handle_status()
            return True
        # Token match keeps "okul" (school POI) from triggering a read.
        if "oku" in tokens or "tabela" in t or "yazıyor" in t or "yazı" in tokens:
            self._handle_read()
            return True
        return False

    def _handle_read(self):
        """"Oku" — OCR the last camera frame and speak what it says (C3)."""
        frame = getattr(self._perception, "last_frame", None)
        if frame is None:
            self._voice.say_prompt("Kamera görüntüsü yok, okuyamıyorum.")
            return
        self._voice.say_prompt("Okuyorum, lütfen sabit durun.")
        from ai.ocr_reader import read_sign
        text = read_sign(frame)
        if text is None:
            self._voice.say_prompt("Okuma özelliği bu cihazda kurulu değil.")
        elif not text:
            self._voice.say_prompt("Okunabilir bir yazı bulamadım.")
        else:
            self._voice.say_prompt("Şunu okudum: %s" % text[:200])

    def _save_place(self, place):
        """Bookmark the current GPS fix under ``place`` ("evi kaydet")."""
        fix = self._gps.get_coord()
        if fix is None:
            self._voice.say_prompt("GPS sinyali yok, konum kaydedilemedi.")
            return
        lat, lon, _ = fix
        try:
            self._places.save(place, lat, lon)
        except Exception:
            logger.exception("[Voice] saved-place write failed")
            self._voice.say_prompt("Konum kaydedilemedi.")
            return
        self._voice.say_prompt("%s konumu kaydedildi." % place.capitalize())

    def _goto_place(self, place):
        """Route to a previously saved bookmark ("eve git")."""
        coord = self._places.get(place)
        if coord is None:
            self._voice.say_prompt(
                "Kayıtlı %s konumu yok. Önce, %s konumunu kaydet, deyin."
                % (place, place)
            )
            return
        self._voice.say_prompt("%s konumuna rota hesaplanıyor." % place.capitalize())
        self.route_to_coord(coord[0], coord[1])

    def _handle_where(self):
        """"Neredeyim?" — nearest named road from the offline OSM graph."""
        fix = self._gps.get_coord()
        if fix is None:
            self._voice.say_prompt("GPS sinyali yok, konum belirlenemiyor.")
            return
        lat, lon, _ = fix
        try:
            road = self._nav.where_am_i(Coord(lat, lon))
        except Exception:
            logger.exception("[Voice] where_am_i failed")
            road = None
        if road:
            text = "%s üzerindesiniz." % road
        else:
            text = "Konumunuz haritadaki yolların dışında görünüyor."
        if self._nav.is_active:
            route = self._nav.get_route()
            if route:
                from navigation.router.geo_utils import haversine_distance
                dest = route[-1].location
                d = haversine_distance(lat, lon, dest.lat, dest.lon)
                text += " Hedefe %d metre kaldı." % int(d)
        self._voice.say_prompt(text)

    def _handle_status(self):
        """Spoken health summary: GPS quality, SoC temperature, route state."""
        parts = []
        try:
            h = self._gps.get_health()
            sats = getattr(h, "satellites", None)
            if self._gps.get_coord() is None:
                parts.append("GPS sinyali yok")
            elif sats is not None and sats < 5:
                parts.append("GPS sinyali zayıf")
            else:
                parts.append("GPS iyi")
        except Exception:  # noqa: BLE001
            parts.append("GPS durumu okunamadı")
        try:
            from main.session_recorder import read_soc_temps
            temps = read_soc_temps()
            if temps:
                hottest = max(temps.values())
                parts.append(
                    "sıcaklık yüksek" if hottest >= 70.0 else "sıcaklık normal"
                )
        except Exception:  # noqa: BLE001
            pass
        parts.append("rota aktif" if self._nav.is_active else "aktif rota yok")
        self._voice.say_prompt("Sistem durumu: %s." % ", ".join(parts))

    # -- Helpers ------------------------------------------------------------

    def _extract_category(self, text):
        text_lower = text.lower()
        for keyword, category in self.NAV_KEYWORDS.items():
            if keyword in text_lower:
                return category
        return None
