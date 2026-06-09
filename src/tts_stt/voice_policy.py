"""
VoicePolicy — TTS gating for ALAS.

Single chokepoint for all speech output. Enforces:
  - Active-utterance gate: skips perception inference while a nav/priority
    utterance is playing.
  - Post-nav silence window: mutes obstacle alerts for N seconds after
    a navigation instruction finishes.

Methods:
    announce_* / emergency  — priority, blocking
    say_nav                 — priority, blocking; sets post-nav silence window
    say_progress            — non-blocking, no silence window
    say_prompt              — priority, blocking, no silence window
    say_obstacle            — non-blocking, suppressed during silence window
"""

import logging
import threading
import time

from main.config import ALASConfig
from tts_stt.tts import shutdown_tts, speak, wait_until_done

logger = logging.getLogger("ALAS.voice")


class VoicePolicy:
    """Single source of truth for *all* TTS in ALAS."""

    def __init__(self, config: ALASConfig) -> None:
        self._cfg = config
        self._lock = threading.Lock()
        self._speaking_priority = False
        self._suppress_obstacles_until = 0.0
        # Set while a priority utterance is playing; perception waits on this
        # event instead of polling so it wakes the instant TTS finishes.
        self._idle_event = threading.Event()
        self._idle_event.set()
        from main.session_recorder import NullRecorder
        self._rec = NullRecorder()  # field-test recorder; replaced via set_recorder

    def set_recorder(self, recorder) -> None:
        """Attach the field-test recorder so every utterance is logged."""
        self._rec = recorder

    # ── High-level semantic methods ──────────────────────────────

    def announce_boot(self) -> None:
        self._priority_speak("ALAS sistemi hazırlanıyor.")
        self._rec.log_speak("announce", "ALAS sistemi hazırlanıyor.", True)

    def announce_ready(self) -> None:
        self._priority_speak("Sistem hazır. Butona basarak komut verebilirsiniz.")
        self._rec.log_speak("announce", "Sistem hazır. Butona basarak komut verebilirsiniz.", True)

    def announce_shutdown(self) -> None:
        self._priority_speak("Sistem kapatılıyor, iyi günler.")
        self._rec.log_speak("announce", "Sistem kapatılıyor, iyi günler.", True)

    def announce_sleep(self) -> None:
        self._priority_speak("Uyku moduna geçiliyor.")
        self._rec.log_speak("announce", "Uyku moduna geçiliyor.", True)

    def announce_wake(self) -> None:
        # Optional short audio cue (WAV) before the spoken confirmation, so the
        # user gets immediate feedback that the PTT press woke the system.
        wav = getattr(getattr(self._cfg, "idle", None), "wake_cue_wav", "")
        if wav:
            self._play_wake_cue(wav)
        self._priority_speak("Sistem aktif.")
        self._rec.log_speak("announce", "Sistem aktif.", True)

    @staticmethod
    def _play_wake_cue(path: str) -> None:
        """Play a short WAV via aplay (non-blocking). Silently no-ops on failure."""
        import os
        import subprocess
        if not path or not os.path.isfile(path):
            return
        try:
            subprocess.Popen(
                ["aplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001
            logger.debug("[Voice] wake cue playback failed", exc_info=True)

    def emergency(self, text: str) -> None:
        self._priority_speak(text)
        self._rec.log_speak("emergency", text, True)

    def say_nav(self, text: str) -> None:
        """Blocking nav instruction. Sets active gate AND post-utterance window."""
        self._priority_speak(text)
        with self._lock:
            self._suppress_obstacles_until = time.monotonic() + self._cfg.voice.post_nav_silence_sec
        self._rec.log_speak("nav", text, True)

    def say_progress(self, text: str) -> None:
        """Non-blocking distance announcement. No gates. No silence window."""
        speak(text)
        self._rec.log_speak("progress", text, True)

    def say_prompt(self, text: str) -> None:
        """Voice UI prompt (e.g. 'Sizi dinliyorum'). Blocking, no silence window."""
        self._priority_speak(text)
        self._rec.log_speak("prompt", text, True)

    def say_drift(self, direction: str, fallback_text: str) -> None:
        """Path-keeping cue (C4): panned earcon when available, else speech.

        ``direction`` is "left" | "right" | "straight". The repetitive
        "hafif sola/sağa" corrections are the main source of verbal fatigue
        in field tests; a 0.3 s stereo beep carries the same information.
        Gated exactly like obstacle speech (post-nav silence window).
        """
        with self._lock:
            if time.monotonic() < self._suppress_obstacles_until:
                self._rec.log_speak("earcon", direction, False, reason="post_nav_silence")
                return
        wav = self._earcon_path(direction)
        if wav is not None:
            try:
                import subprocess
                subprocess.Popen(
                    ["aplay", "-q", wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._rec.log_speak("earcon", direction, True)
                return
            except Exception:  # noqa: BLE001 — no aplay → speech fallback
                logger.debug("[Voice] earcon playback failed", exc_info=True)
        self.say_obstacle(fallback_text)

    def _earcon_path(self, direction: str):
        """Resolve drift_<direction>.wav, or None when earcons can't play."""
        import os
        if not getattr(self._cfg.voice, "earcons_enabled", False):
            return None
        base = getattr(self._cfg.voice, "earcon_dir", "") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "earcons")
        path = os.path.join(base, f"drift_{direction}.wav")
        return path if os.path.isfile(path) else None

    def say_obstacle(self, text: str, urgent: bool = False, preempt: bool = False) -> None:
        """Non-blocking obstacle alert. Suppressed during the post-nav silence window.

        ``urgent=True`` (closing-threat warnings) is spoken faster + higher-pitched.
        ``preempt=True`` (imminent collision) cuts off whatever is currently
        playing AND bypasses the post-nav silence window — a collision warning
        must be heard immediately, even right after a turn instruction.
        """
        if not preempt:
            with self._lock:
                if time.monotonic() < self._suppress_obstacles_until:
                    self._rec.log_speak("obstacle", text, False, reason="post_nav_silence")
                    return
        speak(text, kind="obstacle", urgent=urgent, preempt=preempt)
        self._rec.log_speak("obstacle", text, True)

    # ── Polled by services ───────────────────────────────────────

    def is_speaking_priority(self) -> bool:
        with self._lock:
            return self._speaking_priority

    def in_post_nav_silence(self) -> bool:
        """True while obstacle alerts are muted after a navigation utterance."""
        with self._lock:
            return time.monotonic() < self._suppress_obstacles_until

    def wait_until_idle(self, timeout: float) -> bool:
        """Block until no priority utterance is playing, or timeout elapses."""
        return self._idle_event.wait(timeout)

    # ── Lifecycle ────────────────────────────────────────────────

    def flush(self) -> None:
        wait_until_done()

    def shutdown(self) -> None:
        shutdown_tts()

    # ── Internal ─────────────────────────────────────────────────

    def _priority_speak(self, text: str) -> None:
        with self._lock:
            self._speaking_priority = True
        self._idle_event.clear()
        try:
            speak(text)
            wait_until_done()
        finally:
            with self._lock:
                self._speaking_priority = False
            self._idle_event.set()