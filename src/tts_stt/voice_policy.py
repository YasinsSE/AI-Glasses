"""
VoicePolicy — central speech gate for ALAS.
============================================
The user's #1 complaint about the draft main loop was: "TTS keeps talking and
disturbs me." This module is the single chokepoint that enforces TTS
discipline.

Two orthogonal mechanisms:

    1. **Active-utterance gate** (`is_speaking_priority`): True only *while*
       a high-priority utterance is being spoken. PerceptionService polls
       this and skips inference entirely so it does not queue stale alerts
       behind the live navigation announcement.

    2. **Post-utterance silence window** (`_suppress_obstacles_until`):
       After a navigation utterance finishes, obstacle alerts are silently
       dropped for ``post_nav_silence_sec`` seconds. Perception still runs;
       it just does not speak.

Semantic methods:

    - ``announce_*``  : boot/ready/sleep/wake announcements (priority, blocking)
    - ``emergency``   : error / warmup-timeout / camera-fail (priority, blocking)
    - ``say_nav``     : turn instructions (priority, blocking, sets silence window)
    - ``say_progress``: distance pings (non-blocking, NO silence window)
    - ``say_prompt``  : voice-UI prompts (priority, blocking, NO silence window)
    - ``say_obstacle``: hazard alerts (non-blocking, suppressed during window)

The lock is held only around state mutation, never across the (slow) call to
``speak() / wait_until_done()``. Holding it across audio playback would
deadlock perception.
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

    # ── High-level semantic methods ──────────────────────────────

    def announce_boot(self) -> None:
        self._priority_speak("ALAS sistemi hazırlanıyor.")

    def announce_ready(self) -> None:
        self._priority_speak("Sistem hazır. Butona basarak komut verebilirsiniz.")

    def announce_shutdown(self) -> None:
        self._priority_speak("Sistem kapatılıyor, iyi günler.")

    def announce_sleep(self) -> None:
        self._priority_speak("Uyku moduna geçiliyor.")

    def announce_wake(self) -> None:
        self._priority_speak("Sistem aktif.")

    def emergency(self, text: str) -> None:
        self._priority_speak(text)

    def say_nav(self, text: str) -> None:
        """Blocking nav instruction. Sets active gate AND post-utterance window."""
        self._priority_speak(text)
        with self._lock:
            self._suppress_obstacles_until = time.monotonic() + self._cfg.post_nav_silence_sec

    def say_progress(self, text: str) -> None:
        """Non-blocking distance announcement. No gates. No silence window."""
        speak(text)

    def say_prompt(self, text: str) -> None:
        """Voice UI prompt (e.g. 'Sizi dinliyorum'). Blocking, no silence window."""
        self._priority_speak(text)

    def say_obstacle(self, text: str) -> None:
        """Non-blocking obstacle alert. Suppressed during the post-nav silence window."""
        with self._lock:
            if time.monotonic() < self._suppress_obstacles_until:
                return
        speak(text)

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