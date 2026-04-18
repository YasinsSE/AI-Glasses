"""
ButtonListener — physical GPIO button or keyboard mock.
=======================================================
Two backends:

    - **GPIO** (Jetson Nano)  : pull-up, active-low, polled every 50 ms.
    - **mock** (desktop test) : ENTER key. Uses ``select.select`` so the
                                 thread wakes promptly on shutdown instead
                                 of blocking inside ``input()``.

The listener is **mode-agnostic**. It just calls ``on_press`` — the callback
(``VoiceCommandHandler.handle_press``) is responsible for inspecting the
current ``SystemMode`` and deciding whether to wake from SLEEP or run an
STT session. Keeps this class trivial.
"""

#from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from main.config import ALASConfig
from main.lifecycle import ModeManager

logger = logging.getLogger("ALAS.button")


class ButtonListener:
    """Daemon thread that fires ``on_press`` on each button event."""

    def __init__(
        self,
        config: ALASConfig,
        on_press: Callable[[], None],
        modes: ModeManager,
        stop_event: threading.Event,
        mock: bool = False,
    ) -> None:
        self._config = config
        self._on_press = on_press
        self._modes = modes  # currently unused — kept for symmetry
        self._stop = stop_event
        self._mock = mock
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        target = self._keyboard_loop if self._mock else self._gpio_loop
        self._thread = threading.Thread(target=target, name="ButtonListener", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── GPIO backend (Jetson Nano) ───────────────────────────────

    def _gpio_loop(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore
        except Exception:  # noqa: BLE001
            logger.exception("[Button] Jetson.GPIO unavailable — listener exiting")
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._config.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        logger.info(f"[Button] GPIO pin {self._config.button_pin} ready (active-low).")

        try:
            while not self._stop.is_set():
                if GPIO.input(self._config.button_pin) == GPIO.LOW:
                    self._safe_callback()
                    # Debounce: wait then drain held-down state.
                    time.sleep(self._config.button_debounce_ms / 1000.0)
                    while (
                        GPIO.input(self._config.button_pin) == GPIO.LOW
                        and not self._stop.is_set()
                    ):
                        time.sleep(0.05)
                time.sleep(0.05)
        finally:
            try:
                GPIO.cleanup(self._config.button_pin)
            except Exception:  # noqa: BLE001
                pass

    # ── Mock backend (desktop) ───────────────────────────────────

    def _keyboard_loop(self) -> None:
        """Wakes promptly on shutdown via ``select.select``."""
        import select
        import sys

        logger.info("[Button] Mock mode — press ENTER to simulate button press.")
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            except (ValueError, OSError):
                # stdin was closed mid-shutdown
                break
            if not ready:
                continue
            try:
                line = sys.stdin.readline()
            except (EOFError, OSError):
                break
            if line == "":
                # EOF
                break
            if not self._stop.is_set():
                self._safe_callback()

    # ── Internal ─────────────────────────────────────────────────

    def _safe_callback(self) -> None:
        try:
            self._on_press()
        except Exception:  # noqa: BLE001
            logger.exception("[Button] on_press callback raised")
