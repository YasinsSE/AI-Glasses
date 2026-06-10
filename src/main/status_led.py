"""
StatusLED — single mode-indicator LED for ALAS.
================================================
A status LED on **BCM 24 (J41 physical pin 18)** shows the system mode at a
glance. It is driven by ``alas_main`` — the process that actually knows the
``SystemMode`` — so one LED conveys everything:

    IDLE  (alas_main not running) : OFF   (pin released → LED dark)
    WARMUP (booting)              : steady blink   (~1.25 Hz)
    ACTIVE (running / working)    : SOLID ON
    SLEEP  (auto-STANDBY)         : brief heartbeat pulse every 2 s
    SHUTDOWN (finalizing)         : FAST blink (~4 Hz) — outputs still being
                                    written; do NOT cut LiPo power yet
    stopped (safe to cut power)   : OFF

The fast SHUTDOWN blink exists because the recorder's ``finalize()`` (final
summary, viewer.html, GPX) runs at the very END of the teardown — if the LED
went dark at the start of shutdown, the user would cut battery power while
those files were still being written, truncating the session outputs. The LED
therefore keeps blinking through the whole teardown and only turns off when
``lifecycle.shutdown`` explicitly calls ``StatusLED.stop()`` as its last step.
This is why the driver runs on its OWN stop flag, not the global stop_event
(which is already set while shutdown is still in progress).

**Single owner:** only this driver touches BCM 24, and on exit it cleans up
ONLY that pin (never a global ``GPIO.cleanup()``), so the launcher's BCM 23 and
the PTT button's BCM 25 are left untouched.

**Active-LOW wiring (important).** Jetson GPIO pins idle HIGH when no process
drives them, so in IDLE (alas_main not running) an active-HIGH LED would stay
lit. We therefore SINK current: wire ``3.3 V → ~330–470 Ω → LED anode →
LED cathode → pin 18``. The pin LOW = LED on; pin HIGH (the default) = LED off,
so the LED is dark in IDLE with nothing driving it. Set ``active_low=False``
only if you wired the LED the other way (pin → R → LED → GND).

Keep current low — Jetson GPIO drive is limited; use a transistor/MOSFET for a
bright LED. See ``hardware/PINOUT.md``.
"""

import logging
import threading
import time
from typing import Optional

from main.lifecycle import ModeManager, SystemMode

logger = logging.getLogger("ALAS.status_led")

STATUS_LED_PIN = 24  # BCM numbering — J41 physical pin 18. See hardware/PINOUT.md.


class StatusLED:
    """Daemon thread that drives a mode-indicator LED from the SystemMode."""

    name = "StatusLED"

    def __init__(
        self,
        modes: ModeManager,
        stop_event: threading.Event,
        mock: bool = False,
        pin: int = STATUS_LED_PIN,
        active_low: bool = True,
    ) -> None:
        self._modes = modes
        self._stop = stop_event  # kept for API symmetry; the loop does NOT exit on it
        self._mock = mock
        self._pin = pin
        self._active_low = active_low  # LED sinks current; pin LOW = on (idle-HIGH safe).
        self._thread: Optional[threading.Thread] = None
        # Own stop flag: the LED must outlive the global stop_event so it can
        # blink "finalizing" through the whole shutdown choreography.
        self._led_stop = threading.Event()

    def start(self) -> None:
        target = self._mock_loop if self._mock else self._gpio_loop
        self._thread = threading.Thread(target=target, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Turn the LED off and stop the driver — call ONLY after all outputs
        are flushed (lifecycle.shutdown's last step). LED dark = power-safe."""
        self._led_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── Blink pattern ────────────────────────────────────────────
    @staticmethod
    def _level_for(mode: SystemMode, t: float) -> bool:
        """Return the desired LED level (on/off) for ``mode`` at time ``t``."""
        if mode == SystemMode.ACTIVE:
            return True                       # solid ON
        if mode == SystemMode.WARMUP:
            return (t % 0.8) < 0.4            # steady ~1.25 Hz blink
        if mode == SystemMode.SLEEP:
            return (t % 2.0) < 0.1            # brief heartbeat every 2 s
        if mode == SystemMode.SHUTDOWN:
            return (t % 0.25) < 0.125         # FAST ~4 Hz: finalizing, keep power on
        return False                          # unknown → off

    # ── GPIO backend (Jetson Nano) ───────────────────────────────
    def _gpio_loop(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore
        except Exception:  # noqa: BLE001
            logger.exception("[LED] Jetson.GPIO unavailable — status LED disabled")
            return

        # Map logical on/off to physical pin levels. Active-low (default): pin
        # LOW = LED on, pin HIGH = off — so the idle-HIGH pin keeps the LED dark.
        on_level = GPIO.LOW if self._active_low else GPIO.HIGH
        off_level = GPIO.HIGH if self._active_low else GPIO.LOW

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin, GPIO.OUT, initial=off_level)
        logger.info(
            f"[LED] status LED ready on BCM {self._pin} (physical pin 18, "
            f"active-{'low' if self._active_low else 'high'})."
        )

        last: Optional[bool] = None
        try:
            # Runs until stop() — NOT until the global stop_event — so the
            # fast SHUTDOWN blink covers the recorder finalize at teardown end.
            while not self._led_stop.is_set():
                level = self._level_for(self._modes.mode, time.monotonic())
                if level != last:
                    GPIO.output(self._pin, on_level if level else off_level)
                    last = level
                self._led_stop.wait(0.05)
        finally:
            try:
                GPIO.output(self._pin, off_level)
            except Exception:  # noqa: BLE001
                pass
            try:
                GPIO.cleanup(self._pin)  # ONLY our pin — never a global cleanup().
            except Exception:  # noqa: BLE001
                pass

    # ── Mock backend (desktop) ───────────────────────────────────
    def _mock_loop(self) -> None:
        logger.info("[LED] mock mode — status LED simulated (no GPIO).")
        last: Optional[SystemMode] = None
        while not self._led_stop.is_set():
            mode = self._modes.mode
            if mode != last:
                logger.debug("[LED] mode → %s", mode.value)
                last = mode
            self._led_stop.wait(0.2)
