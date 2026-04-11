"""
ALAS Lifecycle — system modes, signal handling, and ordered shutdown.
=====================================================================
The main loop should not own any of this. Put it here so ``alas_main.py``
reads as a recipe.

Three responsibilities:
    1. ``SystemMode`` / ``ModeManager`` — the WARMUP/ACTIVE/SLEEP/SHUTDOWN
       state machine. Services poll ``modes.mode`` at the top of their loop
       to know whether to do real work or just sleep.
    2. ``install_signal_handlers`` — translates SIGINT/SIGTERM into a
       ``threading.Event`` that every service watches.
    3. ``await_ready`` and ``shutdown`` — choreograph the bring-up and
       tear-down sequences. The shutdown order is *load-bearing* and must
       not be reordered without thinking.
"""

#from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Dict
from enum import Enum
from typing import Optional

logger = logging.getLogger("ALAS.lifecycle")


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM MODES
# ═══════════════════════════════════════════════════════════════════

class SystemMode(Enum):
    WARMUP   = "warmup"     # GPS warming, model loading; no perception/nav loops
    ACTIVE   = "active"     # full pipeline running
    SLEEP    = "sleep"      # low-power: services paused, only button listener active
    SHUTDOWN = "shutdown"   # graceful exit


class ModeManager:
    """Thread-safe holder for the current ``SystemMode``."""

    def __init__(self, initial: SystemMode = SystemMode.WARMUP) -> None:
        self._mode = initial
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @property
    def mode(self) -> SystemMode:
        with self._lock:
            return self._mode

    def transition_to(self, new: SystemMode) -> None:
        with self._cond:
            if self._mode == new:
                return
            old = self._mode
            self._mode = new
            self._cond.notify_all()
        logger.info(f"[Mode] {old.value} → {new.value}")

    def wait_for(self, target: SystemMode, timeout: Optional[float] = None) -> bool:
        """Block until the manager reaches ``target``. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._mode != target:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if deadline is not None and remaining == 0.0:
                    return False
                self._cond.wait(timeout=remaining)
            return True

    def is_running(self) -> bool:
        return self.mode == SystemMode.ACTIVE


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL HANDLING
# ═══════════════════════════════════════════════════════════════════

def install_signal_handlers() -> threading.Event:
    """
    Install SIGINT/SIGTERM handlers that set a shared ``stop_event``.
    Returns the event so the caller can pass it to every service.
    """
    stop_event = threading.Event()

    def _handler(signum, _frame):
        logger.info(f"Shutdown signal received ({signum}).")
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stop_event


def wait_for_shutdown(stop_event: threading.Event) -> None:
    """Idle the main thread until ``stop_event`` is set (or Ctrl+C)."""
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()


# ═══════════════════════════════════════════════════════════════════
#  WARMUP — wait until everything is ready, then promote to ACTIVE
# ═══════════════════════════════════════════════════════════════════

def await_ready(
    modes: ModeManager,
    gps,
    perception,
    voice,
    timeout_sec: float,
) -> None:
    """
    Block until GPS reports a usable fix AND PerceptionService has finished
    loading the model. On success, transition modes to ACTIVE. On timeout,
    speak an emergency message and force-promote so the user is not locked
    out of the system entirely.

    ``perception`` may be ``None`` (when ``--no-camera`` is set) — in that
    case the perception readiness check is skipped.
    """
    from navigation.sensors import GPSStatus

    deadline = time.monotonic() + timeout_sec
    voice.announce_boot()  # idempotent — main() already called it once

    while time.monotonic() < deadline:
        try:
            gps_health = gps.get_health()
            gps_ok = gps_health.status in (GPSStatus.OK, GPSStatus.LOW_ACCURACY)
        except Exception:  # noqa: BLE001 — GPS may not be ready yet
            gps_ok = False

        model_ok = perception is None or perception.model_ready.is_set()

        if gps_ok and model_ok:
            modes.transition_to(SystemMode.ACTIVE)
            return
        time.sleep(0.5)

    logger.warning("[Lifecycle] Warmup timeout — promoting to ACTIVE anyway.")
    voice.emergency("Hazırlık zaman aşımı, sınırlı modda devam ediyorum.")
    modes.transition_to(SystemMode.ACTIVE)


# ═══════════════════════════════════════════════════════════════════
#  ORDERED SHUTDOWN
# ═══════════════════════════════════════════════════════════════════

def shutdown(*, button, services, nav, gps, voice, modes: ModeManager) -> None:
    """
    Stop everything in dependency order:

      1. button.join(timeout=1)        ← stop accepting new commands FIRST
      2. nav.stop_navigation()         ← stop generating nav messages
      3. for s in services: s.join(timeout=3)
      4. voice.flush()                 ← drain TTS queue (wait_until_done)
      5. voice.shutdown()              ← kill TTS worker (shutdown_tts)
      6. gps.stop()                    ← any time after step 3, kept last

    The order matters: pulling the camera before draining audio leaves
    half-spoken alerts queued; closing TTS before nav stops means the
    "navigation cancelled" announcement is silently dropped.
    """
    logger.info("[Lifecycle] Shutdown starting.")
    modes.transition_to(SystemMode.SHUTDOWN)

    # 1. Stop accepting new button presses.
    try:
        button.join(timeout=1.0)
    except Exception:  # noqa: BLE001
        logger.exception("[Lifecycle] button.join failed")

    # 2. Stop generating new nav messages.
    try:
        if getattr(nav, "is_active", False):
            nav.stop_navigation()
    except Exception:  # noqa: BLE001
        logger.exception("[Lifecycle] nav.stop_navigation failed")

    # 3. Join service threads.
    for service in services:
        if service is None:
            continue
        try:
            service.join(timeout=3.0)
        except Exception:  # noqa: BLE001
            logger.exception(f"[Lifecycle] {service} join failed")

    # 4. Drain TTS queue so the goodbye is actually heard.
    try:
        voice.announce_shutdown()
        voice.flush()
    except Exception:  # noqa: BLE001
        logger.exception("[Lifecycle] voice.flush failed")

    # 5. Kill the TTS worker.
    try:
        voice.shutdown()
    except Exception:  # noqa: BLE001
        logger.exception("[Lifecycle] voice.shutdown failed")

    # 6. Stop the GPS reader.
    try:
        gps.stop()
    except Exception:  # noqa: BLE001
        logger.exception("[Lifecycle] gps.stop failed")

    logger.info("[Lifecycle] Shutdown complete.")
