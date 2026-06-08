#!/usr/bin/env python3
"""ALAS Boot Launcher — physical "Launch" button watcher.
=========================================================
Runs at boot (systemd: ``alas-launcher.service``) and watches a dedicated GPIO
push-button — **BCM 23 / J41 physical pin 16** (see ``hardware/PINOUT.md``).
The button TOGGLES the ALAS main loop:

    IDLE     --(press)-->  start alas_main subprocess  -->  RUNNING
    RUNNING  --(press)-->  stop alas_main (whole group) -->  IDLE

The Jetson itself NEVER powers off here — only ``alas_main`` starts and stops.
Power is cut at the wall. This gives a monitor-free, reset-free test loop:
press to go live, test outside, press to return to standby, press to go live
again — all without re-attaching a screen.

Design constraints (deliberate, do not "simplify" away):
  * **Process-group termination.** alas_main spawns background threads and
    child processes (pyttsx3 TTS engine, black-box recorder). We launch it with
    ``start_new_session=True`` (its own session/process-group) and stop it with
    ``os.killpg(os.getpgid(pid), SIGINT)`` so NO orphan/zombie is left holding a
    GPIO pin or the audio device on the next launch.
  * **Lightweight ISR.** The GPIO edge callback only sets a ``threading.Event``
    and returns; all real work happens in the main thread.
  * **Targeted GPIO cleanup.** We only ever clean OUR pin (BCM 23). A bare
    ``GPIO.cleanup()`` would reset every pin and kill alas_main's PTT button.

Install (on the Jetson):
    sudo cp deploy/alas-launcher.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now alas-launcher.service
    journalctl -u alas-launcher -f
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time

# ── Configuration ────────────────────────────────────────────────────────
LAUNCH_BUTTON_PIN = 23      # BCM numbering — J41 physical pin 16. See hardware/PINOUT.md.
BUTTON_DEBOUNCE_MS = 500    # Hardware switch bounce guard (complements the RC debounce).
STOP_GRACE_SEC = 30.0       # Allow alas_main this long for an ordered shutdown
                            # (camera + TRT teardown + TTS drain) before escalating
                            # to SIGTERM/SIGKILL. Generous so a normal stop stays a
                            # single clean SIGINT (no double "shutdown signal").

# Defence-in-depth against a noisy/floating line (the real fix is the RC cap +
# pull-up — see hardware/PINOUT.md):
#   * MIN_TOGGLE_INTERVAL_SEC: ignore any press within this window of the last
#     action, so phantom double-edges can't start-then-instantly-kill ALAS and
#     thrash the Jetson.
#   * PRESS_CONFIRM_SAMPLES/_GAP: a real press holds the line LOW; require it to
#     read LOW across several samples before acting, rejecting transient spikes.
MIN_TOGGLE_INTERVAL_SEC = 3.0
PRESS_CONFIRM_SAMPLES = 5
PRESS_CONFIRM_GAP_SEC = 0.01

# Directory that ``python -m main.alas_main`` must run from (the src/ root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/main
SRC_DIR = os.path.dirname(_THIS_DIR)                      # .../src

# The command alas_main is launched with. TEST configuration for now; edit this
# single list once field tests finish.
#   --bypass-stt    : no microphone — skip STT.
#   --record        : field-test black-box recorder (outputs/field_tests/<ts>/).
#   --auto-standby  : power-save STANDBY after ~idle_enter_sec of stillness.
#   --auto-nav eczane : no-mic auto-route to nearest pharmacy on startup; a PTT
#                       press re-triggers the same route.
LAUNCH_CMD = [
    sys.executable, "-m", "main.alas_main",
    "--model", "models/segmentation/alas_engine.trt",
    "--bypass-stt",
    "--record",
    "--auto-standby",
    # Test destination - A PTT press re-routes here from the current spot.
    "--auto-nav-coord", "39.988679,32.863508",
    # press now also saves frames to outputs/dataset_raw/. DELETE this single
    # line once data collection is done — we won't be gathering training data
    # again. (Add "--capture-masks" on its own line for Roboflow label-assist.)
    "--capture-dataset",
]

logger = logging.getLogger("ALAS.launcher")


class LaunchWatcher:
    """Toggle the ALAS main loop on each physical button press."""

    def __init__(self) -> None:
        self._proc = None                  # subprocess.Popen or None
        self._press = threading.Event()    # flagged by the GPIO ISR
        self._stop = threading.Event()     # flagged by SIGINT/SIGTERM (systemd stop)
        self._last_action = 0.0            # monotonic time of last start/stop toggle
        self._gpio = None                  # set in run() — used by _confirm_press

    # ── GPIO ISR (background C-thread — keep trivial) ────────────────────
    def _on_edge(self, channel) -> None:
        self._press.set()

    # ── Subprocess lifecycle ─────────────────────────────────────────────
    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        logger.info("Launch button → starting ALAS main loop ...")
        # start_new_session=True puts alas_main in its own process group so we
        # can signal the whole tree (TTS subprocess, recorder, threads) at once.
        self._proc = subprocess.Popen(LAUNCH_CMD, cwd=SRC_DIR, start_new_session=True)
        logger.info("ALAS started (pid=%d). Press again to return to standby.", self._proc.pid)

    def _terminate(self) -> None:
        if self._proc is None:
            return
        pid = self._proc.pid
        logger.info("Launch button → stopping ALAS main loop (pid=%d) ...", pid)
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            self._proc = None
            return

        try:
            # SIGINT to the GROUP → triggers alas_main's ordered shutdown for the
            # whole tree, leaving no orphan to block the GPIO/audio on next launch.
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            self._proc = None
            return

        try:
            self._proc.wait(timeout=STOP_GRACE_SEC)
        except subprocess.TimeoutExpired:
            logger.warning(
                "ALAS did not exit within %.0fs — escalating to SIGTERM/SIGKILL.",
                STOP_GRACE_SEC,
            )
            self._escalate(pgid)

        self._proc = None
        logger.info("ALAS stopped. Standby (IDLE) — press to go live again.")

    def _escalate(self, pgid) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                self._proc.wait(timeout=5.0)
                return
            except subprocess.TimeoutExpired:
                continue

    # ── Press validation (noise rejection) ───────────────────────────────
    def _confirm_press(self) -> bool:
        """A real press holds the line LOW; reject transient spikes/noise."""
        if self._gpio is None:
            return True
        for _ in range(PRESS_CONFIRM_SAMPLES):
            if self._gpio.input(LAUNCH_BUTTON_PIN) != self._gpio.LOW:
                return False
            time.sleep(PRESS_CONFIRM_GAP_SEC)
        return True

    # ── Main loop ────────────────────────────────────────────────────────
    def run(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore
        except Exception:  # noqa: BLE001
            logger.exception("Jetson.GPIO unavailable — launcher cannot run.")
            return

        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LAUNCH_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        try:
            GPIO.add_event_detect(
                LAUNCH_BUTTON_PIN, GPIO.FALLING,
                callback=self._on_edge, bouncetime=BUTTON_DEBOUNCE_MS,
            )
        except Exception:  # noqa: BLE001
            logger.exception("add_event_detect failed on BCM %d — launcher exiting.", LAUNCH_BUTTON_PIN)
            GPIO.cleanup(LAUNCH_BUTTON_PIN)
            return

        logger.info(
            "Launch watcher ready on BCM %d (physical pin 16). Standby — press to start ALAS.",
            LAUNCH_BUTTON_PIN,
        )
        try:
            while not self._stop.is_set():
                if self._press.wait(timeout=1.0):
                    self._press.clear()
                    now = time.monotonic()
                    # Cooldown: ignore presses too soon after the last toggle.
                    # Kills the start-then-instant-kill thrash from noisy edges.
                    if (now - self._last_action) < MIN_TOGGLE_INTERVAL_SEC:
                        continue
                    # Reject phantom edges that did not settle into a real press.
                    if not self._confirm_press():
                        logger.debug("Ignoring unconfirmed (noise) edge on BCM %d.", LAUNCH_BUTTON_PIN)
                        continue
                    if self._is_running():
                        self._terminate()
                    else:
                        self._spawn()
                    self._last_action = time.monotonic()
                    # Drop any bounce-queued extra press fired during the action.
                    self._press.clear()
                elif self._proc is not None and self._proc.poll() is not None:
                    # alas_main exited by itself (e.g. spoken "kapat" command).
                    logger.info(
                        "ALAS exited on its own (rc=%s). Back to standby.",
                        self._proc.returncode,
                    )
                    self._proc = None
        finally:
            if self._is_running():
                self._terminate()
            try:
                GPIO.remove_event_detect(LAUNCH_BUTTON_PIN)
            except Exception:  # noqa: BLE001
                pass
            # Clean ONLY our launch pin — never a global GPIO.cleanup().
            GPIO.cleanup(LAUNCH_BUTTON_PIN)
            logger.info("Launch watcher stopped.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    watcher = LaunchWatcher()

    def _sig(signum, frame):
        logger.info("Launcher received signal %d — shutting down.", signum)
        watcher._stop.set()
        watcher._press.set()  # wake the main loop immediately

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    watcher.run()


if __name__ == "__main__":
    main()
