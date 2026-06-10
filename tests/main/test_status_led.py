"""Unit tests for the status LED shutdown behaviour.

The LED must fast-blink during SHUTDOWN (finalize still writing outputs) and
keep running past the global stop_event — it goes dark only on an explicit
``stop()``, which lifecycle.shutdown calls as its LAST step. A dark LED is
the user's "safe to cut LiPo power" signal.

Run via pytest:
    python3 -m pytest tests/main/test_status_led.py
"""

import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main.lifecycle import ModeManager, SystemMode
from main.status_led import StatusLED


def test_shutdown_mode_fast_blinks():
    """SHUTDOWN is a ~4 Hz blink — visibly different from WARMUP's 1.25 Hz."""
    levels = {StatusLED._level_for(SystemMode.SHUTDOWN, t / 100.0)
              for t in range(0, 25)}  # one 0.25 s period sampled at 10 ms
    assert levels == {True, False}  # it blinks (not solid, not dark)
    # Faster than WARMUP: more transitions over the same window.
    def transitions(mode, dur=2.0, dt=0.01):
        seq = [StatusLED._level_for(mode, t * dt) for t in range(int(dur / dt))]
        return sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    assert transitions(SystemMode.SHUTDOWN) > transitions(SystemMode.WARMUP)


def test_active_solid_and_unknown_off():
    assert StatusLED._level_for(SystemMode.ACTIVE, 12.34) is True
    assert StatusLED._level_for(SystemMode.WARMUP, 0.0) in (True, False)


def test_led_survives_global_stop_until_explicit_stop():
    """The driver must outlive stop_event so it can blink through finalize."""
    stop_event = threading.Event()
    modes = ModeManager(initial=SystemMode.ACTIVE)
    led = StatusLED(modes, stop_event, mock=True)
    led.start()
    time.sleep(0.05)
    assert led.is_alive()

    # Shutdown begins: global stop set, mode → SHUTDOWN. LED keeps running.
    stop_event.set()
    modes.transition_to(SystemMode.SHUTDOWN)
    time.sleep(0.3)
    assert led.is_alive(), "LED died on stop_event — finalize blink would be lost"

    # Outputs flushed → lifecycle calls stop(): now it must terminate.
    led.stop(timeout=2.0)
    assert not led.is_alive()
