#!/usr/bin/env python3
"""Generate the stereo earcon WAVs used for path-keeping cues (C4).

    python3 scripts/generate_earcons.py [OUT_DIR]

Writes drift_left.wav / drift_right.wav / drift_straight.wav (default:
src/tts_stt/earcons/). Stdlib only — runs the same on the Mac and the Jetson.

Design: direction is carried by STEREO PAN (left ear = drift left → step
left), urgency stays with speech. 880 Hz sits in the most sensitive hearing
band and remains audible over traffic on small glasses speakers; "straight"
is a softer, centred double-blip used rarely as a confirmation.
"""

import math
import os
import struct
import sys
import wave

RATE = 22050


def _tone(freq_hz, dur_s, pan, fade_s=0.02, amp=0.8):
    """Stereo sine samples. pan: -1 = full left, 0 = centre, +1 = full right."""
    n = int(RATE * dur_s)
    left_gain = min(1.0, 1.0 - pan)   # pan -1 → 1.0, pan +1 → 0.0 (plus floor below)
    right_gain = min(1.0, 1.0 + pan)
    # Keep a -20 dB floor in the far ear so single-ear listeners still hear it.
    left_gain = max(left_gain, 0.1)
    right_gain = max(right_gain, 0.1)
    frames = bytearray()
    fade_n = max(1, int(RATE * fade_s))
    for i in range(n):
        env = min(1.0, i / fade_n, (n - 1 - i) / fade_n)  # click-free edges
        s = amp * env * math.sin(2.0 * math.pi * freq_hz * i / RATE)
        frames += struct.pack(
            "<hh", int(32767 * s * left_gain), int(32767 * s * right_gain)
        )
    return frames


def _silence(dur_s):
    return bytes(4 * int(RATE * dur_s))


def _write(path, frames):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames)
    print(f"  wrote {path}")


def main():
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "src", "tts_stt", "earcons")
    out = os.path.normpath(sys.argv[1] if len(sys.argv) > 1 else default)
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(out, "drift_left.wav"), _tone(880, 0.30, pan=-1.0))
    _write(os.path.join(out, "drift_right.wav"), _tone(880, 0.30, pan=+1.0))
    _write(os.path.join(out, "drift_straight.wav"),
           bytes(_tone(660, 0.12, pan=0.0, amp=0.5)) + _silence(0.08)
           + bytes(_tone(660, 0.12, pan=0.0, amp=0.5)))


if __name__ == "__main__":
    main()
