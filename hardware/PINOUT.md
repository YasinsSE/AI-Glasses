# ALAS — Jetson Nano Pin Map (Single Source of Truth)

Jetson Nano 40-pin header = **J41**. Two numbering schemes appear below — keep
them straight:

- **BCM** — what the code uses (`GPIO.setmode(GPIO.BCM)`).
- **Physical (J41)** — the actual pin you plug a wire into (1–40).

> ⚠️ **BCM ≠ physical.** "BCM 23" is **physical pin 16** — it is *not* physical
> pin 23 (that pin is an SPI line). Always wire by the **Physical** column.

---

## Buttons

| Function | BCM | Physical pin | Logic | Code reference |
| :-- | :-: | :-: | :-- | :-- |
| **Launch button** (boot/standby toggle) | **23** | **16** | active-low, pull-up | `src/main/boot_launcher.py` → `LAUNCH_BUTTON_PIN` |
| **PTT button** (push-to-talk / STT) | **18** | **12** | active-low, pull-up | `src/tts_stt/voice_config.py` → `button_pin` |

Both buttons are **active-low**: the line idles HIGH (pulled to 3.3 V) and goes
LOW when pressed (connected to GND). Falling edge = press.

### Per-button wiring (each button, identical pattern)

```
  3.3V ──[ 10kΩ pull-up ]──┬────────────────  GPIO pin (BCM 23 or BCM 18)
                           │
                           ├──[ push button ]──  GND
                           │
                           └──[ 100nF cap ]────  GND
```

- **10 kΩ pull-up to 3.3 V** — Jetson's internal pull-up (`PUD_UP`) is weak and
  unreliable; the external resistor holds a firm idle-HIGH.
- **100 nF (0.1 µF) ceramic cap, pin → GND** — with the 10 kΩ this forms an RC
  low-pass filter (τ ≈ 1 ms) that absorbs contact bounce. **This is the
  capacitor** that fixes "missed / multiple" presses — *not* a transistor.
- Software side complements this with interrupt edge-detect + `bouncetime`
  (`add_event_detect`), so presses are never lost between poll cycles.

### Suggested power/ground pins for the buttons

| Use | Physical pins |
| :-- | :-- |
| 3.3 V (pull-ups) | 1, 17 |
| GND | 6, 9, 14, 20, 25, 30, 34, 39 |

Recommended: **Launch** button GND = pin 14 (next to pin 16); **PTT** button
GND = pin 20 (next to pin 12).

---

## Reserved / in-use — DO NOT repurpose

| Function | Physical pins | Notes |
| :-- | :-- | :-- |
| **GPS UART** (`/dev/ttyTHS1`) | 8 (TXD), 10 (RXD) | `src/navigation/sensors/sensor_config.py` |
| I²C (bus 1) | 3 (SDA), 5 (SCL) | + ID-EEPROM on 27/28 |
| SPI | 19, 21, 23, 24, 26 | physical pin 23 = SPI CLK |
| Power | 1 (3.3V), 2/4 (5V), 17 (3.3V) | |
| GND | 6, 9, 14, 20, 25, 30, 34, 39 | |

---

## Free GPIO (for future peripherals)

BCM 4 (pin 7), BCM 17 (pin 11), BCM 27 (pin 13), BCM 22 (pin 15),
BCM 24 (pin 18), BCM 10/9/25/11/8/7, BCM 12 (pin 32), BCM 13 (pin 33),
BCM 19 (pin 35), BCM 16 (pin 36), BCM 26 (pin 37), BCM 20/21.

---

## Quick test checklist

1. **Launch button (BCM 23 / pin 16):** with the launcher running
   (`python -m main.boot_launcher` or the systemd service), press → ALAS starts;
   press again → ALAS stops (Jetson stays on); press again → ALAS restarts.
2. **PTT button (BCM 18 / pin 12):** with ALAS in ACTIVE mode, press → hear
   "Sizi dinliyorum." Rapid/double presses must not be lost (interrupt + RC cap).
3. Verify GPS still fixes — confirms UART pins 8/10 were left untouched.
