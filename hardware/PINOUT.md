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

### Two-button board — shared 3.3 V and GND rails

Both buttons live on one board and **share a common 3.3 V rail and a common GND
rail**. Shared = power + ground; per-button = its own signal pin + its own
10 kΩ pull-up. Only **4 wires** go to the Jetson.

```
3.3V (pin 1) ─────────────┬──────────────────┬─────────   ← shared 3.3V rail
                       [10kΩ]              [10kΩ]
                          │                   │
PTT  (pin 12 / BCM 18) ───┤                   ├──── Launch (pin 16 / BCM 23)
                          │                   │
                    [PTT button]        [Launch button]
                          │                   │
GND  (pin 14) ────────────┴──────────────────┴─────────   ← shared GND rail
```

| Wire | Jetson pin |
| :-- | :-- |
| Shared 3.3 V (feeds both pull-ups) | pin 1 |
| Shared GND (both buttons)          | pin 14 |
| PTT signal                         | pin 12 (BCM 18) |
| Launch signal                      | pin 16 (BCM 23) |

- **10 kΩ pull-up to 3.3 V** per button — Jetson's internal `PUD_UP` is weak and
  unreliable; the external resistor holds a firm idle-HIGH. (Internal `PUD_UP`
  is left enabled in code as a harmless parallel fallback.)
- **NO capacitor** on either button — see the warning below.

### ⚠️ Why no capacitor? (hardware debounce removed — do NOT add it back)

An earlier design put a 100 nF cap pin→GND for RC debounce. It made the Jetson
**freeze and reset on button *release***. Root cause: 10 kΩ × 100 nF ≈ **1 ms**
rise time. On release the pin voltage crawls up through the input's "undefined
region" (~1.0–2.0 V); the Tegra X1 GPIO input has weak/no Schmitt-trigger
hysteresis, so the input buffer's N-MOS and P-MOS conduct simultaneously (CMOS
**shoot-through / crowbar** current) → a momentary VDD→GND short inside the SoC
→ local brownout → reset. Removing the cap (edge now snaps via stray
capacitance) fixed it instantly.

**Debounce is therefore handled in SOFTWARE, not hardware. Exact locations:**

| Button | File | Parameter(s) |
| :-- | :-- | :-- |
| **PTT** | `src/tts_stt/button_listener.py` → `_gpio_loop()` `GPIO.add_event_detect(..., bouncetime=…)` | `bouncetime` ← `src/tts_stt/voice_config.py` → `button_debounce_ms` (**300 ms**) |
| **Launch** | `src/main/boot_launcher.py` | `BUTTON_DEBOUNCE_MS` (**500 ms**) + `MIN_TOGGLE_INTERVAL_SEC` (**3.0 s** cooldown) + `_confirm_press()` (`PRESS_CONFIRM_SAMPLES` = **5**) |

> If hardware debounce is ever truly required, do NOT use a bare RC cap on the
> GPIO. Use a **74HC14 Schmitt-trigger** buffer between the RC node and the pin,
> or a much smaller cap (≤ 10 nF) so the edge clears the undefined region in
> nanoseconds.

### Suggested power/ground pins for the buttons

| Use | Physical pins |
| :-- | :-- |
| 3.3 V (pull-ups) | 1, 17 |
| GND | 6, 9, 14, 20, 25, 30, 34, 39 |

For the **shared two-button board**: tap 3.3 V from **pin 1** and GND from
**pin 14**; both buttons share those two rails (see the schematic above).

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
   "Sizi dinliyorum." Rapid/double presses must not be lost (interrupt + software
   debounce). In STANDBY a PTT press wakes the system instead (no STT).
3. Verify GPS still fixes — confirms UART pins 8/10 were left untouched.