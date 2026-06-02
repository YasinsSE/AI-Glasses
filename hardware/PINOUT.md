# ALAS — Jetson Nano Pin Map

Jetson Nano 40-pin header = **J41**. Code uses **BCM** numbering
(`GPIO.setmode(GPIO.BCM)`); you wire by **physical** pin — they differ
(e.g. **BCM 23 = physical pin 16**). Wire by the *Physical* column.

## Connections

| Physical | BCM | Signal | Dir | Code reference | Note |
| :-: | :-: | :-- | :-: | :-- | :-- |
| **1** | — | 3.3 V | pwr | — | shared pull-up rail |
| **8** | 14 | GPS UART TX → GPS RX | out | `navigation/sensors/sensor_config.py` | `/dev/ttyTHS1` @ 9600 |
| **10** | 15 | GPS UART RX ← GPS TX | in | ″ | **TX/RX crossed** |
| **12** | 18 | PTT button | in | `tts_stt/voice_config.py` → `button_pin` | active-low; STT / wake |
| **14** | — | GND | gnd | — | shared ground rail |
| **16** | 23 | Launch button | in | `main/boot_launcher.py` → `LAUNCH_BUTTON_PIN` | active-low; start/stop alas_main |
| **18** | 24 | Status LED | out | `main/status_led.py` → `STATUS_LED_PIN` | **active-low** (sink); mode indicator |

> GPS module also needs **VCC** (3.3 V or 5 V per module) + **GND**.
> Navigation (A\* / VFH) is pure software — no pins of its own.

## Button + LED board

Two buttons + LED on one board, sharing 3.3 V (pin 1) and GND (pin 14).
Per button: own signal pin + own **10 kΩ** pull-up. **5 wires** to the Jetson
(3.3 V, GND, PTT, Launch, LED).

```
3.3V(1) ──┬──[10k]── PTT(12)        button: signal → GND when pressed (active-low)
          ├──[10k]── Launch(16)
          └──[330–470Ω]──▶|── pin18  LED active-low: anode→3.3V, cathode→GPIO
GND(14) ── button commons
```

> **LED is active-low on purpose.** Jetson pins idle HIGH when undriven, so in
> IDLE (alas_main not running) an active-high LED would stay lit. Sinking
> current (pin LOW = on) keeps it dark in IDLE. Software matches via
> `status_led.py` `active_low=True`.

**⚠️ No debounce capacitor — do NOT add one.** An RC cap (10 k × 100 nF, ~1 ms)
made the SoC **reset on button *release***: the slow rising edge lingers in the
input's undefined region and triggers CMOS shoot-through on the Tegra X1 GPIO
buffer → brownout → reset. Debounce is done in **software** instead:

| Button | File | Parameter |
| :-- | :-- | :-- |
| PTT | `tts_stt/button_listener.py` (`add_event_detect` `bouncetime`) | `voice_config.button_debounce_ms` = **300 ms** |
| Launch | `main/boot_launcher.py` | `BUTTON_DEBOUNCE_MS` **500 ms** + `MIN_TOGGLE_INTERVAL_SEC` **3 s** + `_confirm_press()` |

> If HW debounce is ever needed: a **74HC14 Schmitt** buffer or a **≤10 nF** cap —
> never a bare RC cap on the GPIO. Jetson GPIO drive is low: keep LED ≤ 5 mA or
> use a transistor.

**Status LED patterns:** off = IDLE · blink = WARMUP · solid = ACTIVE ·
heartbeat = STANDBY · off = stopping.

## Reserved — do not repurpose

UART **8 / 10** (GPS) · I²C **3 / 5** (+ EEPROM 27/28) · SPI **19/21/23/24/26** ·
power **1 / 2 / 4 / 17** · GND **6/9/14/20/25/30/34/39**.

## Quick test

1. **Launch (pin 16):** press → starts (LED off→blink→solid); press → stops (Jetson stays on, LED off).
2. **PTT (pin 12):** ACTIVE → "Sizi dinliyorum"; STANDBY → wakes (no STT). Rapid presses must not be lost.
3. **LED (pin 18):** tracks mode (see patterns above).
4. **GPS** still gets a fix → UART 8/10 untouched.

```python
# Quick pin-level check (idle should read 1, pressed 0; no flicker):
import Jetson.GPIO as GPIO, time
GPIO.setmode(GPIO.BCM); GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
while True: print(GPIO.input(23)); time.sleep(0.1)
```
