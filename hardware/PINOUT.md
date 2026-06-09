# ALAS — Jetson Nano Pin Map (J41 40-pin header)

Code uses **BCM** numbering (`GPIO.setmode(GPIO.BCM)`); you wire by **physical**
pin — they differ (e.g. **BCM 23 = physical pin 16**). Always wire by the
*Phys* column.

> **What changed (audio amp added):** the MAX98357A I2S amplifier now uses the
> hardware I2S bus on **pins 12 / 35 / 38 / 40**. The PTT button therefore moved
> off pin 12 to **pin 22 (BCM 25)**.

## Full header map — what is occupied vs free

Legend: 🔴 used by ALAS · 🟠 reserved (bus/EEPROM/future) · ⚪ free · ⚡ power · ⏚ GND

| Phys | BCM | Default function | ALAS use | Status |
| :-: | :-: | :-- | :-- | :-: |
| 1 | — | 3.3 V | amp **SD** pull-high (opt.) · mic VCC · button rail | ⚡ |
| 2 | — | 5 V | amp **VIN** | ⚡ |
| 3 | 2 | I2C1 SDA | — | 🟠 I2C |
| 4 | — | 5 V | (spare 5 V) | ⚡ |
| 5 | 3 | I2C1 SCL | — | 🟠 I2C |
| 6 | — | GND | amp **GND** | ⏚ |
| 7 | 4 | AUDIO_MCLK | (not needed by MAX98357A) | ⚪ |
| 8 | 14 | UART1 TX | **GPS RX** | 🔴 GPS |
| 9 | — | GND | — | ⏚ |
| 10 | 15 | UART1 RX | **GPS TX** | 🔴 GPS |
| 11 | 17 | GPIO | — | ⚪ |
| 12 | 18 | **I2S4_SCLK** | amp **BCLK** (+ mic SCK) | 🔴 I2S |
| 13 | 27 | GPIO | — | ⚪ |
| 14 | — | GND | amp GND / button common | ⏚ |
| 15 | 22 | GPIO | — | ⚪ |
| 16 | 23 | GPIO | **Launch button** | 🔴 |
| 17 | — | 3.3 V | — | ⚡ |
| 18 | 24 | GPIO | **Status LED** (active-low) | 🔴 |
| 19 | 10 | SPI1 MOSI | — | 🟠 SPI |
| 20 | — | GND | — | ⏚ |
| 21 | 9 | SPI1 MISO | — | 🟠 SPI |
| 22 | 25 | GPIO | **PTT button** (moved here) | 🔴 |
| 23 | 11 | SPI1 SCK | — | 🟠 SPI |
| 24 | 8 | SPI1 CS0 | — | 🟠 SPI |
| 25 | — | GND | — | ⏚ |
| 26 | 7 | SPI1 CS1 | — | 🟠 SPI |
| 27 | 0 | I2C0 SDA (EEPROM) | — | 🟠 do not use |
| 28 | 1 | I2C0 SCL (EEPROM) | — | 🟠 do not use |
| 29 | 5 | GPIO | — | ⚪ |
| 31 | 6 | GPIO | — | ⚪ |
| 32 | 12 | GPIO / PWM | — | ⚪ |
| 33 | 13 | GPIO / PWM | — | ⚪ |
| 34 | — | GND | — | ⏚ |
| 35 | 19 | **I2S4_LRCK** | amp **LRC** (+ mic WS) | 🔴 I2S |
| 36 | 16 | GPIO | — | ⚪ |
| 37 | 26 | GPIO | — | ⚪ |
| 38 | 20 | **I2S4_SDIN** | (future **mic** data) | 🟠 mic |
| 39 | — | GND | — | ⏚ |
| 40 | 21 | **I2S4_SDOUT** | amp **DIN** | 🔴 I2S |

**Free GPIOs for future use:** 11, 13, 15, 29, 31, 32, 33, 36, 37 (+ pin 7).

---

## Audio amplifier — MAX98357A (I2S Class-D, mono, → 8 Ω 1 W speaker)

The MAX98357A is a DAC **+** amplifier with **no control bus** — it just consumes
the I2S stream. Five wires carry audio + power; **GAIN** and **SD** are
configuration pins (no Jetson GPIO needed).

| Amp pin | Wire | → Jetson | Jetson signal |
| :-- | :-- | :-- | :-- |
| **VIN** | red | **pin 2** | 5 V |
| **GND** | black | **pin 6** | GND |
| **BCLK** | orange | **pin 12** | I2S4_SCLK (bit clock) |
| **LRC** | blue | **pin 35** | I2S4_LRCK (word select) |
| **DIN** | green | **pin 40** | I2S4_SDOUT (data **out** of Jetson) |
| **GAIN** | — | *floating* | 9 dB default (→ GND for 12 dB if too quiet) |
| **SD** | — | *floating* | onboard 1 MΩ→VIN ⇒ (L+R)/2 **mono**, enabled |

> ⚠️ **Correction to the first draft:** **DIN goes to pin 40 (SDOUT)**, not pin
> 38. Pin 38 is I2S4_**SDIN** (data *into* the Jetson) and is reserved for the
> future microphone. Putting amp DIN on pin 38 would give no sound.
>
> **GAIN:** leave the pin unconnected → 9 dB. Too quiet on the 1 W speaker?
> Connect GAIN→GND for 12 dB (or a 100 kΩ GAIN→GND for 15 dB).
>
> **SD:** leave it as the board default. The breakout has a 1 MΩ from SD to VIN
> which, with the chip's internal 100 kΩ pulldown, sets the **(L+R)/2 mono** mix
> and keeps the amp **enabled**. If you get silence, tie SD directly to
> **3.3 V (pin 1)** to force-enable. (Optionally route SD to a free GPIO — e.g.
> pin 15/BCM 22 — for software mute/standby; not required.) Never wire SD to an
> I2S pin.

The speaker's two leads (Molex 1.25 mm) connect to the amp's **screw/solder
terminals** (`+` / `−`), not to the Jetson.

---

## Future microphone — INMP441 (I2S MEMS), *not wired yet*

The INMP441 shares the **same I2S clocks** as the amp and uses the **input**
data line (pin 38), so it coexists with the amplifier on I2S4. Plan for later:

| Mic pin | → Jetson | Jetson signal |
| :-- | :-- | :-- |
| VDD / VCC | pin 1 or 17 | 3.3 V |
| GND | GND | GND |
| **SCK** | **pin 12** | I2S4_SCLK (shared with amp BCLK) |
| **WS** | **pin 35** | I2S4_LRCK (shared with amp LRC) |
| **SD** (data out) | **pin 38** | I2S4_SDIN (data **into** Jetson) |
| **L/R** | GND | GND = left-channel slot |

No new clock pins are needed — only the data line (pin 38) is added when the mic
is fitted.

---

## Enabling I2S on the Jetson (do this once, before audio works)

By default pins 12/35/38/40 are GPIO. The hardware I2S4 bus must be switched on
in the pinmux:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py      # interactive: "Configure 40-pin
                                             # header" → enable "i2s4" → save → reboot
# or non-interactively:
sudo /opt/nvidia/jetson-io/config-by-function.py -o dtbo i2s4
sudo reboot
```

After reboot, pins 12/35/38/40 are I2S (no longer GPIO). Because the MAX98357A
has no codec to probe, ALSA needs a **dummy-codec sound card** bound to I2S4 (a
`simple-audio-card` device-tree overlay) before a playback device appears; then
set it as the ALSA default and unmute the I2S route with `amixer`. See the NVIDIA
forum threads *"Audio I2S on 40-Pin Connector"* and *"ALSA configuration (I2S4)
Jetson Nano"*.

**Waveshare Jetson Nano note:** the SoC/module is a standard Nano, so the pin
functions and `jetson-io` flow are identical. The catch is the carrier's flashed
image: if `/opt/nvidia/jetson-io/` is missing or a custom DTB is used, apply the
I2S4 + dummy-codec overlay manually (place the `.dtbo` in `/boot/`, reference it
from `extlinux.conf`, then reboot). Verify with `aplay -l` (the I2S card should be
listed) and test with `speaker-test -D <card> -c 2 -t sine`.

---

## Button + LED board

Two buttons + one LED on a small board, sharing **3.3 V (pin 1)** and
**GND (pin 14)**. Each button has its own signal pin + own **10 kΩ** pull-up.

| Element | Phys | BCM | Code reference | Note |
| :-- | :-: | :-: | :-- | :-- |
| **PTT button** | **22** | **25** | `tts_stt/voice_config.py` → `button_pin` | active-low; STT / wake |
| **Launch button** | 16 | 23 | `main/boot_launcher.py` → `LAUNCH_BUTTON_PIN` | active-low; start/stop alas_main |
| **Status LED** | 18 | 24 | `main/status_led.py` → `STATUS_LED_PIN` | **active-low** (sink) |

```
3.3V(1) ──┬──[10k]── PTT(22)        button: signal → GND when pressed (active-low)
          ├──[10k]── Launch(16)
          └──[330–470Ω]──▶|── pin18  LED active-low: anode→3.3V, cathode→GPIO
GND(14) ── button commons
```

> **LED is active-low on purpose.** Jetson pins idle HIGH when undriven, so in
> IDLE an active-high LED would stay lit. Sinking current (pin LOW = on) keeps it
> dark in IDLE. Software matches via `status_led.py` `active_low=True`.
> **LED patterns:** off = IDLE · blink = WARMUP · solid = ACTIVE · heartbeat = STANDBY.

**⚠️ No debounce capacitor — do NOT add one.** An RC cap (10 k × 100 nF, ~1 ms)
made the SoC **reset on button *release***: the slow rising edge lingers in the
input's undefined region and triggers CMOS shoot-through on the Tegra X1 GPIO
buffer → brownout → reset. Debounce is done in **software**:

| Button | File | Parameter |
| :-- | :-- | :-- |
| PTT | `tts_stt/button_listener.py` (`add_event_detect` `bouncetime`) | `voice_config.button_debounce_ms` = **300 ms** |
| Launch | `main/boot_launcher.py` | `BUTTON_DEBOUNCE_MS` **500 ms** + `MIN_TOGGLE_INTERVAL_SEC` **3 s** + `_confirm_press()` |

> If HW debounce is ever needed: a **74HC14 Schmitt** buffer or a **≤10 nF** cap —
> never a bare RC cap on the GPIO. Jetson GPIO drive is low: keep LED ≤ 5 mA.

---

## Reserved — do not repurpose

UART **8 / 10** (GPS) · I²S **12 / 35 / 38 / 40** (audio) · I²C **3 / 5**
(+ EEPROM 27 / 28) · SPI **19 / 21 / 23 / 24 / 26** · power **1 / 2 / 4 / 17** ·
GND **6 / 9 / 14 / 20 / 25 / 30 / 34 / 39**.

## Quick test

1. **Launch (pin 16):** press → starts (LED off→blink→solid); press → stops.
2. **PTT (pin 22):** ACTIVE → "Sizi dinliyorum"; STANDBY → wakes. Rapid presses must not be lost.
3. **LED (pin 18):** tracks mode (see patterns above).
4. **GPS:** still gets a fix → UART 8/10 untouched.
5. **Audio:** `aplay -l` lists the I2S card; `speaker-test -t sine` → tone from the speaker.

```python
# Pin-level button check (idle reads 1, pressed 0; no flicker):
import Jetson.GPIO as GPIO, time
GPIO.setmode(GPIO.BCM); GPIO.setup(25, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # PTT = BCM25
while True: print(GPIO.input(25)); time.sleep(0.1)
```
