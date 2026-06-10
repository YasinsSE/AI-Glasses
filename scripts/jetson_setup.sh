#!/usr/bin/env bash
# ALAS — Jetson Nano (Waveshare) performance setup.
#
# Run ONCE per boot (or install the systemd unit below) BEFORE launching ALAS:
#     sudo bash scripts/jetson_setup.sh
#
# Why this exists: the Waveshare Nano carrier ships with weaker cooling than
# the NVIDIA devkit. Left on default DVFS the SoC thermal-throttles within
# minutes of TensorRT load and inference time roughly doubles (150-180 ms →
# 300+ ms), which silently halves the effective perception FPS. Pinning MAXN
# clocks + forcing the fan keeps the inference budget stable; the in-app
# thermal guard (AIConfig.thermal_*) is the second line of defence.

set -u

log() { echo "[jetson_setup] $*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "Bu script root ister: sudo bash scripts/jetson_setup.sh" >&2
    exit 1
fi

# ── 1. Power model: MAXN by default, overridable for weak supplies ──────────
# If the system freezes on ALAS launch (model-load current spike), the 5V
# supply cannot feed MAXN — run with NVP_MODE=1 (5W) until the supply is fixed.
NVP_MODE="${NVP_MODE:-0}"
if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -m "$NVP_MODE" && log "nvpmodel: mode $NVP_MODE set ($([ "$NVP_MODE" = "0" ] && echo MAXN || echo 5W))."
else
    log "WARN: nvpmodel not found — is this a Jetson?"
fi

# ── 2. (Opt-in) Lock clocks at the current power model's maximum ────────────
# DISABLED by default: pinned max clocks raise the sustained current draw and
# the TensorRT model-load spike on launch can brown-out a marginal 5V supply
# (LiPo + converter) — symptom: LED flashes, then the whole system freezes.
# MAXN alone still allows full boost; DVFS just downclocks at idle. Enable
# pinning only on a solid bench supply: PIN_CLOCKS=1 sudo bash scripts/jetson_setup.sh
PIN_CLOCKS="${PIN_CLOCKS:-0}"
if [ "$PIN_CLOCKS" = "1" ]; then
    if command -v jetson_clocks >/dev/null 2>&1; then
        jetson_clocks && log "jetson_clocks: clocks pinned."
    else
        log "WARN: jetson_clocks not found."
    fi
else
    log "jetson_clocks: skipped (PIN_CLOCKS=0 default — brownout-safe)."
fi

# ── 3. Fan: force full duty cycle (Waveshare cooling is marginal) ───────────
# 0-255; full power by default — outdoor summer field tests need every bit of
# cooling headroom. Override for quiet indoor demos: FAN_PWM=140 sudo bash ...
FAN_PWM="${FAN_PWM:-255}"
FAN_NODE="/sys/devices/pwm-fan/target_pwm"
if [ -w "$FAN_NODE" ]; then
    echo "$FAN_PWM" > "$FAN_NODE" && log "fan: target_pwm=$FAN_PWM."
else
    log "WARN: $FAN_NODE not writable — no PWM fan header on this carrier?"
fi

# ── 4. zram swap (compressed RAM swap — never swap to the SD card) ──────────
# JetPack ships an nvzramconfig service; just make sure it is on.
if systemctl list-unit-files 2>/dev/null | grep -q nvzramconfig; then
    systemctl enable --now nvzramconfig >/dev/null 2>&1 && log "zram: nvzramconfig enabled."
fi

# ── 5. Report ───────────────────────────────────────────────────────────────
for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    t=$(cat "$z/temp"); ty=$(cat "$z/type" 2>/dev/null || echo "?")
    log "thermal: $ty = $((t / 1000)) C"
done

log "Done. One-time extras (run manually once):"
log "  GUI'yi kapat (~600 MB RAM):  sudo systemctl set-default multi-user.target && sudo reboot"
log "  Boot'ta otomatik calistir:   sudo cp scripts/alas-perf.service /etc/systemd/system/ && sudo systemctl enable alas-perf"
