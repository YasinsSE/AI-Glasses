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

# ── 1. Power model: MAXN (all cores, max GPU/EMC clocks) ────────────────────
if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -m 0 && log "nvpmodel: MAXN (mode 0) set."
else
    log "WARN: nvpmodel not found — is this a Jetson?"
fi

# ── 2. Lock clocks at the current power model's maximum ─────────────────────
if command -v jetson_clocks >/dev/null 2>&1; then
    jetson_clocks && log "jetson_clocks: clocks pinned."
else
    log "WARN: jetson_clocks not found."
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
