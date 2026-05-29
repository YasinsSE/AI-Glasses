#!/usr/bin/env python3
"""ALAS — GPS hardware test (Jetson Nano).

Hardware-in-the-loop diagnostic for the NEO-7M UART module. Requires a real
GPS device and pyserial; it is not a desktop unit test. It verifies raw NMEA
flow, then reads positions through ``GPSReader`` and logs the fixes.

    Outputs: outputs/tests/navigation/sensors/gps_log.json

How to run (from the repository root):
    python tests/navigation/sensors/test_gps.py
    python tests/navigation/sensors/test_gps.py --port /dev/ttyUSB0 --warmup 5
"""

import sys
import os
import time
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import serial

# Make src/ importable whether run via pytest or as a standalone script.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from navigation.sensors.gps_reader import GPSReader, GPSStatus
from navigation.sensors.sensor_config import CANDIDATE_PORTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("gps_test")

_DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "tests" / "navigation" / "sensors" / "gps_log.json"


def detect_port() -> str:
    """Return the first present candidate serial port, or exit if none exist."""
    for port in CANDIDATE_PORTS:
        if os.path.exists(port):
            logger.info(f"Found port: {port}")
            return port

    logger.error("No serial port found!")
    logger.error("Check:")
    logger.error("  ls /dev/tty*")
    logger.error("  sudo dmesg | grep tty")
    sys.exit(1)


def test_raw_nmea(port: str, baud: int = 9600, seconds: int = 5) -> bool:
    """Check whether raw NMEA data is being received on the port."""
    logger.info(f"--- RAW NMEA TEST ({seconds}s) ---")
    logger.info(f"Port: {port} @ {baud}")

    try:
        ser = serial.Serial(port, baud, timeout=2)
    except Exception as e:
        logger.error(f"Could not open port: {e}")
        logger.error("  Run 'sudo usermod -aG dialout $USER' and reboot")
        return False

    count = 0
    rmc = False
    gga = False
    start = time.time()

    while time.time() - start < seconds:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
        except Exception:
            continue

        if not line:
            continue

        if count < 10:
            print(f"  RAW: {line}")
        count += 1

        if "RMC" in line:
            rmc = True
        if "GGA" in line:
            gga = True

    ser.close()

    logger.info(f"NMEA lines: {count}, RMC: {rmc}, GGA: {gga}")

    if count == 0:
        logger.error("No data received! Check the wiring.")
        logger.error("  GPS TX -> Jetson RX (Pin 10)")
        logger.error("  GPS RX -> Jetson TX (Pin 8)")
        logger.error("  GND -> GND (Pin 6)")
        return False

    if not rmc:
        logger.warning("No RMC — satellites may not be locked; try an open area")

    logger.info("NMEA data is flowing.")
    return True


def main():
    parser = argparse.ArgumentParser(description="ALAS GPS Test — Jetson Nano")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--warmup", type=float, default=10.0,
                        help="Test warmup (s). Use 60 in real operation.")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()

    port = args.port or detect_port()

    # Step 1: raw NMEA test.
    if not args.skip_raw:
        if not test_raw_nmea(port, args.baud):
            sys.exit(1)
        print()

    # Step 2: read positions through GPSReader.
    logger.info("--- GPSReader TEST ---")

    fixes_log = []

    with GPSReader(
        port=port,
        baudrate=args.baud,
        warmup_sec=args.warmup,
        min_sats=0,      # Disable the satellite-count gate for testing.
        max_hdop=99.0,   # Open the HDOP gate fully for testing.
    ) as gps:
        logger.info(f"GPS started. {args.warmup}s warmup...")
        t0 = time.time()

        try:
            while time.time() - t0 < args.duration:
                time.sleep(args.interval)

                health = gps.get_health()
                coord = gps.get_coord()
                elapsed = time.time() - t0
                ts = datetime.now(timezone.utc).isoformat()

                line = (
                    f"[{elapsed:6.1f}s] "
                    f"{health.status.value:<14s} "
                    f"Sats={health.satellites:<2d} "
                    f"HDOP={health.hdop:<5.1f} "
                    f"Fixes={health.fix_count}"
                )

                if coord:
                    lat, lon, age = coord
                    line += f"  LAT={lat:.6f} LON={lon:.6f} Age={age:.1f}s"
                    fixes_log.append({
                        "ts": ts,
                        "elapsed": round(elapsed, 1),
                        "lat": round(lat, 7),
                        "lon": round(lon, 7),
                        "age": round(age, 2),
                        "sats": health.satellites,
                        "hdop": health.hdop,
                        "status": health.status.value,
                    })
                else:
                    line += "  No fix"

                print(line)

        except KeyboardInterrupt:
            logger.info("Ctrl+C — stopping...")

    # Step 3: persist the collected fixes.
    if fixes_log:
        out = {
            "device": "NEO-7M",
            "platform": "Jetson Nano",
            "port": port,
            "date": datetime.now(timezone.utc).isoformat(),
            "total_fixes": len(fixes_log),
            "fixes": fixes_log,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        logger.info(f"{len(fixes_log)} fixes -> {output_path}")

        lats = [f["lat"] for f in fixes_log]
        lons = [f["lon"] for f in fixes_log]
        logger.info(f"  Lat: {min(lats):.6f} – {max(lats):.6f}")
        logger.info(f"  Lon: {min(lons):.6f} – {max(lons):.6f}")
    else:
        logger.warning("No fixes acquired!")
        logger.warning("  1) Try an open area")
        logger.warning("  2) Try --warmup 60")
        logger.warning("  3) Check that the antenna is connected")


if __name__ == "__main__":
    main()
