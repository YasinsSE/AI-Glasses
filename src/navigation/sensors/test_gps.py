#!/usr/bin/env python3
"""
ALAS — GPS Test (Jetson Nano)

Kullanım:
  cd ~/ALAS/src
  python3 -m navigation.sensors.test_gps

  # Farklı port / kısa warmup:
  python3 -m navigation.sensors.test_gps --port /dev/ttyUSB0 --warmup 5
"""

import sys
import os
import time
import json
import argparse
import logging
from datetime import datetime, timezone

import serial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("gps_test")

# ── Olası Jetson Nano portları ──────────────────────────
CANDIDATE_PORTS = [
    "/dev/ttyTHS1",   # Jetson Nano 40-pin UART
    "/dev/ttyTHS2",
    "/dev/ttyUSB0",   # USB-Serial adapter
    "/dev/ttyACM0",
    "/dev/ttyAMA0",   # RPi uyumluluk
    "/dev/ttyAMA10",  # RPi 5
]


def detect_port() -> str:
    for port in CANDIDATE_PORTS:
        if os.path.exists(port):
            logger.info(f"Bulunan port: {port}")
            return port

    logger.error("Serial port bulunamadı!")
    logger.error("Kontrol et:")
    logger.error("  ls /dev/tty*")
    logger.error("  sudo dmesg | grep tty")
    sys.exit(1)


def test_raw_nmea(port: str, baud: int = 9600, seconds: int = 5) -> bool:
    """Ham NMEA verisi gelip gelmediğini kontrol et."""
    logger.info(f"--- HAM NMEA TESTİ ({seconds}sn) ---")
    logger.info(f"Port: {port} @ {baud}")

    try:
        ser = serial.Serial(port, baud, timeout=2)
    except Exception as e:
        logger.error(f"Port açılamadı: {e}")
        logger.error("  sudo usermod -aG dialout $USER  yapıp reboot et")
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

    logger.info(f"NMEA satır: {count}, RMC: {rmc}, GGA: {gga}")

    if count == 0:
        logger.error("Hiç veri gelmedi! Kablo bağlantısını kontrol et.")
        logger.error("  GPS TX → Jetson RX (Pin 10)")
        logger.error("  GPS RX → Jetson TX (Pin 8)")
        logger.error("  GND → GND (Pin 6)")
        return False

    if not rmc:
        logger.warning("RMC yok — uydu bulunamıyor olabilir, açık alanda dene")

    logger.info("NMEA verisi geliyor.")
    return True


def main():
    parser = argparse.ArgumentParser(description="ALAS GPS Test — Jetson Nano")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--warmup", type=float, default=10.0,
                        help="Test warmup (sn). Gerçek kullanımda 60")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", default="gps_log.json")
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()

    port = args.port or detect_port()

    # Adım 1: Ham NMEA testi
    if not args.skip_raw:
        if not test_raw_nmea(port, args.baud):
            sys.exit(1)
        print()

    # Adım 2: GPSReader ile konum oku
    logger.info("--- GPSReader TESTİ ---")

    from .gps_reader import GPSReader, GPSStatus

    fixes_log = []

    with GPSReader(
        port=port,
        baudrate=args.baud,
        warmup_sec=args.warmup,
    ) as gps:

        logger.info(f"GPS başlatıldı. {args.warmup}sn warmup...")
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
            logger.info("Ctrl+C — durduruluyor...")

    # Adım 3: Kaydet
    if fixes_log:
        out = {
            "device": "NEO-7M",
            "platform": "Jetson Nano",
            "port": port,
            "date": datetime.now(timezone.utc).isoformat(),
            "total_fixes": len(fixes_log),
            "fixes": fixes_log,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        logger.info(f"{len(fixes_log)} fix → {args.output}")

        lats = [f["lat"] for f in fixes_log]
        lons = [f["lon"] for f in fixes_log]
        logger.info(f"  Lat: {min(lats):.6f} – {max(lats):.6f}")
        logger.info(f"  Lon: {min(lons):.6f} – {max(lons):.6f}")
    else:
        logger.warning("Hiç fix alınamadı!")
        logger.warning("  1) Açık alanda dene")
        logger.warning("  2) --warmup 60 dene")
        logger.warning("  3) Anten bağlı mı kontrol et")


if __name__ == "__main__":
    main()
