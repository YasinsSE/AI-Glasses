import subprocess
import sys
import traceback
import threading
import queue
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from navigation.router.navigator import NavigationSystem
from navigation.router.models import Coord
from navigation.router.nav_config import NavConfig

_tts_queue = queue.Queue()

def _tts_worker():
    while True:
        text = _tts_queue.get()
        if text is None:
            break
        script = (
            "import pyttsx3\n"
            "engine = pyttsx3.init()\n"
            "engine.setProperty('rate', 150)\n"
            f"engine.say({repr(text)})\n"
            "engine.runAndWait()"
        )
        try:
            subprocess.run([sys.executable, "-c", script])
        except Exception as e:
            print(f"TTS hatası: {e}")
        finally:
            _tts_queue.task_done()

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str):
    text = (text or "").strip()
    if text:
        _tts_queue.put(text)


def parse_floats(parts, count: int):
    if len(parts) != count:
        raise ValueError(f"{count} sayı bekleniyor, gelen: {len(parts)}")
    return [float(x) for x in parts]


def main():
    config = NavConfig(log_dir="logs")
    nav = NavigationSystem("navigation/router/map.osm", config=config)

    print("Komutlar:")
    print("  start <lat> <lon> <poi_tipi>")
    print("  gps <lat> <lon>")
    print("  quit")

    while True:
        line = input("> ").strip()
        if not line:
            continue

        if line.lower() in ("q", "quit", "exit"):
            break

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd == "start":
                s_lat, s_lon = parse_floats(args[:2], 2)
                poi_type = args[2] if len(args) > 2 else "pharmacy"
                success, msg, _ = nav.navigate_to_nearest(
                    Coord(lat=s_lat, lon=s_lon),
                    poi_type
                )
                print("[TTS]", msg)
                speak(msg)

                if not success:
                    print("[NAV]", "Rota hesaplanamadı.")
                    continue

            elif cmd == "gps":
                lat, lon = parse_floats(args, 2)
                result = nav.update(Coord(lat=lat, lon=lon))
                out = f"{result.status}: {result.message}"
                print("[NAV]", out)
                speak(result.message)

            else:
                msg = "Bilinmeyen komut."
                print("[TTS]", msg)
                ##speak(msg)

        except Exception as e:
            err = f"Hata: {e}"
            print(traceback.format_exc())
            print("[ERR]", err)
            ##speak(err)
            
    _tts_queue.join()       # Kuyruktaki tüm seslerin bitmesini bekle
    _tts_queue.put(None)    # Worker thread'e dur sinyali gönder
    _tts_thread.join(timeout=5)

if __name__ == "__main__":
    main()