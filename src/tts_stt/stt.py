import pyttsx3

from navigation.router.navigator import NavigationSystem


def init_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)

    # İsteğe bağlı: mümkünse daha iyi bir voice seç
    preferred = ["Samantha", "Victoria", "Ava", "Moira", "Karen", "Tessa", "Kathy"]
    for v in engine.getProperty("voices"):
        if any(p.lower() in (v.name or "").lower() for p in preferred):
            engine.setProperty("voice", v.id)
            break

    return engine


def speak(engine, text: str):
    text = (text or "").strip()
    if not text:
        return
    engine.say(text)
    engine.runAndWait()


def parse_floats(parts, count: int):
    if len(parts) != count:
        raise ValueError(f"{count} sayı bekleniyor, gelen: {len(parts)}")
    return [float(x) for x in parts]


def main():
    nav = NavigationSystem("navigation/router/map.osm")

    tts = init_tts()

    print("Komutlar:")
    print("  start <start_lat> <start_lon> <end_lat> <end_lon>")
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
                s_lat, s_lon, e_lat, e_lon = parse_floats(args, 4)
                ok = nav.start_navigation(s_lat, s_lon, e_lat, e_lon)
                msg = "Rota oluşturuldu." if ok else "Rota oluşturulamadı."
                print("[TTS]", msg)
                speak(tts, msg)

            elif cmd == "gps":
                lat, lon = parse_floats(args, 2)
                status = nav.check_progress(lat, lon)
                print("[NAV]", status)
                speak(tts, status)

            else:
                msg = "Bilinmeyen komut."
                print("[TTS]", msg)
                speak(tts, msg)

        except Exception as e:
            err = f"Hata: {e}"
            print("[ERR]", err)
            speak(tts, err)


if __name__ == "__main__":
    main()
