import subprocess
import sys
import threading
import queue
import os
from pathlib import Path 

# Kuyruk sistemi (Seslerin birbirini kesmemesi için)
_tts_queue = queue.Queue()

def _tts_worker():
    """Arka planda seslendirme yapan işçi thread."""
    while True:
        text = _tts_queue.get()
        if text is None:
            _tts_queue.task_done()
            break

        # Jetson Nano ve Mac uyumlu sessiz çalıştırma scripti
        # subprocess kullanarak ana programın donmasını engelliyoruz
        script = (
            "import pyttsx3\n"
            "try:\n"
            "    engine = pyttsx3.init()\n"
            "    engine.setProperty('rate', 160)\n"  # Konuşma hızı
            "    engine.setProperty('volume', 1.0)\n"
            f"    engine.say({repr(text)})\n"
            "    engine.runAndWait()\n"
            "except:\n"
            "    pass"
        )
        try:
            subprocess.run(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[TTS ERR] Seslendirme hatası: {e}")
        finally:
            _tts_queue.task_done()


# Worker thread'i başlat
_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text: str):
    """Metni seslendirme kuyruğuna ekler."""
    text = (text or "").strip()
    if text:
        _tts_queue.put(text)

def wait_until_done():
    """Kuyruktaki tüm seslerin bitmesini bekler."""
    _tts_queue.join()

def shutdown_tts():
    """
    Kuyruktaki tüm seslerin bitmesini bekler, sonra worker'ı kapatır.
    Program kapanmadan önce çağrılmalı.
    """
    _tts_queue.join()       # Bekleyen sesler bitsin
    _tts_queue.put(None)    # Worker'a dur sinyali
    _tts_thread.join(timeout=5)
    print("[TTS] Kapatıldı.")


def handle_intent_response(intent: str, text: str = ""):
    """
    SLM'den gelen niyete göre kullanıcıya sesli geri bildirim verir.
    """
    responses = {
        "system_command": "Sistem kapatılıyor, iyi günler dilerim.",
        "navigation": "Anlaşıldı, rota hesaplanıyor.",
        "general": "",  # Genel durumlarda sessiz kal
    }

    msg = responses.get(intent, "")
    if msg:
        speak(msg)
    # NOT: general intent'te kullanıcının söylediği metni papağan gibi
    # tekrar etmiyoruz. İleride LLM entegrasyonu ile anlamlı cevap
    # üretilebilir (ör. "saat kaç" → gerçek saat bilgisi).


# Modülün kendi başına test edilebilmesi için
if __name__ == "__main__":
    print("TTS Test Sistemi Başlatıldı...")
    speak("Sistem hazır. Ses testi yapılıyor.")
    shutdown_tts()