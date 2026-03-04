"""
Kullanım:
    from tts_stt.stt import STTEngine

    stt = STTEngine()
    text = stt.listen()          # tek seferlik dinle
    stt.listen_continuous(cb)    # sürekli dinle, callback'e gönder
"""
import json
import os
import queue
import threading
from pathlib import Path

import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

# Vosk log seviyesini kıs  (-1 = sessiz)
SetLogLevel(-1)

# ──────────────────────────────────────────────────────────────────────────────
# VARSAYILAN AYARLAR
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]          # ai-glasses-1/
_DEFAULT_MODEL = _PROJECT_ROOT / "vosk-model-small-tr-0.3"

SAMPLE_RATE = 16_000
CHUNK_SIZE = 4096
CHANNELS = 1
FORMAT = pyaudio.paInt16


# ──────────────────────────────────────────────────────────────────────────────
# STT ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class STTEngine:
    """Vosk tabanlı offline ses tanıma motoru."""

    def __init__(self, model_path: str | None = None):
        """
        Parameters
        ----------
        model_path : str | None
            Vosk model klasör yolu.
            Verilmezse proje kökündeki vosk-model-small-tr-0.3 kullanılır.
        """
        path = model_path or str(_DEFAULT_MODEL)

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Vosk model bulunamadı: {path}\n"
                "İndirmek için: "
                "wget https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip && unzip ..."
            )

        print(f"[STT] Model yükleniyor: {path}")
        self.model = Model(path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)

        self._audio = pyaudio.PyAudio()
        self._running = False
        print("[STT] Hazır ✓")

    # ──────────────────────────────────────────────────────────────────────
    # TEK SEFERLİK DİNLEME
    # ──────────────────────────────────────────────────────────────────────

    def listen(self, timeout_sec: float = 5.0, silence_sec: float = 1.5) -> str:
        """
        Mikrofonu aç, konuşmayı al, metni döndür.

        Parameters
        ----------
        timeout_sec  : Maksimum dinleme süresi (saniye).
        silence_sec  : Sessizlik sonrası dur (saniye). Vosk otomatik algılar.

        Returns
        -------
        str : Tanınan metin (boş olabilir).
        """
        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("[STT] Dinleniyor...")
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)

        import time
        start = time.time()
        result_text = ""

        try:
            while time.time() - start < timeout_sec:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    result_text = result.get("text", "").strip()
                    if result_text:
                        break

            # Kalan veriyi de al
            if not result_text:
                final = json.loads(recognizer.FinalResult())
                result_text = final.get("text", "").strip()

        finally:
            stream.stop_stream()
            stream.close()

        if result_text:
            print(f"[STT] \"{result_text}\"")
        else:
            print("[STT] Ses algılanamadı.")

        return result_text

    # ──────────────────────────────────────────────────────────────────────
    # SÜREKLİ DİNLEME (arka plan thread)
    # ──────────────────────────────────────────────────────────────────────

    def listen_continuous(self, callback, stop_words: list[str] | None = None):
        """
        Sürekli dinle, tanınan her cümleyi callback'e gönder.

        Parameters
        ----------
        callback   : fn(text: str) -> None
        stop_words : Bu kelimelerden biri gelirse dinlemeyi durdur.
                     Örn: ["dur", "kapat", "çıkış"]
        """
        stop_words = [w.lower() for w in (stop_words or ["çıkış", "kapat"])]
        self._running = True

        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        print("[STT] Sürekli dinleme başladı. Durdurmak için:", stop_words)

        try:
            while self._running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if not text:
                        continue

                    print(f"[STT] \"{text}\"")

                    if any(sw in text.lower() for sw in stop_words):
                        print("[STT] Durdurma komutu algılandı.")
                        self._running = False
                        break

                    callback(text)

        finally:
            stream.stop_stream()
            stream.close()
            print("[STT] Dinleme durduruldu.")

    def stop(self):
        """Sürekli dinlemeyi durdur."""
        self._running = False

    def close(self):
        """Kaynakları serbest bırak."""
        self._running = False
        self._audio.terminate()
        print("[STT] Kapatıldı.")

if __name__ == "__main__":
    stt = STTEngine()

    print("\n=== Tek seferlik dinleme testi ===")
    text = stt.listen(timeout_sec=10)
    print(f"Sonuç: {text}\n")

    print("=== Sürekli dinleme testi ===")
    print("('çıkış' veya 'kapat' diyerek durdurun)\n")
    stt.listen_continuous(lambda t: print(f"  → {t}"))

    stt.close()