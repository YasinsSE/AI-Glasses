"""
Kullanım:
    from tts_stt.stt import STTEngine

    stt = STTEngine()
    text = stt.listen()          # tek seferlik dinle
    stt.listen_continuous(cb)    # sürekli dinle, callback'e gönder
"""
import json
import os
import time
import queue
import threading
import pyaudio
import zipfile
import urllib.request
from pathlib import Path
from vosk import Model, KaldiRecognizer, SetLogLevel
from mlx_lm import load, generate  # MLX kütüphaneleri (Apple Silicon)

# Aynı paketteki tts modülünden import (hem doğrudan hem paket içi çalışır)
try:
    from tts_stt.tts import handle_intent_response, speak,shutdown_tts,wait_until_done
except ImportError:
    from tts import handle_intent_response, speak,shutdown_tts,wait_until_done

# Vosk log seviyesini kıs  (-1 = sessiz)
SetLogLevel(-1)

# ──────────────────────────────────────────────────────────────────────────────
# VARSAYILAN AYARLAR
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]          # ai-glasses-1/
_DEFAULT_MODEL = _PROJECT_ROOT / "vosk-model-small-tr-0.3"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip"

SAMPLE_RATE = 16_000
CHUNK_SIZE = 4096
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ──────────────────────────────────────────────────────────────────────────────
# VOSK MODEL OTOMATİK İNDİRME
# ──────────────────────────────────────────────────────────────────────────────

def ensure_vosk_model(model_dir: Path) -> Path:
    """Model yoksa indir ve çıkart, yolunu döndür."""
    if model_dir.is_dir():
        return model_dir

    zip_path = model_dir.with_suffix(".zip")
    print(f"Vosk model bulunamadı, indiriliyor: {VOSK_MODEL_URL}")
    urllib.request.urlretrieve(VOSK_MODEL_URL, str(zip_path))

    print("Zip açılıyor...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(model_dir.parent))

    zip_path.unlink()
    print(f"Model hazır: {model_dir}")
    return model_dir

# ──────────────────────────────────────────────────────────────────────────────
# MLX TABANLI SLM SINIFLANDIRICI
# ──────────────────────────────────────────────────────────────────────────────

class MLXIntentClassifier:
    """Eğittiğimiz Qwen modelini kullanarak niyet analizi yapan sınıf."""

    VALID_INTENTS = ["system_command", "navigation", "general"]

    def __init__(self, model_path=None):
        # Eğer dışarıdan bir yol verilmezse, stt.py'nin yanındaki klasörü bul
        if model_path is None:
            current_dir = Path(__file__).resolve().parent
            model_path = str(current_dir / "my_custom_slm")

        print(f"[SLM] Model yükleniyor: {model_path} ...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model klasörü bulunamadı: {model_path}\n"
                "Lütfen MLX model eğitiminin (fuse) başarıyla tamamlandığından emin olun."
            )

        # Modeli ve tokenizer'ı yüklüyoruz
        self.model, self.tokenizer = load(model_path)
        print("[SLM] Model hazır ✓")

    def predict(self, text: str):
        """Verilen metni sınıflandırır. (intent, confidence) döndürür."""
        # Eğitimde kullandığımız tam prompt şablonu
        prompt = (
            "<|im_start|>user\n"
            "Aşağıdaki komutun niyetini (intent) sınıflandır. "
            "Seçenekler: 'system_command', 'navigation', 'general'.\n"
            f"Komut: {text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Modelden cevabı al
        response = generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=10, verbose=False,
        )

        # Qwen'in ek token'larını ve fazlalıkları temizle
        intent = response.strip().lower()
        intent = intent.replace("<|im_end|>", "").strip()
        # Sadece ilk kelimeyi al (halüsinasyon koruması)
        intent = intent.split()[0] if intent else "general"

        # Olası bir halüsinasyon durumunda güvenli fallback
        if intent not in self.VALID_INTENTS:
            print(f"[SLM] ⚠️  Bilinmeyen intent: '{intent}' → fallback: 'general'")
            return "general", 0.0

        return intent, 1.0


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
            Model bulunamazsa otomatik indirilir.
        """
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        path = ensure_vosk_model(path)

        print(f"[STT] Model yükleniyor: {path}")
        self.model = Model(str(path))
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)

        self._audio = pyaudio.PyAudio()
        self._running = False

        # MLX tabanlı intent sınıflandırıcı
        self._slm = MLXIntentClassifier()
        print("[STT] Hazır ✓")

    # ──────────────────────────────────────────────────────────────────────
    # TEK SEFERLİK DİNLEME
    # ──────────────────────────────────────────────────────────────────────

    def listen(self, timeout_sec: float = 5.0, silence_sec: float = 1.5) -> str:
        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("[STT] Dinleniyor...")
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)

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

            if not result_text:
                final = json.loads(recognizer.FinalResult())
                result_text = final.get("text", "").strip()

        finally:
            stream.stop_stream()
            stream.close()

        if result_text:
            print(f'[STT] "{result_text}"')
        else:
            print("[STT] Ses algılanamadı.")

        return result_text

    # ──────────────────────────────────────────────────────────────────────
    # SÜREKLİ DİNLEME (arka plan thread)
    # ──────────────────────────────────────────────────────────────────────

    def listen_continuous(self, callback):
        self._running = True

        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        print("[STT] Sürekli dinleme başladı. Konuşabilirsiniz...")

        try:
            while self._running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if not text:
                        continue

                    # Vosk'tan gelen sesi metne çevirdik, şimdi modele soruyoruz
                    intent, conf = self._slm.predict(text)
                    print(f'\n[SİSTEM] Duyulan: "{text}" → Algılanan Niyet: [{intent}] (güven: {conf})')

                    if intent == "system_command":
                        print("[SİSTEM] Kapatma komutu algılandı.")
                        # Kullanıcıya sesli geri bildirim ver, sonra kapat
                        callback(text, intent, conf)
                        self._running = False
                        break

                    callback(text, intent, conf)

        finally:
            stream.stop_stream()
            stream.close()
            print("[STT] Dinleme durduruldu.")

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        shutdown_tts()
        self._audio.terminate()
        print("[STT] Kapatıldı.")
        


def my_callback(text, intent, conf):
    """
    Bu fonksiyon her cümle tanındığında çalışır.
    İşlem sırası: Niyet Algılama -> Sesli Yanıt -> Donanım Tetikleme
    """
    # 1. TEMİZLİK: Gözlük kendi sesini duyduğunda (STT hatası olarak)
    # niyet 'general' çıkabiliyor. Eğer metin asistanın kendi kalıplarını
    # içeriyorsa işlemi atla.
    tts_echo_patterns = ["anlaşıldı", "hesaplanıyor", "rota", "yönlendirme"]
    if any(pattern in text.lower() for pattern in tts_echo_patterns):
        return

    print(f"\n[İŞLEM] Kullanıcı ne dedi: '{text}'")
    print(f"[İŞLEM] Tespit edilen niyet: {intent.upper()} (güven: {conf})")

    # 2. SESLİ GERİ BİLDİRİM (TTS)
    # handle_intent_response fonksiyonu ses bitene kadar beklediği için
    # eko (echo) sorunu burada doğal yolla çözülür.
    handle_intent_response(intent, text)

    # 3. DONANIM VE MANTIK TETİKLEME
    if intent == "navigation":
        print("[SİSTEM] Navigasyon motoru ateşleniyor...")
        # Örnek entegrasyon (Kendi objene göre güncelle):
        # success, msg, _ = nav.navigate_to_nearest(current_coord, text)
        # speak(msg)

    elif intent == "system_command":
        print("[SİSTEM] Kritik sistem komutu uygulandı.")
        # Buraya ek temizlik işlemleri gelebilir (logları kaydetmek gibi)


if __name__ == "__main__":
    # STT motorunu başlat
    stt = STTEngine()

    print("\n=== ALAS Akıllı Gözlük Sistemi Aktif ===")
    print("Dinleme yapılıyor... Durdurmak için 'kapat' deyin.\n")
    speak("Merhaba, Bugün nereye gitmek istersin?")
    wait_until_done()

    try:
        # Sürekli dinlemeyi bizim yeni callback fonksiyonumuzla başlatıyoruz
        stt.listen_continuous(callback=my_callback)
    except KeyboardInterrupt:
        print("\nSistem kullanıcı tarafından kapatıldı.")
    finally:
        stt.close()