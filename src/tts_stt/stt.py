"""Offline speech-to-text (Vosk) with MLX-based intent classification.

Runs Vosk Turkish speech recognition and classifies the recognised intent with
a fine-tuned Qwen SLM (MLX). The Vosk model is downloaded automatically on the
first run.

First-time setup (run once after cloning the repository):
    1. Install portaudio (required by PyAudio) on macOS: brew install portaudio
    2. Install the libraries: pip install vosk pyaudio pyttsx3 mlx mlx-lm huggingface_hub
    3. Log in to Hugging Face: python3 -c "from huggingface_hub import login; login()"
    4. Download the fine-tuned SLM model: python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Sirius95/ai-glasses-safetensor', local_dir='src/tts_stt/my_custom_slm/')"
    5. The Vosk Turkish model downloads automatically on first run.

How to run (from the repository root):
    python3 -m tts_stt.stt

Optional — re-train the SLM model (from the repository root):
    python3 eval/tts_stt/slmprepare.py
    python3 -m mlx_lm lora --model Qwen/Qwen2.5-0.5B-Instruct --data outputs/eval/tts_stt --train --iters 500 --batch-size 4 --num-layers 8 --learning-rate 1e-4 --seed 42
    python3 -m mlx_lm fuse --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters/ --save-path src/tts_stt/my_custom_slm/
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
from mlx_lm import load, generate  # MLX libraries (Apple Silicon).

# Import from sibling modules in the same package. The try/except keeps the
# module runnable both as a package member and as a standalone script.
try:
    from tts_stt.tts import handle_intent_response, speak, shutdown_tts, wait_until_done
    from tts_stt.voice_config import VoiceConfig
except ImportError:
    from tts import handle_intent_response, speak, shutdown_tts, wait_until_done
    from voice_config import VoiceConfig

# Silence Vosk's own logging (-1 = quiet).
SetLogLevel(-1)

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]          # Repository root.
_DEFAULT_MODEL = _PROJECT_ROOT / "vosk-model-small-tr-0.3"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip"

# Audio capture format — sourced from VoiceConfig for a single source of truth.
_VOICE_CFG = VoiceConfig()
SAMPLE_RATE = _VOICE_CFG.sample_rate
CHUNK_SIZE = _VOICE_CFG.chunk_size
CHANNELS = _VOICE_CFG.channels
FORMAT = pyaudio.paInt16

# ──────────────────────────────────────────────────────────────────────────────
# VOSK MODEL AUTO-DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

def ensure_vosk_model(model_dir: Path) -> Path:
    """Download and extract the model if missing; return its path."""
    if model_dir.is_dir():
        return model_dir

    zip_path = model_dir.with_suffix(".zip")
    print(f"Vosk model not found, downloading: {VOSK_MODEL_URL}")
    urllib.request.urlretrieve(VOSK_MODEL_URL, str(zip_path))

    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(model_dir.parent))

    zip_path.unlink()
    print(f"Model ready: {model_dir}")
    return model_dir

# ──────────────────────────────────────────────────────────────────────────────
# MLX-BASED SLM CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

class MLXIntentClassifier:
    """Intent classifier backed by our fine-tuned Qwen model."""

    VALID_INTENTS = ["system_command", "navigation", "general"]

    def __init__(self, model_path=None):
        # If no path is given, use the folder next to stt.py.
        if model_path is None:
            current_dir = Path(__file__).resolve().parent
            model_path = str(current_dir / "my_custom_slm")

        print(f"[SLM] Loading model: {model_path} ...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model folder not found: {model_path}\n"
                "Make sure the MLX model training (fuse) completed successfully."
            )

        # Load the model and tokenizer.
        self.model, self.tokenizer = load(model_path)
        print("[SLM] Model ready ✓")

    def predict(self, text: str):
        """Classify the given text. Returns (intent, confidence)."""
        # Exact prompt template used during fine-tuning (kept in Turkish).
        prompt = (
            "<|im_start|>user\n"
            "Aşağıdaki komutun niyetini (intent) sınıflandır. "
            "Seçenekler: 'system_command', 'navigation', 'general'.\n"
            f"Komut: {text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Get the model's answer.
        response = generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=10, verbose=False,
        )

        # Strip Qwen's extra tokens and noise.
        intent = response.strip().lower()
        intent = intent.replace("<|im_end|>", "").strip()
        # Keep only the first word (hallucination guard).
        intent = intent.split()[0] if intent else "general"

        # Safe fallback in case of a hallucinated intent.
        if intent not in self.VALID_INTENTS:
            print(f"[SLM] ⚠️  Unknown intent: '{intent}' -> fallback: 'general'")
            return "general", 0.0

        return intent, 1.0


# ──────────────────────────────────────────────────────────────────────────────
# STT ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class STTEngine:
    """Offline speech-recognition engine based on Vosk."""

    def __init__(self, model_path: str | None = None):
        """
        Parameters
        ----------
        model_path : str | None
            Path to the Vosk model folder. If omitted, the repository-root
            vosk-model-small-tr-0.3 is used. The model is downloaded
            automatically if it is missing.
        """
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        path = ensure_vosk_model(path)

        print(f"[STT] Loading model: {path}")
        self.model = Model(str(path))
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)

        self._audio = pyaudio.PyAudio()
        self._running = False

        # MLX-based intent classifier.
        self._slm = MLXIntentClassifier()
        print("[STT] Ready ✓")

    # ──────────────────────────────────────────────────────────────────────
    # SINGLE-SHOT LISTENING
    # ──────────────────────────────────────────────────────────────────────

    def listen(self, timeout_sec: float = 5.0, silence_sec: float = 1.5) -> str:
        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("[STT] Listening...")
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
            print("[STT] No speech detected.")

        return result_text

    # ──────────────────────────────────────────────────────────────────────
    # CONTINUOUS LISTENING (background thread)
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
        print("[STT] Continuous listening started. You may speak...")

        try:
            while self._running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if not text:
                        continue

                    # Vosk turned speech into text; now query the model.
                    intent, conf = self._slm.predict(text)
                    print(f'\n[SYSTEM] Heard: "{text}" -> Detected intent: [{intent}] (confidence: {conf})')

                    if intent == "system_command":
                        print("[SYSTEM] Shutdown command detected.")
                        # Give the user audio feedback, then shut down.
                        callback(text, intent, conf)
                        self._running = False
                        break

                    callback(text, intent, conf)

        finally:
            stream.stop_stream()
            stream.close()
            print("[STT] Listening stopped.")

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        shutdown_tts()
        self._audio.terminate()
        print("[STT] Shut down.")
        


def my_callback(text, intent, conf):
    """Runs whenever a sentence is recognised.

    Order of operations: intent detection -> audio response -> hardware trigger.
    """
    # 1. CLEANUP: when the glasses hear their own voice (as an STT error) the
    # intent may come out as 'general'. If the text contains the assistant's
    # own phrases, skip processing. (Patterns are the assistant's own Turkish
    # TTS output, so they must stay in Turkish.)
    tts_echo_patterns = ["anlaşıldı", "hesaplanıyor", "rota", "yönlendirme"]
    if any(pattern in text.lower() for pattern in tts_echo_patterns):
        return

    print(f"\n[ACTION] What the user said: '{text}'")
    print(f"[ACTION] Detected intent: {intent.upper()} (confidence: {conf})")

    # 2. AUDIO FEEDBACK (TTS). handle_intent_response blocks until speech ends,
    # so the echo problem is naturally resolved here.
    handle_intent_response(intent, text)

    # 3. HARDWARE AND LOGIC TRIGGERS.
    if intent == "navigation":
        print("[SYSTEM] Firing up the navigation engine...")
        # Example integration (adapt to your own objects):
        # success, msg, _ = nav.navigate_to_nearest(current_coord, text)
        # speak(msg)

    elif intent == "system_command":
        print("[SYSTEM] Critical system command applied.")
        # Additional cleanup could go here (e.g. saving logs).


if __name__ == "__main__":
    # Start the STT engine.
    stt = STTEngine()

    print("\n=== ALAS Smart Glasses System Active ===")
    print("Listening... Say 'kapat' to stop.\n")
    speak("Merhaba, Bugün nereye gitmek istersin?")
    wait_until_done()

    try:
        # Start continuous listening with our callback.
        stt.listen_continuous(callback=my_callback)
    except KeyboardInterrupt:
        print("\nSystem stopped by the user.")
    finally:
        stt.close()