"""Voice I/O configuration (STT / TTS / button / intent classifier).

Tunables owned by the ``tts_stt`` package: speech-recognition timing, audio
capture format, the GPIO push-button, the TTS post-navigation silence window,
and the SLM intent-classifier confidence threshold.

Composed by :class:`main.config.ALASConfig`.
"""

from dataclasses import dataclass


@dataclass
class VoiceConfig:
    # Speech-to-text timing.
    stt_listen_timeout: float = 5.0   # Maximum listening window in seconds.
    stt_silence_sec: float = 1.5      # End recognition after this much silence.

    # Audio capture format (Vosk / PyAudio).
    sample_rate: int = 16_000
    chunk_size: int = 4096
    channels: int = 1

    # Text-to-speech gating.
    post_nav_silence_sec: float = 3.0  # Mute obstacle alerts after a nav utterance.

    # GPIO push-button (Jetson Nano).
    button_pin: int = 18              # BCM pin number.
    button_debounce_ms: int = 300

    # SLM intent classifier — minimum confidence to accept a predicted intent.
    slm_confidence_threshold: float = 0.60
