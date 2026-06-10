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

    # GPIO push-button (Jetson Nano). BCM 25 = physical pin 22.
    # Moved off BCM 18 (physical pin 12): that pin is now the I2S bit clock
    # (I2S_4_SCLK) for the MAX98357A audio amplifier — see hardware/PINOUT.md.
    button_pin: int = 25              # BCM pin number (physical pin 22).
    button_debounce_ms: int = 300

    # SLM intent classifier — minimum confidence to accept a predicted intent.
    slm_confidence_threshold: float = 0.60

    # Earcons (C4): short stereo tones for the repetitive path-keeping cues
    # ("hafif sola/sağa"). Direction is carried by STEREO PAN, so they are
    # DISABLED by default: the current rig uses a mono USB speaker, where a
    # panned beep degrades to a meaningless centred blip. Flip to True only
    # on stereo output hardware. Speech fallback is automatic while off.
    # Generate the WAVs once with: python3 scripts/generate_earcons.py
    earcons_enabled: bool = False
    earcon_dir: str = ""   # "" = src/tts_stt/earcons/ (package default)
