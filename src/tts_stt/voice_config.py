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

    # Spoken navigation confirmation. A terse command ("eczane", "eczaneye
    # git") starts routing immediately, but a destination INFERRED from a
    # longer utterance ("ilaç almak istiyorum eczaneye gitmem lazım") is
    # confirmed first — STT mishears, and walking a blind user to the wrong
    # place is far costlier than one extra question. Only applies on the
    # microphone path; typed --bypass-stt commands are already explicit.
    nav_confirm_enabled: bool = True
    nav_confirm_min_words: int = 3        # utterances with >= this many words confirm first
    confirm_listen_timeout: float = 6.0   # seconds to wait for the yes/no answer
    confirm_silence_sec: float = 1.2

    # Deferred navigation for urban canyons (Kızılay case): a nav command
    # arriving while GPS has no fix is QUEUED instead of refused — the user
    # keeps walking, and routing starts automatically (with an announcement)
    # the moment a fix arrives. Gives up with a spoken notice after this long.
    pending_route_timeout_sec: float = 180.0

    # Earcons (C4): short stereo tones for the repetitive path-keeping cues
    # ("hafif sola/sağa"). Direction is carried by STEREO PAN, so they are
    # DISABLED by default: the current rig uses a mono USB speaker, where a
    # panned beep degrades to a meaningless centred blip. Flip to True only
    # on stereo output hardware. Speech fallback is automatic while off.
    # Generate the WAVs once with: python3 scripts/generate_earcons.py
    earcons_enabled: bool = False
    earcon_dir: str = ""   # "" = src/tts_stt/earcons/ (package default)
