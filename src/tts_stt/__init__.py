# tts_stt package — speech I/O subsystems and the load_stt() factory.

import logging

logger = logging.getLogger("ALAS.tts_stt")


def load_stt(config):
    """Build the STT engine, or return None when STT is bypassed / unavailable.

    Importing STTEngine is deferred so that --bypass-stt runs do not pay the
    Vosk / pyaudio import cost on hardware where the mic stack is missing.
    """
    if config.bypass_stt:
        logger.info("[tts_stt] --bypass-stt active: microphone disabled, typed input only.")
        return None
    try:
        from tts_stt.stt import STTEngine
        return STTEngine()
    except Exception:
        logger.exception("[tts_stt] STTEngine could not be loaded — falling back to bypass mode.")
        return None
