"""Text-to-speech output via espeak, serialised through a worker thread.

Each utterance is spoken by calling the ``espeak`` binary directly (Turkish
voice). This is far cheaper than the old "spawn python + import pyttsx3 per
utterance" path — no interpreter startup, no module import — which matters on
the Jetson Nano where warnings fire several times per minute. A single
background worker drains the queue so utterances never overlap, and each call
is its own short-lived subprocess so a stuck engine can never freeze the main
program.

Urgency: ``urgent=True`` utterances (the closing-threat warnings from
PerceptionService) are spoken FASTER and HIGHER-pitched, so the user hears the
urgency in the rhythm of the voice, not only in the words.
"""

import logging
import subprocess
import threading
import queue

logger = logging.getLogger("ALAS.tts")

# Queue so utterances do not cut each other off. Items are (kind, text, urgent)
# tuples; ``None`` is the shutdown sentinel.
_tts_queue = queue.Queue()
# Guards the read-modify-write when dropping stale obstacle items.
_queue_lock = threading.Lock()

# Surface each distinct TTS failure ONCE (so a broken audio path is visible in
# the journal without spamming a warning for every single utterance).
_logged_tts_errors = set()

# ── espeak voice parameters ──────────────────────────────────────────────────
_VOICE         = "tr"   # Turkish voice (correct phonetics).
_AMPLITUDE     = 200    # 0..200 — max volume for outdoor use.
_NORMAL_SPEED  = 150    # words/min (espeak default 175; 150 = clearer Turkish).
_NORMAL_PITCH  = 50     # 0..99 (espeak default 50).
# Urgent: faster + higher pitch → conveys "act now" through the rhythm.
_URGENT_SPEED  = 200
_URGENT_PITCH  = 70


def _drop_pending_obstacles():
    """Remove not-yet-started obstacle items from the queue.

    Obstacle alerts are spatial and perishable: replaying warnings about cars
    the user already walked past is worse than silence. We keep only the newest
    obstacle by dropping any pending ones before enqueuing it. Priority items
    (nav / announcements) are preserved. Must be called holding ``_queue_lock``.
    """
    kept = []
    try:
        while True:
            item = _tts_queue.get_nowait()
            _tts_queue.task_done()
            if item is not None and item[0] == "obstacle":
                continue  # drop stale obstacle
            kept.append(item)
    except queue.Empty:
        pass
    for item in kept:
        _tts_queue.put(item)


def _log_tts_error(key: str, msg: str, *args) -> None:
    if key not in _logged_tts_errors:
        _logged_tts_errors.add(key)
        logger.warning(msg, *args)


def _tts_worker():
    """Worker thread that speaks queued text via espeak."""
    while True:
        item = _tts_queue.get()
        if item is None:
            _tts_queue.task_done()
            break
        kind, text, urgent = item

        speed = _URGENT_SPEED if urgent else _NORMAL_SPEED
        pitch = _URGENT_PITCH if urgent else _NORMAL_PITCH
        cmd = ["espeak", "-v", _VOICE, "-s", str(speed), "-p", str(pitch),
               "-a", str(_AMPLITUDE), text]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15,
            )
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace").strip() or "no stderr"
                _log_tts_error(
                    err,
                    "[TTS] espeak FAILED (rc=%s) — no audio. Is the audio device/"
                    "PulseAudio routed under the service? Error: %s",
                    proc.returncode, err,
                )
        except FileNotFoundError:
            _log_tts_error("nofile",
                           "[TTS] 'espeak' not found — install it: sudo apt install espeak")
        except subprocess.TimeoutExpired:
            logger.warning("[TTS] espeak timed out for %r", text[:40])
        except Exception as e:  # noqa: BLE001
            _log_tts_error(str(e), "[TTS] espeak launch failed: %s", e)
        finally:
            _tts_queue.task_done()


# Start the worker thread.
_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text: str, kind: str = "info", urgent: bool = False):
    """Add text to the speech queue.

    ``kind="obstacle"`` marks perishable hazard alerts: enqueuing one first
    drops any pending obstacle still waiting, so a post-freeze backlog never
    dumps stale spatial warnings. ``urgent=True`` speaks faster + higher-pitched.
    """
    text = (text or "").strip()
    if not text:
        return
    item = (kind, text, bool(urgent))
    if kind == "obstacle":
        with _queue_lock:
            _drop_pending_obstacles()
            _tts_queue.put(item)
    else:
        _tts_queue.put(item)


def wait_until_done():
    """Block until all queued utterances have been spoken."""
    _tts_queue.join()


def shutdown_tts():
    """Drain the queue, then stop the worker. Call before the program exits."""
    _tts_queue.join()       # Let pending utterances finish.
    _tts_queue.put(None)    # Signal the worker to stop.
    _tts_thread.join(timeout=5)
    logger.info("[TTS] Shut down.")


def handle_intent_response(intent: str, text: str = ""):
    """Speak feedback to the user based on the intent returned by the SLM."""
    responses = {
        "system_command": "Sistem kapatılıyor, iyi günler dilerim.",
        "navigation": "Anlaşıldı, rota hesaplanıyor.",
        "general": "",  # Stay silent for general intents.
    }
    msg = responses.get(intent, "")
    if msg:
        speak(msg)


# Standalone module test.
if __name__ == "__main__":
    print("TTS test system started...")
    speak("Sistem hazır. Ses testi yapılıyor.")
    speak("Dikkat, çok yakın, durun.", kind="obstacle", urgent=True)
    shutdown_tts()
