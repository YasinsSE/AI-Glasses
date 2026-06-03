"""Text-to-speech output via pyttsx3, serialised through a worker thread.

Each utterance is spoken in a short-lived subprocess so a hung pyttsx3 engine
can never block the main program. A single background worker drains the queue
so utterances never overlap.
"""

import logging
import subprocess
import sys
import threading
import queue

logger = logging.getLogger("ALAS.tts")

# Queue so utterances do not cut each other off. Items are (kind, text)
# tuples; ``None`` is the shutdown sentinel.
_tts_queue = queue.Queue()
# Guards the read-modify-write when dropping stale obstacle items.
_queue_lock = threading.Lock()

# Surface each distinct TTS failure ONCE (so a broken audio path is visible in
# the journal without spamming a warning for every single utterance).
_logged_tts_errors = set()


def _drop_pending_obstacles():
    """Remove not-yet-started obstacle items from the queue.

    Obstacle alerts are spatial and perishable: after the Jetson unfreezes,
    replaying warnings about cars the user already walked past is worse than
    silence. We keep only the newest obstacle by dropping any pending ones
    before enqueuing it. Priority items (nav / announcements) are preserved.
    Must be called holding ``_queue_lock``.
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


def _tts_worker():
    """Worker thread that speaks queued text in the background."""
    while True:
        item = _tts_queue.get()
        if item is None:
            _tts_queue.task_done()
            break
        _, text = item

        # Jetson Nano / Mac compatible runner. A subprocess is used so a stuck
        # engine cannot freeze the main program. Errors are reported to stderr
        # (and surfaced via the logger below) instead of being swallowed — a
        # silent failure is exactly what hides a broken audio path.
        script = (
            "import pyttsx3, sys\n"
            "try:\n"
            "    engine = pyttsx3.init()\n"
            "    engine.setProperty('rate', 150)\n"  # Speech rate (slightly slower = clearer).
            "    engine.setProperty('volume', 1.0)\n"
            # Select the Turkish voice so espeak uses Turkish phonetics instead of
            # reading Turkish text with the default English voice (unintelligible).
            "    try:\n"
            "        engine.setProperty('voice', 'tr')\n"
            "    except Exception:\n"
            "        pass\n"
            f"    engine.say({repr(text)})\n"
            "    engine.runAndWait()\n"
            "except Exception as e:\n"
            "    sys.stderr.write('TTS_SUBPROC_ERROR: %r' % (e,))\n"
            "    sys.exit(1)\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace").strip() or "no stderr"
                if err not in _logged_tts_errors:
                    _logged_tts_errors.add(err)
                    logger.warning(
                        "[TTS] playback FAILED (rc=%s) — no audio will be heard. "
                        "Check the audio device/PulseAudio under the service. Error: %s",
                        proc.returncode, err,
                    )
        except Exception as e:  # noqa: BLE001 — launching the subprocess itself failed
            if str(e) not in _logged_tts_errors:
                _logged_tts_errors.add(str(e))
                logger.warning("[TTS] subprocess launch failed: %s", e)
        finally:
            _tts_queue.task_done()


# Start the worker thread.
_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text: str, kind: str = "info"):
    """Add text to the speech queue.

    ``kind="obstacle"`` marks perishable hazard alerts: enqueuing one first
    drops any pending obstacle still waiting, so a post-freeze backlog never
    dumps stale spatial warnings. Other kinds (nav, announcements, prompts)
    always queue in order.
    """
    text = (text or "").strip()
    if not text:
        return
    if kind == "obstacle":
        with _queue_lock:
            _drop_pending_obstacles()
            _tts_queue.put((kind, text))
    else:
        _tts_queue.put((kind, text))


def wait_until_done():
    """Block until all queued utterances have been spoken."""
    _tts_queue.join()


def shutdown_tts():
    """Drain the queue, then stop the worker.

    Must be called before the program exits.
    """
    _tts_queue.join()       # Let pending utterances finish.
    _tts_queue.put(None)    # Signal the worker to stop.
    _tts_thread.join(timeout=5)
    print("[TTS] Shut down.")


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
    # NOTE: for the "general" intent we do not parrot the user's text back.
    # A future LLM integration could generate a meaningful reply here
    # (e.g. "what time is it" -> the actual time).


# Standalone module test.
if __name__ == "__main__":
    print("TTS test system started...")
    speak("Sistem hazır. Ses testi yapılıyor.")
    shutdown_tts()