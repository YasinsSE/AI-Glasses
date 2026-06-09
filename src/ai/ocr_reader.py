"""On-demand sign/text reading (C3) — Tesseract OCR over the last camera frame.

Voice command "oku" → one frame → Tesseract (Turkish) → TTS. Runs only on
demand, so the 1-3 s OCR cost never touches the perception loop's budget.

Install on the Jetson:
    sudo apt install tesseract-ocr tesseract-ocr-tur
    pip3 install pytesseract

Note: the CPU only ever sees the ISP-downscaled 512x384 frame, so this reads
storefront signs and large lettering, not fine print. The 2x cubic upscale
below measurably helps Tesseract at this resolution.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("ALAS.ocr")


def ocr_available() -> bool:
    """True when pytesseract + the tesseract binary are usable."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def read_sign(frame_bgr, lang: str = "tur") -> Optional[str]:
    """OCR the frame and return speakable text.

    Returns None when OCR is not installed, "" when nothing legible was
    found. Tesseract does its own binarisation, so preprocessing stays at
    grayscale + upscale (a hard threshold hurts in mixed outdoor lighting).
    """
    try:
        import cv2
        import pytesseract
    except ImportError:
        logger.warning("[OCR] pytesseract not installed — 'oku' disabled.")
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    try:
        # PSM 11: sparse text — finds isolated words/lines, the typical
        # geometry of signs and shop fronts (not a dense paragraph).
        raw = pytesseract.image_to_string(gray, lang=lang, config="--psm 11")
    except Exception:  # noqa: BLE001
        logger.exception("[OCR] tesseract failed")
        return ""
    return _clean(raw)


def _clean(raw: str) -> str:
    """Keep only word-like tokens so TTS does not read OCR noise aloud."""
    words = []
    for token in raw.split():
        # A speakable token has at least two letters/digits in it.
        if len(re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]", token)) >= 2:
            words.append(token.strip(".,;:!?()[]{}\"'"))
    return " ".join(w for w in words if w)
