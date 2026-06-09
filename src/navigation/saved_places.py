"""SavedPlaces — named GPS bookmarks ("ev", "iş") persisted as JSON.

Backs the "konumu ev olarak kaydet" / "eve git" voice commands: the current
fix is stored under a spoken name and later fed straight into
``VoiceCommandHandler.route_to_coord``. One flat JSON file, written atomically
so a power cut mid-save cannot corrupt the existing bookmarks.
"""

import json
import logging
import os
import tempfile
from typing import Dict, Optional, Tuple

logger = logging.getLogger("ALAS.saved_places")


class SavedPlaces:
    """Tiny name → (lat, lon) store on disk."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._places: Dict[str, Tuple[float, float]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            self._places = {
                str(k): (float(v[0]), float(v[1]))
                for k, v in raw.items()
            }
            logger.info("[Places] %d saved place(s) loaded.", len(self._places))
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 — a corrupt file must not kill boot
            logger.exception("[Places] could not read %s — starting empty.", self._path)

    def save(self, name: str, lat: float, lon: float) -> None:
        self._places[name.strip().lower()] = (lat, lon)
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self._path) or ".",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._places, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("[Places] saved '%s' = (%.6f, %.6f).", name, lat, lon)

    def get(self, name: str) -> Optional[Tuple[float, float]]:
        return self._places.get(name.strip().lower())

    def names(self):
        return sorted(self._places)
