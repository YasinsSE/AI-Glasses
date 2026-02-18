# nav_logger.py
# Handles all file I/O for the navigation system.
# Saves routes and navigation events as JSON.

import json
import os
import logging
from datetime import datetime
from typing import List, Optional

from models import RouteStep, ProgressResult
from nav_config import NavConfig

# Standard Python logger — configure at app entry point if needed
logger = logging.getLogger(__name__)


class NavLogger:
    """
    Persists route data and navigation events to JSON files.

    Args:
        config: NavConfig instance for file paths and directories.
    """

    def __init__(self, config: Optional[NavConfig] = None) -> None:
        self.config = config or NavConfig()
        os.makedirs(self.config.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Route persistence
    # ------------------------------------------------------------------

    def save_route(self, steps: List[RouteStep]) -> bool:
        """
        Serialize a route to JSON.

        Args:
            steps: List of RouteStep objects.

        Returns:
            True on success, False on failure.
        """
        filepath = self.config.route_filepath
        try:
            data = {
                "saved_at": datetime.now().isoformat(),
                "step_count": len(steps),
                "steps": [s.to_dict() for s in steps],
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Route saved to {filepath} ({len(steps)} steps).")
            return True
        except IOError as e:
            logger.error(f"Failed to save route to {filepath}: {e}")
            return False

    def load_route(self, filepath: Optional[str] = None) -> Optional[List[RouteStep]]:
        """
        Load a previously saved route from JSON.

        Args:
            filepath: Path override; uses config default if omitted.

        Returns:
            List of RouteStep objects, or None if loading failed.
        """
        path = filepath or self.config.route_filepath
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = [RouteStep.from_dict(s) for s in data["steps"]]
            logger.info(f"Route loaded from {path} ({len(steps)} steps).")
            return steps
        except (IOError, KeyError, ValueError) as e:
            logger.error(f"Failed to load route from {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Session event logging
    # ------------------------------------------------------------------

    def log_event(self, result: ProgressResult, position_lat: float, position_lon: float) -> None:
        """
        Append a single navigation event to a session log file.

        Args:
            result:       ProgressResult from RouteTracker.
            position_lat: Current latitude.
            position_lon: Current longitude.
        """
        event_file = os.path.join(self.config.log_dir, "nav_session.jsonl")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "lat": position_lat,
            "lon": position_lon,
            "status": result.status.value,
            "message": result.message,
            "distance_to_next": result.distance_to_next,
        }
        try:
            with open(event_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to write event log: {e}")
