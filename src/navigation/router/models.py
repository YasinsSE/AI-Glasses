# models.py
# Shared data structures and enums used across all modules.

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Coordinate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coord:
    """Immutable geographic coordinate."""
    lat: float
    lon: float


# ---------------------------------------------------------------------------
# Route step
# ---------------------------------------------------------------------------

@dataclass
class RouteStep:
    """A single navigation instruction in a route."""
    step_id: int
    text: str
    location: Coord
    action: str                  # "start" | "turn_right" | "turn_left" | "continue" | "finish"
    distance_meters: int
    road_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "text": self.text,
            "location": {"lat": self.location.lat, "lon": self.location.lon},
            "action": self.action,
            "distance_meters": self.distance_meters,
            "road_name": self.road_name,
        }

    @staticmethod
    def from_dict(d: dict) -> "RouteStep":
        return RouteStep(
            step_id=d["step_id"],
            text=d["text"],
            location=Coord(d["location"]["lat"], d["location"]["lon"]),
            action=d["action"],
            distance_meters=d["distance_meters"],
            road_name=d.get("road_name"),
        )


# ---------------------------------------------------------------------------
# Navigation status
# ---------------------------------------------------------------------------

class RouteStatus(Enum):
    INACTIVE       = "inactive"
    PROGRESSING    = "progressing"
    WAYPOINT_HIT   = "waypoint_hit"
    OFF_ROUTE      = "off_route"
    FINISHED       = "finished"


@dataclass
class ProgressResult:
    """Returned by RouteTracker.check_progress() every GPS update."""
    status: RouteStatus
    message: str
    distance_to_next: Optional[float] = None   # metres
    current_step: Optional[RouteStep] = None
