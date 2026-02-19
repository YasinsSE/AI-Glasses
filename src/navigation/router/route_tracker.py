# route_tracker.py
# State machine that tracks a user's position against an active route.
# Call load_route() once, then check_progress() on every GPS update.

from typing import List, Optional

from models import Coord, RouteStep, RouteStatus, ProgressResult
from geo_utils import haversine_distance
from nav_config import NavConfig


class RouteTracker:
    """
    Stateful progress tracker for a single navigation session.

    Usage:
        tracker = RouteTracker(config)
        tracker.load_route(steps)

        # Inside GPS loop:
        result = tracker.check_progress(current_coord)
    """

    def __init__(self, config: Optional[NavConfig] = None) -> None:
        self.config = config or NavConfig()
        self._route: List[RouteStep] = []
        self._step_index: int = 0
        self._active: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_route(self, steps: List[RouteStep]) -> None:
        """Load a new route and reset state."""
        self._route = steps
        self._step_index = 0
        self._active = True

    def stop(self) -> None:
        """Forcibly end navigation."""
        self._active = False

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_step(self) -> Optional[RouteStep]:
        if 0 <= self._step_index < len(self._route):
            return self._route[self._step_index]
        return None

    @property
    def remaining_steps(self) -> int:
        return max(0, len(self._route) - self._step_index)

    # ------------------------------------------------------------------
    # Core method — call on every GPS update
    # ------------------------------------------------------------------

    def check_progress(self, position: Coord) -> ProgressResult:
        """
        Compare current GPS position to the active route.

        Args:
            position: Current geographic position.

        Returns:
            ProgressResult with status, message, and contextual data.
        """
        if not self._active:
            return ProgressResult(
                status=RouteStatus.INACTIVE,
                message="Navigation is not active.",
            )

        # All waypoints passed
        if self._step_index >= len(self._route):
            self._active = False
            return ProgressResult(
                status=RouteStatus.FINISHED,
                message="You have reached your destination.",
            )

        target = self._route[self._step_index]
        dist = haversine_distance(
            position.lat, position.lon,
            target.location.lat, target.location.lon,
        )

        # 1. Waypoint reached
        if dist < self.config.waypoint_threshold_m:
            self._step_index += 1

            if self._step_index >= len(self._route):
                self._active = False
                return ProgressResult(
                    status=RouteStatus.FINISHED,
                    message="You have reached your destination.",
                    distance_to_next=0.0,
                    current_step=target,
                )

            next_step = self._route[self._step_index]
            return ProgressResult(
                status=RouteStatus.WAYPOINT_HIT,
                message=next_step.text,
                distance_to_next=0.0,
                current_step=next_step,
            )

        # 2. Off-route check
        if dist > self.config.off_route_threshold_m and self._step_index > 0:
            prev_step = self._route[self._step_index - 1]
            dist_from_prev = haversine_distance(
                position.lat, position.lon,
                prev_step.location.lat, prev_step.location.lon,
            )
            if dist_from_prev > self.config.off_route_threshold_m:
                return ProgressResult(
                status=RouteStatus.OFF_ROUTE,
                message="You are off the route. Recalculating may be needed.",
                distance_to_next=dist,
                current_step=target,
            )

        # 3. Still on route — progressing
        return ProgressResult(
            status=RouteStatus.PROGRESSING,
            message=f"{int(dist)} m to next waypoint. ({target.action})",
            distance_to_next=dist,
            current_step=target,
        )
