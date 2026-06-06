"""Navigation (router) configuration."""

import os
from dataclasses import dataclass

# Route files (active_route.json, nav_session.jsonl) live alongside the
# router source so they are easy to find and don't pollute other directories.
_ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Road type constants (used by OSM parser)
# ---------------------------------------------------------------------------

WALKABLE_TYPES: frozenset = frozenset({
    'footway', 'pedestrian', 'path', 'steps', 'cycleway',
    'living_street', 'track', 'crossing', 'residential',
    'service', 'unclassified', 'primary', 'primary_link',
    'secondary', 'secondary_link', 'tertiary', 'tertiary_link',
    'trunk', 'trunk_link',
})

FORBIDDEN_TYPES: frozenset = frozenset({'motorway', 'motorway_link', 'construction'})

WALKING_SPEED_KMH: float = 5.0  # km/h


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class NavConfig:
    # Routing
    walking_speed_kmh: float = WALKING_SPEED_KMH
    steps_time_penalty: float = 2.0        # multiplier for 'steps' road type

    # Progress tracking
    waypoint_threshold_m: float = 15.0     # Distance to mark a waypoint as reached.
    off_route_threshold_m: float = 60.0    # Extra metres beyond step distance -> off-route.
                                           # Raised: brittle on poor pedestrian maps; the
                                           # destination-progress check below is the real signal.

    # Navigation announcement cadence (consumed by NavigationService)
    approach_threshold_m: float = 30.0       # Pre-warn when distance to next step < N.
    long_stretch_threshold_m: float = 100.0  # > N -> fall back to the periodic reminder.
    progress_announce_interval: float = 25.0  # positive "on track" / distance reminder period.

    # Destination-progress (Faz 2): "are we actually getting closer to the target?"
    progress_window_sec: float = 20.0      # sliding window over which to judge progress.
    wrong_way_gain_m: float = 25.0         # moved this much FARTHER over the window -> wrong way.

    # Route file output
    log_dir: str = _ROUTER_DIR             # always src/navigation/router/
    route_filename: str = "active_route.json"

    @property
    def route_filepath(self) -> str:
        import os
        return os.path.join(self.log_dir, self.route_filename)
