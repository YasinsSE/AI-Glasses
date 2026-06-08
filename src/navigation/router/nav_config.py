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
    waypoint_threshold_m: float = 15.0     # "On the node" tolerance — mark a waypoint reached.
    # "On the segment" tolerance — perpendicular distance to the route polyline
    # beyond which we are genuinely off-route. Deliberately WIDE: it must absorb
    # OSM pedestrian-map drift + GPS noise + nearest-node snapping at the start
    # of a route (keci_testv4 went off-route ~immediately at 35 m on a longer
    # route). The destination-progress check is the real wrong-way signal.
    off_route_corridor_m: float = 50.0

    # Navigation announcement cadence (consumed by NavigationService)
    approach_threshold_m: float = 45.0       # Pre-warn when distance to next turn < N.
                                             # Raised 30→45: at walking speed 30 m left
                                             # almost no lead time once the obstacle-alert
                                             # queue drained (turn announced too late).
    long_stretch_threshold_m: float = 100.0  # > N -> fall back to the periodic reminder.
    progress_announce_interval: float = 25.0  # positive "on track" / distance reminder period.
    # After a turn instruction is spoken, suppress the destination-distance ping
    # ("hedefe X metre") for this long, so the user does not hear "26 m turn"
    # immediately followed by "62 m to target" (confusing in the field test).
    progress_suppress_after_turn_sec: float = 12.0

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
