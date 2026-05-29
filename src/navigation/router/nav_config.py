"""Navigation (router) configuration.

All tunable constants for the routing subsystem in one place. A ``NavConfig``
instance is passed to every router module that needs settings, and is composed
by :class:`main.config.ALASConfig`.
"""

from dataclasses import dataclass


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
    off_route_threshold_m: float = 40.0    # Extra metres beyond step distance -> off-route.

    # Navigation announcement cadence (consumed by NavigationService)
    approach_threshold_m: float = 30.0       # Pre-warn when distance to next step < N.
    long_stretch_threshold_m: float = 100.0  # > N -> fall back to the periodic reminder.
    progress_announce_interval: float = 30.0  # "X metres to go" reminder period.

    # Logging
    log_dir: str = "."                     # directory for saved JSON files
    route_filename: str = "active_route.json"

    @property
    def route_filepath(self) -> str:
        import os
        return os.path.join(self.log_dir, self.route_filename)
