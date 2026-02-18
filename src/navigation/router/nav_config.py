# nav_config.py
# All tuneable constants in one place.
# Pass a NavConfig instance to every module that needs settings.

from dataclasses import dataclass, field


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
    waypoint_threshold_m: float = 15.0     # distance to mark a waypoint as reached
    off_route_threshold_m: float = 40.0    # extra metres beyond step distance → off-route

    # Logging
    log_dir: str = "."                     # directory for saved JSON files
    route_filename: str = "active_route.json"

    @property
    def route_filepath(self) -> str:
        import os
        return os.path.join(self.log_dir, self.route_filename)
