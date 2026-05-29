"""Local planner (VFH) configuration.

Tunables for the image-space Vector Field Histogram planner in
:mod:`navigation.local_planner.vfh`. The planner also reads camera geometry and
model input dimensions from the AI config; only the parameters specific to the
histogram and its activation gate live here.

Composed by :class:`main.config.ALASConfig`.
"""

from dataclasses import dataclass


@dataclass
class VFHConfig:
    enabled: bool = True

    # Histogram resolution.
    num_sectors: int = 7        # Odd, so a centre sector exists.
    grid_rows: int = 8          # Near-field grid resolution (rows).
    grid_cols: int = 16         # Near-field grid resolution (cols).
    near_rows_ratio: float = 0.55   # Fraction of mask height (from bottom) treated as near-field.
    blocked_threshold: float = 0.35  # Sector cost above this is considered "blocked".

    # Activation gate — keeps the planner silent on a clear sidewalk.
    activation_ratio: float = 0.06       # Centre obstacle pixel ratio below this -> skip.
    activation_distance_m: float = 5.0   # Vehicles closer than this trigger VFH even at low ratio.
    low_walkable_ratio: float = 0.25     # Walkable area below this -> trigger VFH.

    # Announcement cadence.
    announce_cooldown_sec: float = 6.0   # Min gap between identical VFH utterances.
