"""Unit tests for _compute_safety_level (Faz 6).

UNSAFE is reserved for a genuinely blocked forward path; a hazard that is merely
present and passable is CAUTION; a far/small vehicle is SAFE. This stops the
viewer from going ~92% red on normal streets.

Run standalone or via pytest:
    python3 tests/ai/test_safety_level.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.perception import (  # noqa: E402
    ZoneInfo, ClassID, CLASS_NAMES, _compute_safety_level,
    SAFETY_SAFE, SAFETY_CAUTION, SAFETY_UNSAFE,
)


def _vehicle(corridor_overlap=0.0, bottom_half_ratio=0.4, pixel_ratio=0.10,
             distance_m=4.0):
    return ZoneInfo(
        class_id=int(ClassID.VEHICLE), class_name=CLASS_NAMES[ClassID.VEHICLE],
        pixel_ratio=pixel_ratio, dominant_zone="center",
        walkable_overlap=1.0, estimated_distance_m=distance_m,
        bottom_half_ratio=bottom_half_ratio, corridor_overlap=corridor_overlap,
    )


def test_side_parked_car_is_caution_not_unsafe():
    """A close car beside the path (not in corridor, path passable) → CAUTION."""
    zones = [_vehicle(corridor_overlap=0.0)]
    assert _compute_safety_level(zones, walkable_ratio=0.5) == SAFETY_CAUTION


def test_blocking_car_is_unsafe():
    """A car in the corridor with little room to pass → UNSAFE."""
    zones = [_vehicle(corridor_overlap=0.5)]
    assert _compute_safety_level(zones, walkable_ratio=0.08) == SAFETY_UNSAFE


def test_far_small_vehicle_is_safe():
    """A small vehicle high in the frame (far) is not even a caution → SAFE."""
    zones = [_vehicle(corridor_overlap=0.0, bottom_half_ratio=0.0, pixel_ratio=0.03)]
    assert _compute_safety_level(zones, walkable_ratio=0.6) == SAFETY_SAFE


def test_no_walkable_is_unsafe():
    """Almost no walkable area at all → blocked → UNSAFE regardless of zones."""
    assert _compute_safety_level([], walkable_ratio=0.02) == SAFETY_UNSAFE


def test_road_alongside_is_caution():
    """Vehicle road visible but off the corridor → CAUTION, not UNSAFE."""
    road = ZoneInfo(
        class_id=int(ClassID.VEHICLE_ROAD), class_name="vehicle_road",
        pixel_ratio=0.2, dominant_zone="right", corridor_overlap=0.05,
    )
    assert _compute_safety_level([road], walkable_ratio=0.5) == SAFETY_CAUTION


if __name__ == "__main__":
    test_side_parked_car_is_caution_not_unsafe()
    test_blocking_car_is_unsafe()
    test_far_small_vehicle_is_safe()
    test_no_walkable_is_unsafe()
    test_road_alongside_is_caution()
    print("all safety_level tests passed")
