"""Unit tests for analyse_corridor — crossing detection + near-weighted offset.

Crossing detection (Faz 4) is safety-critical: a false positive could imply a
street crossing is there when it is not. These tests lock in the two shields
that make a sparse far-pixel hallucination FAIL (density + contiguous width),
and that the near-row-weighted offset does not get dragged into a far lateral
opening (the "centroid fallacy").

Run standalone or via pytest:
    python3 tests/ai/test_corridor.py
"""

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.perception import ClassID, analyse_corridor  # noqa: E402

WALK = int(ClassID.WALKABLE_SURFACE)   # 0
ROAD = int(ClassID.VEHICLE_ROAD)       # 2

H = W = 100
# Central strip bounds used by _detect_crossing (mirror the module constants).
CB_LO, CB_HI = 33, 67          # CENTERLINE_BAND_FRAC = 0.34 → central 34 cols
TOP_CUT = 30                   # CROSSING_TOP_IGNORE_FRAC = 0.30


def _crossing_scaffold():
    """Mask filled with road, with the central strip split into vertical thirds.

    Returns (mask, far_slice) so each test can paint the far band differently.
    near third = walkable (your sidewalk), mid third = road, far third = caller.
    """
    mask = np.full((H, W), ROAD, dtype=np.uint8)
    strip_h = H - TOP_CUT                  # 70
    t = strip_h // 3                       # 23
    far = slice(TOP_CUT, TOP_CUT + t)             # img rows 30..52
    mid = slice(TOP_CUT + t, TOP_CUT + 2 * t)     # img rows 53..75
    near = slice(TOP_CUT + 2 * t, H)              # img rows 76..99
    mask[near, CB_LO:CB_HI] = WALK
    mask[mid, CB_LO:CB_HI] = ROAD
    return mask, far


def test_crossing_true_for_clean_pattern():
    """walkable → road → CONTIGUOUS walkable beyond → crossing candidate True."""
    mask, far = _crossing_scaffold()
    mask[far, CB_LO:CB_HI] = WALK          # solid far sidewalk
    info = analyse_corridor(mask)
    assert info.crossing is True, "clean walkable→road→walkable must be a candidate"


def test_crossing_false_for_sparse_far_noise():
    """Scattered far walkable columns (model noise) must NOT pass the width shield."""
    mask, far = _crossing_scaffold()
    # Alternating columns walkable: high density but max contiguous run == 1.
    for c in range(CB_LO, CB_HI, 2):
        mask[far, c] = WALK
    info = analyse_corridor(mask)
    assert info.crossing is False, "non-contiguous far walkable must be rejected"


def test_crossing_false_without_road_ahead():
    """No road in the middle band → not a crossing (just open walkable ahead)."""
    mask, far = _crossing_scaffold()
    mask[far, CB_LO:CB_HI] = WALK
    strip_h = H - TOP_CUT
    t = strip_h // 3
    mask[slice(TOP_CUT + t, TOP_CUT + 2 * t), CB_LO:CB_HI] = WALK  # mid = walkable, not road
    info = analyse_corridor(mask)
    assert info.crossing is False, "no road ahead → no crossing"


def test_offset_not_dragged_by_far_lateral_opening():
    """Near rows centred, a far-left opening present → offset stays near centre.

    The centroid fallacy: a wide sidewalk with open space far to one side drags
    an unweighted centroid sideways and the assistant nags "hafif sola". The
    near-row weighting must keep the steering offset close to centre when the
    ground at the user's feet is centred.
    """
    mask = np.full((H, W), ROAD, dtype=np.uint8)
    # Near rows (bottom) — walkable centred under the user's feet.
    mask[80:H, 40:60] = WALK
    # Far rows (upper corridor) — a big opening off to the LEFT only.
    mask[50:70, 15:32] = WALK
    info = analyse_corridor(mask)
    assert info.valid is True
    assert info.offset > -0.30, (
        "near-weighted offset should not be pulled hard left by a far opening",
        info.offset)


if __name__ == "__main__":
    test_crossing_true_for_clean_pattern()
    test_crossing_false_for_sparse_far_noise()
    test_crossing_false_without_road_ahead()
    test_offset_not_dragged_by_far_lateral_opening()
    print("all corridor tests passed")
