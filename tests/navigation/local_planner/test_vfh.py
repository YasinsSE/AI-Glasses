"""Unit tests for the VFH local planner (src/navigation/local_planner/vfh.py).

Standalone script (no pytest dependency) — matches the rest of the tests/
directory. Runs without a model, camera, or TTS engine: synthesises fake
segmentation masks and asserts the planner picks the right open sector.

How to run (from the repository root):
    python tests/navigation/local_planner/test_vfh.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable whether run via pytest or as a standalone script.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai.perception import ClassID, analyse_scene
from main.config import ALASConfig
from navigation.local_planner import VFHAction, VFHPlanner


def _new_planner() -> VFHPlanner:
    return VFHPlanner(ALASConfig())


def _blank_mask(h: int = 384, w: int = 512) -> np.ndarray:
    """All-walkable baseline mask."""
    return np.full((h, w), int(ClassID.WALKABLE_SURFACE), dtype=np.uint8)


def _paint_block(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int, cid: int) -> None:
    mask[y0:y1, x0:x1] = cid


def test_clear_path_no_activation() -> None:
    """Walkable everywhere -> planner should NOT activate."""
    planner = _new_planner()
    mask = _blank_mask()
    scene = analyse_scene(mask)
    assert planner.should_activate(scene) is False
    assert planner.plan(mask, scene) is None


def test_centre_obstacle_picks_open_side() -> None:
    """Big collision obstacle in the centre-bottom -> open sectors on the sides.

    With a target_action of None (= centre), the planner should pick a non-centre
    sector and yield a LEFT/RIGHT-flavoured action with a non-empty histogram.
    """
    planner = _new_planner()
    mask = _blank_mask()
    h, w = mask.shape
    # Vertical box covering the central third of the bottom half.
    _paint_block(mask,
                 x0=w // 3, x1=2 * w // 3,
                 y0=h // 2, y1=h,
                 cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    num_sectors = ALASConfig().vfh.num_sectors

    assert planner.should_activate(scene) is True
    guidance = planner.plan(mask, scene)
    assert guidance is not None
    assert len(guidance.histogram) == num_sectors, len(guidance.histogram)
    assert guidance.sector_index != num_sectors // 2, guidance.sector_index
    assert guidance.action in {
        VFHAction.LEFT, VFHAction.LEFT_SLIGHT,
        VFHAction.RIGHT, VFHAction.RIGHT_SLIGHT,
        VFHAction.STOP,
    }, guidance.action


def test_left_blocked_picks_right() -> None:
    """Block the entire LEFT half — only right sectors are open."""
    planner = _new_planner()
    mask = _blank_mask()
    h, w = mask.shape
    _paint_block(mask, x0=0, x1=w // 2, y0=h // 2, y1=h,
                 cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    guidance = planner.plan(mask, scene)
    assert guidance is not None
    centre = ALASConfig().vfh.num_sectors // 2
    assert guidance.sector_index >= centre, guidance.sector_index
    assert guidance.action in {
        VFHAction.RIGHT, VFHAction.RIGHT_SLIGHT, VFHAction.STRAIGHT,
    }, guidance.action


def test_fully_blocked_returns_stop() -> None:
    """Whole near-field covered in collision class -> STOP."""
    planner = _new_planner()
    mask = _blank_mask()
    h, w = mask.shape
    _paint_block(mask, x0=0, x1=w, y0=h // 3, y1=h,
                 cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    guidance = planner.plan(mask, scene)
    assert guidance is not None
    assert guidance.action == VFHAction.STOP, guidance.action
    assert "Durun" in guidance.text, guidance.text


def test_small_obstacle_below_activation() -> None:
    """A tiny obstacle in the corner must not wake the planner up."""
    planner = _new_planner()
    mask = _blank_mask()
    # 4x4 pixel speck in the upper-left — well under activation_ratio and
    # not in the central zone.
    _paint_block(mask, x0=2, x1=6, y0=2, y1=6, cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    assert planner.should_activate(scene) is False


# Test functions for both pytest and the standalone runner below.
_TESTS = [
    test_clear_path_no_activation,
    test_centre_obstacle_picks_open_side,
    test_left_blocked_picks_right,
    test_fully_blocked_returns_stop,
    test_small_obstacle_below_activation,
]


def main() -> int:
    """Standalone runner: execute every test and print a pass/fail summary."""
    failures = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}  {exc}")
    print()
    if failures == 0:
        print("OK — all VFH unit tests passed.")
        return 0
    print(f"FAILED — {failures} test(s) did not pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
