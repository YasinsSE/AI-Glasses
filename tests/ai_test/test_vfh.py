"""Unit tests for the VFH local planner.

Standalone script (no pytest dependency) — matches the rest of the tests/
directory. Runs without a model, camera, or TTS engine: synthesises fake
segmentation masks and asserts the planner picks the right open sector.

Run from the repo root:
    python tests/ai_test/test_vfh.py
"""

import os
import sys

import numpy as np

# Make src/ importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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


def _check(label: str, cond: bool, detail: str = "") -> int:
    """Print one line; return 0 on pass, 1 on fail."""
    if cond:
        print(f"  PASS  {label}")
        return 0
    print(f"  FAIL  {label}  {detail}")
    return 1


def test_clear_path_no_activation() -> int:
    """Walkable everywhere → planner should NOT activate."""
    planner = _new_planner()
    mask = _blank_mask()
    scene = analyse_scene(mask)
    fails = 0
    fails += _check(
        "should_activate False on clear mask",
        planner.should_activate(scene) is False,
    )
    guidance = planner.plan(mask, scene)
    fails += _check("plan() returns None on clear mask", guidance is None)
    return fails


def test_centre_obstacle_picks_open_side() -> int:
    """Big collision obstacle in the centre-bottom → open sectors on the sides.

    With a target_action of None (= centre), the planner should pick a non-centre
    sector and yield a LEFT/RIGHT-flavoured action with non-empty histogram.
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
    fails = 0
    fails += _check(
        "should_activate True with central obstacle",
        planner.should_activate(scene) is True,
    )
    guidance = planner.plan(mask, scene)
    fails += _check("plan() returns guidance", guidance is not None)
    if guidance is not None:
        fails += _check(
            "histogram length == num_sectors",
            len(guidance.histogram) == ALASConfig().vfh_num_sectors,
            detail=f"got {len(guidance.histogram)}",
        )
        fails += _check(
            "selected sector is not the centre",
            guidance.sector_index != ALASConfig().vfh_num_sectors // 2,
            detail=f"sector={guidance.sector_index}",
        )
        fails += _check(
            "action is a steer-away (LEFT*/RIGHT*) or STOP",
            guidance.action in {
                VFHAction.LEFT, VFHAction.LEFT_SLIGHT,
                VFHAction.RIGHT, VFHAction.RIGHT_SLIGHT,
                VFHAction.STOP,
            },
            detail=f"action={guidance.action}",
        )
    return fails


def test_left_blocked_picks_right() -> int:
    """Block the entire LEFT half — only right sectors are open."""
    planner = _new_planner()
    mask = _blank_mask()
    h, w = mask.shape
    _paint_block(mask, x0=0, x1=w // 2, y0=h // 2, y1=h,
                 cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    guidance = planner.plan(mask, scene)
    fails = 0
    fails += _check("plan() returns guidance (left blocked)", guidance is not None)
    if guidance is not None:
        centre = ALASConfig().vfh_num_sectors // 2
        fails += _check(
            "selected sector is on the right half",
            guidance.sector_index >= centre,
            detail=f"sector={guidance.sector_index}",
        )
        fails += _check(
            "action is RIGHT-flavoured or STRAIGHT",
            guidance.action in {VFHAction.RIGHT, VFHAction.RIGHT_SLIGHT, VFHAction.STRAIGHT},
            detail=f"action={guidance.action}",
        )
    return fails


def test_fully_blocked_returns_stop() -> int:
    """Whole near-field covered in collision class → STOP."""
    planner = _new_planner()
    mask = _blank_mask()
    h, w = mask.shape
    _paint_block(mask, x0=0, x1=w, y0=h // 3, y1=h,
                 cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    guidance = planner.plan(mask, scene)
    fails = 0
    fails += _check("plan() returns guidance (fully blocked)", guidance is not None)
    if guidance is not None:
        fails += _check(
            "action is STOP",
            guidance.action == VFHAction.STOP,
            detail=f"action={guidance.action}",
        )
        fails += _check(
            "TTS text contains 'Durun'",
            "Durun" in guidance.text,
            detail=f"text={guidance.text!r}",
        )
    return fails


def test_small_obstacle_below_activation() -> int:
    """A tiny obstacle in the corner must not wake the planner up."""
    planner = _new_planner()
    mask = _blank_mask()
    # 4×4 pixel speck in the upper-left — well under activation_ratio and
    # not in the central zone.
    _paint_block(mask, x0=2, x1=6, y0=2, y1=6, cid=int(ClassID.COLLISION_OBSTACLE))
    scene = analyse_scene(mask)
    return _check(
        "should_activate False for tiny corner speck",
        planner.should_activate(scene) is False,
    )


def main() -> int:
    suites = [
        ("clear_path_no_activation",       test_clear_path_no_activation),
        ("centre_obstacle_picks_open_side", test_centre_obstacle_picks_open_side),
        ("left_blocked_picks_right",        test_left_blocked_picks_right),
        ("fully_blocked_returns_stop",      test_fully_blocked_returns_stop),
        ("small_obstacle_below_activation", test_small_obstacle_below_activation),
    ]
    total_fail = 0
    for name, fn in suites:
        print(f"[{name}]")
        total_fail += fn()
    print()
    if total_fail == 0:
        print("OK — all VFH unit tests passed.")
        return 0
    print(f"FAILED — {total_fail} assertion(s) did not hold.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
