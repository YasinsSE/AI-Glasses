"""Camera FOV / tilt calibration helper for ground-plane distance estimation.

ALAS turns an obstacle blob into a "N metres away" warning with inverse
perspective mapping (see ``src/ai/geometry.py``). That math is only as good as
three mounting parameters in ``AIConfig``: ``camera_height_m``,
``camera_tilt_deg`` and ``camera_vfov_deg``. The IMX219-120 in sensor-mode 4 is
a cropped readout, so its effective VERTICAL FOV is neither 120 nor a clean 60 —
it must be measured. This tool does that measurement with a tape measure and one
saved frame, no model required.

────────────────────────────────────────────────────────────────────────────
HOW TO MEASURE (do this once, on the final rig)
────────────────────────────────────────────────────────────────────────────
1. Mount the glasses as worn. Measure the camera lens height above the ground
   (``--height``, metres).
2. Put a small marker on the ground a KNOWN distance straight ahead (e.g. a
   piece of tape 3.00 m in front of you). Measure it with a tape (``--dist``).
3. Capture one frame (the overlay recorder already saves frames, or use
   ``eval/ai/image_seg_demo.py``). Open it in any image viewer and read the
   PIXEL ROW (y, 0 = top) where the marker touches the ground (``--row``), plus
   the image height in pixels (``--img-h``, default 384 = model input).
4. Run this tool. It inverts the projection to report the ``camera_vfov_deg``
   that makes the math agree with your tape measure, then prints a distance
   table so you can sanity-check other rows.

Examples (from the repository root):
    # Solve vertical FOV from one measurement (tilt assumed known):
    python3 eval/ai/calibrate_camera.py --height 1.65 --tilt 5 \
        --dist 3.0 --row 300 --img-h 384

    # Solve tilt instead (FOV assumed known):
    python3 eval/ai/calibrate_camera.py --solve tilt --vfov 60 \
        --height 1.65 --dist 3.0 --row 300 --img-h 384

    # Just print the distance-per-row table for the CURRENT config values:
    python3 eval/ai/calibrate_camera.py --check --height 1.65 --tilt 5 --vfov 60
"""

import argparse
import math
import sys
from pathlib import Path

# Reuse the production projection so calibration matches runtime exactly.
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(_REPO / "src"))
from ai.geometry import CameraGeometry, pixel_to_ground_distance  # noqa: E402


def solve_vfov_deg(height_m, tilt_deg, dist_m, row, img_h):
    """Vertical FOV (deg) that projects pixel ``row`` to ground distance ``dist_m``.

    Inverts geometry.pixel_to_ground_distance:
        theta_total = atan(height / dist)
        theta_pix   = theta_total - tilt
        vfov        = theta_pix / ((row+0.5)/img_h - 0.5)
    """
    norm = (row + 0.5) / img_h - 0.5  # signed offset from image centre
    if abs(norm) < 1e-3:
        raise ValueError(
            "Marker row is too close to the vertical centre of the image; "
            "the FOV is indeterminate there. Re-measure with the marker lower "
            "in the frame (nearer to you).")
    theta_total = math.atan(height_m / dist_m)
    theta_pix = theta_total - math.radians(tilt_deg)
    return math.degrees(theta_pix / norm)


def solve_tilt_deg(height_m, vfov_deg, dist_m, row, img_h):
    """Downward tilt (deg) that projects pixel ``row`` to ground distance ``dist_m``."""
    norm = (row + 0.5) / img_h - 0.5
    theta_total = math.atan(height_m / dist_m)
    theta_pix = norm * math.radians(vfov_deg)
    return math.degrees(theta_total - theta_pix)


def distance_table(geom, img_h, rows=None):
    """Return [(row, distance_m_or_None)] for a spread of rows down the frame."""
    if rows is None:
        rows = [int(img_h * f) for f in (0.50, 0.60, 0.70, 0.80, 0.90, 0.99)]
    return [(r, pixel_to_ground_distance(r, img_h, geom)) for r in rows]


def _print_table(title, geom, img_h):
    print(title)
    print("    row   |  ground distance")
    print("    ------+------------------")
    for r, d in distance_table(geom, img_h):
        shown = "— (above horizon)" if d is None else f"{d:6.2f} m"
        print(f"    {r:5d} |  {shown}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solve", choices=("vfov", "tilt"), default="vfov",
                    help="Which parameter to solve for (default: vfov).")
    ap.add_argument("--check", action="store_true",
                    help="Skip solving; just print the distance table for the given params.")
    ap.add_argument("--height", type=float, required=True, help="Camera height above ground (m).")
    ap.add_argument("--tilt", type=float, default=5.0, help="Downward tilt (deg). Used/initial value.")
    ap.add_argument("--vfov", type=float, default=60.0, help="Vertical FOV (deg). Used/initial value.")
    ap.add_argument("--dist", type=float, help="Measured ground distance to the marker (m).")
    ap.add_argument("--row", type=int, help="Pixel row where the marker meets the ground (0 = top).")
    ap.add_argument("--img-h", type=int, default=384, help="Image height in pixels (default 384).")
    args = ap.parse_args(argv)

    if not args.check:
        if args.dist is None or args.row is None:
            ap.error("--dist and --row are required unless --check is given.")
        if not (0 <= args.row < args.img_h):
            ap.error(f"--row must be within [0, {args.img_h}).")

        try:
            if args.solve == "vfov":
                solved = solve_vfov_deg(args.height, args.tilt, args.dist, args.row, args.img_h)
            else:
                solved = solve_tilt_deg(args.height, args.vfov, args.dist, args.row, args.img_h)
        except ValueError as exc:
            ap.error(str(exc))

        if args.solve == "vfov":
            args.vfov = solved
            print(f"\n  Solved camera_vfov_deg = {solved:.2f}   (was {60.0:.1f} default)")
            print(f"  → set camera_vfov_deg: float = {solved:.1f} in src/ai/ai_config.py")
        else:
            args.tilt = solved
            print(f"\n  Solved camera_tilt_deg = {solved:.2f}")
            print(f"  → set camera_tilt_deg: float = {solved:.1f} in src/ai/ai_config.py")

        # Round-trip sanity: the solved params must reproduce the measured distance.
        geom = CameraGeometry(height_m=args.height, tilt_deg=args.tilt, vfov_deg=args.vfov)
        back = pixel_to_ground_distance(args.row, args.img_h, geom)
        if back is not None:
            err = abs(back - args.dist)
            flag = "OK" if err < 0.05 else "CHECK INPUTS"
            print(f"  Verify: row {args.row} → {back:.2f} m "
                  f"(measured {args.dist:.2f} m, error {err:.3f} m) [{flag}]")

    geom = CameraGeometry(height_m=args.height, tilt_deg=args.tilt, vfov_deg=args.vfov)
    print()
    _print_table(
        f"  Distance table  (height={args.height} m, tilt={args.tilt:.1f}°, "
        f"vfov={args.vfov:.1f}°, img_h={args.img_h}):",
        geom, args.img_h)
    print("\n  Sanity check the table against a tape measure at a couple of rows.")
    print("  If distances look optimistic on stairs/slopes/curbs, that is expected")
    print("  (the projection assumes flat ground — see src/ai/geometry.py).")


if __name__ == "__main__":
    main()
