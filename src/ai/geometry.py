"""
ALAS Camera Geometry — distance estimation from semantic segmentation pixels
============================================================================
A semantic mask only tells us *which class* each pixel belongs to. It does not
give us depth. To turn an obstacle blob into a "this is N metres away" warning
we use **ground-plane projection** (a.k.a. inverse perspective mapping).

Assumption: the obstacle's contact with the ground is its lowest pixel row in
the segmentation mask. For a known camera height, downward tilt, and vertical
field of view, every pixel that lies on the flat ground plane has a single,
computable distance from the camera.

The math:
    For pixel row y in an image of height H:
        theta_pix   = ((y + 0.5) / H - 0.5) * vfov_rad
        theta_total = tilt_rad + theta_pix
        distance    = camera_height_m / tan(theta_total)

    If theta_total <= 0 the ray is at or above the horizon → no ground
    intersection, return None.

Limitations (intentional, documented for callers):
    - Assumes the obstacle touches the ground. Hanging signs / low ceilings
      will report incorrectly.
    - Assumes flat ground. Stairs, slopes, and curbs will be optimistic.
    - Assumes camera is level (no roll). Head-tilt sideways skews the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CameraGeometry:
    """Static camera mounting parameters used for ground-plane projection."""
    height_m: float        # height of camera above the ground (metres)
    tilt_deg: float        # downward tilt — positive means looking down
    vfov_deg: float        # vertical field of view in degrees


def pixel_to_ground_distance(
    pixel_y: int,
    image_height: int,
    geom: CameraGeometry,
) -> Optional[float]:
    """
    Estimate the ground-plane distance (metres) from camera to the point that
    projects to pixel row ``pixel_y``.

    Args:
        pixel_y:      0-indexed row of the pixel (0 = top of image).
        image_height: total image height in pixels (must be > 0).
        geom:         calibrated CameraGeometry for the device.

    Returns:
        Distance in metres, or None if the pixel is at or above the horizon.
    """
    if image_height <= 0:
        return None

    vfov_rad = math.radians(geom.vfov_deg)
    tilt_rad = math.radians(geom.tilt_deg)

    theta_pix = ((pixel_y + 0.5) / image_height - 0.5) * vfov_rad
    theta_total = tilt_rad + theta_pix

    if theta_total <= 1e-4:
        return None  # at or above horizon — no ground intersection

    return geom.height_m / math.tan(theta_total)
