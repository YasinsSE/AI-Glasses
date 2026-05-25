"""VFH local planner — image-space Vector Field Histogram on a segmentation mask.

The classic VFH builds a polar histogram of obstacle density around a mobile
robot's pose using a 2-D occupancy grid. Here we have no LIDAR and no metric
occupancy grid — only a per-pixel class mask from the segmentation network.
We adapt the idea: treat the near-field portion of the mask as a (rows × cols)
cost grid, fold image columns into ``num_sectors`` angular bins, weight each
row by its real-world distance via ground-plane projection, and pick the
sector with lowest cost as the suggested heading.

Gating: ``should_activate`` returns False when there is no central obstacle
worth dodging, so the planner contributes zero CPU and zero TTS noise on a
clear sidewalk. The dispatcher is expected to check it before calling plan().
"""

import logging
from typing import List, Optional

import numpy as np

from ai.geometry import CameraGeometry, pixel_to_ground_distance
from ai.perception import ClassID, SceneAnalysis
from main.config import ALASConfig
from navigation.local_planner.models import VFHAction, VFHGuidance

logger = logging.getLogger("ALAS.vfh")


# Per-class cost in [0, 1]. Walkable / crosswalk = free; collisions, falls,
# vehicles, dynamic obstacles = hard block; vehicle road = soft (legal-only).
_CLASS_COSTS = {
    int(ClassID.WALKABLE_SURFACE):   0.0,
    int(ClassID.CROSSWALK):          0.0,
    int(ClassID.VEHICLE_ROAD):       0.3,
    int(ClassID.COLLISION_OBSTACLE): 1.0,
    int(ClassID.FALL_HAZARD):        1.0,
    int(ClassID.DYNAMIC_HAZARD):     1.0,
    int(ClassID.VEHICLE):            1.0,
}

# Distance floor so the inverse-distance weight does not explode at the
# very bottom row where pixel_to_ground_distance() would project to ~0 m.
_DISTANCE_FLOOR_M = 0.5

# Classes that contribute to the "is there an obstacle worth planning around"
# activation gate.
_ACTIVATION_HAZARD_CLASSES = {
    int(ClassID.COLLISION_OBSTACLE),
    int(ClassID.FALL_HAZARD),
    int(ClassID.DYNAMIC_HAZARD),
    int(ClassID.VEHICLE),
}


class VFHPlanner:
    """Stateless-ish (only smoothing state) image-space VFH planner."""

    def __init__(self, config: ALASConfig, camera_geometry: Optional[CameraGeometry] = None) -> None:
        self._cfg = config
        self._geom = camera_geometry or CameraGeometry(
            height_m=config.camera_height_m,
            tilt_deg=config.camera_tilt_deg,
            vfov_deg=config.camera_vfov_deg,
        )
        self._image_h = config.model_input_h
        self._image_w = config.model_input_w

        # Pre-compute the bottom-near region rows and the per-row distance
        # weight. These depend only on geometry + config, not on the frame.
        self._near_h = max(1, int(self._image_h * config.vfh_near_rows_ratio))
        self._near_start = self._image_h - self._near_h
        self._row_weights = self._compute_row_weights()

        # Pre-compute column → sector mapping.
        self._sector_for_col = self._compute_sector_for_col()
        self._centre_sector = config.vfh_num_sectors // 2

        # Lookup-table form of _CLASS_COSTS for vectorised mapping.
        max_cid = max(_CLASS_COSTS.keys()) + 1
        self._cost_lut = np.zeros(max_cid, dtype=np.float32)
        for cid, cost in _CLASS_COSTS.items():
            self._cost_lut[cid] = cost

        # Last histogram retained for optional EMA smoothing across frames.
        self._last_hist: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────

    def should_activate(self, scene: SceneAnalysis) -> bool:
        """Decide whether the dispatcher should bother calling plan().

        Triggers, any of which is enough:
          1. Walkable area has collapsed below ``vfh_low_walkable_ratio``.
          2. A hazard class is in the centre zone above ``vfh_activation_ratio``.
          3. A hazard class has some of its mass in the centre band
             (zone_ratios["center"] > 0.15) — catches obstacles whose dominant
             pixel mass is to one side but still bleed into the user's path.
          4. A hazard class is large overall (pixel_ratio > 0.15) — a half-image
             blocker still warrants a plan even if its dominant zone is left/right.
          5. A vehicle is in the centre and closer than ``vfh_activation_distance_m``.
        """
        if not self._cfg.vfh_enabled:
            return False
        if scene.walkable_ratio < self._cfg.vfh_low_walkable_ratio:
            return True
        big_threshold = 0.15
        for zone in scene.zones:
            if zone.class_id not in _ACTIVATION_HAZARD_CLASSES:
                continue
            if zone.pixel_ratio < self._cfg.vfh_activation_ratio:
                continue
            centre_share = float(zone.zone_ratios.get("center", 0.0))
            if zone.dominant_zone == "center":
                return True
            if centre_share > big_threshold:
                return True
            if zone.pixel_ratio > big_threshold:
                return True
            if (
                zone.class_id == int(ClassID.VEHICLE)
                and zone.estimated_distance_m is not None
                and zone.estimated_distance_m < self._cfg.vfh_activation_distance_m
            ):
                return True
        return False

    def plan(
        self,
        mask: np.ndarray,
        scene: SceneAnalysis,
        target_action: Optional[str] = None,
    ) -> Optional[VFHGuidance]:
        """Run a single VFH pass. Returns ``None`` when the activation gate fails."""
        if not self.should_activate(scene):
            return None

        cost_grid = self._build_cost_grid(mask)
        hist = self._build_histogram(cost_grid)
        sector = self._select_sector(hist, target_action)
        action = self._sector_to_action(sector)
        text = self._action_to_text(action)
        centre_blocked = bool(hist[self._centre_sector] > self._cfg.vfh_blocked_threshold)

        return VFHGuidance(
            action=action,
            sector_index=sector if sector is not None else -1,
            blocked_center=centre_blocked,
            text=text,
            histogram=hist.tolist(),
        )

    # ── Public helpers exposed for the demo / visualiser ──────────

    def build_cost_grid(self, mask: np.ndarray) -> np.ndarray:
        """Public alias of the private cost-grid builder (used by vfh_demo)."""
        return self._build_cost_grid(mask)

    def build_histogram(self, cost_grid: np.ndarray) -> np.ndarray:
        return self._build_histogram(cost_grid)

    def select_sector(self, hist: np.ndarray, target_action: Optional[str] = None) -> Optional[int]:
        return self._select_sector(hist, target_action)

    # ── Internals ─────────────────────────────────────────────────

    def _compute_row_weights(self) -> np.ndarray:
        """One weight per cost-grid row (closer → larger)."""
        rows = self._cfg.vfh_grid_rows
        weights = np.zeros(rows, dtype=np.float32)
        for r in range(rows):
            # Centre pixel-y of this grid row in mask coordinates.
            centre_in_near = (r + 0.5) * (self._near_h / rows)
            pixel_y = int(self._near_start + centre_in_near)
            dist = pixel_to_ground_distance(pixel_y, self._image_h, self._geom)
            if dist is None or dist <= _DISTANCE_FLOOR_M:
                dist = _DISTANCE_FLOOR_M
            weights[r] = 1.0 / dist
        # Normalise so weights sum to 1 — keeps the histogram values bounded.
        total = float(weights.sum())
        if total > 0:
            weights /= total
        return weights

    def _compute_sector_for_col(self) -> np.ndarray:
        cols = self._cfg.vfh_grid_cols
        sectors = self._cfg.vfh_num_sectors
        idx = np.arange(cols, dtype=np.int32)
        return (idx * sectors // cols).astype(np.int32)

    def _build_cost_grid(self, mask: np.ndarray) -> np.ndarray:
        """(grid_rows × grid_cols) float32 cost grid by majority-class block pooling.

        Each grid cell takes the most-frequent class in its (row_block × col_block)
        of the near-field mask, then maps it through ``_cost_lut``.
        """
        rows = self._cfg.vfh_grid_rows
        cols = self._cfg.vfh_grid_cols
        h, w = mask.shape

        # If mask shape disagrees with config (e.g. demo with a raw PNG of
        # different size) operate on the actual shape.
        near_h = max(1, int(h * self._cfg.vfh_near_rows_ratio))
        near = mask[h - near_h:, :]
        # Resize the near-field mask down to (rows × cols) using INTER_NEAREST
        # so class IDs are preserved (no interpolation across class boundaries).
        # cv2 is already a project dependency.
        import cv2
        small = cv2.resize(near, (cols, rows), interpolation=cv2.INTER_NEAREST)

        # Clip class IDs to the LUT range defensively, then look up costs.
        clipped = np.clip(small, 0, self._cost_lut.shape[0] - 1).astype(np.int32)
        return self._cost_lut[clipped]

    def _build_histogram(self, cost_grid: np.ndarray) -> np.ndarray:
        """Per-sector aggregated cost, distance-weighted across rows."""
        sectors = self._cfg.vfh_num_sectors
        # Row-weight each cell first → shape (rows, cols).
        weighted = cost_grid * self._row_weights[:, None]
        hist = np.zeros(sectors, dtype=np.float32)
        counts = np.zeros(sectors, dtype=np.float32)
        for s in range(sectors):
            mask_cols = self._sector_for_col == s
            if not mask_cols.any():
                continue
            # Mean cell value within the sector keeps the histogram on a
            # consistent [0, 1] scale regardless of grid_cols.
            hist[s] = float(weighted[:, mask_cols].sum() / mask_cols.sum())
            counts[s] = float(mask_cols.sum())

        # EMA smoothing across frames to damp single-frame flicker. Skipped
        # on the first call.
        if self._last_hist is not None and self._last_hist.shape == hist.shape:
            hist = 0.6 * hist + 0.4 * self._last_hist
        self._last_hist = hist
        return hist

    def _select_sector(self, hist: np.ndarray, target_action: Optional[str]) -> Optional[int]:
        """Pick the open sector closest to the target direction.

        target_action mirrors RouteStep.action from the global planner:
          - "turn_left"  → preferred bias toward the left-most sectors
          - "turn_right" → toward the right-most sectors
          - anything else (None, "continue", "start", "finish") → centre
        """
        threshold = self._cfg.vfh_blocked_threshold
        open_mask = hist < threshold
        if not open_mask.any():
            return None

        if target_action == "turn_left":
            target = 0
        elif target_action == "turn_right":
            target = self._cfg.vfh_num_sectors - 1
        else:
            target = self._centre_sector

        # Among open sectors, choose the one nearest to ``target``. Break ties
        # by lower cost (prefer the safer option when two are equidistant).
        candidates = np.where(open_mask)[0]
        # Stable tie-breaker: (|distance|, cost).
        best = min(
            candidates,
            key=lambda s: (abs(int(s) - target), float(hist[s])),
        )
        return int(best)

    def _sector_to_action(self, sector: Optional[int]) -> VFHAction:
        if sector is None:
            return VFHAction.STOP
        centre = self._centre_sector
        delta = sector - centre  # negative = left, positive = right
        if delta == 0:
            return VFHAction.STRAIGHT
        if delta == -1:
            return VFHAction.LEFT_SLIGHT
        if delta == 1:
            return VFHAction.RIGHT_SLIGHT
        if delta < 0:
            return VFHAction.LEFT
        return VFHAction.RIGHT

    @staticmethod
    def _action_to_text(action: VFHAction) -> str:
        return {
            VFHAction.STRAIGHT:     "Düz devam edin",
            VFHAction.LEFT_SLIGHT:  "Hafif sola kayın",
            VFHAction.LEFT:         "Sola kayın, engelden kaçının",
            VFHAction.RIGHT_SLIGHT: "Hafif sağa kayın",
            VFHAction.RIGHT:        "Sağa kayın, engelden kaçının",
            VFHAction.STOP:         "Durun, geçiş yok",
        }[action]
