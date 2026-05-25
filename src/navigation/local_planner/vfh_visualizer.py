"""Visualisation helpers for the VFH planner.

Pure cv2 + numpy — no matplotlib so the demo runs on the same dependency set
as the rest of the project. Used by vfh_demo.py and reusable as a debug HUD.

Layout of ``draw_overlay`` output (single composite BGR image):

    ┌──────────────────────────┐
    │   histogram bar chart    │  ← height = _HIST_BAND_H
    ├────────────┬─────────────┤
    │   frame +  │  cost grid  │
    │   mask     │  heatmap    │
    │   overlay  │             │
    ├────────────┴─────────────┤
    │ action + TTS text strip  │  ← height = _TEXT_BAND_H
    └──────────────────────────┘
"""

from typing import Optional

import cv2
import numpy as np

from ai.perception import CLASS_COLORS_BGR, render_overlay
from navigation.local_planner.models import VFHAction, VFHGuidance


_HIST_BAND_H = 100
_TEXT_BAND_H = 70
_PANEL_W = 480           # Each of the two main panels is resized to this width.
_PANEL_H = 360


def _resize_panel(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (_PANEL_W, _PANEL_H), interpolation=cv2.INTER_NEAREST)


def _cost_grid_heatmap(cost_grid: np.ndarray) -> np.ndarray:
    """Convert (rows × cols) float32 cost grid to a colour heatmap."""
    # Map cost in [0, 1] to a green→yellow→red gradient via OpenCV's JET LUT
    # inverted (we want low cost = green).
    norm = np.clip(cost_grid, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    # Upscale before colouring so the colour blocks are crisp.
    big = cv2.resize(u8, (_PANEL_W, _PANEL_H), interpolation=cv2.INTER_NEAREST)
    coloured = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    # Draw grid lines for readability.
    rows, cols = cost_grid.shape
    for r in range(1, rows):
        y = int(r * _PANEL_H / rows)
        cv2.line(coloured, (0, y), (_PANEL_W, y), (40, 40, 40), 1)
    for c in range(1, cols):
        x = int(c * _PANEL_W / cols)
        cv2.line(coloured, (x, 0), (x, _PANEL_H), (40, 40, 40), 1)
    cv2.putText(coloured, "VFH cost grid", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return coloured


def _histogram_band(
    histogram,
    selected_sector: int,
    blocked_threshold: float,
    total_w: int,
) -> np.ndarray:
    band = np.zeros((_HIST_BAND_H, total_w, 3), dtype=np.uint8)
    if not histogram:
        return band
    n = len(histogram)
    bar_w = total_w // n
    max_cost = max(max(histogram), blocked_threshold * 1.2, 0.01)
    baseline_y = _HIST_BAND_H - 18
    # Threshold line.
    th_y = int(baseline_y - (blocked_threshold / max_cost) * (baseline_y - 12))
    cv2.line(band, (0, th_y), (total_w, th_y), (80, 80, 200), 1)
    for s, cost in enumerate(histogram):
        x0 = s * bar_w + 4
        x1 = x0 + bar_w - 8
        bar_h = int((cost / max_cost) * (baseline_y - 12))
        y0 = baseline_y - bar_h
        # Green when open, red when blocked, yellow when selected.
        if s == selected_sector:
            colour = (0, 220, 220)
        elif cost >= blocked_threshold:
            colour = (60, 60, 220)
        else:
            colour = (60, 200, 60)
        cv2.rectangle(band, (x0, y0), (x1, baseline_y), colour, -1)
        cv2.putText(band, f"{s}", (x0 + 4, baseline_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(band, "VFH histogram (sector cost)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return band


def _text_band(guidance: Optional[VFHGuidance], activated: bool, total_w: int) -> np.ndarray:
    band = np.zeros((_TEXT_BAND_H, total_w, 3), dtype=np.uint8)
    if guidance is None:
        line1 = "VFH idle" if not activated else "VFH: no guidance"
        line2 = "(activation gate not triggered)" if not activated else ""
        cv2.putText(band, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(band, line2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (120, 120, 120), 1, cv2.LINE_AA)
        return band
    colour = (60, 200, 60) if guidance.action != VFHAction.STOP else (60, 60, 220)
    cv2.putText(band, f"action={guidance.action.value}  sector={guidance.sector_index}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    cv2.putText(band, f'TTS: "{guidance.text}"', (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return band


def draw_overlay(
    frame_bgr: Optional[np.ndarray],
    mask: np.ndarray,
    cost_grid: np.ndarray,
    guidance: Optional[VFHGuidance],
    blocked_threshold: float,
    activated: bool,
) -> np.ndarray:
    """Compose a single BGR image visualising one VFH frame.

    ``frame_bgr`` may be None (e.g. ``--no-model --mask-image`` mode); in that
    case we render the mask alone using class colours.
    """
    if frame_bgr is not None:
        left = render_overlay(frame_bgr, mask, alpha=0.45)
    else:
        left = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cid, color in CLASS_COLORS_BGR.items():
            left[mask == int(cid)] = color
    left = _resize_panel(left)
    cv2.putText(left, "Segmentation + frame", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

    right = _cost_grid_heatmap(cost_grid)
    middle = np.hstack([left, right])

    total_w = middle.shape[1]
    selected = guidance.sector_index if guidance is not None else -1
    hist_band = _histogram_band(
        guidance.histogram if guidance is not None else [],
        selected,
        blocked_threshold,
        total_w,
    )
    text_band = _text_band(guidance, activated, total_w)

    return np.vstack([hist_band, middle, text_band])
