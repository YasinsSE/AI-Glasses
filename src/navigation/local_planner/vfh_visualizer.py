"""Visualisation helpers for the VFH planner.

Renders one composite BGR image per frame, organised as three vertical bands:

    ┌──────────────────────────────────────────────────────┐
    │  Polar Histogram — sektör maliyetleri               │  hist band
    ├──────────────────────────────────────────────────────┤
    │                                                      │
    │  Görüntü + Segmentasyon + VFH Maliyet Izgarası      │  main panel
    │  (cost grid is overlaid on the photo with low alpha, │
    │   sector dividers + chosen-direction arrow on top)   │
    │                                                      │
    ├──────────────────────────────────────────────────────┤
    │  Karar: <action>  —  TTS: "<turkish text>"          │  decision band
    └──────────────────────────────────────────────────────┘

Turkish text is rendered through PIL (cv2's Hershey font cannot render "ı",
"ç", etc.). Falls back to ASCII transliteration if PIL is unavailable.
"""

from typing import Optional

import cv2
import numpy as np

from ai.perception import CLASS_COLORS_BGR, render_overlay
from navigation.local_planner.models import VFHAction, VFHGuidance


_HIST_BAND_H = 110
_DECISION_BAND_H = 90
_MAIN_PANEL_W = 960
_MAIN_PANEL_H = 540
_COST_OVERLAY_ALPHA = 0.45    # Cost-grid heatmap blended onto the main image.
_SEG_OVERLAY_ALPHA = 0.40     # Segmentation colours blended onto the photo.

_ACTION_LABEL_TR = {
    VFHAction.STRAIGHT:     "DÜZ DEVAM",
    VFHAction.LEFT_SLIGHT:  "HAFİF SOLA",
    VFHAction.LEFT:         "SOLA KAY",
    VFHAction.RIGHT_SLIGHT: "HAFİF SAĞA",
    VFHAction.RIGHT:        "SAĞA KAY",
    VFHAction.STOP:         "DUR",
}


# ── Turkish-safe text rendering via PIL ─────────────────────────────

_PIL_AVAILABLE = True
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    _PIL_AVAILABLE = False


_FONT_CACHE: dict = {}
_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # Linux / Jetson
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _load_font(size: int):
    if not _PIL_AVAILABLE:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    _FONT_CACHE[size] = font
    return font


def _transliterate_tr(text: str) -> str:
    """Strip Turkish-only accents to ASCII for the cv2 fallback path."""
    table = str.maketrans({
        "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
        "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U",
    })
    return text.translate(table)


def _put_text(
    image: np.ndarray,
    text: str,
    pos,
    size: int = 18,
    colour=(240, 240, 240),
    bold: bool = False,
) -> None:
    """Draw ``text`` at ``pos`` (top-left) onto ``image`` in BGR.

    Uses PIL when available so Turkish characters render correctly; otherwise
    falls back to cv2 with an ASCII-transliterated string.
    """
    if _PIL_AVAILABLE:
        # cv2 image is BGR — PIL needs RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = _load_font(size if not bold else size + 1)
        # PIL uses RGB tuples.
        pil_colour = (int(colour[2]), int(colour[1]), int(colour[0]))
        draw.text(pos, text, font=font, fill=pil_colour)
        image[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        fallback = _transliterate_tr(text)
        scale = max(0.4, size / 30.0)
        thickness = 2 if bold else 1
        # cv2.putText anchors at the text BASELINE — shift down by size so the
        # caller's `pos` behaves like a top-left anchor.
        cv2.putText(image, fallback, (int(pos[0]), int(pos[1]) + size),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


# ── Panel builders ──────────────────────────────────────────────────

def _build_main_panel(
    frame_bgr: Optional[np.ndarray],
    mask: np.ndarray,
    cost_grid: np.ndarray,
    guidance: Optional[VFHGuidance],
    near_rows_ratio: float,
    num_sectors: int,
) -> np.ndarray:
    """Photo + segmentation + cost grid layered together with sector cues."""
    # Base: original photo, or pure class-coloured mask if no photo.
    if frame_bgr is not None:
        base = render_overlay(frame_bgr, mask, alpha=_SEG_OVERLAY_ALPHA)
    else:
        base = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cid, colour in CLASS_COLORS_BGR.items():
            base[mask == int(cid)] = colour
    base = cv2.resize(base, (_MAIN_PANEL_W, _MAIN_PANEL_H), interpolation=cv2.INTER_LINEAR)

    # Cost-grid heatmap, upscaled to fill only the near-field bottom region of
    # the main panel — that is the region VFH actually analyses.
    near_h_px = max(1, int(_MAIN_PANEL_H * near_rows_ratio))
    near_y0 = _MAIN_PANEL_H - near_h_px

    norm = np.clip(cost_grid, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    heat_small = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat_small, (_MAIN_PANEL_W, near_h_px), interpolation=cv2.INTER_NEAREST)

    near_region = base[near_y0:, :, :].copy()
    blended = cv2.addWeighted(near_region, 1.0 - _COST_OVERLAY_ALPHA, heat, _COST_OVERLAY_ALPHA, 0)
    base[near_y0:, :, :] = blended

    # Horizontal line marking where the VFH near-field begins.
    cv2.line(base, (0, near_y0), (_MAIN_PANEL_W, near_y0), (255, 255, 255), 2)
    _put_text(base, "VFH yakın-alan sınırı", (8, near_y0 - 24), size=16, colour=(255, 255, 255))

    # Sector dividers — vertical lines splitting the panel into num_sectors
    # equal columns. Selected sector gets a highlight band.
    sector_w = _MAIN_PANEL_W / num_sectors
    for s in range(1, num_sectors):
        x = int(s * sector_w)
        cv2.line(base, (x, near_y0), (x, _MAIN_PANEL_H), (220, 220, 220), 1)

    if guidance is not None and guidance.sector_index >= 0:
        sx0 = int(guidance.sector_index * sector_w)
        sx1 = int((guidance.sector_index + 1) * sector_w)
        # Translucent highlight band over the chosen sector.
        highlight = base[near_y0:, sx0:sx1].copy()
        tint = np.zeros_like(highlight)
        tint[:] = (60, 220, 60) if guidance.action != VFHAction.STOP else (60, 60, 220)
        base[near_y0:, sx0:sx1] = cv2.addWeighted(highlight, 0.55, tint, 0.45, 0)
        # Bordered rectangle around the chosen sector for crispness.
        border_colour = (60, 220, 60) if guidance.action != VFHAction.STOP else (60, 60, 220)
        cv2.rectangle(base, (sx0, near_y0), (sx1, _MAIN_PANEL_H - 1), border_colour, 3)

        # Direction arrow centred on the chosen sector, pointing the way out.
        arrow_y = (near_y0 + _MAIN_PANEL_H) // 2
        arrow_cx = (sx0 + sx1) // 2
        _draw_action_arrow(base, guidance.action, arrow_cx, arrow_y)

    # Panel caption (top-left).
    _put_text(base, "1) Görüntü + Segmentasyon + VFH Maliyet Izgarası",
              (10, 10), size=20, colour=(255, 255, 255), bold=True)
    return base


def _draw_action_arrow(image: np.ndarray, action: VFHAction, cx: int, cy: int) -> None:
    """Big arrow inside the highlighted sector, hinting the steering direction."""
    colour = (255, 255, 255)
    thickness = 6
    length = 70
    if action == VFHAction.STRAIGHT:
        p1, p2 = (cx, cy + length // 2), (cx, cy - length // 2)
    elif action == VFHAction.LEFT_SLIGHT:
        p1, p2 = (cx + 25, cy + length // 2), (cx - 25, cy - length // 2)
    elif action == VFHAction.LEFT:
        p1, p2 = (cx + length // 2, cy), (cx - length // 2, cy)
    elif action == VFHAction.RIGHT_SLIGHT:
        p1, p2 = (cx - 25, cy + length // 2), (cx + 25, cy - length // 2)
    elif action == VFHAction.RIGHT:
        p1, p2 = (cx - length // 2, cy), (cx + length // 2, cy)
    elif action == VFHAction.STOP:
        # Big X for "stop".
        r = length // 2
        cv2.line(image, (cx - r, cy - r), (cx + r, cy + r), (50, 50, 230), thickness + 2)
        cv2.line(image, (cx - r, cy + r), (cx + r, cy - r), (50, 50, 230), thickness + 2)
        return
    else:
        return
    cv2.arrowedLine(image, p1, p2, colour, thickness, tipLength=0.35)


def _build_hist_band(
    histogram,
    selected_sector: int,
    blocked_threshold: float,
    total_w: int,
) -> np.ndarray:
    band = np.full((_HIST_BAND_H, total_w, 3), 18, dtype=np.uint8)  # near-black bg
    _put_text(band, "2) Polar Histogram — sektör maliyetleri (yeşil=boş, kırmızı=tıkalı)",
              (10, 6), size=18, colour=(245, 245, 245), bold=True)
    if not histogram:
        _put_text(band, "(histogram yok — VFH henüz aktive olmadı)",
                  (10, 50), size=16, colour=(180, 180, 180))
        return band

    n = len(histogram)
    bar_area_top = 38
    bar_area_bottom = _HIST_BAND_H - 24
    bar_w = total_w // n
    max_cost = max(max(histogram), blocked_threshold * 1.2, 0.01)

    # Threshold line drawn across the band.
    th_y = int(bar_area_bottom - (blocked_threshold / max_cost) * (bar_area_bottom - bar_area_top))
    cv2.line(band, (0, th_y), (total_w, th_y), (90, 90, 200), 1, cv2.LINE_AA)
    _put_text(band, f"eşik={blocked_threshold:.2f}", (total_w - 130, th_y - 18),
              size=12, colour=(160, 160, 220))

    for s, cost in enumerate(histogram):
        x0 = s * bar_w + 4
        x1 = x0 + bar_w - 8
        bar_h = int((cost / max_cost) * (bar_area_bottom - bar_area_top))
        y0 = bar_area_bottom - bar_h
        if s == selected_sector:
            colour = (60, 220, 220)   # cyan = selected
        elif cost >= blocked_threshold:
            colour = (60, 60, 220)    # red = blocked
        else:
            colour = (60, 200, 60)    # green = open
        cv2.rectangle(band, (x0, y0), (x1, bar_area_bottom), colour, -1)
        _put_text(band, f"S{s}", (x0 + 4, bar_area_bottom + 2),
                  size=12, colour=(210, 210, 210))
    return band


def _build_decision_band(
    guidance: Optional[VFHGuidance],
    activated: bool,
    total_w: int,
) -> np.ndarray:
    band = np.full((_DECISION_BAND_H, total_w, 3), 24, dtype=np.uint8)
    _put_text(band, "3) Karar:", (10, 8), size=18, colour=(220, 220, 220), bold=True)
    if guidance is None:
        title = "VFH BOŞTA"
        subtitle = "(aktivasyon eşiği aşılmadı)" if not activated else "(öneri üretilemedi)"
        _put_text(band, title, (110, 8), size=22, colour=(180, 180, 180), bold=True)
        _put_text(band, subtitle, (10, 50), size=16, colour=(150, 150, 150))
        return band
    label = _ACTION_LABEL_TR.get(guidance.action, guidance.action.value.upper())
    title_colour = (80, 230, 80) if guidance.action != VFHAction.STOP else (80, 80, 230)
    _put_text(band, label, (110, 6), size=24, colour=title_colour, bold=True)
    _put_text(band, f"Sektör {guidance.sector_index}/{len(guidance.histogram)}",
              (350, 12), size=16, colour=(200, 200, 200))
    _put_text(band, f'TTS: "{guidance.text}"', (10, 50),
              size=18, colour=(240, 240, 240))
    return band


# ── Public entry point ──────────────────────────────────────────────

def draw_overlay(
    frame_bgr: Optional[np.ndarray],
    mask: np.ndarray,
    cost_grid: np.ndarray,
    guidance: Optional[VFHGuidance],
    blocked_threshold: float,
    activated: bool,
    near_rows_ratio: float = 0.55,
) -> np.ndarray:
    """Compose one labelled BGR image for the demo / debug HUD."""
    num_sectors = len(guidance.histogram) if guidance is not None else max(1, cost_grid.shape[1])
    if guidance is None:
        # Match the planner's default when we have no histogram yet — only used
        # for drawing sector dividers, so num_sectors is the only thing needed.
        num_sectors = 7

    main = _build_main_panel(
        frame_bgr=frame_bgr,
        mask=mask,
        cost_grid=cost_grid,
        guidance=guidance,
        near_rows_ratio=near_rows_ratio,
        num_sectors=num_sectors,
    )
    selected = guidance.sector_index if guidance is not None else -1
    hist = _build_hist_band(
        guidance.histogram if guidance is not None else [],
        selected,
        blocked_threshold,
        _MAIN_PANEL_W,
    )
    decision = _build_decision_band(guidance, activated, _MAIN_PANEL_W)
    return np.vstack([hist, main, decision])
