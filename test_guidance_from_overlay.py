"""
Test path guidance from a segmented overlay image.
Zero external dependencies (uses macOS sips + numpy only).

Usage:
    venv/bin/python test_guidance_from_overlay.py <image_path>
"""

import sys
import os
import time
import numpy as np
from PIL import Image

# ── Inline class definitions (mirrors perception.py) ────────────────────────

WALKABLE_SURFACE   = 0
CROSSWALK          = 1
VEHICLE_ROAD       = 2
COLLISION_OBSTACLE = 3
FALL_HAZARD        = 4
DYNAMIC_HAZARD     = 5
VEHICLE            = 6

CLASS_NAMES = {
    0: "walkable_surface",
    1: "crosswalk",
    2: "vehicle_road",
    3: "collision_obstacle",
    4: "fall_hazard",
    5: "dynamic_hazard",
    6: "vehicle",
}

CLASS_ALERT_CONFIG = {
    WALKABLE_SURFACE:   {"priority": 0, "alert": None,                                              "cooldown": 0},
    CROSSWALK:          {"priority": 1, "alert": "Yaya geçidi algılandı, geçiş güvenli",           "cooldown": 8.0},
    VEHICLE_ROAD:       {"priority": 4, "alert": "Dikkat, araç yolu, girmeyin",                     "cooldown": 5.0},
    COLLISION_OBSTACLE: {"priority": 3, "alert": "Önünüzde engel var, durun veya yön değiştirin",  "cooldown": 3.0},
    FALL_HAZARD:        {"priority": 3, "alert": "Zemin tehlikesi, yavaşlayın",                     "cooldown": 3.0},
    DYNAMIC_HAZARD:     {"priority": 4, "alert": "Hareketli tehlike yakınızda",                     "cooldown": 2.0},
    VEHICLE:            {"priority": 5, "alert": "Durun, araç algılandı",                           "cooldown": 1.5},
}

MIN_ALERT_RATIO      = 0.02
VERY_CLOSE_RATIO     = 0.15
NEARBY_RATIO         = 0.05
PATH_BOTTOM_FRACTION = 0.5
MIN_WALKABLE_FOR_GUIDANCE = 0.08


# ── Image loading (macOS sips → PPM → numpy) ────────────────────────────────

def load_image_as_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


# ── Colour → class mask ──────────────────────────────────────────────────────

def rgb_to_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Map overlay colours to class IDs based on the actual legend:
      walkable   = bright green
      crosswalk  = cyan
      road       = dark navy blue
      obstacle   = orange
      fall_hazard= dark orange / brownish-orange
      dynamic    = pink
      vehicle    = red
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    mx  = np.maximum(np.maximum(r, g), b)
    sat = (mx - np.minimum(np.minimum(r, g), b)) / (mx + 1e-6)
    colourful = sat > 0.15

    # Bright green: G dominant, R and B low
    is_green    = colourful & (g > r * 1.4) & (g > b * 1.4) & (g > 100)

    # Cyan: G and B both high, R low
    is_cyan     = colourful & (g > r * 1.3) & (b > r * 1.3) & (g > 80) & (b > 80)

    # Dark navy blue: B dominant, R and G both low
    is_navy     = colourful & (b > r * 1.5) & (b > g * 1.5)

    # Orange: R high, G medium, B low
    is_orange   = colourful & (r > 150) & (g > 60) & (g < r * 0.75) & (b < 60)

    # Dark orange / brown-orange (fall hazard): similar to orange but darker
    is_dk_orange = colourful & (r > 100) & (r < 200) & (g > 40) & (g < r * 0.65) & (b < 40)

    # Pink: R dominant, G and B similar and not too low (distinct from orange)
    is_pink     = colourful & (r > g * 1.2) & (r > b * 1.2) & (g > 80) & (b > 80) & (np.abs(g - b) < 40)

    # Red: R dominant, G and B both low
    is_red      = colourful & (r > g * 1.5) & (r > b * 1.5) & (r > 120) & (g < 80)

    # Apply in order (most specific first, least specific last)
    mask[is_navy]     = VEHICLE_ROAD
    mask[is_cyan]     = CROSSWALK
    mask[is_green]    = WALKABLE_SURFACE
    mask[is_orange]   = COLLISION_OBSTACLE
    mask[is_dk_orange]= FALL_HAZARD
    mask[is_red]      = VEHICLE
    mask[is_pink]     = DYNAMIC_HAZARD

    return mask


# ── Scene analysis ───────────────────────────────────────────────────────────

def analyse_and_alert(mask: np.ndarray):
    h, w    = mask.shape
    third   = w // 3
    total   = float(h * w)
    alerts  = []
    now     = time.time()
    max_pri = 0
    dominant_hazard = None

    print("── Piksel dağılımı ──────────────────────────────────")
    for cid in range(7):
        px = float(np.sum(mask == cid))
        if px == 0:
            continue
        ratio = px / total
        print(f"  {CLASS_NAMES[cid]:<22} {ratio:5.1%}")

        if cid != WALKABLE_SURFACE and ratio >= MIN_ALERT_RATIO:
            cfg = CLASS_ALERT_CONFIG[cid]
            pri = cfg["priority"]
            if pri > max_pri:
                max_pri = pri
                dominant_hazard = CLASS_NAMES[cid]

            # Direction
            left_px   = float(np.sum((mask == cid)[:, :third]))
            center_px = float(np.sum((mask == cid)[:, third:2*third]))
            right_px  = float(np.sum((mask == cid)[:, 2*third:]))
            zone = max({"left": left_px, "center": center_px, "right": right_px}, key=lambda k: {"left": left_px, "center": center_px, "right": right_px}[k])

            parts = [cfg["alert"]]
            if ratio > VERY_CLOSE_RATIO:
                parts.append("çok yakın")
            elif ratio > NEARBY_RATIO:
                parts.append("yakın")
            if zone == "left":
                parts.append("solunuzda")
            elif zone == "right":
                parts.append("sağınızda")
            alerts.append((pri, ", ".join(parts)))

    alerts.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in alerts], (max_pri < 3)


# ── Path guidance ─────────────────────────────────────────────────────────

def generate_path_guidance(mask: np.ndarray):
    h, w = mask.shape
    bottom_start = int(h * (1.0 - PATH_BOTTOM_FRACTION))
    bottom = mask[bottom_start:, :]

    walkable    = (bottom == WALKABLE_SURFACE)
    walkable_px = float(np.sum(walkable))
    total_px    = float(bottom.size)

    if walkable_px == 0:
        return None

    walkable_ratio = walkable_px / total_px
    if walkable_ratio < MIN_WALKABLE_FOR_GUIDANCE:
        return "Yürünebilir alan çok azalıyor, dikkatli ilerleyin"

    centroid_x = float(np.mean(np.where(walkable)[1])) / w

    cx_l, cx_r = int(w * 0.35), int(w * 0.65)
    center_ratio = float(np.sum(walkable[:, cx_l:cx_r])) / float(bottom[:, cx_l:cx_r].size)

    if centroid_x < 0.25:
        primary = "Sola dönün"
    elif centroid_x < 0.38:
        primary = "Hafif sola yönelin"
    elif centroid_x <= 0.62:
        primary = "Düz yürüyün"
    elif centroid_x < 0.75:
        primary = "Hafif sağa yönelin"
    else:
        primary = "Sağa dönün"

    if center_ratio >= 0.40 and primary != "Düz yürüyün":
        return f"{primary}, düz yürüyün"
    if walkable_ratio < 0.18:
        return f"{primary}, yol daralıyor"
    return primary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Kullanım: venv/bin/python test_guidance_from_overlay.py <foto.png>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"HATA: Dosya bulunamadı: {img_path}")
        sys.exit(1)

    print(f"[Yükleniyor] {img_path}")
    rgb  = load_image_as_rgb(img_path)
    print(f"[Boyut] {rgb.shape[1]}×{rgb.shape[0]}\n")

    mask = rgb_to_mask(rgb)

    alerts, is_safe = analyse_and_alert(mask)
    print()

    print("── Sistem Çıktısı ───────────────────────────────────")
    # Simüle et: navigasyon aktif mi? (argümanla geçilebilir, default False)
    nav_active = "--nav" in sys.argv

    # Yaya geçidini sadece navigasyon aktifken söyle
    filtered = [a for a in alerts if not ("geçidi" in a and not nav_active)]
    top_hazard = filtered[0] if filtered else None

    guidance = generate_path_guidance(mask)

    if guidance:
        if top_hazard:
            print(f"  🗣  {guidance} — {top_hazard}")
        else:
            print(f"  🗣  {guidance}")
    elif top_hazard:
        print(f"  🗣  {top_hazard}")
    else:
        print("  (yürünebilir alan algılanamadı)")

    if not nav_active and any("geçidi" in a for a in alerts):
        print("  ℹ  (yaya geçidi algılandı ama navigasyon aktif değil, sessiz geçildi)")
    print()


if __name__ == "__main__":
    main()
