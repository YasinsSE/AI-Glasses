"""
SANPO Dataset -> YOLO Segmentation Format Converter  (v5)
============================================================
ALAS Project - AI Glasses for Visually Impaired

v5 CHANGES
──────────
- Expanded dataset volume with targeted sessions to mitigate class imbalance.
- Dynamic camera detection ('camera_chest' vs 'camera_head') fully integrated.
- Removed 'building' class to strictly focus on active navigation hazards.
- Frame skipping via FRAME_STEP to reduce near-duplicate frames across sessions.
  More sessions + fewer frames per session = better diversity.
- Debug overlay: draws 1 sample per session with polygon masks + class labels
  on the real image, saved to <output_root>/_debug_overlays/.
  Lets you visually verify YOLO conversion correctness.
- Fixed MACRO_CLASS_MAP keys to match exact SANPO labelmap strings.
- Updated directory paths to target 'camera_chest' instead of 'camera_head'.
- Building class removed to prevent class imbalance and false positives.

SANPO LABELMAP (v3)
─────────────────────
  {
    "unlabeled": 0, "road": 1, "curb": 2, "sidewalk": 3,
    "guard rail/road barrier": 4, "crosswalk": 5, "paved trail": 6,
    "building": 7, "wall/fence": 8, "hand rail": 9,
    "opening-door": 10, "opening-gate": 11, "pedestrian": 12,
    "rider": 13, "animal": 14, "stairs": 15, "water body": 16,
    "other walkable surface": 17, "inaccessible surface": 18,
    "railway track": 19, "obstacle": 20, "vehicle": 21,
    "traffic sign": 22, "traffic light": 23, "pole": 24,
    "bus stop": 25, "bike rack": 26, "sky": 27, "tree": 28,
    "vegetation": 29, "terrain": 30
  }

MACRO-CLASS DESIGN RATIONALE
─────────────────────────────
The original SANPO dataset has 31 classes (0-30). For the ALAS system,
the core question is not "what exactly is this object?" but:
  (a) "Where can I walk safely?"
  (b) "What type of danger is ahead, and how should I react?"

This leads to 7 semantically distinct groups:

  Group 0 — Walkable Surface   : safe pedestrian zones (no alert)
  Group 1 — Crosswalk          : kept separate — critical for safe road crossing
  Group 2 — Vehicle Road       : must be avoided (road + railway)
  Group 3 — Collision Obstacle : head/body-level impact risk (poles, walls, trees)
  Group 4 — Fall Hazard        : ground-level drop/trip risk (stairs, curbs, water)
  Group 5 — Dynamic Hazard     : moving entities requiring immediate reaction
  Group 6 — Vehicle            : highest-priority moving threat

KEY DESIGN DECISIONS
─────────────────────
1. crosswalk is its own class (Group 1).
   The system must tell the user "you are at a crosswalk" —
   merging it with sidewalk blinds the navigation layer to safe crossings.

2. stairs -> fall_hazard (Group 4), not structure_context.
   A staircase is a fall risk, not an orientation reference.

3. opening-door / opening-gate -> dynamic_hazard (Group 5).
   These are sudden, time-sensitive threats — semantically identical to moving pedestrians.

4. collision_obstacle (Group 3) and fall_hazard (Group 4) are split.
   "Obstacle ahead" vs. "Ground hazard ahead" require different user reactions:
     - Collision: stop or change direction horizontally
     - Fall hazard: slow down, check footing, do NOT step forward

5. building is skipped (-1).
   Building masks are large and distant — generating collision alerts for far-away
   walls creates noise. The navigation layer (OSM) already handles buildings.

6. traffic_sign is skipped (-1).
   The sign panel itself is not a collision risk; the pole beneath it is already
   covered by 'pole' in Group 3.

7. vegetation -> collision_obstacle. Merged with trees to prevent model confusion.
   hand rail -> fall_hazard. Presence/absence near stairs is meaningful.
   terrain -> fall_hazard. Unpaved ground = trip risk for visually impaired users.

8. traffic_light and bus_stop are skipped (-1).
   These are navigation/POI references better handled by the OSM/routing layer.

9. inaccessible surface -> vehicle_road (Group 2).
   Areas explicitly marked inaccessible should trigger the same "stay away" alert.

10. guard rail/road barrier -> collision_obstacle (Group 3).
    Physical barrier at body height — same collision risk as a fence.

11. bike rack -> collision_obstacle (Group 3).
    Metal structure at body/leg height — collision risk when walking.

CLASSES SKIPPED (mapped to -1, not included in training)
─────────────────────────────────────────────────────────
  unlabeled      (pixel 0)  — no annotation
  sky            (pixel 27) — no semantic value for navigation
  building       (pixel 7)  — distant, large mask; OSM handles it
  traffic sign   (pixel 22) — panel only; pole already in Group 3
  traffic light  (pixel 23) — POI, handled by navigation layer
  bus stop       (pixel 25) — POI, handled by navigation layer

SESSION-BASED TRAIN/VAL SPLIT
───────────────────────────────
Consecutive frames in a SANPO session are ~99% visually identical (video source).
Splitting by frame causes data leakage (val frames seen during training).
Solution: entire sessions are assigned to either train or val — never split across.

FRAME SKIPPING (v4)
─────────────────────
Even after session-based splitting, consecutive frames within a session carry
almost identical visual information. FRAME_STEP=2 takes every 2nd frame,
halving per-session frame count while preserving temporal coverage.
This lets us include MORE sessions without inflating dataset size with redundant images.

CONTOUR EXTRACTION FIX
────────────────────────
Binary masks use value 255 (not 1) before cv2.findContours.
cv2.findContours expects a proper binary image (0 / 255). Using 0/1 values
causes OpenCV to silently skip contours on certain builds.

Usage:
    python src/ai/dataset_format.py
    python src/ai/dataset_format.py --frame_step 3
    python src/ai/dataset_format.py --max_sessions 5  # quick debug run
"""

import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

SANPO_ROOT   = Path("/Users/yasinyldrm/Downloads/SANPO_DATASET/sanpo_dataset_v5")
OUTPUT_ROOT  = Path("/Users/yasinyldrm/Downloads/SANPO_DATASET/ALAS_YOLO_Dataset_v5")
LABELMAP_PATH = Path("/Users/yasinyldrm/Library/Mobile Documents/com~apple~CloudDocs/Graduation-Project/AI-Glasses/labelmap.json")


# ──────────────────────────────────────────────────────────────────────────────
# MACRO-CLASS MAPPING  (SANPO class name -> YOLO class id)
#
# IMPORTANT: Keys must match labelmap.json strings EXACTLY (case, spaces, slashes).
# Classes not listed here are implicitly skipped (-1).
# ──────────────────────────────────────────────────────────────────────────────

MACRO_CLASS_MAP: dict[str, int] = {

    # ── Group 0: Walkable Surface ─────────────────────────────────────────────
    "sidewalk"               : 0,   # pixel 3
    "paved trail"            : 0,   # pixel 6
    "other walkable surface" : 0,   # pixel 17

    # ── Group 1: Crosswalk ────────────────────────────────────────────────────
    "crosswalk"              : 1,   # pixel 5

    # ── Group 2: Vehicle Road ─────────────────────────────────────────────────
    "road"                   : 2,   # pixel 1
    "railway track"          : 2,   # pixel 19
    "inaccessible surface"   : 2,   # pixel 18

    # ── Group 3: Collision Obstacle ───────────────────────────────────────────
    "pole"                   : 3,   # pixel 24
    "wall/fence"             : 3,   # pixel 8
    "guard rail/road barrier": 3,   # pixel 4
    "tree"                   : 3,   # pixel 28
    "vegetation"             : 3,   # pixel 29
    "obstacle"               : 3,   # pixel 20 — generic static obstacle
    "bus stop"               : 3,   # pixel 25
    "bike rack"              : 3,   # pixel 26

    # ── Group 4: Fall Hazard ──────────────────────────────────────────────────
    "stairs"                 : 4,   # pixel 15
    "curb"                   : 4,   # pixel 2
    "water body"             : 4,   # pixel 16
    "hand rail"              : 4,   # pixel 9
    "terrain"                : 4,   # pixel 30

    # ── Group 5: Dynamic Hazard ───────────────────────────────────────────────
    "pedestrian"             : 5,   # pixel 12
    "rider"                  : 5,   # pixel 13
    "animal"                 : 5,   # pixel 14
    "opening-door"           : 5,   # pixel 10
    "opening-gate"           : 5,   # pixel 11

    # ── Group 6: Vehicle ──────────────────────────────────────────────────────
    "vehicle"                : 6,   # pixel 21

    # SKIPPED (not in map -> pixel_to_macro returns -1):
    #   "unlabeled"    (pixel 0)  — no annotation
    #    "building"               — pixel 7 (Skipped to prevent class imbalance)
    #   "traffic sign" (pixel 22) — panel only; pole in Group 3
    #   "traffic light"(pixel 23) — POI; navigation layer
    #   "sky"          (pixel 27) — no semantic value
}

CLASS_NAMES: dict[int, str] = {
    0: "walkable_surface",
    1: "crosswalk",
    2: "vehicle_road",
    3: "collision_obstacle",
    4: "fall_hazard",
    5: "dynamic_hazard",
    6: "vehicle",
}

NUM_CLASSES = len(CLASS_NAMES)   # 7

# Distinct colors per class for debug overlay (BGR format for OpenCV)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),       # walkable_surface   — green
    1: (255, 255, 0),     # crosswalk          — cyan
    2: (0, 0, 200),       # vehicle_road       — red
    3: (200, 100, 0),     # collision_obstacle — dark blue
    4: (0, 165, 255),     # fall_hazard        — orange
    5: (255, 0, 255),     # dynamic_hazard     — magenta
    6: (0, 0, 255),       # vehicle            — bright red
}


# ──────────────────────────────────────────────────────────────────────────────
# TUNING PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

MAX_POLYGON_POINTS = 80    # points per contour (lower = faster Jetson inference)
MIN_CONTOUR_AREA   = 200   # px² — filters annotation noise
MIN_MASK_PIXELS    = 100   # total foreground pixels before processing a class
TRAIN_RATIO        = 0.85  # session-level split
FRAME_STEP         = 3     # take every Nth frame (1 = all, 2 = skip 1, 3 = skip 2)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_labelmap(labelmap_path: Path) -> dict[str, int]:
    """
    Load labelmap.json -> {class_name: pixel_value}.
    Handles both dict {"name": id} and list [{"name":..., "id":...}] formats.
    """
    if not labelmap_path.exists():
        raise FileNotFoundError(f"labelmap.json not found: {labelmap_path}")

    with open(labelmap_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {item["name"]: item["id"] for item in data}
    raise ValueError("Unrecognized labelmap.json format.")


def build_pixel_to_macro(labelmap: dict[str, int]) -> dict[int, int]:
    """
    Build pixel_value -> macro_class_id lookup from the loaded labelmap.
    Returns -1 for classes that are intentionally skipped.
    Prints a full mapping report so the user can verify correctness.
    """
    pixel_to_macro: dict[int, int] = {v: -1 for v in labelmap.values()}

    mapped_lines, skipped_lines = [], []

    for name, pixel in labelmap.items():
        if name in MACRO_CLASS_MAP:
            macro = MACRO_CLASS_MAP[name]
            pixel_to_macro[pixel] = macro
            mapped_lines.append(
                f"  pixel {pixel:2d}  {name:<30} -> group {macro} [{CLASS_NAMES[macro]}]"
            )
        else:
            skipped_lines.append(f"  pixel {pixel:2d}  {name}")

    # Detect keys in MACRO_CLASS_MAP that don't exist in labelmap (typo catcher)
    labelmap_names = set(labelmap.keys())
    for key in MACRO_CLASS_MAP:
        if key not in labelmap_names:
            print(f"  [ERROR] MACRO_CLASS_MAP key '{key}' not found in labelmap!")

    print(f"\n[build_pixel_to_macro]")
    print(f"  Mapped  : {len(mapped_lines)} / {len(labelmap)} classes")
    print(f"  Skipped : {len(skipped_lines)}")
    print("\nMapped classes:")
    for line in sorted(mapped_lines):
        print(line)
    if skipped_lines:
        print("\nSkipped classes (excluded from training):")
        for line in sorted(skipped_lines):
            print(line)

    return pixel_to_macro


def mask_to_yolo_polygons(mask_path: Path,
                           pixel_to_macro: dict[int, int],
                           img_w: int,
                           img_h: int) -> list[str]:
    """
    Convert a SANPO segmentation mask PNG to YOLO polygon annotation lines.

    SANPO format : each pixel value = SANPO class ID (0-30)
    YOLO format  : "<class_id> x1 y1 x2 y2 ... xn yn" (values normalized 0-1)
    """
    mask = np.array(Image.open(mask_path))

    if mask.ndim >= 3:
        mask = mask[:, :, 0]

    lines: list[str] = []

    for pixel_val in np.unique(mask):
        macro_id = pixel_to_macro.get(int(pixel_val), -1)
        if macro_id == -1:
            continue

        # --- Build binary mask (0 / 255) ---
        binary = np.where(mask == pixel_val, 255, 0).astype(np.uint8)

        if int(binary.sum()) < MIN_MASK_PIXELS * 255:
            continue

        # --- Extract contours ---
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            # Simplify polygon
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx  = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Downsample to MAX_POLYGON_POINTS
            if len(approx) > MAX_POLYGON_POINTS:
                idx    = np.linspace(0, len(approx) - 1, MAX_POLYGON_POINTS, dtype=int)
                approx = approx[idx]

            # Normalize and clamp coordinates
            pts = approx.reshape(-1, 2)
            normalized: list[float] = []
            for x, y in pts:
                normalized.extend([
                    round(float(np.clip(x, 0, img_w - 1)) / img_w, 6),
                    round(float(np.clip(y, 0, img_h - 1)) / img_h, 6),
                ])

            if len(normalized) >= 6:   # minimum 3 (x, y) pairs
                lines.append(f"{macro_id} " + " ".join(map(str, normalized)))

    return lines


def get_sessions(sanpo_root: Path) -> list[Path]:
    """
    Collect valid session directories.
    Uses all_candidates.txt if present; otherwise, auto-scans for either 
    'camera_head' or 'camera_chest' directories.
    """
    candidates = sanpo_root / "all_candidates.txt"

    if candidates.exists():
        with open(candidates) as f:
            ids = [line.strip() for line in f if line.strip()]
        sessions = [sanpo_root / s for s in ids if (sanpo_root / s).exists()]
        print(f"[get_sessions] {len(sessions)} sessions loaded from all_candidates.txt")
    else:
        # CHECK: Look for either chest or head directory
        sessions = [
            d for d in sorted(sanpo_root.iterdir())
            if d.is_dir() and ((d / "camera_chest").exists() or (d / "camera_head").exists())
        ]
        print(f"[get_sessions] {len(sessions)} sessions found by directory scan")

    return sessions


def process_session(session_path: Path,
                    pixel_to_macro: dict[int, int],
                    frame_step: int = 1) -> list[tuple[Path, list[str]]]:
    """
    Process annotated frames in one session with temporal subsampling.
    Dynamically targets 'camera_chest' or 'camera_head' based on directory availability.

    frame_step=2 means take every 2nd frame (indices 0, 2, 4, ...),
    effectively halving the per-session count while keeping temporal coverage.
    """
    # DYNAMIC PATH: Determine which camera directory exists for this specific session
    if (session_path / "camera_chest").exists():
        camera_dir = "camera_chest"
    elif (session_path / "camera_head").exists():
        camera_dir = "camera_head"
    else:
        return []

    frames_dir = session_path / camera_dir / "left" / "video_frames"
    masks_dir  = session_path / camera_dir / "left" / "segmentation_masks"

    if not frames_dir.exists() or not masks_dir.exists():
        return []

    all_frames = sorted(frames_dir.glob("*.png"))

    # Temporal subsampling — take every Nth frame
    sampled_frames = all_frames[::frame_step]

    results = []
    for frame_path in sampled_frames:
        mask_path = masks_dir / frame_path.name
        if not mask_path.exists():
            continue

        try:
            w, h = Image.open(frame_path).size
        except Exception as e:
            print(f"[WARN] Cannot open image {frame_path.name}: {e}")
            continue

        try:
            anno = mask_to_yolo_polygons(mask_path, pixel_to_macro, w, h)
        except Exception as e:
            print(f"[WARN] Mask error {mask_path.name}: {e}")
            continue

        if anno:
            results.append((frame_path, anno))

    return results


def write_split(samples: list[tuple[Path, list[str]]],
                split: str,
                output_root: Path,
                session_id: str) -> None:
    """
    Copy images and write YOLO .txt labels for one session into a split folder.
    Filename: <session_id>_<frame_stem>.png / .txt  (collision-safe across sessions)
    """
    img_dir = output_root / "images" / split
    lbl_dir = output_root / "labels" / split

    for frame_path, anno_lines in samples:
        stem = f"{session_id}_{frame_path.stem}"
        shutil.copy2(frame_path, img_dir / f"{stem}.png")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(anno_lines))


# ──────────────────────────────────────────────────────────────────────────────
# DEBUG OVERLAY — visual verification of YOLO conversion
# ──────────────────────────────────────────────────────────────────────────────

def draw_debug_overlay(image_path: Path,
                       anno_lines: list[str],
                       output_path: Path) -> None:
    """
    Draw YOLO polygon annotations back onto the original image for visual QA.

    Reads normalized YOLO lines, denormalizes to pixel coords, and draws
    semi-transparent filled polygons + class labels on the image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image for overlay: {image_path}")
        return

    h, w = img.shape[:2]
    overlay = img.copy()

    for line in anno_lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Denormalize polygon points
        pts = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i + 1] * h)
            pts.append([px, py])

        pts_np = np.array(pts, dtype=np.int32)
        color = CLASS_COLORS.get(class_id, (128, 128, 128))

        # Filled polygon on overlay layer
        cv2.fillPoly(overlay, [pts_np], color)

        # Draw polygon outline on original for clarity
        cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=2)

        # Class label at polygon centroid
        cx = int(np.mean(pts_np[:, 0]))
        cy = int(np.mean(pts_np[:, 1]))
        label = f"{class_id}:{CLASS_NAMES.get(class_id, '?')}"
        cv2.putText(img, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Blend: 60% original + 40% filled overlay
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Add legend in top-left corner
    legend_y = 20
    for cid, cname in CLASS_NAMES.items():
        c = CLASS_COLORS[cid]
        cv2.rectangle(blended, (10, legend_y - 12), (28, legend_y + 4), c, -1)
        cv2.putText(blended, f"{cid}: {cname}", (34, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blended, f"{cid}: {cname}", (34, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        legend_y += 22

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blended)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def convert(sanpo_root: Path,
            output_root: Path,
            labelmap_path: Path,
            max_sessions: int | None = None,
            frame_step: int = FRAME_STEP) -> None:
    """
    Full conversion pipeline with session-based train/val split.

    Why session-based split?
    ───────────────────────
    Frames 000000.png and 000001.png from the same session are ~99% visually
    identical (source: continuous video recording). A frame-based random split
    would place these near-duplicates in train and val respectively, causing
    severe data leakage — the model effectively "memorizes" val during training,
    inflating validation metrics without reflecting real-world generalization.

    Solution: shuffle entire sessions and assign each session wholly to either
    train or val. No frame from the same scene can appear in both splits.
    """
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    debug_dir = output_root / "_debug_overlays"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load labelmap
    print("\n[Step 1/4] Loading labelmap...")
    labelmap = load_labelmap(labelmap_path)
    print(f"           SANPO total classes: {len(labelmap)}")
    print(f"           Labelmap path      : {labelmap_path}")

    # Step 2 — Build pixel -> macro mapping
    print("\n[Step 2/4] Building pixel-to-macro mapping...")
    pixel_to_macro = build_pixel_to_macro(labelmap)

    # Step 3 — Session-level split
    print("\n[Step 3/4] Splitting sessions (session-based, no data leakage)...")
    sessions = get_sessions(sanpo_root)
    if not sessions:
        print("[ERROR] No valid sessions found. Check directory structure.")
        return

    if max_sessions:
        sessions = sessions[:max_sessions]
        print(f"         Debug mode: using {max_sessions} sessions only")

    random.shuffle(sessions)
    cut            = int(len(sessions) * TRAIN_RATIO)
    train_sessions = sessions[:cut]
    val_sessions   = sessions[cut:]

    print(f"         Total   : {len(sessions)} sessions")
    print(f"         Train   : {len(train_sessions)} sessions")
    print(f"         Val     : {len(val_sessions)} sessions")
    print(f"         Frame step : {frame_step} (taking every {frame_step}. frame)")

    # Step 4 — Process and write
    print("\n[Step 4/4] Processing frames...")
    stats = {"train": 0, "val": 0, "empty": 0, "overlays": 0}

    for split, session_list in (("train", train_sessions), ("val", val_sessions)):
        for session_path in tqdm(session_list, desc=f"  {split:5s}"):
            samples = process_session(session_path, pixel_to_macro, frame_step)
            if not samples:
                stats["empty"] += 1
                continue
            write_split(samples, split, output_root, session_path.name)
            stats[split] += len(samples)

            # Pick the middle sample for the most representative overlay
            mid_idx = len(samples) // 2
            overlay_sample = samples[mid_idx]
            overlay_path = debug_dir / f"{split}_{session_path.name}.png"
            draw_debug_overlay(overlay_sample[0], overlay_sample[1], overlay_path)
            stats["overlays"] += 1

    # Write dataset.yaml
    yaml_path = output_root / "dataset.yaml"
    yaml_text = "\n".join([
        "# ALAS Project - SANPO Dataset (YOLO Segmentation Format)",
        f"# Auto-generated by ALAS dataset_format.py (v5, frame_step={frame_step})",
        "",
        f"path: {output_root.absolute()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {NUM_CLASSES}",
        "",
        "names:",
        *[f"  {i}: {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)],
    ])
    yaml_path.write_text(yaml_text)

    # Final report
    total_frames = stats["train"] + stats["val"]
    print("\n" + "=" * 62)
    print("  CONVERSION COMPLETE")
    print("=" * 62)
    print(f"  Output directory    : {output_root}")
    print(f"  Frame step          : {frame_step}")
    print(f"  Train frames        : {stats['train']}")
    print(f"  Val frames          : {stats['val']}")
    print(f"  Empty sessions      : {stats['empty']}")
    print(f"  Total frames        : {total_frames}")
    print(f"  Debug overlays      : {stats['overlays']}  ({debug_dir})")
    print(f"  dataset.yaml        : {yaml_path}")
    print(f"\n  Macro classes ({NUM_CLASSES}):")
    for i, name in CLASS_NAMES.items():
        print(f"    [{i}] {name}")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SANPO dataset to YOLO segmentation format (session-based split)."
    )
    parser.add_argument("--sanpo_root",    type=str, default=str(SANPO_ROOT),
                        help=f"Root directory of SANPO dataset (default: {SANPO_ROOT})")
    parser.add_argument("--output_root",   type=str, default=str(OUTPUT_ROOT),
                        help=f"Output directory for YOLO-format dataset (default: {OUTPUT_ROOT})")
    parser.add_argument("--labelmap_path", type=str, default=str(LABELMAP_PATH),
                        help=f"Path to labelmap.json (default: {LABELMAP_PATH})")
    parser.add_argument("--max_sessions",  type=int, default=None,
                        help="Limit session count for debug runs (default: all)")
    parser.add_argument("--frame_step",    type=int, default=FRAME_STEP,
                        help=f"Take every Nth frame per session (default: {FRAME_STEP})")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Random seed for reproducible session split (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)
    convert(
        sanpo_root=Path(args.sanpo_root),
        output_root=Path(args.output_root),
        labelmap_path=Path(args.labelmap_path),
        max_sessions=args.max_sessions,
        frame_step=args.frame_step,
    )