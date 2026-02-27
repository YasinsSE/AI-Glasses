"""
SANPO Dataset -> YOLO Segmentation Format Converter  (v3 - Final)
=================================================================
ALAS Project - AI Glasses for Visually Impaired

MACRO-CLASS DESIGN RATIONALE
─────────────────────────────
The original SANPO dataset has 29 fine-grained classes. For the ALAS system,
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

KEY DESIGN DECISIONS vs. v2
─────────────────────────────
1. crosswalk is its own class (was merged into walkable_surface in v2).
   The system must be able to tell the user "you are at a crosswalk" —
   merging it with sidewalk would blind the navigation layer to safe crossings.

2. stairs moved from structure_context (v2) to fall_hazard (v3).
   A staircase is a fall risk, not an orientation reference.

3. opening-door / opening-gate moved from static_obstacle to dynamic_hazard.
   These are sudden, time-sensitive threats — semantically identical to a moving person.

4. collision_obstacle and fall_hazard are split (were both static_obstacle in v2).
   "Obstacle ahead" vs. "Ground hazard ahead" require different user reactions:
     - Collision: stop or change direction horizontally
     - Fall hazard: slow down, check footing, do NOT step forward

5. building is skipped (-1).
   Building masks are large and distant — generating collision alerts for far-away
   walls creates noise. The navigation layer (OSM) already handles buildings.

6. traffic_sign is skipped (-1).
   The sign panel itself is not a collision risk; the pole beneath it is already
   covered by 'pole' in Group 3.

7. vegetation and hand_rail are retained.
   - vegetation -> fall_hazard (signals end of paved surface / terrain change)
   - hand_rail  -> fall_hazard (presence/absence of handrail near stairs is meaningful)

8. traffic_light and bus_stop are skipped (-1).
   These are navigation/POI references better handled by the OSM/routing layer,
   not the perception layer.

CLASSES SKIPPED (mapped to -1, not included in training)
─────────────────────────────────────────────────────────
  unlabeled      (pixel 0)  — no annotation
  sky            (pixel 27) — no semantic value for navigation
  building                  — distant, large mask; OSM handles it
  traffic_sign              — panel only; pole already in Group 3
  traffic_light             — POI, handled by navigation layer
  bus_stop                  — POI, handled by navigation layer

SESSION-BASED TRAIN/VAL SPLIT
───────────────────────────────
Consecutive frames in a SANPO session are ~99% visually identical (video source).
Splitting by frame causes data leakage (val frames seen during training).
Solution: entire sessions are assigned to either train or val — never split across.

CONTOUR EXTRACTION FIX
────────────────────────
Binary masks use value 255 (not 1) before cv2.findContours.
cv2.findContours expects a proper binary image (0 / 255). Using 0/1 values
causes OpenCV to silently skip contours on certain builds.

Usage:
    python src/ai/dataset_format.py \
  --sanpo_root /Users/yasinyldrm/Downloads/SANPO_DATASET/sanpo_dataset \
  --output_root /Users/yasinyldrm/Downloads/SANPO_DATASET/ALAS_YOLO_Dataset

Kaggle / Colab:
    !python dataset_format.py \\
        --sanpo_root /kaggle/input/sanpo \\
        --output_root /kaggle/working/yolo_dataset
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
# MACRO-CLASS MAPPING  (SANPO class name -> YOLO class id)
# Classes not listed here are implicitly skipped.
# ──────────────────────────────────────────────────────────────────────────────

MACRO_CLASS_MAP: dict[str, int] = {

    # ── Group 0: Walkable Surface ─────────────────────────────────────────────
    # Safe zones where the user can walk without risk.
    # TTS: no alert (green light).
    "sidewalk"               : 0,
    "paved trail"            : 0,
    "other walkable surface" : 0,

    # ── Group 1: Crosswalk ────────────────────────────────────────────────────
    # Kept separate from walkable_surface — signals a safe road-crossing point.
    # TTS: "You are at a crosswalk."
    "crosswalk"              : 1,

    # ── Group 2: Vehicle Road ─────────────────────────────────────────────────
    # Zones the user must not enter. Includes railway tracks (equally lethal).
    # TTS: "Warning — vehicle road ahead."
    "road"                   : 2,
    "railway track"          : 2,

    # ── Group 3: Collision Obstacle ───────────────────────────────────────────
    # Head- or body-level hard objects. Correct reaction: stop / step sideways.
    # TTS: "Obstacle ahead — stop or turn."
    "pole"                   : 3,
    "obstacle"               : 3,
    "bike rack"              : 3,
    "guard rail/road barrier": 3,
    "tree"                   : 3,
    "wall/fence"             : 3,
    # NOTE: building -> skipped (see design notes above)
    # NOTE: traffic_sign -> skipped (pole already covers the physical hazard)

    # ── Group 4: Fall / Ground Hazard ────────────────────────────────────────
    # Ground-level drop or trip risks. Correct reaction: slow down, check footing.
    # TTS: "Caution — ground hazard ahead."
    "stairs"                 : 4,   # moved from structure_context (v2 bug)
    "curb"                   : 4,   # moved from static_obstacle (v2 bug)
    "water body"             : 4,   # moved from static_obstacle (v2 bug)
    "terrain"                : 4,
    "inaccessible surface"   : 4,
    "vegetation"             : 4,   # signals end of paved surface / terrain change
    "hand rail"              : 4,   # absence near stairs is safety-relevant

    # ── Group 5: Dynamic Hazard ───────────────────────────────────────────────
    # Moving or suddenly-appearing entities. Require immediate reaction.
    # Includes doors/gates — these are time-critical like moving pedestrians.
    # TTS: "Moving hazard nearby — [direction]."
    "pedestrian"             : 5,
    "rider"                  : 5,
    "animal"                 : 5,
    "opening-door"           : 5,   # moved from static_obstacle (v2 bug)
    "opening-gate"           : 5,   # moved from static_obstacle (v2 bug)

    # ── Group 6: Vehicle ──────────────────────────────────────────────────────
    # Highest-priority moving threat. Triggers emergency alert.
    # TTS: "STOP — vehicle detected."
    "vehicle"                : 6,

    # SKIPPED (not in map -> pixel_to_macro returns -1):
    #   unlabeled (0), sky (27), building, traffic_sign, traffic_light, bus_stop
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

# ──────────────────────────────────────────────────────────────────────────────
# TUNING PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

MAX_POLYGON_POINTS = 50    # points per contour (lower = faster Jetson inference)
MIN_CONTOUR_AREA   = 200   # px² — filters annotation noise
MIN_MASK_PIXELS    = 100   # total foreground pixels before processing a class
TRAIN_RATIO        = 0.85  # session-level split


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_labelmap(sanpo_root: Path) -> dict[str, int]:
    """
    Load labelmap.json -> {class_name: pixel_value}.
    Handles both dict {"name": id} and list [{"name":..., "id":...}] formats.
    """
    path = sanpo_root / "labelmap.json"
    if not path.exists():
        raise FileNotFoundError(f"labelmap.json not found: {path}")

    with open(path) as f:
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

    print(f"\n[build_pixel_to_macro]")
    print(f"  Mapped  : {len(mapped_lines)}")
    print(f"  Skipped : {len(skipped_lines)}")
    print("\nMapped classes:")
    for line in sorted(mapped_lines):
        print(line)
    if skipped_lines:
        print("\nSkipped classes (excluded from training):")
        for line in skipped_lines:
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

    Contour fix:
        Binary mask is built with np.where(..., 255, 0).astype(np.uint8).
        cv2.findContours requires a proper 0/255 image — using 0/1 causes
        silent shape-skipping on certain OpenCV builds.
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
            epsilon = 0.005 * cv2.arcLength(contour, True)
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
    Uses all_candidates.txt if present; otherwise auto-scans.
    """
    candidates = sanpo_root / "all_candidates.txt"

    if candidates.exists():
        with open(candidates) as f:
            ids = [line.strip() for line in f if line.strip()]
        sessions = [sanpo_root / s for s in ids if (sanpo_root / s).exists()]
        print(f"[get_sessions] {len(sessions)} sessions loaded from all_candidates.txt")
    else:
        sessions = [
            d for d in sorted(sanpo_root.iterdir())
            if d.is_dir() and (d / "camera_head").exists()
        ]
        print(f"[get_sessions] {len(sessions)} sessions found by directory scan")

    return sessions


def process_session(session_path: Path,
                    pixel_to_macro: dict[int, int]) -> list[tuple[Path, list[str]]]:
    """
    Process all annotated frames in one session.
    Returns list of (image_path, yolo_annotation_lines) — only non-empty frames.
    """
    frames_dir = session_path / "camera_head" / "left" / "video_frames"
    masks_dir  = session_path / "camera_head" / "left" / "segmentation_masks"

    if not frames_dir.exists() or not masks_dir.exists():
        return []

    results = []
    for frame_path in sorted(frames_dir.glob("*.png")):
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
# MAIN CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def convert(sanpo_root: str,
            output_root: str,
            max_sessions: int | None = None) -> None:
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
    sanpo_root  = Path(sanpo_root)
    output_root = Path(output_root)

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Step 1 — Load labelmap
    print("\n[Step 1/4] Loading labelmap...")
    labelmap = load_labelmap(sanpo_root)
    print(f"           SANPO total classes: {len(labelmap)}")

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
    cut           = int(len(sessions) * TRAIN_RATIO)
    train_sessions = sessions[:cut]
    val_sessions   = sessions[cut:]

    print(f"         Total   : {len(sessions)} sessions")
    print(f"         Train   : {len(train_sessions)} sessions")
    print(f"         Val     : {len(val_sessions)} sessions")

    # Step 4 — Process and write
    print("\n[Step 4/4] Processing frames...")
    stats = {"train": 0, "val": 0, "empty": 0}

    for split, session_list in (("train", train_sessions), ("val", val_sessions)):
        for session_path in tqdm(session_list, desc=f"  {split:5s}"):
            samples = process_session(session_path, pixel_to_macro)
            if not samples:
                stats["empty"] += 1
                continue
            write_split(samples, split, output_root, session_path.name)
            stats[split] += len(samples)

    # Write dataset.yaml
    yaml_path = output_root / "dataset.yaml"
    yaml_text = "\n".join([
        "# ALAS Project - SANPO Dataset (YOLO Segmentation Format)",
        "# Auto-generated by ALAS dataset_format.py",
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
    print(f"  Train frames        : {stats['train']}")
    print(f"  Val frames          : {stats['val']}")
    print(f"  Empty sessions      : {stats['empty']}")
    print(f"  Total frames        : {total_frames}")
    print(f"  dataset.yaml        : {yaml_path}")
    print(f"\n  Macro classes ({NUM_CLASSES}):")
    for i, name in CLASS_NAMES.items():
        print(f"    [{i}] {name}")
    print(f"\n  To start training:")
    print(f"    yolo train model=yolo26n-seg.pt data={yaml_path} epochs=100 imgsz=640")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SANPO dataset to YOLO segmentation format (session-based split)."
    )
    parser.add_argument("--sanpo_root",   required=True,
                        help="Root directory of SANPO dataset (contains labelmap.json)")
    parser.add_argument("--output_root",  required=True,
                        help="Output directory for YOLO-format dataset")
    parser.add_argument("--max_sessions", type=int, default=None,
                        help="Limit session count for debug runs (default: all)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for reproducible session split (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)
    convert(args.sanpo_root, args.output_root, args.max_sessions)