"""
ALAS — SANPO U-Net Semantic Dataset Builder
==================================================
Converts original SANPO masks directly to 7-class semantic masks.
 
Usage:
  python src/ai/dataset_unet_format.py \
    --sanpo_root /Users/yasinyldrm/Downloads/SANPO_DATASET/sanpo_dataset_v7 \
    --output_root /Users/yasinyldrm/Downloads/SANPO_DATASET/ALAS_Unet_FINAL
 
Output:
  ALAS_Semantic_v1/
  ├── dataset.yaml
  ├── images/train/*.png   (RGB frames, original resolution)
  ├── images/val/*.png
  ├── masks/train/*.png    (class-ID masks: 0-6 valid, 255=ignore)
  └── masks/val/*.png
  
  python src/ai/dataset_unet_format.py \
  --sanpo_root /Users/yasinyldrm/Downloads/SANPO_DATASET/test_dataset \
  --output_root /Users/yasinyldrm/Downloads/SANPO_DATASET/ALAS_Unet_FINAL \
  --max_sessions 3
"""

import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2


# ------------------------------------------------------------------------------
# CLASS MAPPING & CONFIGURATION
# ------------------------------------------------------------------------------

MACRO_MAP: dict[str, int] = {
    # 0: Walkable Surface (Safe pedestrian paths)
    "sidewalk": 0, "paved trail": 0, "other walkable surface": 0,

    # 1: Crosswalk (Designated road-crossing zones)
    "crosswalk": 1,

    # 2: Vehicle Road (Motorized traffic lanes to avoid)
    "road": 2, "railway track": 2,

    # 3: Collision Obstacle (Stationary head/body-level impact risks)
    # Note: 'tree' is included (trunk hazard); 'vegetation' is ignored.
    "pole": 3, "wall/fence": 3, "guard rail/road barrier": 3,
    "obstacle": 3, "bike rack": 3, "opening-door": 3, "opening-gate": 3,
    "hand rail": 3, "bus stop": 3, "traffic sign": 3, "traffic light": 3,
    "tree": 3,

    # 4: Fall Hazard (Ground-level trip or drop risks)
    # Note: 'terrain' is ignored to reduce noise.
    "curb": 4, "stairs": 4, "inaccessible surface": 4, "water body": 4,

    # 5: Dynamic Hazard (Moving entities requiring immediate reaction)
    "pedestrian": 5, "rider": 5, "animal": 5,

    # 6: Vehicle (Highest-priority moving threat)
    "vehicle": 6,
}

NUM_CLASSES = 7
CLASS_NAMES = {
    0: "walkable_surface", 1: "crosswalk", 2: "vehicle_road",
    3: "collision_obstacle", 4: "fall_hazard", 5: "dynamic_hazard", 6: "vehicle",
}

# BGR Colors for OpenCV Debug Overlays
COLOR_PALETTE = {
    0: (0, 200, 0),       # walkable_surface   - green
    1: (255, 255, 0),     # crosswalk          - cyan
    2: (0, 0, 200),       # vehicle_road       - red
    3: (200, 100, 0),     # collision_obstacle - dark blue
    4: (0, 165, 255),     # fall_hazard        - orange
    5: (255, 0, 255),     # dynamic_hazard     - magenta
    6: (0, 0, 255),       # vehicle            - bright red
    255: (0, 0, 0)        # ignore             - black
}

IGNORE = 255
TRAIN_RATIO = 0.85
FRAME_STEP = 4  # Process every Nth frame to reduce dataset size and increase temporal diversity
RARE_CLASSES = {1, 4, 5}  # Classes with low pixel representation that benefit from denser sampling : crosswalk, fall_hazard, dynamic_hazard
RARE_CLASS_THRESHOLD = 0.02  # Minimum rare-class pixel ratio to trigger denser sampling, 2% of valid (non-ignore) pixels


# ------------------------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------------------------

def load_labelmap(labelmap_path: Path) -> dict[str, int]:
    """Loads the original SANPO labelmap from the specified path."""
    if not labelmap_path.exists():
        raise FileNotFoundError(f"labelmap.json not found: {labelmap_path}")
    with open(labelmap_path) as f:
        data = json.load(f)
    
    # Handle both dict and list formats just in case
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {item["name"]: item["id"] for item in data}
    raise ValueError("Unrecognized labelmap.json format.")


def build_lookup(labelmap: dict[str, int]) -> np.ndarray:
    """Builds a fast numpy lookup table: pixel_value -> macro_class_id."""
    max_val = max(labelmap.values()) + 1
    lut = np.full(max_val, IGNORE, dtype=np.uint8)

    mapped, skipped = 0, 0
    for name, pixel in labelmap.items():
        if name in MACRO_MAP:
            lut[pixel] = MACRO_MAP[name]
            mapped += 1
        else:
            skipped += 1

    print(f"  Mapped: {mapped} SANPO classes -> {NUM_CLASSES} macro groups")
    print(f"  Skipped: {skipped} classes -> ignore ({IGNORE})")
    return lut


def remap_mask(mask_path: Path, lut: np.ndarray) -> np.ndarray:
    """Remaps the original SANPO mask to the 7-class semantic mask using a LUT."""
    raw = np.array(Image.open(mask_path))
    if raw.ndim >= 3:
        raw = raw[:, :, 0]

    clipped = np.clip(raw, 0, len(lut) - 1)
    return lut[clipped]

def get_sessions(sanpo_root: Path) -> list[Path]:
    """Collects valid sessions, supporting both camera_chest and camera_head."""
    candidates = sanpo_root / "all_candidates.txt"
    if candidates.exists():
        with open(candidates) as f:
            ids = [l.strip() for l in f if l.strip()]
        sessions = [sanpo_root / s for s in ids if (sanpo_root / s).exists()]
    else:
        sessions = [
            d for d in sanpo_root.iterdir()
            if d.is_dir() and ((d / "camera_chest").exists() or (d / "camera_head").exists())
        ]

    print(f"  Found {len(sessions)} valid sessions")
    return sessions


def get_camera_dir(session_path: Path) -> str | None:
    """Determines the correct camera directory for a given session."""
    if (session_path / "camera_chest").exists():
        return "camera_chest"
    elif (session_path / "camera_head").exists():
        return "camera_head"
    return None

def get_frame_step(session: Path, lut: np.ndarray, default_step: int = 4) -> int:
    """Returns a reduced frame step for sessions containing rare classes.

    Samples 15 evenly spaced masks throughout the session. If any rare class 
    (crosswalk, fall_hazard, dynamic_hazard) covers at least RARE_CLASS_THRESHOLD 
    of the VALID pixels in any single probe frame, the step is halved.
    """
    camera_dir = get_camera_dir(session)
    if not camera_dir:
        return default_step

    masks_dir = session / camera_dir / "left" / "segmentation_masks"
    if not masks_dir.exists():
        return default_step

    all_masks = sorted(masks_dir.glob("*.png"))
    if not all_masks:
        return default_step

    # Sample 15 frames evenly distributed across the entire session
    num_probes = min(15, len(all_masks))
    probe_indices = np.linspace(0, len(all_masks) - 1, num_probes, dtype=int)
    sample_masks = [all_masks[i] for i in probe_indices]

    for mp in sample_masks:
        semantic = remap_mask(mp, lut)
        
        # Calculate ratio based ONLY on valid pixels (excluding IGNORE/255)
        valid_mask = semantic != IGNORE
        valid_pixels = np.sum(valid_mask)
        
        if valid_pixels == 0:
            continue

        for cls in RARE_CLASSES:
            cls_pixels = np.sum(semantic == cls)
            if (cls_pixels / valid_pixels) > RARE_CLASS_THRESHOLD:
                return max(1, default_step // 2)  # Halve the step (e.g., 4 -> 2)

    return default_step

def process_session(session: Path, lut: np.ndarray, frame_step: int) -> list[tuple[Path, Path, np.ndarray]]:
    """Processes frames with temporal subsampling (frame_step)."""
    camera_dir = get_camera_dir(session)
    if not camera_dir:
        return []

    frames_dir = session / camera_dir / "left" / "video_frames"
    masks_dir  = session / camera_dir / "left" / "segmentation_masks"

    if not frames_dir.exists() or not masks_dir.exists():
        return []

    frame_step = get_frame_step(session, lut, default_step=frame_step)
    
    all_frames = sorted(frames_dir.glob("*.png"))
    sampled_frames = all_frames[::frame_step]

    results = []
    for frame in sampled_frames:
        mask_path = masks_dir / frame.name
        if not mask_path.exists():
            continue

        semantic = remap_mask(mask_path, lut)

        # Skip frames where >90% of pixels are ignore (mostly sky/building)
        valid_ratio = np.mean(semantic != IGNORE)
        if valid_ratio < 0.10:
            continue

        results.append((frame, mask_path, semantic))

    return results


def draw_debug_overlay(img_path: Path, mask: np.ndarray, save_path: Path) -> None:
    """Blends the raw RGB image with a colorized version of the semantic mask."""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    # Create a colorized mask based on the PALETTE
    color_mask = np.zeros_like(img)
    for class_id, color in COLOR_PALETTE.items():
        color_mask[mask == class_id] = color

    # Blend original image and the colorized mask
    blended = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), blended)


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------

def convert(sanpo_root: Path, output_root: Path, labelmap_path: Path, max_sessions: int | None = None, frame_step: int = FRAME_STEP):
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "masks"  / split).mkdir(parents=True, exist_ok=True)
    
    debug_dir = output_root / "_debug_masks"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Building class lookup table...")
    labelmap = load_labelmap(labelmap_path)
    lut = build_lookup(labelmap)

    print("\n[2/3] Session-based split...")
    sessions = get_sessions(sanpo_root)
    if not sessions:
        print("[ERROR] No sessions found.")
        return

    if max_sessions:
        sessions = sessions[:max_sessions]
        print(f"  Debug mode: {max_sessions} sessions")

    random.seed(42)
    random.shuffle(sessions)
    cut = int(len(sessions) * TRAIN_RATIO)
    splits = {
        "train": sessions[:cut],
        "val": sessions[cut:],
    }
    print(f"  Train: {len(splits['train'])} sessions | Val: {len(splits['val'])} sessions")

    print(f"\n[3/3] Processing frames (Step: {frame_step}) - USING SHUTIL.MOVE...")
    stats = {"train": 0, "val": 0, "skipped": 0, "debug_overlays": 0}
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for split, session_list in splits.items():
        for session in tqdm(session_list, desc=f"  {split:5s}"):
            sid = session.name
            pairs = process_session(session, lut, frame_step)

            if not pairs:
                stats["skipped"] += 1
                continue
            
            mid_idx = len(pairs) // 2

            for i, (frame_path, mask_path, semantic_mask) in enumerate(pairs):
                stem = f"{sid}_{frame_path.stem}"
                new_img_path = output_root / "images" / split / f"{stem}.png"
                new_mask_path = output_root / "masks" / split / f"{stem}.png"

                # 0. Hata almamak için taşıma işleminden ÖNCE debug görselini oluştur
                if i == mid_idx:
                    overlay_path = debug_dir / f"{split}_{sid}.png"
                    draw_debug_overlay(frame_path, semantic_mask, overlay_path)
                    stats["debug_overlays"] += 1

                # 1. Orijinal RGB kareyi yeni yere TAŞI (0 alan harcar, anında gerçekleşir)
                shutil.move(str(frame_path), str(new_img_path))

                # 2. Yeni 7-class semantic maskeyi kaydet
                Image.fromarray(semantic_mask).save(new_mask_path)

                # 3. Orijinal SANPO maskesini SİL (Ekstra alan açar)
                mask_path.unlink(missing_ok=True)

                # Sınıf istatistiklerini topla
                for c in range(NUM_CLASSES):
                    class_pixel_counts[c] += np.sum(semantic_mask == c)

                stats[split] += 1

    # Generate dataset.yaml for reference
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text("\n".join([
        "# ALAS Semantic Segmentation Dataset",
        f"path: .",
        "train_images: images/train",
        "val_images: images/val",
        "train_masks: masks/train",
        "val_masks: masks/val",
        f"nc: {NUM_CLASSES}",
        "ignore_index: 255",
        "names:",
        *[f"  {i}: {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)],
    ]))

    # Print Final Report
    total_pixels = class_pixel_counts.sum()
    print(f"\n{'=' * 62}")
    print(f"  DATASET READY (Moved & Cleaned)")
    print(f"{'=' * 62}")
    print(f"  Output    : {output_root}")
    print(f"  Frame step: {frame_step} (Adaptive)")
    print(f"  Train     : {stats['train']} frames")
    print(f"  Val       : {stats['val']} frames")
    print(f"  Skipped   : {stats['skipped']} empty sessions")
    print(f"  Overlays  : {stats['debug_overlays']} generated in _debug_masks/")
    print(f"\n  Class distribution:")
    print(f"  {'Class':<25} {'Pixels':>14} {'%':>7}")
    print("  " + "-" * 48)
    for c in range(NUM_CLASSES):
        pct = class_pixel_counts[c] / max(total_pixels, 1) * 100
        print(f"  {CLASS_NAMES[c]:<25} {class_pixel_counts[c]:>14,} {pct:>6.1f}%")
    print(f"\n  Mask format: class IDs 0-6, ignore=255")
    print(f"{'=' * 62}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SANPO -> ALAS direct semantic dataset")
    parser.add_argument("--sanpo_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--labelmap_path", type=str, default="labelmap.json",
                        help="Relative or absolute path to labelmap.json (default: labelmap.json)")
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=FRAME_STEP)
    args = parser.parse_args()
    
    convert(
        Path(args.sanpo_root), 
        Path(args.output_root), 
        Path(args.labelmap_path), 
        args.max_sessions, 
        args.frame_step
    )