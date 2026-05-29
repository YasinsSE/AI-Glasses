"""ALAS — VFH local planner demo.

Standalone demonstration of the VFH local planner on image, video, or live
camera input. Mirrors the structure of the AI segmentation eval scripts so it
slots into the same evaluation workflow.

Modes:
    1) Image    — run PerceptionPipeline + VFH on a single JPG/PNG.
    2) Video    — the same, frame-by-frame on a video file.
    3) Camera   — live camera input.
    4) No-model — skip segmentation and feed a pre-computed class-ID mask PNG
                  straight into VFH (fastest iteration loop).

In every mode the demo prints the chosen VFHAction + Turkish TTS text to stdout
and (unless --headless) opens a window with the composite overlay. Press 'q' to
quit, 's' to save a snapshot. Snapshots are written under
outputs/eval/navigation/local_planner/.

How to run (from the repository root):
    python eval/navigation/local_planner/vfh_demo.py --model models/segmentation/alas_engine.trt --image eval/ai/samples/test1.jpeg --save
    python eval/navigation/local_planner/vfh_demo.py --model models/segmentation/alas_engine.trt --video walk.mp4
    python eval/navigation/local_planner/vfh_demo.py --model models/segmentation/alas_engine.trt --camera 0
    python eval/navigation/local_planner/vfh_demo.py --no-model --mask-image mask.png
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make src/ importable, and the sibling visualizer module too, whether run from
# the repository root or from anywhere else.
_HERE = Path(__file__).resolve()
_REPO_ROOT = next(p for p in _HERE.parents if (p / "src").is_dir())
for _path in (str(_REPO_ROOT / "src"), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ai.geometry import CameraGeometry
from ai.perception import PerceptionPipeline, analyse_scene
from main.config import ALASConfig
from navigation.local_planner import VFHPlanner
from vfh_visualizer import draw_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vfh_demo")

_OUTPUT_DIR = _REPO_ROOT / "outputs" / "eval" / "navigation" / "local_planner"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="VFH local planner demo")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",      help="Path to a single image file")
    src.add_argument("--video",      help="Path to a video file")
    src.add_argument("--camera",     type=int, nargs="?", const=0,
                     help="Use a live camera (optionally device index, default 0)")
    src.add_argument("--mask-image", help="Pre-computed class-ID mask PNG (with --no-model)")
    p.add_argument("--model",     help="Path to .trt/.engine or .onnx segmentation model")
    p.add_argument("--no-model",  action="store_true",
                   help="Skip segmentation pipeline (only valid with --mask-image)")
    p.add_argument("--save",      action="store_true",
                   help="Auto-save each rendered overlay to outputs/vfh_demo_*.png")
    p.add_argument("--headless",  action="store_true",
                   help="Do not open a window (CI / no display)")
    p.add_argument("--target",    default=None,
                   help='Optional target action ("turn_left" / "turn_right" / "continue")')
    p.add_argument("--fps-cap",   type=float, default=10.0,
                   help="Cap loop FPS for video/camera modes (default 10)")
    return p.parse_args(argv)


def _build_planner(config: ALASConfig) -> VFHPlanner:
    geom = CameraGeometry(
        height_m=config.ai.camera_height_m,
        tilt_deg=config.ai.camera_tilt_deg,
        vfov_deg=config.ai.camera_vfov_deg,
    )
    return VFHPlanner(config, geom)


def _save_snapshot(image: np.ndarray, tag: str = "demo") -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = _OUTPUT_DIR / f"vfh_{tag}_{ts}.png"
    cv2.imwrite(str(out), image)
    logger.info("Saved snapshot: %s", out)
    return out


def _process_one(
    frame_bgr,
    mask,
    pipeline,
    planner: VFHPlanner,
    target_action,
):
    """Run scene analysis + VFH on a (frame, mask) pair.

    ``frame_bgr`` may be None (no-model mode). When ``pipeline`` is provided
    and ``mask`` is None, the mask is produced by running the model on the
    frame; otherwise the supplied mask is used as-is.
    """
    if mask is None:
        if pipeline is None or frame_bgr is None:
            raise ValueError("Either a model+frame or a pre-computed mask is required.")
        result = pipeline.process(frame_bgr)
        mask = result.mask
        scene = result.scene
    else:
        scene = analyse_scene(
            mask,
            camera_geom=CameraGeometry(
                height_m=planner._cfg.ai.camera_height_m,
                tilt_deg=planner._cfg.ai.camera_tilt_deg,
                vfov_deg=planner._cfg.ai.camera_vfov_deg,
            ),
        )

    activated = planner.should_activate(scene)
    guidance = planner.plan(mask, scene, target_action=target_action)
    cost_grid = planner.build_cost_grid(mask)
    overlay = draw_overlay(
        frame_bgr=frame_bgr,
        mask=mask,
        cost_grid=cost_grid,
        guidance=guidance,
        blocked_threshold=planner._cfg.vfh.blocked_threshold,
        activated=activated,
        near_rows_ratio=planner._cfg.vfh.near_rows_ratio,
    )
    return guidance, activated, overlay, scene


def _log_result(scene, guidance, activated):
    walkable_pct = scene.walkable_ratio * 100.0
    if guidance is None:
        logger.info("activated=%s | walkable=%.1f%% | VFH: (no guidance)",
                    activated, walkable_pct)
    else:
        logger.info(
            "activated=%s | walkable=%.1f%% | VFH: action=%s sector=%d hist=%s",
            activated, walkable_pct,
            guidance.action.value, guidance.sector_index,
            ["%.2f" % h for h in guidance.histogram],
        )


def _run_image(args, pipeline, planner):
    frame = cv2.imread(args.image)
    if frame is None:
        logger.error("Could not read image: %s", args.image)
        return 1
    guidance, activated, overlay, scene = _process_one(frame, None, pipeline, planner, args.target)
    _log_result(scene, guidance, activated)
    if args.save:
        _save_snapshot(overlay, tag="image")
    if not args.headless:
        cv2.imshow("VFH demo", overlay)
        logger.info("Press any key to close window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _run_mask_image(args, planner):
    mask = cv2.imread(args.mask_image, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error("Could not read mask image: %s", args.mask_image)
        return 1
    guidance, activated, overlay, scene = _process_one(None, mask, None, planner, args.target)
    _log_result(scene, guidance, activated)
    if args.save:
        _save_snapshot(overlay, tag="mask")
    if not args.headless:
        cv2.imshow("VFH demo (mask only)", overlay)
        logger.info("Press any key to close window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _run_stream(args, pipeline, planner, cap, label: str):
    if not cap.isOpened():
        logger.error("Could not open source: %s", label)
        return 1
    min_interval = 1.0 / max(args.fps_cap, 0.1)
    last = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("Stream ended.")
                break
            now = time.monotonic()
            if now - last < min_interval:
                cv2.waitKey(1) if not args.headless else None
                continue
            last = now
            guidance, activated, overlay, scene = _process_one(
                frame, None, pipeline, planner, args.target,
            )
            _log_result(scene, guidance, activated)

            if args.save and guidance is not None and activated:
                # Save only "interesting" frames to keep disk usage sane.
                _save_snapshot(overlay, tag=label)

            if not args.headless:
                cv2.imshow("VFH demo", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    _save_snapshot(overlay, tag=f"{label}_manual")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.no_model and not args.mask_image:
        logger.error("--no-model requires --mask-image")
        return 2
    if args.mask_image and not args.no_model:
        logger.error("--mask-image must be paired with --no-model")
        return 2
    if not args.no_model and not args.model:
        logger.error("--model is required unless --no-model is set")
        return 2

    config = ALASConfig()
    planner = _build_planner(config)

    pipeline = None
    if not args.no_model:
        logger.info("Loading segmentation model: %s", args.model)
        pipeline = PerceptionPipeline(
            model_path=args.model,
            input_h=config.ai.model_input_h,
            input_w=config.ai.model_input_w,
            camera_geometry=CameraGeometry(
                height_m=config.ai.camera_height_m,
                tilt_deg=config.ai.camera_tilt_deg,
                vfov_deg=config.ai.camera_vfov_deg,
            ),
        )

    if args.image:
        return _run_image(args, pipeline, planner)
    if args.mask_image:
        return _run_mask_image(args, planner)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        return _run_stream(args, pipeline, planner, cap, label="video")
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ai.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ai.camera_height)
        return _run_stream(args, pipeline, planner, cap, label="camera")

    logger.error("No input mode selected.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
