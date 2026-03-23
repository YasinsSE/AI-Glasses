# =============================================================================
# ALAS — Scene Inference Test Runner
# =============================================================================
# Standalone script to test the segmentation model on Jetson Nano.
# Captures frames from camera, runs inference, saves annotated outputs
# and logs all results to CSV for offline review.
#
# Output structure:
#   test_output/
#   ├── frames/          Annotated frames with overlay (PNG)
#   ├── masks/           Raw segmentation masks (PNG, class ID per pixel)
#   ├── inference_log.csv   Per-frame metrics and alerts
#   └── session_summary.txt Overall session statistics
#
# Usage:
#   python src/ai/test_seg_inference.py --model models/segmentation/alas_engine.trt --gstreamer --fps 4
#   python src/ai/test_seg_inference.py --model models/segmentation/alas_engine.trt --gstreamer --fps 15
# Alternatively, for ONNX (slower but no TensorRT build needed):
#   python src/ai/test_seg_inference.py --model models/segmentation/alas_model.onnx --camera 0
#   python src/ai/test_seg_inference.py --model models/segmentation/alas_model.onnx --no-display

#
# TensorRT engine build (run once on Jetson):
#   trtexec --onnx=alas_model.onnx --saveEngine=alas_engine.trt \
#           --fp16 --workspace=1024
# =============================================================================

from __future__ import annotations

import os
import csv
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alas_test")


# =============================================================================
# Class Definitions
# =============================================================================

class ClassID(IntEnum):
    WALKABLE_SURFACE   = 0
    CROSSWALK          = 1
    VEHICLE_ROAD       = 2
    COLLISION_OBSTACLE = 3
    FALL_HAZARD        = 4
    DYNAMIC_HAZARD     = 5
    VEHICLE            = 6


CLASS_NAMES = {
    ClassID.WALKABLE_SURFACE:   "walkable_surface",
    ClassID.CROSSWALK:          "crosswalk",
    ClassID.VEHICLE_ROAD:       "vehicle_road",
    ClassID.COLLISION_OBSTACLE: "collision_obstacle",
    ClassID.FALL_HAZARD:        "fall_hazard",
    ClassID.DYNAMIC_HAZARD:     "dynamic_hazard",
    ClassID.VEHICLE:            "vehicle",
}

# BGR colors for overlay rendering
CLASS_COLORS_BGR = {
    ClassID.WALKABLE_SURFACE:   (0, 200, 0),
    ClassID.CROSSWALK:          (200, 255, 0),
    ClassID.VEHICLE_ROAD:       (180, 30, 30),
    ClassID.COLLISION_OBSTACLE: (255, 140, 0),
    ClassID.FALL_HAZARD:        (255, 80, 0),
    ClassID.DYNAMIC_HAZARD:     (255, 100, 100),
    ClassID.VEHICLE:            (255, 0, 0),
}

CLASS_CONFIG = {
    ClassID.WALKABLE_SURFACE:   {"priority": 0, "alert": None, "cooldown": 0},
    ClassID.CROSSWALK:          {"priority": 1, "alert": "Crosswalk detected", "cooldown": 8.0},
    ClassID.VEHICLE_ROAD:       {"priority": 2, "alert": "Vehicle road ahead", "cooldown": 5.0},
    ClassID.COLLISION_OBSTACLE: {"priority": 3, "alert": "Obstacle ahead", "cooldown": 2.0},
    ClassID.FALL_HAZARD:        {"priority": 3, "alert": "Fall hazard", "cooldown": 2.0},
    ClassID.DYNAMIC_HAZARD:     {"priority": 4, "alert": "Moving hazard nearby", "cooldown": 1.5},
    ClassID.VEHICLE:            {"priority": 5, "alert": "Vehicle detected", "cooldown": 1.0},
}


# =============================================================================
# Inference Backend
# =============================================================================

class TensorRTBackend:

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa

        self.cuda = cuda
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.bindings = []
        self.d_input = self.d_output = None
        self.h_output = None
        self.output_shape = None

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.d_input = device_mem
            else:
                self.output_shape = shape
                self.d_output = device_mem
                self.h_output = cuda.pagelocked_empty(int(np.prod(shape)), dtype=dtype)

        self.stream = cuda.Stream()
        logger.info(f"TensorRT engine loaded, output shape: {self.output_shape}")

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        self.cuda.memcpy_htod_async(self.d_input, np.ascontiguousarray(tensor.ravel()), self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape(self.output_shape)


class ONNXBackend:

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()
        logger.info(f"ONNX Runtime loaded, providers: {active}")

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]


def load_backend(model_path: str):
    ext = Path(model_path).suffix.lower()
    if ext in (".trt", ".engine"):
        return TensorRTBackend(model_path)
    elif ext == ".onnx":
        return ONNXBackend(model_path)
    raise ValueError(f"Unsupported format: {ext}. Use .onnx or .trt/.engine")


# =============================================================================
# Camera
# =============================================================================

GST_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width={w}, height={h}, "
    "format=(string)NV12, framerate=(fraction){fps}/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width={w}, height={h}, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink drop=1"
)


def open_camera(
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    use_gstreamer: bool = False,
) -> cv2.VideoCapture:
    if use_gstreamer:
        pipeline = GST_PIPELINE.format(w=width, h=height, fps=fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (id={camera_id})")
    logger.info(f"Camera opened: {width}x{height}@{fps}fps (gst={use_gstreamer})")
    return cap


# =============================================================================
# Processing Functions
# =============================================================================

def preprocess(frame_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """BGR frame -> (1, H, W, 3) float32 [0, 1]. Model handles [-1, 1] internally."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def postprocess(logits: np.ndarray) -> np.ndarray:
    """Model output -> class ID mask (H, W)."""
    if logits.ndim == 4:
        return np.argmax(logits[0], axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        return np.argmax(logits, axis=-1).astype(np.uint8)
    return logits.astype(np.uint8)


def compute_class_stats(mask: np.ndarray) -> dict:
    """Per-class pixel ratio and dominant zone (left/center/right)."""
    h, w = mask.shape
    third = w // 3
    total = float(h * w)
    stats = {}

    for cid in ClassID:
        binary = (mask == cid)
        total_px = float(np.sum(binary))
        if total_px == 0:
            continue

        left_px   = float(np.sum(binary[:, :third]))
        center_px = float(np.sum(binary[:, third:2*third]))
        right_px  = float(np.sum(binary[:, 2*third:]))

        zones = {"left": left_px, "center": center_px, "right": right_px}
        dominant = max(zones, key=zones.get)

        stats[cid] = {
            "name": CLASS_NAMES[cid],
            "ratio": total_px / total,
            "dominant_zone": dominant,
        }

    return stats


def generate_alerts(
    stats: dict,
    last_alert_time: dict,
    now: float,
) -> list[str]:
    """Priority-sorted alerts with cooldown."""
    raw: list[tuple[int, str]] = []

    for cid, info in stats.items():
        cfg = CLASS_CONFIG[cid]
        if cfg["priority"] == 0 or info["ratio"] < 0.02:
            continue

        elapsed = now - last_alert_time.get(int(cid), 0.0)
        if elapsed < cfg["cooldown"]:
            continue

        parts = [cfg["alert"]]

        if info["ratio"] > 0.15:
            parts.append("very close")
        elif info["ratio"] > 0.05:
            parts.append("nearby")

        if info["dominant_zone"] != "center":
            parts.append(f"to your {info['dominant_zone']}")

        raw.append((cfg["priority"], ", ".join(parts)))
        last_alert_time[int(cid)] = now

    raw.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in raw]


def render_overlay(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend segmentation colors onto the camera frame."""
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(frame_bgr)
    for cid, color in CLASS_COLORS_BGR.items():
        overlay[mask_resized == cid] = color

    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


# =============================================================================
# CSV Logger
# =============================================================================

class FrameLogger:
    """Write per-frame metrics and alerts to a CSV file."""

    COLUMNS = [
        "frame_id", "timestamp", "inference_ms", "total_ms",
        "walkable_pct", "crosswalk_pct", "vehicle_road_pct",
        "collision_obstacle_pct", "fall_hazard_pct",
        "dynamic_hazard_pct", "vehicle_pct",
        "alerts",
    ]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.file = open(csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.COLUMNS)
        self.writer.writeheader()

    def log(self, frame_id: int, timestamp: float, inference_ms: float,
            total_ms: float, stats: dict, alerts: list[str]):
        row = {
            "frame_id": frame_id,
            "timestamp": f"{timestamp:.3f}",
            "inference_ms": f"{inference_ms:.1f}",
            "total_ms": f"{total_ms:.1f}",
            "alerts": " | ".join(alerts) if alerts else "",
        }
        for cid in ClassID:
            col = f"{CLASS_NAMES[cid]}_pct"
            if cid in stats:
                row[col] = f"{stats[cid]['ratio'] * 100:.1f}"
            else:
                row[col] = "0.0"
        self.writer.writerow(row)

    def close(self):
        self.file.close()


# =============================================================================
# Main Test Loop
# =============================================================================

def run_test(
    model_path: str,
    camera_id: int = 0,
    target_fps: float = 4.0,
    max_frames: int = 0,
    output_dir: str = "test_output",
    use_gstreamer: bool = False,
    show_display: bool = True,
    input_h: int = 384,
    input_w: int = 512,
):
    out = Path(output_dir)
    frames_dir = out / "frames"
    masks_dir  = out / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    backend = load_backend(model_path)
    cap = open_camera(camera_id, use_gstreamer=use_gstreamer)
    csv_logger = FrameLogger(out / "inference_log.csv")

    throttle_interval = 1.0 / target_fps
    last_yield = 0.0
    last_alert_time: dict[int, float] = {}
    frame_count = 0
    latencies: list[float] = []

    logger.info(f"Test started. Output: {out.absolute()}")
    logger.info(f"Target FPS: {target_fps}, model: {model_path}")
    if max_frames > 0:
        logger.info(f"Will stop after {max_frames} frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed, retrying...")
                continue

            now = time.monotonic()
            if (now - last_yield) < throttle_interval:
                continue
            last_yield = now

            t_start = time.monotonic()
            timestamp = time.time()

            tensor = preprocess(frame, input_h, input_w)

            t_inf = time.monotonic()
            logits = backend.predict(tensor)
            inference_ms = (time.monotonic() - t_inf) * 1000

            mask = postprocess(logits)
            stats = compute_class_stats(mask)
            alerts = generate_alerts(stats, last_alert_time, timestamp)

            total_ms = (time.monotonic() - t_start) * 1000
            latencies.append(total_ms)

            for alert in alerts:
                logger.info(f"[ALERT] {alert}")

            csv_logger.log(frame_count, timestamp, inference_ms, total_ms, stats, alerts)

            # Save annotated frame and raw mask
            annotated = render_overlay(frame, mask)
            fps_text = f"FPS: {1000.0 / total_ms:.1f} | Inf: {inference_ms:.0f}ms"
            cv2.putText(annotated, fps_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, alert in enumerate(alerts):
                cv2.putText(annotated, alert, (10, 58 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

            cv2.imwrite(str(frames_dir / f"frame_{frame_count:05d}.png"), annotated)
            cv2.imwrite(str(masks_dir / f"mask_{frame_count:05d}.png"), mask)

            if show_display:
                cv2.imshow("ALAS Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed 'q', stopping")
                    break

            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                logger.info(f"Reached {max_frames} frames, stopping")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        cap.release()
        csv_logger.close()
        if show_display:
            cv2.destroyAllWindows()

        # Session summary
        summary_lines = [
            "ALAS Scene Inference — Test Session Summary",
            "=" * 50,
            f"Model           : {model_path}",
            f"Frames processed: {frame_count}",
            f"Target FPS      : {target_fps}",
        ]
        if latencies:
            summary_lines.extend([
                f"Latency mean    : {np.mean(latencies):.1f} ms",
                f"Latency p50     : {np.percentile(latencies, 50):.1f} ms",
                f"Latency p95     : {np.percentile(latencies, 95):.1f} ms",
                f"Actual FPS      : {1000.0 / np.mean(latencies):.1f}",
            ])
        summary_lines.extend([
            f"Output dir      : {out.absolute()}",
            "",
            "Files:",
            f"  frames/       : {frame_count} annotated PNGs",
            f"  masks/        : {frame_count} raw segmentation masks",
            f"  inference_log.csv : per-frame metrics and alerts",
        ])

        summary_text = "\n".join(summary_lines)
        (out / "session_summary.txt").write_text(summary_text)
        print(f"\n{summary_text}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ALAS — Scene Inference Test Runner")
    parser.add_argument("--model", required=True,
                        help="Model file (.onnx or .trt/.engine)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--fps", type=float, default=4.0,
                        help="Target processing FPS (default: 4.0)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = unlimited)")
    parser.add_argument("--output", type=str, default="test_output",
                        help="Output directory (default: test_output)")
    parser.add_argument("--gstreamer", action="store_true",
                        help="Use GStreamer pipeline for CSI camera (Jetson)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no OpenCV window)")

    args = parser.parse_args()

    run_test(
        model_path=args.model,
        camera_id=args.camera,
        target_fps=args.fps,
        max_frames=args.max_frames,
        output_dir=args.output,
        use_gstreamer=args.gstreamer,
        show_display=not args.no_display,
    )