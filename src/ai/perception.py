"""
ALAS Perception Pipeline
========================
Camera frame → preprocess → TensorRT/ONNX inference → scene understanding → alerts.

This module wraps the full perception chain into a single class that the main
loop can call with one method: ``pipeline.process(frame_bgr)``.

Pipeline stages:
    1. Preprocess  : BGR → RGB → resize → float32 [0,1] → (1, H, W, 3)
    2. Inference   : TensorRT or ONNX backend → raw logits
    3. Postprocess : logits → argmax → class-ID mask (H, W)
    4. Scene analysis : per-class pixel ratios + dominant zone (left/center/right)
    5. Alert generation : priority-sorted TTS strings with cooldown

Usage:
    from ai.perception import PerceptionPipeline

    pipeline = PerceptionPipeline("models/segmentation/alas_engine.trt")

    # In the camera loop:
    result = pipeline.process(frame_bgr)
    for alert in result.alerts:
        speak(alert)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ALAS.perception")


# ═══════════════════════════════════════════════════════════════════
#  SEMANTIC CLASS DEFINITIONS (7 macro classes)
# ═══════════════════════════════════════════════════════════════════

class ClassID(IntEnum):
    WALKABLE_SURFACE   = 0
    CROSSWALK          = 1
    VEHICLE_ROAD       = 2
    COLLISION_OBSTACLE = 3
    FALL_HAZARD        = 4
    DYNAMIC_HAZARD     = 5
    VEHICLE            = 6


CLASS_NAMES: dict[int, str] = {
    ClassID.WALKABLE_SURFACE:   "walkable_surface",
    ClassID.CROSSWALK:          "crosswalk",
    ClassID.VEHICLE_ROAD:       "vehicle_road",
    ClassID.COLLISION_OBSTACLE: "collision_obstacle",
    ClassID.FALL_HAZARD:        "fall_hazard",
    ClassID.DYNAMIC_HAZARD:     "dynamic_hazard",
    ClassID.VEHICLE:            "vehicle",
}

# BGR colours for optional overlay rendering
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    ClassID.WALKABLE_SURFACE:   (0, 200, 0),
    ClassID.CROSSWALK:          (200, 255, 0),
    ClassID.VEHICLE_ROAD:       (180, 30, 30),
    ClassID.COLLISION_OBSTACLE: (255, 140, 0),
    ClassID.FALL_HAZARD:        (255, 80, 0),
    ClassID.DYNAMIC_HAZARD:     (255, 100, 100),
    ClassID.VEHICLE:            (255, 0, 0),
}

# Alert config per class — priority 0 = silent
CLASS_ALERT_CONFIG: dict[int, dict] = {
    ClassID.WALKABLE_SURFACE:   {"priority": 0, "alert": None,                                           "cooldown": 0},
    ClassID.CROSSWALK:          {"priority": 1, "alert": "Yaya geçidi algılandı, geçiş güvenli",        "cooldown": 8.0},
    ClassID.VEHICLE_ROAD:       {"priority": 4, "alert": "Dikkat, araç yolu, girmeyin",                  "cooldown": 5.0},
    ClassID.COLLISION_OBSTACLE: {"priority": 3, "alert": "Önünüzde engel var, durun veya yön değiştirin","cooldown": 3.0},
    ClassID.FALL_HAZARD:        {"priority": 3, "alert": "Zemin tehlikesi, yavaşlayın",                  "cooldown": 3.0},
    ClassID.DYNAMIC_HAZARD:     {"priority": 4, "alert": "Hareketli tehlike yakınızda",                  "cooldown": 2.0},
    ClassID.VEHICLE:            {"priority": 5, "alert": "Durun, araç algılandı",                        "cooldown": 1.5},
}

# Minimum pixel ratio for a class to trigger an alert
MIN_ALERT_RATIO = 0.02

# Pixel ratio thresholds for proximity wording
VERY_CLOSE_RATIO = 0.15
NEARBY_RATIO = 0.05


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES — pipeline output
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ZoneInfo:
    """Per-class analysis result."""
    class_id: int
    class_name: str
    pixel_ratio: float          # fraction of frame occupied [0, 1]
    dominant_zone: str           # "left" | "center" | "right"
    zone_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class SceneAnalysis:
    """Full scene understanding result for a single frame."""
    walkable_ratio: float        # how much of the frame is walkable
    zones: list[ZoneInfo]        # per-class breakdowns (non-zero only)
    is_safe: bool                # True if no high-priority hazards
    dominant_hazard: str | None  # name of the biggest threat, or None


@dataclass
class PerceptionResult:
    """Everything the main loop needs from a single frame."""
    alerts: list[str]            # TTS-ready strings, priority-sorted
    scene: SceneAnalysis         # detailed scene breakdown
    mask: np.ndarray             # class-ID mask (H_model, W_model)
    inference_ms: float          # TRT/ONNX inference time
    total_ms: float              # full pipeline time


# ═══════════════════════════════════════════════════════════════════
#  INFERENCE BACKENDS
# ═══════════════════════════════════════════════════════════════════

class TensorRTBackend:
    """TensorRT engine backend for Jetson Nano."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises CUDA context

        self.cuda = cuda
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.bindings: list[int] = []
        self.d_input = self.d_output = None
        self.h_output: np.ndarray | None = None
        self.output_shape: tuple | None = None

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.d_input = device_mem
                self.input_shape = tuple(shape)
            else:
                self.output_shape = tuple(shape)
                self.d_output = device_mem
                self.h_output = cuda.pagelocked_empty(int(np.prod(shape)), dtype=dtype)

        self.stream = cuda.Stream()
        logger.info(f"TensorRT engine loaded — input: {self.input_shape}, output: {self.output_shape}")

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        self.cuda.memcpy_htod_async(
            self.d_input,
            np.ascontiguousarray(tensor.ravel()),
            self.stream,
        )
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape(self.output_shape)


class ONNXBackend:
    """ONNX Runtime backend (CPU or CUDA)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()
        logger.info(f"ONNX Runtime loaded — providers: {active}")

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]


def _load_backend(model_path: str):
    """Auto-select backend from file extension."""
    ext = Path(model_path).suffix.lower()
    if ext in (".trt", ".engine"):
        return TensorRTBackend(model_path)
    if ext == ".onnx":
        return ONNXBackend(model_path)
    raise ValueError(f"Unsupported model format: {ext}. Use .onnx, .trt, or .engine")


# ═══════════════════════════════════════════════════════════════════
#  PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def preprocess(frame_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """BGR camera frame → model input tensor (1, H, W, 3) float32 [0, 1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def postprocess(logits: np.ndarray) -> np.ndarray:
    """Model logits → class-ID mask (H, W) uint8."""
    if logits.ndim == 4:
        return np.argmax(logits[0], axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        return np.argmax(logits, axis=-1).astype(np.uint8)
    return logits.astype(np.uint8)


def analyse_scene(mask: np.ndarray) -> SceneAnalysis:
    """
    Extract meaningful information from the segmentation mask.

    For each detected class, compute:
      - pixel ratio (how much of the frame it occupies)
      - dominant zone (left / center / right third of the frame)

    Also determine overall walkability and primary hazard.
    """
    h, w = mask.shape
    third = w // 3
    total_px = float(h * w)

    zones: list[ZoneInfo] = []
    walkable_ratio = 0.0
    max_hazard_priority = 0
    dominant_hazard: str | None = None

    for cid in ClassID:
        binary = (mask == cid)
        px_count = float(np.sum(binary))
        if px_count == 0:
            continue

        ratio = px_count / total_px

        left_px   = float(np.sum(binary[:, :third]))
        center_px = float(np.sum(binary[:, third:2 * third]))
        right_px  = float(np.sum(binary[:, 2 * third:]))

        zone_ratios = {
            "left":   left_px / px_count if px_count > 0 else 0,
            "center": center_px / px_count if px_count > 0 else 0,
            "right":  right_px / px_count if px_count > 0 else 0,
        }
        dominant = max(zone_ratios, key=zone_ratios.get)

        zones.append(ZoneInfo(
            class_id=int(cid),
            class_name=CLASS_NAMES[cid],
            pixel_ratio=ratio,
            dominant_zone=dominant,
            zone_ratios=zone_ratios,
        ))

        if cid == ClassID.WALKABLE_SURFACE:
            walkable_ratio = ratio

        # Track biggest hazard
        cfg = CLASS_ALERT_CONFIG[cid]
        if cfg["priority"] > max_hazard_priority and ratio >= MIN_ALERT_RATIO:
            max_hazard_priority = cfg["priority"]
            dominant_hazard = CLASS_NAMES[cid]

    is_safe = max_hazard_priority < 3

    return SceneAnalysis(
        walkable_ratio=walkable_ratio,
        zones=zones,
        is_safe=is_safe,
        dominant_hazard=dominant_hazard,
    )


def generate_alerts(
    scene: SceneAnalysis,
    last_alert_time: dict[int, float],
    now: float,
) -> list[str]:
    """
    Generate priority-sorted TTS alert strings from scene analysis.

    Rules:
      - Skip walkable_surface (silent = safe)
      - Skip classes below MIN_ALERT_RATIO (too small to matter)
      - Respect per-class cooldown to avoid TTS spam
      - Add proximity wording ("çok yakın" / "yakın")
      - Add direction if not center ("solunuzda" / "sağınızda")
    """
    raw: list[tuple[int, str]] = []

    for zone in scene.zones:
        cfg = CLASS_ALERT_CONFIG[zone.class_id]

        # Skip silent classes and tiny detections
        if cfg["priority"] == 0 or zone.pixel_ratio < MIN_ALERT_RATIO:
            continue

        # Cooldown check
        elapsed = now - last_alert_time.get(zone.class_id, 0.0)
        if elapsed < cfg["cooldown"]:
            continue

        parts: list[str] = [cfg["alert"]]

        # Proximity
        if zone.pixel_ratio > VERY_CLOSE_RATIO:
            parts.append("çok yakın")
        elif zone.pixel_ratio > NEARBY_RATIO:
            parts.append("yakın")

        # Direction
        if zone.dominant_zone == "left":
            parts.append("solunuzda")
        elif zone.dominant_zone == "right":
            parts.append("sağınızda")

        raw.append((cfg["priority"], ", ".join(parts)))
        last_alert_time[zone.class_id] = now

    # Highest priority first
    raw.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in raw]


def render_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend segmentation colours onto the camera frame (for debugging/display)."""
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(frame_bgr)
    for cid, color in CLASS_COLORS_BGR.items():
        overlay[mask_resized == cid] = color

    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


# ═══════════════════════════════════════════════════════════════════
#  PERCEPTION PIPELINE — single entry point
# ═══════════════════════════════════════════════════════════════════

class PerceptionPipeline:
    """
    Full perception chain: preprocess → inference → postprocess → scene analysis → alerts.

    Usage:
        pipeline = PerceptionPipeline("alas_engine.trt")

        # In your loop:
        result = pipeline.process(frame_bgr)
        for alert in result.alerts:
            speak(alert)
    """

    def __init__(
        self,
        model_path: str,
        input_h: int = 384,
        input_w: int = 512,
    ):
        """
        Args:
            model_path: Path to .trt/.engine or .onnx model file.
            input_h:    Model input height (must match training).
            input_w:    Model input width (must match training).
        """
        logger.info(f"[Perception] Loading model: {model_path}")
        self._backend = _load_backend(model_path)
        self._input_h = input_h
        self._input_w = input_w

        # Per-class cooldown tracker (class_id → last alert unix time)
        self._last_alert_time: dict[int, float] = {}

        logger.info(f"[Perception] Ready — input size: {input_w}x{input_h}")

    def process(self, frame_bgr: np.ndarray) -> PerceptionResult:
        """
        Run the full pipeline on one BGR camera frame.

        Args:
            frame_bgr: Raw BGR frame from cv2.VideoCapture.

        Returns:
            PerceptionResult with alerts, scene analysis, mask, and timing.
        """
        t_start = time.monotonic()
        now = time.time()

        # 1. Preprocess
        tensor = preprocess(frame_bgr, self._input_h, self._input_w)

        # 2. Inference
        t_inf = time.monotonic()
        logits = self._backend.predict(tensor)
        inference_ms = (time.monotonic() - t_inf) * 1000

        # 3. Postprocess → class-ID mask
        mask = postprocess(logits)

        # 4. Scene analysis
        scene = analyse_scene(mask)

        # 5. Generate alerts
        alerts = generate_alerts(scene, self._last_alert_time, now)

        total_ms = (time.monotonic() - t_start) * 1000

        return PerceptionResult(
            alerts=alerts,
            scene=scene,
            mask=mask,
            inference_ms=inference_ms,
            total_ms=total_ms,
        )

    def reset_cooldowns(self) -> None:
        """Reset all alert cooldowns (e.g. after a long pause)."""
        self._last_alert_time.clear()
