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

#from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ai.geometry import CameraGeometry, pixel_to_ground_distance

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


CLASS_NAMES = {
    ClassID.WALKABLE_SURFACE:   "walkable_surface",
    ClassID.CROSSWALK:          "crosswalk",
    ClassID.VEHICLE_ROAD:       "vehicle_road",
    ClassID.COLLISION_OBSTACLE: "collision_obstacle",
    ClassID.FALL_HAZARD:        "fall_hazard",
    ClassID.DYNAMIC_HAZARD:     "dynamic_hazard",
    ClassID.VEHICLE:            "vehicle",
}

# BGR colours for optional overlay rendering
CLASS_COLORS_BGR = {
    ClassID.WALKABLE_SURFACE:   (0, 200, 0),
    ClassID.CROSSWALK:          (200, 255, 0),
    ClassID.VEHICLE_ROAD:       (180, 30, 30),
    ClassID.COLLISION_OBSTACLE: (255, 140, 0),
    ClassID.FALL_HAZARD:        (255, 80, 0),
    ClassID.DYNAMIC_HAZARD:     (255, 100, 100),
    ClassID.VEHICLE:            (255, 0, 0),
}

# Alert config per class — priority 0 = silent
CLASS_ALERT_CONFIG= {
    ClassID.WALKABLE_SURFACE:   {"priority": 0, "alert": None,                                           "cooldown": 0},
    ClassID.CROSSWALK:          {"priority": 1, "alert": "Yaya geçidi algılandı, geçiş güvenli",        "cooldown": 8.0},
    ClassID.VEHICLE_ROAD:       {"priority": 4, "alert": "Dikkat, araç yolu, girmeyin",                  "cooldown": 5.0},
    ClassID.COLLISION_OBSTACLE: {"priority": 3, "alert": "Önünüzde engel var, durun veya yön değiştirin","cooldown": 3.0},
    ClassID.FALL_HAZARD:        {"priority": 3, "alert": "Zemin tehlikesi, yavaşlayın",                  "cooldown": 3.0},
    ClassID.DYNAMIC_HAZARD:     {"priority": 4, "alert": "Nesne algılandı",                              "cooldown": 4.0},
    ClassID.VEHICLE:            {"priority": 5, "alert": "Durun, önünüzde engel var",                   "cooldown": 1.5},
}

# Minimum pixel ratio for a class to trigger an alert
MIN_ALERT_RATIO = 0.02

# Pixel ratio thresholds for proximity wording (fallback when no distance estimate)
VERY_CLOSE_RATIO = 0.15
NEARBY_RATIO = 0.05

# ── Walkable-overlap gating (applied to static obstacles only) ──
# A COLLISION_OBSTACLE / FALL_HAZARD is suppressed unless at least
# MIN_WALKABLE_OVERLAP of its pixels overlap the (dilated) walkable surface.
# Vehicles and dynamic hazards are NEVER gated this way — users still need
# warnings about a car that has not yet entered the sidewalk.
DILATE_FRAC = 0.06                       # dilation kernel as fraction of mask width
MIN_WALKABLE_OVERLAP = 0.10              # ≥10% of pixels must overlap walkable
MIN_WALKABLE_RATIO_FOR_GATING = 0.05     # safety hatch: bypass gate if scene
                                         # has too little walkable area to trust
WALKABLE_GATED_CLASSES = {ClassID.COLLISION_OBSTACLE, ClassID.FALL_HAZARD}

# Metric distance thresholds (used when CameraGeometry is provided)
VERY_CLOSE_M = 2.0
NEARBY_M = 5.0


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
    zone_ratios: dict = field(default_factory=dict)
    walkable_overlap: float = 1.0
        # Fraction of this class's pixels that overlap the (dilated) walkable
        # surface. Only meaningful for classes in WALKABLE_GATED_CLASSES;
        # defaults to 1.0 so non-gated classes always pass the gate.
    estimated_distance_m: Optional[float] = None
        # Ground-plane projected distance to the bottom-most pixel of the
        # blob, in metres. None when no CameraGeometry was provided or when
        # the bottom pixel projects above the horizon.


@dataclass
class SceneAnalysis:
    """Full scene understanding result for a single frame."""
    walkable_ratio: float        # how much of the frame is walkable
    zones: list                  # per-class breakdowns (non-zero only)
    is_safe: bool                # True if no high-priority hazards
    dominant_hazard: Optional[str]  # name of the biggest threat, or None


@dataclass
class Alert:
    """A single hazard announcement candidate.

    Carries enough structured context for the dispatcher to filter by class
    (e.g. only crosswalks while navigating) and to stamp the per-class
    cooldown only after the alert is actually spoken.
    """
    class_id: int
    text: str
    priority: int


@dataclass
class PerceptionResult:
    """Everything the main loop needs from a single frame."""
    alerts: list                 # List[Alert], priority-sorted (highest first)
    scene: SceneAnalysis         # detailed scene breakdown
    mask: np.ndarray             # class-ID mask (H_model, W_model)
    inference_ms: float          # TRT/ONNX inference time
    total_ms: float              # full pipeline time
    path_guidance: Optional[str] = None  # directional walking instruction, or None


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
        self.bindings: list = []
        self.d_input = self.d_output = None
        self.h_output: Optional[np.ndarray] = None
        self.output_shape: Optional[tuple] = None

        self.h_input: Optional[np.ndarray] = None
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.d_input = device_mem
                self.input_shape = tuple(shape)
                # Pinned host input buffer — copying frames into this is much
                # faster than the implicit pageable-to-pinned staging that
                # memcpy_htod_async would do otherwise.
                self.h_input = cuda.pagelocked_empty(int(np.prod(shape)), dtype=dtype)
            else:
                self.output_shape = tuple(shape)
                self.d_output = device_mem
                self.h_output = cuda.pagelocked_empty(int(np.prod(shape)), dtype=dtype)

        self.stream = cuda.Stream()
        logger.info(f"TensorRT engine loaded — input: {self.input_shape}, output: {self.output_shape}")

        # Warm-up: TRT JIT-compiles execution plans on the first real call.
        # Run one zeroed inference so the first user-visible frame does not
        # eat that one-off latency hit.
        try:
            warm = np.zeros(self.input_shape, dtype=np.float32)
            self.predict(warm)
            logger.info("TensorRT engine warmed up.")
        except Exception:
            logger.exception("TensorRT warm-up failed — first real inference will be slow.")

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        # Copy into the pinned host buffer in-place — no extra ascontiguousarray
        # allocation per frame.
        np.copyto(self.h_input, tensor.ravel())
        self.cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
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

        # Warm-up: first run does provider/kernel allocation. Skip the cost on
        # the first user-visible frame.
        try:
            shape = tuple(self.session.get_inputs()[0].shape)
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in shape)
            warm = np.zeros(shape, dtype=np.float32)
            self.predict(warm)
            logger.info("ONNX Runtime warmed up.")
        except Exception:
            logger.exception("ONNX warm-up failed — first real inference will be slow.")

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

def preprocess(
    frame_bgr: np.ndarray,
    target_h: int,
    target_w: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """BGR camera frame → model input tensor (1, H, W, 3) float32 [0, 1].

    When ``out`` is provided, the result is written into it in-place to avoid
    per-frame allocations. Resize happens before colorspace conversion so the
    cvtColor pass works on the smaller buffer.
    """
    if frame_bgr.shape[0] != target_h or frame_bgr.shape[1] != target_w:
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame_bgr
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if out is None:
        return np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

    # In-place: cast + scale into the pre-allocated buffer.
    np.divide(rgb, 255.0, out=out[0], casting="unsafe", dtype=np.float32)
    return out


def postprocess(logits: np.ndarray) -> np.ndarray:
    """Model logits → class-ID mask (H, W) uint8."""
    if logits.ndim == 4:
        return np.argmax(logits[0], axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        return np.argmax(logits, axis=-1).astype(np.uint8)
    return logits.astype(np.uint8)


def analyse_scene(
    mask: np.ndarray,
    camera_geom: Optional[CameraGeometry] = None,
) -> SceneAnalysis:
    """Extract scene understanding from the segmentation mask.

    Two-pass design: a single bincount gives the per-class pixel totals in
    one O(N) sweep, then the per-class deep analysis (zoning, walkable
    overlap, distance) only runs for classes that crossed
    ``MIN_ALERT_RATIO`` — the rest cannot trigger an alert anyway, so
    skipping them avoids the bulk of the work on a typical frame.

    Zone classification uses the bottom half of the frame only: a tall pole
    leaning across the upper image should still be classified by where its
    base actually sits, since that is what threatens the walker's foot.
    """
    h, w = mask.shape
    total_px = float(h * w)
    n_classes = len(ClassID)

    # Single O(N) sweep for per-class pixel counts. Replaces seven boolean
    # masks + np.sum calls.
    flat = mask.ravel()
    counts = np.bincount(flat, minlength=n_classes)
    if counts.size > n_classes:
        counts = counts[:n_classes]

    walkable_count = int(counts[int(ClassID.WALKABLE_SURFACE)])
    walkable_ratio = walkable_count / total_px if total_px > 0 else 0.0

    # Dilated walkable mask: only needed when at least one gated class is
    # present in this frame, otherwise the dilate cost is wasted.
    walkable_dilated: Optional[np.ndarray] = None
    if any(counts[int(cid)] > 0 for cid in WALKABLE_GATED_CLASSES):
        walkable_binary = (mask == int(ClassID.WALKABLE_SURFACE)).astype(np.uint8)
        k = max(3, int(DILATE_FRAC * w))
        walkable_dilated = cv2.dilate(walkable_binary, np.ones((k, k), np.uint8))

    bottom_start = h // 2
    bottom = mask[bottom_start:, :]
    third = w // 3

    zones: list = []
    max_hazard_priority = 0
    dominant_hazard: Optional[str] = None

    for cid in ClassID:
        cid_int = int(cid)
        px_count = int(counts[cid_int])
        if px_count == 0:
            continue
        ratio = px_count / total_px

        # Skip the per-class deep work for classes too small to ever alert,
        # except for WALKABLE_SURFACE itself which we still need to expose.
        if ratio < MIN_ALERT_RATIO and cid != ClassID.WALKABLE_SURFACE:
            continue

        # Bottom-half zone classification: matches what threatens the foot,
        # not what fills the upper periphery.
        bottom_binary = (bottom == cid_int)
        b_total = int(bottom_binary.sum())
        if b_total > 0:
            left_px   = int(bottom_binary[:, :third].sum())
            center_px = int(bottom_binary[:, third:2 * third].sum())
            right_px  = int(bottom_binary[:, 2 * third:].sum())
            zone_ratios = {
                "left":   left_px / b_total,
                "center": center_px / b_total,
                "right":  right_px / b_total,
            }
            dominant = max(zone_ratios, key=zone_ratios.get)
        else:
            # Class only exists in the upper half — too far to matter for
            # a pedestrian's next step. Mark as "center" with empty ratios.
            zone_ratios = {"left": 0.0, "center": 0.0, "right": 0.0}
            dominant = "center"

        overlap = 1.0
        if cid in WALKABLE_GATED_CLASSES and walkable_dilated is not None:
            class_binary = (mask == cid_int)
            overlap_px = int(np.logical_and(class_binary, walkable_dilated).sum())
            overlap = overlap_px / px_count if px_count > 0 else 0.0

        distance_m: Optional[float] = None
        if camera_geom is not None and cid != ClassID.WALKABLE_SURFACE:
            # Bottom-most row that contains this class — np.argmax on a
            # reversed any-per-row vector is faster than np.where on the
            # full mask.
            row_has = (mask == cid_int).any(axis=1)
            if row_has.any():
                bottom_y = int(h - 1 - np.argmax(row_has[::-1]))
                distance_m = pixel_to_ground_distance(bottom_y, h, camera_geom)

        zones.append(ZoneInfo(
            class_id=cid_int,
            class_name=CLASS_NAMES[cid],
            pixel_ratio=ratio,
            dominant_zone=dominant,
            zone_ratios=zone_ratios,
            walkable_overlap=overlap,
            estimated_distance_m=distance_m,
        ))

        cfg = CLASS_ALERT_CONFIG[cid]
        if cfg["priority"] > max_hazard_priority:
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
    last_alert_time: dict,
    now: float,
) -> list:
    """Build priority-sorted ``Alert`` candidates from a scene analysis.

    Pure function: ``last_alert_time`` is *read-only* here. The dispatcher is
    responsible for stamping the cooldown after the alert is actually spoken,
    so a deduped or filtered alert does not silently consume the cooldown
    budget for a class that never reached the user's ear.

    Filtering rules:
      - Skip silent classes (priority 0) and tiny detections (< MIN_ALERT_RATIO).
      - Respect per-class cooldown so the same class does not spam TTS.
      - **Walkable-area gate** for static obstacles (COLLISION_OBSTACLE,
        FALL_HAZARD): require >= MIN_WALKABLE_OVERLAP overlap with the
        dilated walkable surface. Bypassed when the scene has too little
        walkable area to trust (safety hatch).
      - **Lateral suppression** for DYNAMIC_HAZARD: a person on the side
        below VERY_CLOSE_RATIO is silenced; only center-zone or very-close
        dynamic hazards alert.
      - Distance-aware proximity wording when CameraGeometry is available;
        otherwise fall back to the pixel-ratio heuristic.
      - Add direction if not center ("solunuzda" / "sağınızda").
    """
    raw: list = []

    for zone in scene.zones:
        cfg = CLASS_ALERT_CONFIG[zone.class_id]

        if cfg["priority"] == 0 or zone.pixel_ratio < MIN_ALERT_RATIO:
            continue

        elapsed = now - last_alert_time.get(zone.class_id, 0.0)
        if elapsed < cfg["cooldown"]:
            continue

        if (
            zone.class_id in WALKABLE_GATED_CLASSES
            and scene.walkable_ratio >= MIN_WALKABLE_RATIO_FOR_GATING
            and zone.walkable_overlap < MIN_WALKABLE_OVERLAP
        ):
            continue

        if zone.class_id == ClassID.DYNAMIC_HAZARD:
            is_center = zone.dominant_zone == "center"
            is_close = zone.pixel_ratio > VERY_CLOSE_RATIO
            if not (is_center or is_close):
                continue

        parts: list = [cfg["alert"]]

        if zone.estimated_distance_m is not None:
            if zone.estimated_distance_m < VERY_CLOSE_M:
                parts.append("çok yakın")
            elif zone.estimated_distance_m < NEARBY_M:
                parts.append("yakın")
        else:
            # Fallback for callers without CameraGeometry (e.g. offline tests).
            if zone.pixel_ratio > VERY_CLOSE_RATIO:
                parts.append("çok yakın")
            elif zone.pixel_ratio > NEARBY_RATIO:
                parts.append("yakın")

        if zone.dominant_zone == "left":
            parts.append("solunuzda")
        elif zone.dominant_zone == "right":
            parts.append("sağınızda")

        raw.append(Alert(
            class_id=zone.class_id,
            text=", ".join(parts),
            priority=cfg["priority"],
        ))

    raw.sort(key=lambda a: a.priority, reverse=True)
    return raw


# ── Path guidance constants ──────────────────────────────────────────────────
PATH_BOTTOM_FRACTION   = 0.5   # maskenin alt yarısına bak (kullanıcıya en yakın zemin)
CORRIDOR_MARGIN        = 0.15  # sol ve sağ kenarda bu kadar piksel yoksay —
                                # kullanıcı zaten oraya yürümeyecek
MIN_WALKABLE_FOR_GUIDANCE = 0.08  # koridorda bu oranın altındaysa "yol daralıyor" uyarısı


def generate_path_guidance(mask: np.ndarray) -> Optional[str]:
    """
    Maskenin alt yarısındaki merkezi yürüyüş koridorunu analiz ederek
    yön kılavuzu üretir.

    Algoritma:
      1. Alt yarıyı al (kullanıcıya en yakın zemin bölgesi).
      2. Sol ve sağ kenar piksellerini yoksay (CORRIDOR_MARGIN).
         — Kullanıcı ekranın tam kenarına yürümez, oradaki walkable
           pikseller centroid'i yanıltır.
      3. Koridoru 3 eşit dilime böl (sol / orta / sağ).
      4. Her dilimin walkable piksel sayısını karşılaştır:
         - Orta dilim yeterliyse → "Düz yürüyün"
         - Değilse en dolu yana yönlendir.
      5. Toplam walkable oranı düşükse "yol daralıyor" ekle.

    Returns:
        "Düz yürüyün", "Hafif sola yönelin", "Hafif sağa yönelin" gibi
        Türkçe TTS metni; hiç walkable yoksa None.
    """
    h, w = mask.shape
    bottom_start = int(h * (1.0 - PATH_BOTTOM_FRACTION))
    bottom = mask[bottom_start:, :]

    # Merkezi koridor: sol/sağ kenar piksellerini çıkar
    c_left  = int(w * CORRIDOR_MARGIN)
    c_right = int(w * (1.0 - CORRIDOR_MARGIN))
    corridor = bottom[:, c_left:c_right]
    corridor_w = c_right - c_left

    walkable    = (corridor == ClassID.WALKABLE_SURFACE)
    walkable_px = float(np.sum(walkable))

    if walkable_px == 0:
        return None

    walkable_ratio = walkable_px / float(corridor.size)
    if walkable_ratio < MIN_WALKABLE_FOR_GUIDANCE:
        return "Yürünebilir alan çok azalıyor, dikkatli ilerleyin"

    # Koridoru 3 dilime böl ve her dilimin walkable piksel yoğunluğunu hesapla
    third = corridor_w // 3
    left_px   = float(np.sum(walkable[:, :third]))
    center_px = float(np.sum(walkable[:, third: 2 * third]))
    right_px  = float(np.sum(walkable[:, 2 * third:]))

    left_ratio   = left_px   / float(walkable[:, :third].size)
    center_ratio = center_px / float(walkable[:, third: 2 * third].size)
    right_ratio  = right_px  / float(walkable[:, 2 * third:].size)

    narrow_suffix = ", yol daralıyor" if walkable_ratio < 0.18 else ""

    # Centre lane is clear enough — go straight.
    if center_ratio >= 0.40:
        return "Düz yürüyün" + narrow_suffix

    # Centre is poor — pick the fuller side. Symmetric / both-empty cases
    # fall through to "go straight" rather than producing no guidance at all,
    # since silence here would be the worst outcome for the walker.
    if left_ratio > right_ratio:
        return "Hafif sola yönelin" + narrow_suffix
    if right_ratio > left_ratio:
        return "Hafif sağa yönelin" + narrow_suffix
    return "Düz yürüyün" + narrow_suffix

    # Symmetric or both empty — fall back to "go straight" so the user never
    # gets silence at this stage.
    return "Düz yürüyün" + narrow_suffix


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
        camera_geometry: Optional[CameraGeometry] = None,
    ):
        """
        Args:
            model_path:      Path to .trt/.engine or .onnx model file.
            input_h:         Model input height (must match training).
            input_w:         Model input width (must match training).
            camera_geometry: Optional CameraGeometry for distance estimation.
                             When None, alerts use the pixel-ratio heuristic.
        """
        logger.info(f"[Perception] Loading model: {model_path}")
        self._backend = _load_backend(model_path)
        self._input_h = input_h
        self._input_w = input_w
        self._camera_geometry = camera_geometry

        # Per-class cooldown tracker (class_id → last alert unix time).
        self._last_alert_time: dict = {}

        # Reused input tensor — preprocess() writes into it in-place to avoid
        # per-frame allocations on Jetson Nano's slow CPU.
        self._input_buffer = np.empty((1, input_h, input_w, 3), dtype=np.float32)

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

        # 1. Preprocess (writes into the reusable input buffer).
        tensor = preprocess(frame_bgr, self._input_h, self._input_w, out=self._input_buffer)

        # 2. Inference
        t_inf = time.monotonic()
        logits = self._backend.predict(tensor)
        inference_ms = (time.monotonic() - t_inf) * 1000

        # 3. Postprocess → class-ID mask
        mask = postprocess(logits)

        # 4. Scene analysis
        scene = analyse_scene(mask, camera_geom=self._camera_geometry)

        # 5. Generate alert candidates (pure — cooldown is stamped by the
        #    dispatcher only after an alert is actually spoken).
        alerts = generate_alerts(scene, self._last_alert_time, now)

        # 6. Path guidance
        guidance = generate_path_guidance(mask)

        total_ms = (time.monotonic() - t_start) * 1000

        return PerceptionResult(
            alerts=alerts,
            scene=scene,
            mask=mask,
            inference_ms=inference_ms,
            total_ms=total_ms,
            path_guidance=guidance,
        )

    def mark_alert_spoken(self, class_id: int, when: Optional[float] = None) -> None:
        """Record that an alert for ``class_id`` was actually spoken.

        Called by the dispatcher *only* when an alert reaches the user, so
        cooldown is consumed by speech, not by candidate generation.
        """
        self._last_alert_time[int(class_id)] = when if when is not None else time.time()

    def reset_cooldowns(self) -> None:
        """Reset all alert cooldowns (e.g. after a long pause)."""
        self._last_alert_time.clear()