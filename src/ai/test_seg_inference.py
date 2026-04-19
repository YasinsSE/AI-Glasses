# =============================================================================
# ALAS — Live Segmentation Viewer (Lightweight)
# =============================================================================
# Display-only version — no PNG/CSV writes, minimal I/O.
# Press 'q' to quit, 's' to save a single snapshot, 'r' to toggle raw mask view.
#
# Usage:
#   python src/ai/live_seg_viewer.py --model models/segmentation/alas_engine.trt --gstreamer
#   python3 src/ai/test_seg_inference.py --model models/segmentation/alas_engine.trt --gstreamer --fps 10
#   python src/ai/live_seg_viewer.py --model models/segmentation/alas_model.onnx
# =============================================================================

import os
import sys
import time
import ctypes
import logging
from pathlib import Path
from enum import IntEnum

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alas_live")


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
    ClassID.WALKABLE_SURFACE:   "walkable",
    ClassID.CROSSWALK:          "crosswalk",
    ClassID.VEHICLE_ROAD:       "road",
    ClassID.COLLISION_OBSTACLE: "obstacle",
    ClassID.FALL_HAZARD:        "fall_hazard",
    ClassID.DYNAMIC_HAZARD:     "dynamic",
    ClassID.VEHICLE:            "vehicle",
}

CLASS_COLORS_BGR = {
    ClassID.WALKABLE_SURFACE:   (0, 200, 0),
    ClassID.CROSSWALK:          (200, 255, 0),
    ClassID.VEHICLE_ROAD:       (180, 30, 30),
    ClassID.COLLISION_OBSTACLE: (0, 140, 255),
    ClassID.FALL_HAZARD:        (0, 80, 255),
    ClassID.DYNAMIC_HAZARD:     (100, 100, 255),
    ClassID.VEHICLE:            (0, 0, 255),
}

CLASS_CONFIG = {
    ClassID.WALKABLE_SURFACE:   {"priority": 0, "alert": None, "cooldown": 0},
    ClassID.CROSSWALK:          {"priority": 1, "alert": "Crosswalk", "cooldown": 8.0},
    ClassID.VEHICLE_ROAD:       {"priority": 2, "alert": "Road ahead", "cooldown": 5.0},
    ClassID.COLLISION_OBSTACLE: {"priority": 3, "alert": "Obstacle", "cooldown": 2.0},
    ClassID.FALL_HAZARD:        {"priority": 3, "alert": "Fall hazard", "cooldown": 2.0},
    ClassID.DYNAMIC_HAZARD:     {"priority": 4, "alert": "Moving hazard", "cooldown": 1.5},
    ClassID.VEHICLE:            {"priority": 5, "alert": "Vehicle!", "cooldown": 1.0},
}


# =============================================================================
# CUDA Runtime (ctypes — no pycuda needed)
# =============================================================================

class CUDARuntime:
    H2D = 1
    D2H = 2

    def __init__(self):
        self._lib = ctypes.CDLL("libcudart.so")

        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMallocHost.restype = ctypes.c_int
        self._lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self._lib.cudaFreeHost.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int

    @staticmethod
    def check(status, msg="CUDA error"):
        if status != 0:
            raise RuntimeError("{}: code {}".format(msg, status))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self._lib.cudaMalloc(ctypes.byref(ptr), nbytes))
        return ptr

    def free(self, ptr):
        self._lib.cudaFree(ptr)

    def malloc_host(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self._lib.cudaMallocHost(ctypes.byref(ptr), nbytes))
        return ptr

    def free_host(self, ptr):
        self._lib.cudaFreeHost(ptr)

    def memcpy_h2d_async(self, dst, src_np, nbytes, stream):
        self.check(self._lib.cudaMemcpyAsync(
            dst, src_np.ctypes.data_as(ctypes.c_void_p), nbytes, self.H2D, stream))

    def memcpy_d2h_async(self, dst_host, src_dev, nbytes, stream):
        self.check(self._lib.cudaMemcpyAsync(
            dst_host, src_dev, nbytes, self.D2H, stream))

    def stream_create(self):
        s = ctypes.c_void_p()
        self.check(self._lib.cudaStreamCreate(ctypes.byref(s)))
        return s

    def stream_sync(self, stream):
        self.check(self._lib.cudaStreamSynchronize(stream))

    def stream_destroy(self, stream):
        self._lib.cudaStreamDestroy(stream)


# =============================================================================
# Inference Backends
# =============================================================================

class TensorRTBackend:
    def __init__(self, engine_path):
        import tensorrt as trt
        self._crt = CUDARuntime()
        self._stream = self._crt.stream_create()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.bindings = []
        self._d_ptrs = []
        self._h_output_ptr = None
        self.h_output = None
        self.output_shape = None
        self._input_idx = 0
        self._output_idx = 1
        self._input_nbytes = 0
        self._output_nbytes = 0

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

            d_ptr = self._crt.malloc(nbytes)
            self.bindings.append(int(d_ptr.value))
            self._d_ptrs.append(d_ptr)

            if self.engine.binding_is_input(i):
                self._input_idx = i
                self._input_nbytes = nbytes
            else:
                self._output_idx = i
                self.output_shape = shape
                self._output_nbytes = nbytes
                self._output_dtype = dtype
                h_ptr = self._crt.malloc_host(nbytes)
                self._h_output_ptr = h_ptr
                c_arr = (ctypes.c_byte * nbytes).from_address(h_ptr.value)
                self.h_output = np.frombuffer(c_arr, dtype=dtype)

        logger.info("TensorRT loaded, output: {}".format(self.output_shape))

    def predict(self, tensor):
        host_input = np.ascontiguousarray(tensor.ravel())
        self._crt.memcpy_h2d_async(
            self._d_ptrs[self._input_idx], host_input,
            self._input_nbytes, self._stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=int(self._stream.value))
        self._crt.memcpy_d2h_async(
            self._h_output_ptr, self._d_ptrs[self._output_idx],
            self._output_nbytes, self._stream)
        self._crt.stream_sync(self._stream)
        return self.h_output.reshape(self.output_shape)

    def __del__(self):
        try:
            for d in self._d_ptrs:
                self._crt.free(d)
            if self._h_output_ptr:
                self._crt.free_host(self._h_output_ptr)
            if self._stream:
                self._crt.stream_destroy(self._stream)
        except Exception:
            pass


class ONNXBackend:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ONNX loaded, providers: {}".format(self.session.get_providers()))

    def predict(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0]


def load_backend(model_path):
    ext = Path(model_path).suffix.lower()
    if ext in (".trt", ".engine"):
        return TensorRTBackend(model_path)
    elif ext == ".onnx":
        return ONNXBackend(model_path)
    raise ValueError("Unsupported: {}".format(ext))


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


def open_camera(camera_id=0, width=640, height=480, fps=30, use_gstreamer=False):
    if use_gstreamer:
        pipeline = GST_PIPELINE.format(w=width, h=height, fps=fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    logger.info("Camera: {}x{}@{}fps gst={}".format(width, height, fps, use_gstreamer))
    return cap


# =============================================================================
# Processing — optimized for Jetson Nano
# =============================================================================

def preprocess(frame_bgr, target_h, target_w):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def postprocess(logits):
    if logits.ndim == 4:
        return np.argmax(logits[0], axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        return np.argmax(logits, axis=-1).astype(np.uint8)
    return logits.astype(np.uint8)


def render_overlay(frame_bgr, mask, alpha=0.45):
    """Fast overlay — resize mask to frame size, blend colors."""
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(frame_bgr)
    for cid, color in CLASS_COLORS_BGR.items():
        overlay[mask_resized == cid] = color

    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


def render_color_mask(mask, height, width):
    """Pure color mask — no camera, just segmentation colors."""
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS_BGR.items():
        color_mask[mask_resized == cid] = color
    return color_mask


def get_alerts(mask, last_alert_time, now):
    """Lightweight alert generation — no full stats, just check danger zones."""
    h, w = mask.shape
    total = float(h * w)
    third = w // 3
    alerts = []

    for cid in ClassID:
        cfg = CLASS_CONFIG[cid]
        if cfg["priority"] == 0:
            continue

        ratio = float(np.sum(mask == cid)) / total
        if ratio < 0.02:
            continue

        elapsed = now - last_alert_time.get(int(cid), 0.0)
        if elapsed < cfg["cooldown"]:
            continue

        # Zone detection on mask directly (cheaper than full stats)
        binary = (mask == cid)
        left  = float(np.sum(binary[:, :third]))
        center = float(np.sum(binary[:, third:2*third]))
        right = float(np.sum(binary[:, 2*third:]))

        parts = [cfg["alert"]]
        if ratio > 0.15:
            parts.append("CLOSE")
        elif ratio > 0.05:
            parts.append("near")

        best = max(left, center, right)
        if best == left:
            parts.append("L")
        elif best == right:
            parts.append("R")

        alerts.append((cfg["priority"], " ".join(parts)))
        last_alert_time[int(cid)] = now

    alerts.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in alerts]


def draw_hud(frame, fps, inf_ms, alerts, frame_count):
    """Draw FPS, inference time, alerts, and legend on frame."""
    h, w = frame.shape[:2]

    # FPS bar
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, "FPS:{:.1f} Inf:{:.0f}ms F:{}".format(fps, inf_ms, frame_count),
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Alerts
    y = 60
    for alert in alerts[:3]:
        cv2.putText(frame, alert, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 26

    # Legend (bottom-left)
    legend_y = h - 10
    for cid in reversed(list(ClassID)):
        color = CLASS_COLORS_BGR[cid]
        name = CLASS_NAMES[cid]
        cv2.rectangle(frame, (5, legend_y - 14), (20, legend_y), color, -1)
        cv2.putText(frame, name, (24, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y -= 20

    return frame


# =============================================================================
# Main Live Loop
# =============================================================================

def run_live(
    model_path,
    camera_id=0,
    target_fps=4.0,
    use_gstreamer=False,
    input_h=384,
    input_w=512,
):
    backend = load_backend(model_path)
    cap = open_camera(camera_id, use_gstreamer=use_gstreamer)

    throttle = 1.0 / target_fps
    last_yield = 0.0
    last_alert_time = {}
    frame_count = 0
    show_raw_mask = False
    snap_dir = Path("outputs/snapshots")

    # EMA for smooth FPS display
    ema_fps = target_fps
    ema_alpha = 0.3

    logger.info("=== LIVE MODE ===")
    logger.info("Keys: [q] quit | [s] snapshot | [r] toggle mask view")
    logger.info("Target FPS: {}".format(target_fps))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.monotonic()
            if (now - last_yield) < throttle:
                continue
            last_yield = now

            t0 = time.monotonic()

            # Preprocess + Inference
            tensor = preprocess(frame, input_h, input_w)
            t_inf = time.monotonic()
            logits = backend.predict(tensor)
            inf_ms = (time.monotonic() - t_inf) * 1000.0

            # Postprocess
            mask = postprocess(logits)

            # Alerts (on mask resolution — fast)
            timestamp = time.time()
            alerts = get_alerts(mask, last_alert_time, timestamp)

            total_ms = (time.monotonic() - t0) * 1000.0
            current_fps = 1000.0 / total_ms if total_ms > 0 else 0
            ema_fps = ema_alpha * current_fps + (1 - ema_alpha) * ema_fps

            # Render
            if show_raw_mask:
                display = render_color_mask(mask, frame.shape[0], frame.shape[1])
            else:
                display = render_overlay(frame, mask)

            display = draw_hud(display, ema_fps, inf_ms, alerts, frame_count)

            # Show
            cv2.imshow("ALAS Live", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit")
                break
            elif key == ord("s"):
                # Save snapshot on demand
                snap_dir.mkdir(exist_ok=True)
                ts = int(time.time())
                cv2.imwrite(str(snap_dir / "snap_{}_{}.jpg".format(frame_count, ts)), display)
                cv2.imwrite(str(snap_dir / "mask_{}_{}.png".format(frame_count, ts)), mask)
                logger.info("Snapshot saved: frame {}".format(frame_count))
            elif key == ord("r"):
                show_raw_mask = not show_raw_mask
                logger.info("Raw mask view: {}".format(show_raw_mask))

            for alert in alerts:
                logger.info("[ALERT] {}".format(alert))

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cap.release()
        time.sleep(0.3)
        cv2.destroyAllWindows()
        logger.info("Processed {} frames, avg FPS: {:.1f}".format(frame_count, ema_fps))


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ALAS -- Live Segmentation Viewer")
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--gstreamer", action="store_true")

    args = parser.parse_args()

    run_live(
        model_path=args.model,
        camera_id=args.camera,
        target_fps=args.fps,
        use_gstreamer=args.gstreamer,
    )