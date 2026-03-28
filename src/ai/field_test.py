# =============================================================================
# ALAS — Headless Field Test
# =============================================================================
# No display, no lag. Logs everything to CSV, saves every Nth frame as JPEG.
# Run outdoors, review logs at home.
#
# Usage:
#   python src/ai/field_test.py --model models/segmentation/alas_engine.trt --gstreamer
#   python src/ai/field_test.py --model models/segmentation/alas_engine.trt --gstreamer --duration 300
#   python src/ai/field_test.py --model models/segmentation/alas_model.onnx --duration 120
#
# Output:  field_test_<timestamp>/
#          ├── log.csv              (every frame: fps, inference_ms, alerts, class %)
#          ├── summary.txt          (session stats)
#          └── frames/              (every 10th frame as JPEG — overlay + raw)
# =============================================================================

import os
import csv
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
logger = logging.getLogger("alas_field")


# =============================================================================
# Classes
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
    ClassID.FALL_HAZARD:        "fall_hzrd",
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
    ClassID.VEHICLE_ROAD:       {"priority": 2, "alert": "Road", "cooldown": 5.0},
    ClassID.COLLISION_OBSTACLE: {"priority": 3, "alert": "Obstacle", "cooldown": 2.0},
    ClassID.FALL_HAZARD:        {"priority": 3, "alert": "Fall hazard", "cooldown": 2.0},
    ClassID.DYNAMIC_HAZARD:     {"priority": 4, "alert": "Moving hazard", "cooldown": 1.5},
    ClassID.VEHICLE:            {"priority": 5, "alert": "Vehicle!", "cooldown": 1.0},
}


# =============================================================================
# CUDA Runtime (ctypes)
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
            ctypes.c_int, ctypes.c_void_p]
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

    def malloc(self, n):
        p = ctypes.c_void_p()
        self.check(self._lib.cudaMalloc(ctypes.byref(p), n)); return p

    def free(self, p):
        self._lib.cudaFree(p)

    def malloc_host(self, n):
        p = ctypes.c_void_p()
        self.check(self._lib.cudaMallocHost(ctypes.byref(p), n)); return p

    def free_host(self, p):
        self._lib.cudaFreeHost(p)

    def h2d(self, dst, src_np, n, stream):
        self.check(self._lib.cudaMemcpyAsync(
            dst, src_np.ctypes.data_as(ctypes.c_void_p), n, self.H2D, stream))

    def d2h(self, dst, src, n, stream):
        self.check(self._lib.cudaMemcpyAsync(dst, src, n, self.D2H, stream))

    def stream_create(self):
        s = ctypes.c_void_p()
        self.check(self._lib.cudaStreamCreate(ctypes.byref(s))); return s

    def stream_sync(self, s):
        self.check(self._lib.cudaStreamSynchronize(s))

    def stream_destroy(self, s):
        self._lib.cudaStreamDestroy(s)


# =============================================================================
# TensorRT Backend
# =============================================================================

class TRTBackend:
    def __init__(self, path):
        import tensorrt as trt
        self._crt = CUDARuntime()
        self._stream = self._crt.stream_create()

        tl = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(tl).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.bindings = []
        self._dp = []
        self._hp = None
        self.h_out = None
        self.out_shape = None
        self._ii = 0
        self._oi = 1
        self._in = 0
        self._on = 0

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dt = trt.nptype(self.engine.get_binding_dtype(i))
            nb = int(np.prod(shape)) * np.dtype(dt).itemsize
            dp = self._crt.malloc(nb)
            self.bindings.append(int(dp.value))
            self._dp.append(dp)

            if self.engine.binding_is_input(i):
                self._ii = i; self._in = nb
            else:
                self._oi = i; self.out_shape = shape; self._on = nb
                hp = self._crt.malloc_host(nb)
                self._hp = hp
                ca = (ctypes.c_byte * nb).from_address(hp.value)
                self.h_out = np.frombuffer(ca, dtype=dt)

        logger.info("TRT loaded, output: {}".format(self.out_shape))

    def predict(self, tensor):
        hi = np.ascontiguousarray(tensor.ravel())
        self._crt.h2d(self._dp[self._ii], hi, self._in, self._stream)
        self.ctx.execute_async_v2(bindings=self.bindings, stream_handle=int(self._stream.value))
        self._crt.d2h(self._hp, self._dp[self._oi], self._on, self._stream)
        self._crt.stream_sync(self._stream)
        return self.h_out.reshape(self.out_shape)

    def __del__(self):
        try:
            for d in self._dp: self._crt.free(d)
            if self._hp: self._crt.free_host(self._hp)
            if self._stream: self._crt.stream_destroy(self._stream)
        except Exception: pass


class ONNXBackend:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        logger.info("ONNX loaded: {}".format(self.sess.get_providers()))

    def predict(self, tensor):
        return self.sess.run(None, {self.iname: tensor})[0]


def load_backend(path):
    ext = Path(path).suffix.lower()
    if ext in (".trt", ".engine"): return TRTBackend(path)
    if ext == ".onnx": return ONNXBackend(path)
    raise ValueError("Unsupported: " + ext)


# =============================================================================
# Camera
# =============================================================================

GST = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width={w}, height={h}, "
    "format=(string)NV12, framerate=(fraction){fps}/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width={w}, height={h}, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink drop=1"
)

def open_cam(cam_id=0, w=640, h=480, fps=30, gst=False):
    if gst:
        cap = cv2.VideoCapture(GST.format(w=w, h=h, fps=fps), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("Camera failed")
    logger.info("Camera: {}x{}@{}fps gst={}".format(w, h, fps, gst))
    return cap


# =============================================================================
# Processing
# =============================================================================

def preprocess(frame, th, tw):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

def postprocess(logits):
    if logits.ndim == 4:
        return np.argmax(logits[0], axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        return np.argmax(logits, axis=-1).astype(np.uint8)
    return logits.astype(np.uint8)

def class_ratios(mask):
    total = float(mask.shape[0] * mask.shape[1])
    ratios = {}
    for cid in ClassID:
        r = float(np.sum(mask == cid)) / total
        if r > 0.005:
            ratios[cid] = r
    return ratios

def get_alerts(mask, last_t, now):
    h, w = mask.shape
    total = float(h * w)
    third = w // 3
    alerts = []

    for cid in ClassID:
        cfg = CLASS_CONFIG[cid]
        if cfg["priority"] == 0: continue
        ratio = float(np.sum(mask == cid)) / total
        if ratio < 0.02: continue
        if (now - last_t.get(int(cid), 0.0)) < cfg["cooldown"]: continue

        binary = (mask == cid)
        l = float(np.sum(binary[:, :third]))
        c = float(np.sum(binary[:, third:2*third]))
        r = float(np.sum(binary[:, 2*third:]))

        parts = [cfg["alert"]]
        if ratio > 0.15: parts.append("CLOSE")
        elif ratio > 0.05: parts.append("near")
        best = max(l, c, r)
        if best == l: parts.append("L")
        elif best == r: parts.append("R")

        alerts.append((cfg["priority"], " ".join(parts)))
        last_t[int(cid)] = now

    alerts.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in alerts]

def render_overlay(frame, mask, alpha=0.45):
    h, w = frame.shape[:2]
    mr = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    ov = np.zeros_like(frame)
    for cid, col in CLASS_COLORS_BGR.items():
        ov[mr == cid] = col
    return cv2.addWeighted(frame, 1.0 - alpha, ov, alpha, 0)


# =============================================================================
# Main
# =============================================================================

def run(model_path, cam_id=0, target_fps=8.0, duration=0, gst=False,
        save_every=10, input_h=384, input_w=512):

    # Output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path("field_test_{}".format(ts))
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)

    backend = load_backend(model_path)
    cap = open_cam(cam_id, gst=gst)

    # CSV
    csv_cols = ["frame", "time", "inf_ms", "total_ms", "fps",
                "walkable", "crosswalk", "road", "obstacle",
                "fall_hzrd", "dynamic", "vehicle", "alerts"]
    csv_f = open(str(out / "log.csv"), "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=csv_cols)
    writer.writeheader()

    throttle = 1.0 / target_fps
    last_yield = 0.0
    last_alert_t = {}
    fc = 0
    lats = []
    t_start_session = time.monotonic()

    logger.info("=" * 50)
    logger.info("FIELD TEST STARTED")
    logger.info("Output: {}".format(out.absolute()))
    logger.info("FPS: {} | Save every {}th frame | Duration: {}s".format(
        target_fps, save_every, duration if duration > 0 else "unlimited"))
    logger.info("Ctrl+C to stop")
    logger.info("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue

            now = time.monotonic()
            if (now - last_yield) < throttle: continue
            last_yield = now

            t0 = time.monotonic()
            tensor = preprocess(frame, input_h, input_w)

            ti = time.monotonic()
            logits = backend.predict(tensor)
            inf_ms = (time.monotonic() - ti) * 1000.0

            mask = postprocess(logits)
            ratios = class_ratios(mask)
            timestamp = time.time()
            alerts = get_alerts(mask, last_alert_t, timestamp)

            total_ms = (time.monotonic() - t0) * 1000.0
            fps = 1000.0 / total_ms if total_ms > 0 else 0
            lats.append(total_ms)

            # CSV row
            row = {
                "frame": fc,
                "time": "{:.3f}".format(timestamp),
                "inf_ms": "{:.1f}".format(inf_ms),
                "total_ms": "{:.1f}".format(total_ms),
                "fps": "{:.1f}".format(fps),
                "alerts": " | ".join(alerts),
            }
            for cid in ClassID:
                row[CLASS_NAMES[cid]] = "{:.1f}".format(ratios.get(cid, 0) * 100)
            writer.writerow(row)

            # Save every Nth frame as JPEG (fast, small)
            if fc % save_every == 0:
                overlay = render_overlay(frame, mask)
                # JPEG quality 80 — fast encode, small file
                cv2.imwrite(
                    str(fdir / "f{:05d}.jpg".format(fc)),
                    overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Log alerts to console
            for a in alerts:
                logger.info("[ALERT] {}".format(a))

            # Progress every 50 frames
            if fc > 0 and fc % 50 == 0:
                avg = sum(lats[-50:]) / len(lats[-50:])
                logger.info("Frame {} | avg {:.0f}ms ({:.1f} FPS)".format(
                    fc, avg, 1000.0 / avg))

            fc += 1

            # Duration limit
            if duration > 0 and (time.monotonic() - t_start_session) >= duration:
                logger.info("Duration limit reached ({} sec)".format(duration))
                break

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        cap.release()
        csv_f.close()
        time.sleep(0.3)

        # Summary
        elapsed = time.monotonic() - t_start_session
        lines = [
            "ALAS Field Test Summary",
            "=" * 40,
            "Frames      : {}".format(fc),
            "Duration    : {:.0f} sec".format(elapsed),
            "Model       : {}".format(model_path),
            "Target FPS  : {}".format(target_fps),
            "Saved frames: {} (every {}th)".format(fc // save_every if fc > 0 else 0, save_every),
        ]
        if lats:
            arr = np.array(lats)
            lines.extend([
                "",
                "Latency (ms):",
                "  mean  : {:.1f}".format(np.mean(arr)),
                "  p50   : {:.1f}".format(np.percentile(arr, 50)),
                "  p95   : {:.1f}".format(np.percentile(arr, 95)),
                "  min   : {:.1f}".format(np.min(arr)),
                "  max   : {:.1f}".format(np.max(arr)),
                "  FPS   : {:.1f}".format(1000.0 / np.mean(arr)),
            ])
        lines.extend(["", "Output: {}".format(out.absolute())])

        txt = "\n".join(lines)
        with open(str(out / "summary.txt"), "w") as f:
            f.write(txt)
        print("\n" + txt)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ALAS Field Test (headless)")
    p.add_argument("--model", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--duration", type=int, default=0,
                   help="Max seconds (0=unlimited)")
    p.add_argument("--gstreamer", action="store_true")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save frame every N (default: 10)")
    a = p.parse_args()

    run(model_path=a.model, cam_id=a.camera, target_fps=a.fps,
        duration=a.duration, gst=a.gstreamer, save_every=a.save_every)
