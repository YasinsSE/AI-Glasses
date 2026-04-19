# =============================================================================
# ALAS — Single Image Segmentation Test
# =============================================================================
# Runs segmentation on a single image file and displays the result.
#
# Usage:
#   python src/ai/test_single_image.py --model models/segmentation/alas_engine.trt --image test.jpg
#   python src/ai/test_single_image.py --model models/segmentation/alas_model.onnx --image test.jpg
#   python src/ai/test_single_image.py --model models/segmentation/alas_engine.trt --image test.jpg --save
# =============================================================================

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
logger = logging.getLogger("alas_img")


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
    ClassID.WALKABLE_SURFACE:   {"priority": 0, "alert": None},
    ClassID.CROSSWALK:          {"priority": 1, "alert": "Crosswalk"},
    ClassID.VEHICLE_ROAD:       {"priority": 2, "alert": "Road ahead"},
    ClassID.COLLISION_OBSTACLE: {"priority": 3, "alert": "Obstacle"},
    ClassID.FALL_HAZARD:        {"priority": 3, "alert": "Fall hazard"},
    ClassID.DYNAMIC_HAZARD:     {"priority": 4, "alert": "Moving hazard"},
    ClassID.VEHICLE:            {"priority": 5, "alert": "Vehicle!"},
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
# Processing
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
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(frame_bgr)
    for cid, color in CLASS_COLORS_BGR.items():
        overlay[mask_resized == cid] = color
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


def render_color_mask(mask, height, width):
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS_BGR.items():
        color_mask[mask_resized == cid] = color
    return color_mask


def analyze_scene(mask):
    """Analyze mask and return per-class stats + alerts."""
    h, w = mask.shape
    total = float(h * w)
    third = w // 3
    results = []

    for cid in ClassID:
        binary = (mask == cid)
        ratio = float(np.sum(binary)) / total
        if ratio < 0.01:
            continue

        left   = float(np.sum(binary[:, :third]))
        center = float(np.sum(binary[:, third:2*third]))
        right  = float(np.sum(binary[:, 2*third:]))
        best = max(left, center, right)

        if best == left:
            zone = "LEFT"
        elif best == right:
            zone = "RIGHT"
        else:
            zone = "CENTER"

        results.append({
            "class": CLASS_NAMES[cid],
            "ratio": ratio,
            "zone": zone,
            "priority": CLASS_CONFIG[cid]["priority"],
            "alert": CLASS_CONFIG[cid]["alert"],
        })

    results.sort(key=lambda x: x["priority"], reverse=True)
    return results


def draw_info(frame, inf_ms, scene_results):
    """Draw inference info, class stats, alerts and legend."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, "Inference: {:.1f}ms".format(inf_ms),
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Alerts + stats (right side)
    y = 60
    for r in scene_results:
        if r["priority"] == 0:
            continue
        text = "{} {:.0f}% {}".format(r["alert"], r["ratio"] * 100, r["zone"])
        color = (0, 0, 255) if r["priority"] >= 3 else (0, 180, 255)
        cv2.putText(frame, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 24

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
# Main
# =============================================================================

def run_image(model_path, image_path, input_h=384, input_w=512, save=False):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Cannot read image: {}".format(image_path))
        sys.exit(1)

    logger.info("Image loaded: {} ({}x{})".format(image_path, img.shape[1], img.shape[0]))

    # Load model
    backend = load_backend(model_path)

    # Warmup (first inference is slow on TensorRT)
    logger.info("Warmup inference...")
    dummy = preprocess(img, input_h, input_w)
    backend.predict(dummy)

    # Real inference
    tensor = preprocess(img, input_h, input_w)
    t0 = time.monotonic()
    logits = backend.predict(tensor)
    inf_ms = (time.monotonic() - t0) * 1000.0

    mask = postprocess(logits)
    logger.info("Inference: {:.1f}ms".format(inf_ms))

    # Analyze
    scene = analyze_scene(mask)
    logger.info("--- Scene Analysis ---")
    for r in scene:
        logger.info("  {:16s} {:5.1f}%  zone={}  priority={}".format(
            r["class"], r["ratio"] * 100, r["zone"], r["priority"]))

    # Render views
    overlay = render_overlay(img, mask)
    color_mask = render_color_mask(mask, img.shape[0], img.shape[1])

    # Add HUD to overlay
    overlay = draw_info(overlay, inf_ms, scene)

    # Side-by-side: original | overlay | color mask
    # Resize all to same height
    target_h = 480
    def resize_to_h(frame, th):
        ratio = th / float(frame.shape[0])
        tw = int(frame.shape[1] * ratio)
        return cv2.resize(frame, (tw, th))

    img_r = resize_to_h(img, target_h)
    overlay_r = resize_to_h(overlay, target_h)
    mask_r = resize_to_h(color_mask, target_h)

    combined = np.hstack([img_r, overlay_r, mask_r])

    # Save if requested
    if save:
        stem = Path(image_path).stem
        out_dir = Path("outputs/segmentation_samples")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_overlay = str(out_dir / "result_{}_overlay.jpg".format(stem))
        out_mask = str(out_dir / "result_{}_mask.png".format(stem))
        out_combined = str(out_dir / "result_{}_combined.jpg".format(stem))

        cv2.imwrite(out_overlay, overlay)
        cv2.imwrite(out_mask, mask)
        cv2.imwrite(out_combined, combined)
        logger.info("Saved: {}, {}, {}".format(out_overlay, out_mask, out_combined))

    # Display
    cv2.imshow("ALAS: Original | Overlay | Mask", combined)
    logger.info("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ALAS -- Image Segmentation")
    parser.add_argument("--model", required=True,
                        help="Model file (.onnx or .trt/.engine)")
    
    parser.add_argument("--images", nargs='+', required=False,
                        help="Input image paths",
                        default=[
                            "/home/alas/ALAS_PROJECT/AI-Glasses/tests/ai_test/test1.jpeg",
                            "/home/alas/ALAS_PROJECT/AI-Glasses/tests/ai_test/test2.jpeg",
                            "/home/alas/ALAS_PROJECT/AI-Glasses/tests/ai_test/test3.jpeg",
                            "/home/alas/ALAS_PROJECT/AI-Glasses/tests/ai_test/30a1b8f6-e058-4b11-a5b7-c958b323393f.jpeg"
                        ])
    parser.add_argument("--save", action="store_true",
                        help="Save overlay, mask and combined output")

    args = parser.parse_args()

    for img_path in args.images:
        print(f"\n[INFO] İşleniyor: {img_path}")
        run_image(
            model_path=args.model,
            image_path=img_path,
            save=args.save,
        )