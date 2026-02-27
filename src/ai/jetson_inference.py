"""
ALAS - Real-Time Inference on Jetson Nano  (v3 - Final)
========================================================
Runs the trained 7-class segmentation model on live camera input.
Generates semantically differentiated TTS alerts per macro class.

Jetson Nano setup (run once after copying the model):
    trtexec --onnx=alas_model.onnx --saveEngine=alas_int8.engine --int8

Run:
    python jetson_inference_v3.py --model alas_int8.engine
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CLASS CONFIGURATION
# 7 macro classes — must match CLASS_NAMES in sanpo_to_yolo_v3.py exactly.
#
# TTS alert design principles:
#   - Collision Obstacle : horizontal reaction  ("stop / turn")
#   - Fall Hazard        : vertical/footing reaction ("slow down / check ground")
#   - Dynamic Hazard     : time-critical ("moving ... nearby")
#   - Vehicle            : emergency stop ("STOP")
#   - Crosswalk          : positive signal ("safe to cross")
#   - Vehicle Road       : avoidance ("road — do not enter")
#   - Walkable Surface   : silent (no alert needed)
# ──────────────────────────────────────────────────────────────────────────────

CLASS_CONFIG: dict[int, dict] = {
    0: {
        "name"    : "walkable_surface",
        "color"   : (0, 200, 0),         # green — safe zone
        "priority": "low",
        "alert"   : None,                # no TTS — silence means safe
    },
    1: {
        "name"    : "crosswalk",
        "color"   : (0, 255, 200),       # cyan-green — safe crossing point
        "priority": "medium",
        "alert"   : "Crosswalk ahead — safe to cross",
    },
    2: {
        "name"    : "vehicle_road",
        "color"   : (30, 30, 180),       # dark red — danger zone
        "priority": "critical",
        "alert"   : "Warning — vehicle road, do not enter",
    },
    3: {
        "name"    : "collision_obstacle",
        "color"   : (0, 140, 255),       # orange — impact hazard
        "priority": "high",
        "alert"   : "Obstacle ahead — stop or turn",
    },
    4: {
        "name"    : "fall_hazard",
        "color"   : (0, 80, 255),        # red-orange — ground hazard
        "priority": "high",
        "alert"   : "Ground hazard — slow down and check footing",
    },
    5: {
        "name"    : "dynamic_hazard",
        "color"   : (100, 100, 255),     # light red — moving entity
        "priority": "high",
        "alert"   : "Moving hazard nearby",
    },
    6: {
        "name"    : "vehicle",
        "color"   : (0, 0, 255),         # pure red — emergency
        "priority": "critical",
        "alert"   : "Stop — vehicle detected",
    },
}

# Seconds before the same class alert can fire again (prevents TTS spam)
ALERT_COOLDOWN: dict[str, float] = {
    "low"     : 999.0,  # never
    "medium"  : 5.0,
    "high"    : 3.0,
    "critical": 2.0,
}

# Overlay transparency (0.0 = invisible, 1.0 = fully opaque)
MASK_ALPHA = 0.42


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ALASInference:

    def __init__(self, model_path: str, conf_threshold: float = 0.40):
        """
        Args:
            model_path      : Path to .pt (PyTorch) or .engine (TensorRT) file.
            conf_threshold  : Minimum confidence threshold for detections.
        """
        print(f"[ALAS] Loading model : {model_path}")
        self.model             = YOLO(model_path)
        self.conf              = conf_threshold
        self.last_alert: dict[int, float] = {}
        print(f"[ALAS] Ready  (conf={conf_threshold}) ✓")

    # ──────────────────────────────────────────────────────────────────────
    # DIRECTION HELPER
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _direction(center_x: float, frame_width: int) -> str:
        """Map bounding-box horizontal center to a spoken direction."""
        rel = center_x / frame_width
        if rel < 0.33:
            return "to your left"
        elif rel > 0.67:
            return "to your right"
        return "directly ahead"

    # ──────────────────────────────────────────────────────────────────────
    # SINGLE FRAME
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Run segmentation inference on one BGR frame.

        Returns:
            annotated  : Frame with colored mask overlays and HUD text.
            tts_alerts : Alert strings ordered by priority (critical first).
                         In production: pass each string to your TTS engine.
        """
        results   = self.model(frame, conf=self.conf, verbose=False)[0]
        annotated = frame.copy()
        alerts: list[tuple[int, str]] = []  # (priority_rank, text)
        now       = time.time()

        if results.masks is None:
            return annotated, []

        h, w = frame.shape[:2]

        priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        for mask_t, box, cls_t in zip(
            results.masks.data,
            results.boxes.xyxy,
            results.boxes.cls.int(),
        ):
            class_id = cls_t.item()
            cfg      = CLASS_CONFIG.get(class_id)
            if cfg is None:
                continue

            # --- Segmentation overlay ---
            mask_np    = mask_t.cpu().numpy()
            mask_sized = cv2.resize(mask_np, (w, h))
            mask_bool  = mask_sized > 0.5

            overlay            = np.zeros_like(frame, dtype=np.uint8)
            overlay[mask_bool] = cfg["color"]
            annotated          = cv2.addWeighted(annotated, 1.0, overlay, MASK_ALPHA, 0)

            # --- TTS alert logic ---
            priority   = cfg["priority"]
            alert_text = cfg["alert"]
            cooldown   = ALERT_COOLDOWN[priority]

            if alert_text and (now - self.last_alert.get(class_id, 0)) > cooldown:
                x1, y1, x2, y2 = box.int().tolist()
                direction       = self._direction((x1 + x2) / 2.0, w)

                # Collision and fall hazards include direction; others are general
                if priority in ("high", "critical"):
                    full_alert = f"{alert_text}, {direction}"
                else:
                    full_alert = alert_text

                alerts.append((priority_rank[priority], full_alert))
                self.last_alert[class_id] = now

        # Sort by priority rank (critical first) and return text only
        alerts.sort(key=lambda x: x[0])
        return annotated, [text for _, text in alerts]

    # ──────────────────────────────────────────────────────────────────────
    # CAMERA LOOP
    # ──────────────────────────────────────────────────────────────────────

    def run_camera(self, camera_index: int = 0) -> None:
        """
        Start real-time inference from a camera stream.
        Press 'q' to quit.
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera index {camera_index}")
            return

        fps_buf: list[float] = []
        print(f"[ALAS] Camera started (index={camera_index}). Press 'q' to quit.")

        while True:
            t0       = time.time()
            ret, frame = cap.read()

            if not ret:
                print("[WARN] Frame grab failed — skipping")
                continue

            annotated, tts_alerts = self.process_frame(frame)

            # FPS
            fps = 1.0 / max(time.time() - t0, 1e-6)
            fps_buf.append(fps)
            if len(fps_buf) > 30:
                fps_buf.pop(0)

            # HUD
            cv2.putText(annotated, f"FPS: {np.mean(fps_buf):.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            for i, alert in enumerate(tts_alerts):
                # ── Production: replace print() with TTS engine call ──────
                print(f"[TTS] {alert}")
                # e.g. tts_engine.say(alert)
                # ─────────────────────────────────────────────────────────
                cv2.putText(annotated, alert,
                            (10, 62 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 60, 255), 2)

            cv2.imshow("ALAS — AI Glasses", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[ALAS] Session ended. Average FPS: {np.mean(fps_buf):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ALAS — Jetson Nano Real-Time Inference (7-class segmentation)"
    )
    parser.add_argument("--model",  default="alas_model.pt",
                        help="Model file (.pt or TensorRT .engine)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--conf",   type=float, default=0.40,
                        help="Detection confidence threshold (default: 0.40)")

    args  = parser.parse_args()
    alas  = ALASInference(model_path=args.model, conf_threshold=args.conf)
    alas.run_camera(camera_index=args.camera)