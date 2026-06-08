"""Raw-frame dataset capture for offline model fine-tuning.

ALAS's segmentation model was trained on the SANPO dataset (Japanese streets);
on local streets it sometimes confuses vehicle_road with walkable surface and
misguides the user. The cure is domain adaptation — fine-tune the EXISTING model
on a small set of locally captured + labelled frames, not a from-scratch retrain.

This module is the capture half: enabled with ``--capture-dataset``, it saves
clean camera frames (NO colour overlay) at the model input resolution as
lossless PNGs, throttled, plus a manifest. Workflow:

    1.  Walk the test route with ``--capture-dataset`` → frames land in
        ``<dir>/images/``.
    2.  Upload to Roboflow, label as Semantic Segmentation with the 7 ALAS
        classes (walkable_surface, crosswalk, vehicle_road, collision_obstacle,
        fall_hazard, dynamic_hazard, vehicle — see dataset_unet_format.py).
    3.  Export class-ID masks; drop into the images/masks train/val layout that
        ``ai/dataset/dataset_unet_format.py`` produces for SANPO.
    4.  Fine-tune: load the current U-Net weights, low LR, train a few epochs on
        SANPO + local data MIXED (avoids catastrophic forgetting), re-export
        ONNX (opset 12) → rebuild the TensorRT engine on the Jetson.

``--capture-dataset`` may additionally dump the model's PREDICTED mask
(``--capture-masks`` via config) as a label-assist starting point in Roboflow.
"""

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("ALAS.dataset_collector")


class NullCollector:
    """No-op collector used when ``--capture-dataset`` is not set."""
    enabled = False

    def maybe_capture(self, frame_bgr, mask=None, gps=None) -> None:  # noqa: D401
        return

    def close(self) -> None:
        return


class DatasetCollector:
    """Throttled raw-frame saver for building a fine-tuning dataset."""

    def __init__(self, out_dir, interval_s: float = 2.0, save_mask: bool = False) -> None:
        self.enabled = True
        self._dir = Path(out_dir)
        self._img_dir = self._dir / "images"
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._save_mask = save_mask
        self._mask_dir = self._dir / "masks_pred"
        if save_mask:
            self._mask_dir.mkdir(parents=True, exist_ok=True)
        self._interval = max(0.2, float(interval_s))
        self._last = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._manifest = (self._dir / "manifest.jsonl").open("a", encoding="utf-8")
        logger.info(
            "[Dataset] Capturing raw frames to %s every %.1fs (save_mask=%s)",
            self._dir, self._interval, save_mask,
        )

    def maybe_capture(self, frame_bgr, mask=None, gps=None) -> None:
        """Save the raw frame if the throttle interval has elapsed.

        ``frame_bgr`` is the clean camera frame at model resolution (no overlay);
        ``mask`` is the model's predicted class-ID mask (optional, label-assist).
        """
        if frame_bgr is None:
            return
        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now
        try:
            import cv2
            with self._lock:
                self._seq += 1
                seq = self._seq
            name = f"frame_{seq:05d}.png"
            cv2.imwrite(str(self._img_dir / name), frame_bgr)   # raw BGR, lossless
            if self._save_mask and mask is not None:
                cv2.imwrite(str(self._mask_dir / name), mask)   # class-ID 0..6
            rec = {"file": f"images/{name}", "ts": round(time.time(), 2)}
            if gps is not None:
                rec["gps"] = gps
            self._manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._manifest.flush()
        except Exception:  # noqa: BLE001 — capture must never crash the perception loop
            logger.exception("[Dataset] frame capture failed")

    def close(self) -> None:
        try:
            self._manifest.close()
        except Exception:  # noqa: BLE001
            pass


def build_collector(config):
    """Return a DatasetCollector when --capture-dataset is set, else NullCollector."""
    if not getattr(config, "capture_dataset", False):
        return NullCollector()
    return DatasetCollector(
        config.capture_dataset_dir,
        interval_s=config.rec.capture_interval_s,
        save_mask=config.rec.capture_save_mask,
    )
