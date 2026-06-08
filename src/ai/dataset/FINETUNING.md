# Local fine-tuning (domain adaptation) recipe

The segmentation model is trained on **SANPO** (Japanese streets). On local
streets it sometimes confuses `vehicle_road` with `walkable_surface` and
misguides the user. Fix it by **fine-tuning the existing model** on a small set
of locally captured + labelled frames — *not* a from-scratch retrain.

## 1. Capture frames on the device
```
python -m main.alas_main --model models/segmentation/alas_engine.trt \
    --capture-dataset                # raw frames → outputs/dataset_raw/images/
    # add --capture-masks            # also dump predicted masks (label-assist)
```
Walk the routes that confuse the model (roads, crossings, the sidewalk/road
boundary). Frames are clean 512×384 PNGs (no overlay), throttled to ~1 every 2 s
(`RecorderConfig.capture_interval_s`).

## 2. Label on Roboflow
- New **Semantic Segmentation** project. Upload `outputs/dataset_raw/images/`.
- Classes = the 7 ALAS macro classes (order matters — match the IDs):
  `0 walkable_surface, 1 crosswalk, 2 vehicle_road, 3 collision_obstacle,
   4 fall_hazard, 5 dynamic_hazard, 6 vehicle`.
- (Optional) import `masks_pred/` as label-assist to pre-fill, then correct —
  pay special attention to the road↔sidewalk boundary that fails today.
- Export as **semantic masks (PNG, class-ID)**.

## 3. Merge into the training layout
Match what `dataset_unet_format.py` produces for SANPO:
```
images/train/*.png   masks/train/*.png   (class-ID 0–6, 255 = ignore)
images/val/*.png     masks/val/*.png
```
Put the Roboflow export into this structure (keep an 85/15 train/val split).

## 4. Fine-tune (transfer learning)
In the original U-Net training environment (Colab/PyTorch — not in this repo):
- **Load the existing checkpoint** (don't reinitialise).
- Train on **SANPO + local data MIXED** so the model adapts to local streets
  without catastrophic forgetting. Oversample the local frames (e.g. 3–5×).
- **Low LR** (e.g. 1e-4 or lower), freeze the encoder for the first epochs, then
  unfreeze with a small LR; class-weighted cross-entropy (same weighting as the
  SANPO run — walkable dominates pixel area).
- A few epochs is usually enough for domain adaptation; watch val mIoU,
  especially `vehicle_road` vs `walkable_surface`.

## 5. Re-deploy
- Export ONNX **opset 12** (matches `alas_model_opset12.onnx`).
- On the Jetson, rebuild the TensorRT FP16 engine (`alas_engine.trt`).
- Re-run a field test; the road↔sidewalk confusion should drop.

> Until the model is re-trained, the runtime keeps conservative safety wording
> (road ahead → "girmeyin"; crossings are cautionary, never "safe to cross").
