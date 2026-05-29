# ALAS Evaluation & Demo Scripts

This directory holds **demos, benchmarks, and inspection tools** — interactive or
model/camera-driven scripts used to evaluate the system by hand. They are kept
separate from [`../tests/`](../tests), which contains only assertion-based unit
tests. The layout mirrors `src/`.

```
eval/
├── ai/
│   ├── field_test.py                 # Headless outdoor field test -> CSV + frames
│   ├── seg_inference_eval.py         # Live segmentation viewer (camera/gstreamer)
│   ├── image_seg_demo.py             # Single-image segmentation demo
│   ├── guidance_from_overlay_eval.py # Path-guidance replay / camera eval
│   └── samples/                      # Sample input images
├── navigation/
│   └── local_planner/
│       ├── vfh_demo.py               # VFH planner demo (image/video/camera/mask)
│       └── vfh_visualizer.py         # Overlay rendering helper for vfh_demo
└── tts_stt/
    ├── slmprepare.py                 # Build SLM fine-tuning data (train/valid jsonl)
    └── slmtester.py                  # Interactive SLM intent-classifier tester
```

## Conventions

- Run every script **from the repository root**; each script's docstring shows
  the exact `python eval/...` command(s) on a single line.
- Scripts locate `src/` and the repository root automatically, so they work
  regardless of the current working directory.
- **Artifacts** are written under `outputs/eval/<module>/`, e.g.
  `outputs/eval/ai/field_tests/<timestamp>/`,
  `outputs/eval/ai/segmentation_samples/`,
  `outputs/eval/navigation/local_planner/`.

Most AI/VFH scripts require a segmentation model (`.onnx` or `.trt/.engine`) and
a camera or sample image; the SLM scripts require the MLX stack (Apple Silicon).
