#!/usr/bin/env bash
# ALAS — TensorRT engine build + verification (run ON the Jetson Nano).
#
#   bash scripts/build_trt_engine.sh check                # is the current engine fast?
#   bash scripts/build_trt_engine.sh build                # rebuild from ONNX with FP16
#   bash scripts/build_trt_engine.sh bench NEW_ENGINE.trt # compare a candidate engine
#
# Why: a .trt engine silently keeps whatever precision it was built with. If
# alas_engine.trt was exported without --fp16, the Nano runs the U-Net in FP32
# and leaves a ~1.8x speedup on the table (the Nano's Maxwell GPU has double-
# rate FP16). trtexec prints per-layer precision and a latency benchmark, so
# "check" answers the question definitively; "build" produces an FP16 engine
# from the committed ONNX. (No INT8: the Nano has no DLA and INT8 calibration
# on Maxwell rarely beats FP16 for a model this small.)

set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="$REPO/models/segmentation/alas_model.onnx"
ENGINE="$REPO/models/segmentation/alas_engine.trt"

if [ ! -x "$TRTEXEC" ]; then
    echo "trtexec not found at $TRTEXEC — run this on the Jetson (JetPack installs it)." >&2
    exit 1
fi

case "${1:-check}" in
    check)
        echo "── Benchmarking current engine: $ENGINE"
        "$TRTEXEC" --loadEngine="$ENGINE" --iterations=50 --avgRuns=50 \
            --dumpProfile 2>&1 | tee /tmp/alas_trt_check.log | \
            grep -E "mean|median|percentile|Throughput|GPU Compute" || true
        echo
        echo "Yorum: 512x384 U-Net icin Nano'da FP16 ~150-180 ms beklenir."
        echo "250 ms+ goruyorsaniz engine buyuk ihtimalle FP32 — 'build' calistirin."
        ;;
    build)
        OUT="${2:-$REPO/models/segmentation/alas_engine_fp16.trt}"
        echo "── Building FP16 engine: $ONNX → $OUT"
        "$TRTEXEC" --onnx="$ONNX" --fp16 --saveEngine="$OUT" \
            --workspace=1024 --iterations=50 --avgRuns=50
        echo
        echo "Dogrulama: ayni kareler uzerinde eski/yeni maskeleri karsilastirin:"
        echo "  python3 eval/ai/image_seg_demo.py --model $OUT eval/ai/samples/..."
        echo "Sonuc iyiyse: mv '$OUT' '$ENGINE'"
        ;;
    bench)
        CAND="${2:?usage: build_trt_engine.sh bench ENGINE.trt}"
        echo "── Benchmarking candidate: $CAND"
        "$TRTEXEC" --loadEngine="$CAND" --iterations=50 --avgRuns=50 \
            2>&1 | grep -E "mean|median|Throughput|GPU Compute" || true
        ;;
    *)
        echo "usage: $0 {check|build [OUT.trt]|bench ENGINE.trt}" >&2
        exit 1
        ;;
esac
