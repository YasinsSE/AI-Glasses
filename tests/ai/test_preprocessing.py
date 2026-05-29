"""Unit test for the AI preprocessing pipeline (src/ai/preprocessing.py).

Loads a fixture image, runs it through ``PreprocessingPipeline``, asserts the
output tensor shape/range, and saves the preprocessed image for visual
inspection.

    Inputs : tests/fixtures/image.png
    Outputs: outputs/tests/ai/preprocessed.png

How to run (from the repository root):
    python tests/ai/test_preprocessing.py
    pytest tests/ai/test_preprocessing.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make src/ importable whether run via pytest or as a standalone script.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_models import Frame
from ai.preprocessing import PreprocessingPipeline

FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "image.png"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "tests" / "ai"
OUTPUT_PATH = OUTPUT_DIR / "preprocessed.png"

TARGET_SIZE = (640, 480)  # (width, height)


def test_preprocess_shape_and_range() -> None:
    """Preprocessing yields an NCHW float tensor normalised to [0, 1]."""
    bgr = cv2.imread(str(FIXTURE_PATH))
    assert bgr is not None, f"fixture not found: {FIXTURE_PATH}"

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame = Frame(rgb=rgb, timestamp=time.time(), frame_id=0, metadata={"source": "image.png"})

    pipeline = PreprocessingPipeline(target_size=TARGET_SIZE)
    tensor = pipeline.preprocess(frame)

    # (1, 3, H, W) with width/height matching the target size.
    assert tensor.shape == (1, 3, TARGET_SIZE[1], TARGET_SIZE[0]), tensor.shape
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0

    # Save the preprocessed frame for inspection.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessed_rgb = np.transpose(tensor[0], (1, 2, 0))
    preprocessed_rgb = (preprocessed_rgb * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_PATH), cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    test_preprocess_shape_and_range()
    print(f"OK — preprocessing test passed; output saved to {OUTPUT_PATH}")
