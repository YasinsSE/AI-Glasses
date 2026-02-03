import time
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_models import Frame
from ai.preprocessing import PreprocessingPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR / "image.png"
OUTPUT_PATH = SCRIPT_DIR / "preprocessed.png"

# 1) Read the image (cv2 loads BGR)
bgr = cv2.imread(str(IMAGE_PATH))
if bgr is None:
    raise FileNotFoundError(f"image.png not found: {IMAGE_PATH}")

# 2) Convert to RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# 3) Build the frame
frame = Frame(
    rgb=rgb,
    timestamp=time.time(),
    frame_id=0,
    metadata={"source": "image.png"},
)

# 4) Preprocess
pipeline = PreprocessingPipeline(target_size=(640, 480))
tensor = pipeline.preprocess(frame)
print(tensor.shape)  # (1, 3, 480, 640)

# 5) Save the preprocessed image
preprocessed_rgb = np.transpose(tensor[0], (1, 2, 0))
preprocessed_rgb = (preprocessed_rgb * 255.0).clip(0, 255).astype(np.uint8)
preprocessed_bgr = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUTPUT_PATH), preprocessed_bgr)
print(f"Preprocessed image saved: {OUTPUT_PATH}")
