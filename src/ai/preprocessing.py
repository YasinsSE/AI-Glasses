import cv2
import numpy as np
from typing import Tuple

from data_models import Frame


class PreprocessingPipeline:
    """Prepare RGB camera frames for the detection model."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)) -> None:
        self.target_size = target_size

    def preprocess(
        self,
        frame: Frame,
        apply_blur: bool = False,
        blur_kernel_size: int = 5,
    ) -> np.ndarray:
        """Preprocess an RGB frame for YOLO-style models.

        Steps:
        1. Resize to target size
        2. Normalize to [0, 1]
        3. Convert HWC -> CHW
        4. Add batch dimension -> NCHW
        """
        img = frame.rgb
        self._validate_input(img)

        img = cv2.resize(img, self.target_size)

        if apply_blur:
            img = self.apply_gaussian_blur(img, kernel_size=blur_kernel_size)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def apply_gaussian_blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Noise reduction (FR-2.2)."""
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def _validate_input(self, img: np.ndarray) -> None:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected HWC image with 3 color channels.")
