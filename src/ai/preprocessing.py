import cv2
import numpy as np
from typing import Tuple
from data_models import Frame

class PreprocessingPipeline:
    """
    Kamera görüntüsünü model için hazırlar
    Gereksinimler: NFR-P5 - Pipeline latency < 20ms
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        self.target_size = target_size
        
    def preprocess(self, frame: Frame) -> np.ndarray:
        """
        YOLO için görüntü hazırla
        
        Args:
            frame: Ham RGB frame
            
        Returns:
            Preprocessed numpy array (1, 3, H, W) - NCHW format
        """
        img = frame.rgb.copy()
        
        # 1. Resize
        img = cv2.resize(img, self.target_size)
        
        # 2. Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 3. BGR to RGB (cv2 uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 4. HWC to CHW (YOLO format)
        img = np.transpose(img, (2, 0, 1))
        
        # 5. Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def apply_gaussian_blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Noise reduction (FR-2.2)"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)