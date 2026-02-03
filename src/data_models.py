"""Shared data models for the AI/Perception pipeline.

These dataclasses define the contract between camera, perception, and navigation
modules. Keep them stable and versioned together with the rest of the codebase.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==================== ENUMS ====================
class ObstacleType(Enum):
    """Detected object types."""

    PERSON = "person"
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    SIDEWALK = "sidewalk"
    ROAD = "road"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Risk level for the user."""

    SAFE = 0
    WARNING = 1
    DANGER = 2


# ==================== DATA CLASSES ====================
@dataclass
class Frame:
    """Raw camera frame data."""

    rgb: np.ndarray  # (H, W, 3) RGB image
    timestamp: float  # Unix timestamp
    frame_id: int  # Sequential frame number
    metadata: Optional[Dict[str, Any]]  # Extra info (ISO, exposure, etc.)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BoundingBox:
    """Object location (YOLO output format)."""

    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y
    confidence: float  # Confidence score in [0, 1]

    def get_center(self) -> Tuple[float, float]:
        """Compute center point."""

        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def get_area(self) -> float:
        """Compute area."""

        width = max(0.0, self.x2 - self.x1)
        height = max(0.0, self.y2 - self.y1)
        return width * height

    def to_dict(self) -> Dict[str, float]:
        """Convert to a JSON/logging friendly dict."""

        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "confidence": float(self.confidence),
        }


@dataclass
class ObstacleDescriptor:
    """Perception output entry sent to Navigation."""

    object_type: ObstacleType  # Object type
    bounding_box: BoundingBox  # Location
    distance: float  # Distance in meters
    direction: str  # "left", "center", "right"
    severity: SeverityLevel  # Risk level
    timestamp: float  # Detection time
    confidence: float  # Detection confidence [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON/logging friendly dict."""

        return {
            "type": self.object_type.value,
            "distance": round(self.distance, 2),
            "direction": self.direction,
            "severity": self.severity.name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bounding_box.to_dict(),
        }

    def get_tts_message(self) -> Optional[str]:
        """Create a simple TTS message for user feedback."""

        if self.severity == SeverityLevel.SAFE:
            return None

        object_label = self.object_type.value.replace("_", " ")
        if self.severity == SeverityLevel.DANGER:
            return f"Danger: {object_label} on your {self.direction}."

        if self.severity == SeverityLevel.WARNING:
            return f"Caution: {object_label} on your {self.direction}."

        return None


@dataclass
class PerceptionOutput:
    """Complete analysis result for a single frame."""

    frame_id: int
    timestamp: float
    obstacles: List[ObstacleDescriptor]
    processing_time_ms: float
    fps: float

    def get_critical_obstacles(self) -> List[ObstacleDescriptor]:
        """Return only dangerous obstacles."""

        return [
            obs
            for obs in self.obstacles
            if obs.severity == SeverityLevel.DANGER
        ]


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str
    input_size: Tuple[int, int]  # (width, height)
    model_format: str = "onnx"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100


# ==================== MOCK GENERATORS ====================

def generate_mock_frame(
    width: int = 640,
    height: int = 480,
    frame_id: int = 0,
) -> Frame:
    """Generate a fake frame for testing."""

    import time

    return Frame(
        rgb=np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        timestamp=time.time(),
        frame_id=frame_id,
        metadata={"source": "mock"},
    )


def generate_mock_obstacle() -> ObstacleDescriptor:
    """Generate a fake obstacle for testing."""

    import time

    return ObstacleDescriptor(
        object_type=ObstacleType.PERSON,
        bounding_box=BoundingBox(100, 100, 200, 300, 0.95),
        distance=2.5,
        direction="center",
        severity=SeverityLevel.WARNING,
        timestamp=time.time(),
        confidence=0.95,
    )
