"""Data structures emitted by the VFH local planner."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class VFHAction(Enum):
    """Discrete steering hints derived from the chosen open sector."""
    STRAIGHT     = "straight"
    LEFT_SLIGHT  = "left_slight"
    LEFT         = "left"
    RIGHT_SLIGHT = "right_slight"
    RIGHT        = "right"
    STOP         = "stop"


@dataclass
class VFHGuidance:
    """Single-frame VFH result, ready for the dispatcher to speak / log."""
    action: VFHAction
    sector_index: int          # 0 .. num_sectors-1, -1 when STOP.
    blocked_center: bool       # Centre sector cost > threshold.
    text: str                  # Turkish TTS line.
    histogram: List[float] = field(default_factory=list)   # Per-sector cost, debug only.
