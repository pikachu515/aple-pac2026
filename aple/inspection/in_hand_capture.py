"""Internal mode-B capture payload; does not alter the shared schema contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aple.schema import SceneSnapshot


@dataclass(frozen=True)
class InHandCapture:
    snapshot: SceneSnapshot
    mask: np.ndarray
    rotation_camera_apple: np.ndarray
