"""저장한 D435 스냅샷(.npz)을 SceneSnapshot 으로 읽는다 — 로봇·카메라 없이 M1·M2 개발용.

tools/capture_snapshot.py 로 저장한 파일을 쓴다. 실제 운용에서 PiperStudio.snapshot 이 돌려주는 것과
같은 SceneSnapshot 이므로, 이 데이터로 만든 코드는 그대로 현장에서 동작한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from aple.schema import CaptureMode, SceneSnapshot, Source


def load_snapshot(path: str | Path, scene_id: int = 0) -> SceneSnapshot:
    with np.load(path) as z:
        return SceneSnapshot(
            scene_id=scene_id,
            source=Source.TOP,
            capture_mode=CaptureMode.FULL,
            color=z["color"],
            depth_m=z["depth_m"],
            intrinsics=tuple(float(v) for v in z["intrinsics"]),
            stamp_s=float(z["stamp_s"]) if "stamp_s" in z.files else 0.0,   # 현장 grab 스냅샷에는 없을 수 있다
        )


def iter_snapshots(folder: str | Path) -> Iterator[SceneSnapshot]:
    for i, p in enumerate(sorted(Path(folder).glob("*.npz")), start=1):
        yield load_snapshot(p, scene_id=i)
