"""실기 스냅샷 공급자 — Piper Studio 공유 메모리의 D435 컬러·깊이 → SceneSnapshot (설계서 scene_capture_server). 역할 A.

카메라 장치를 직접 열지 않는다 (AGENTS.md). Piper Studio(rsd)가 `/dev/shm` 으로 상시 내보내는 프레임을 읽어
8비트 깊이를 디코딩하고 컬러에 정렬한다 (aple/hal/depth_align.py). 결과는 합성 장면·저장 스냅샷과 같은
SceneSnapshot 이므로 M1·M2 코드는 바꾸지 않고 그대로 쓴다.

    cam = PiperLiveCamera.from_config(load_config())
    snap = cam()                                  # scene_id 가 1씩 증가
    studio = PiperStudio(cfg, snapshot_provider=cam)   # 역할 B 클라이언트에 꽂아 studio.snapshot() 으로도 쓴다

스냅샷 1장은 연속 frames 장의 깊이 중앙값이다 (팔이 멈춘 장면에서 찍으므로 — 전략 S2-6). 픽셀마다
절반 이상의 프레임에서 유효해야 유효로 본다. 컬러는 마지막 프레임을 쓴다.
설정: configs/live_camera.yaml
"""

from __future__ import annotations

import time
import math
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from aple.hal.depth_align import D435Calib, DepthEncoding, snapshot_from_piper
from aple.schema import CaptureMode, SceneSnapshot, Source

ROOT = Path(__file__).resolve().parents[2]

# (컬러 BGR, 인코딩 깊이, wall_ns) 한 쌍 또는 None 을 돌려주는 함수. 기본은 공유 메모리, 시험에서는 가짜.
PairReader = Callable[[], "tuple[np.ndarray, np.ndarray, int] | None"]


class PiperLiveCamera:
    def __init__(self, calib: D435Calib, encoding: DepthEncoding, read_pair: PairReader, *,
                 frames: int = 3, timeout_s: float = 2.0, first_scene_id: int = 1,
                 max_frame_age_s: float = 1.0, wall_clock_ns=time.time_ns) -> None:
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            raise ValueError("frames 는 1 이상")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.calib, self.encoding, self._read_pair = calib, encoding, read_pair
        self.frames, self.timeout_s = int(frames), float(timeout_s)
        self._next_id = int(first_scene_id)
        if not math.isfinite(max_frame_age_s) or max_frame_age_s <= 0:
            raise ValueError("max_frame_age_s must be positive and finite")
        self._max_age_ns = max_frame_age_s * 1e9
        self._wall_clock_ns = wall_clock_ns
        self._last_frame_ns = 0

    @classmethod
    def from_config(cls, cfg: dict, *, shm_dir: Path | None = None) -> "PiperLiveCamera":
        from aple.hal.shm_frames import SHM_DIR, ShmError, ShmReader, find_realsense_serials, read_pair, segment_path

        lc = cfg["live_camera"]
        calib_path = Path(lc["depth_calib"])
        calib_path = calib_path if calib_path.is_absolute() else ROOT / calib_path
        if not calib_path.exists():
            raise FileNotFoundError(
                f"깊이→컬러 보정 파일이 없다: {calib_path}\n  → systemctl --user stop piper-rsd && "
                f"python tools/rs_check.py --save-calib {lc['depth_calib']} && systemctl --user start piper-rsd")
        shm = shm_dir or SHM_DIR
        serial = lc.get("serial") or next(iter(find_realsense_serials(shm)), None)
        if serial is None:
            raise RuntimeError("공유 메모리에 RealSense 영상이 없다 — Piper Studio 에서 D435 를 켰는지 확인")
        try:
            color_r = ShmReader(segment_path(f"rs:{serial}:color", shm))
            depth_r = ShmReader(segment_path(f"rs:{serial}:depth", shm))
        except ShmError as e:
            raise RuntimeError(str(e)) from e
        skew = float(lc.get("max_skew_ms", 50))

        def reader():
            return read_pair(color_r, depth_r, max_skew_ms=skew, timeout_s=0.5)

        return cls(D435Calib.load(calib_path), DepthEncoding(float(lc["near_mm"]), float(lc["far_mm"])), reader,
                   frames=int(lc.get("frames", 3)), timeout_s=float(lc.get("timeout_s", 2.0)),
                   max_frame_age_s=float(lc.get("max_frame_age_s", 1.0)))

    def __call__(self, capture_mode: CaptureMode = CaptureMode.FULL, *, tray_clear: bool = True,
                 capture_after_ns: int | None = None) -> SceneSnapshot:
        # Require exposures after entering this capture, not a cached frame
        # from before the robot stopped. The threshold stays fixed for the batch.
        after_ns = self._wall_clock_ns() if capture_after_ns is None else capture_after_ns
        deadline = time.monotonic() + self.timeout_s
        snaps: list[SceneSnapshot] = []
        last_ns = max(self._last_frame_ns, after_ns)
        while len(snaps) < self.frames:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.timeout_s:.1f} s 안에 컬러·깊이 프레임 {self.frames}장을 받지 못했다 "
                                   f"(받은 수 {len(snaps)}) — Piper Studio 스트림 확인")
            pair = self._read_pair()
            if (pair is None or pair[2] <= last_ns
                    or not 0 <= self._wall_clock_ns() - pair[2] <= self._max_age_ns):
                time.sleep(0.005)
                continue
            color, depth_enc, last_ns = pair
            self._last_frame_ns = last_ns
            snaps.append(snapshot_from_piper(color, depth_enc, self.encoding, self.calib, 0, last_ns / 1e9))

        scene_id, self._next_id = self._next_id, self._next_id + 1
        last = snaps[-1]
        depth = median_depth([s.depth_m for s in snaps])
        return SceneSnapshot(scene_id, Source.TOP, capture_mode, last.color, depth, last.intrinsics,
                             last.stamp_s, tray_clear=tray_clear)


def median_depth(depths: list[np.ndarray]) -> np.ndarray:
    """프레임별 깊이(0 = 결측) → 픽셀 중앙값. 절반 이상 프레임에서 유효해야 유효."""
    if len(depths) == 1:
        return depths[0].astype(np.float32)
    stack = np.stack(depths).astype(np.float32)
    valid = stack > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # 전 프레임 결측 픽셀 (All-NaN slice)
        med = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    ok = valid.sum(axis=0) * 2 >= len(depths)
    return np.where(ok & np.isfinite(med), med, 0.0).astype(np.float32)
