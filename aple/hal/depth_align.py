"""Piper Studio(rsd)가 공유 메모리로 내보내는 깊이 → 컬러 시점에 정렬된 미터 깊이.

Piper Studio 코드에서 확인한 사실 (wego-piper-web, rs/piper_rs/hub.py · depth.py)
- 깊이 세그먼트(`rs:<serial>:depth`)는 **컬러에 정렬되지 않은 원본 시점**이다.
  D435 는 깊이(왼쪽 IR)와 컬러 센서가 약 15 mm 떨어져 있어 그대로 겹치면 사과 경계가 어긋난다
- 깊이는 **uint8 3채널로 인코딩**되어 있다: near_mm..far_mm → 0..254 선형(내림), 255 = 무효.
  기본 150..1200 mm 이면 한 단계가 약 4 mm 다. 인코딩 창은 카메라 API로 바꿀 수 있다
- 내부 파라미터는 rsd `intrinsics` RPC 로 얻을 수 있지만 **깊이→컬러 외부 파라미터는 제공하지 않는다**
  → 장치 고유 상수이므로 tools/rs_check.py --save-calib 로 한 번 읽어 configs/calib/ 에 저장해 쓴다

이 모듈은 위 세 가지를 풀어 판정 모듈이 쓰는 SceneSnapshot(컬러 + 정렬 깊이)을 만든다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aple.schema import CaptureMode, SceneSnapshot, Source

INVALID = 255
VALID_MAX = 254


@dataclass(frozen=True)
class DepthEncoding:
    near_mm: float = 150.0
    far_mm: float = 1200.0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def tuple(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    @staticmethod
    def from_dict(d: dict) -> "Intrinsics":
        return Intrinsics(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
                          int(d["width"]), int(d["height"]))


@dataclass(frozen=True)
class D435Calib:
    """깊이·컬러 내부 파라미터와 깊이→컬러 외부 파라미터 (P_color = R · P_depth + t, m)."""
    depth: Intrinsics
    color: Intrinsics
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def load(path: str | Path) -> "D435Calib":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return D435Calib(Intrinsics.from_dict(d["depth"]), Intrinsics.from_dict(d["color"]),
                         np.asarray(d["depth_to_color"]["R"], float).reshape(3, 3),
                         np.asarray(d["depth_to_color"]["t"], float))


def decode_depth(encoded: np.ndarray, enc: DepthEncoding) -> np.ndarray:
    """uint8 인코딩(H×W 또는 H×W×3) → 미터 (float32, 0 = 무효).

    인코더가 소수점을 버리므로(astype) 각 단계의 가운데 값으로 되돌린다. 오차는 반 단계 이내다.
    """
    v = encoded[..., 0] if encoded.ndim == 3 else encoded
    v = v.astype(np.float32)
    span = enc.far_mm - enc.near_mm
    mm = enc.near_mm + (np.minimum(v, VALID_MAX - 1) + 0.5) / VALID_MAX * span
    out = (mm / 1000.0).astype(np.float32)
    out[v >= INVALID] = 0.0
    return out


def align_depth_to_color(depth_m: np.ndarray, calib: D435Calib, fill_holes: bool = True) -> np.ndarray:
    """깊이 시점의 미터 깊이 → 컬러 시점 (컬러 해상도). 같은 컬러 픽셀에 여러 점이 오면 가까운 쪽.

    렌즈 왜곡은 무시한다 (D435 깊이는 무왜곡, 컬러는 작다). 해상도 차이로 생기는 한두 픽셀 구멍은
    주변 최솟값으로 메운다.
    """
    dI, cI = calib.depth, calib.color
    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u].astype(np.float64)
    P = np.stack([(u - dI.cx) / dI.fx * z, (v - dI.cy) / dI.fy * z, z], axis=1)
    Q = P @ calib.R.T + calib.t
    ok = Q[:, 2] > 1e-6
    Q = Q[ok]
    uc = np.round(Q[:, 0] / Q[:, 2] * cI.fx + cI.cx).astype(np.int64)
    vc = np.round(Q[:, 1] / Q[:, 2] * cI.fy + cI.cy).astype(np.int64)
    inside = (uc >= 0) & (uc < cI.width) & (vc >= 0) & (vc < cI.height)
    out = np.full((cI.height, cI.width), np.inf)
    np.minimum.at(out, (vc[inside], uc[inside]), Q[inside, 2])
    if fill_holes:
        filled = cv2.erode(out.astype(np.float32), np.ones((3, 3), np.uint8))
        hole = ~np.isfinite(out) & np.isfinite(filled)
        out[hole] = filled[hole]
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def snapshot_from_piper(color_bgr: np.ndarray, depth_encoded: np.ndarray, enc: DepthEncoding,
                        calib: D435Calib, scene_id: int, stamp_s: float = 0.0) -> SceneSnapshot:
    """Piper Studio 세그먼트 두 장(컬러, 인코딩 깊이) → 판정 모듈 입력."""
    depth_c = align_depth_to_color(decode_depth(depth_encoded, enc), calib)
    return SceneSnapshot(scene_id, Source.TOP, CaptureMode.FULL, color_bgr, depth_c,
                         calib.color.tuple(), stamp_s)
