"""레퍼런스 색 패치 조건부 보정 (전략 S3-5). 역할 C.

회색 카드 게인은 채널마다 곱하는 대각 보정이라, 조명 스펙트럼이 바뀌어 채널이 섞이는 변화는 못 되돌린다.
장면에 적·황·백 패치(+ 회색 카드)를 두고, 보정 시점의 패치 색(선형 RGB)을 기준으로 저장한 뒤
조명 감시가 틀어짐을 잡았을 때만 "지금 패치 색 → 기준 패치 색" 3×3 변환을 구해 영상 전체에 적용한다.
산업 측색에서 기준 색 타일을 함께 재는 것과 같은 원리다 (컬러체커 보정).

  평상시(ΔE ≤ warn)  적용하지 않는다 — 고정 기준의 단순성 유지
  틀어짐 탐지        3×3 변환 → 보정 후 패치 ΔE 가 warn 이하이면 보정 영상으로 판정 (상태 CORRECTING)
  보정 불가          패치가 가려졌거나 보정 후에도 어긋나면 FAULT 그대로 (판정 보류)

패치 기준값은 tools/color_session.py patches 로 만든다 (patch.json, 회색 카드와 같은 세션 사진).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aple.inspection.color import _roi_pixels, bgr_to_linear_rgb, linear_rgb_to_lab, linear_to_srgb

Roi = tuple[int, int, int, int]


@dataclass(frozen=True)
class PatchCalib:
    """패치 이름 → 영역(x, y, w, h)과 보정 시점의 선형 RGB 중앙값."""
    rois: dict[str, Roi]
    ref_lin: dict[str, tuple[float, float, float]]

    def to_dict(self) -> dict:
        return {name: {"roi": list(self.rois[name]), "ref_lin": list(self.ref_lin[name])} for name in self.rois}

    @staticmethod
    def from_dict(d: dict) -> "PatchCalib":
        return PatchCalib({k: tuple(int(x) for x in v["roi"]) for k, v in d.items()},
                          {k: tuple(float(x) for x in v["ref_lin"]) for k, v in d.items()})


def _patch_lin(lin_img: np.ndarray, roi: Roi, min_valid: int = 50) -> np.ndarray | None:
    px = _roi_pixels(lin_img, roi)
    px = px[(px.max(axis=1) < 0.98) & (px.min(axis=1) > 0.005)]          # 포화·암부 제외
    return np.median(px, axis=0) if len(px) >= min_valid else None


def calibrate_patches(bgr: np.ndarray, rois: dict[str, Roi]) -> PatchCalib:
    if len(rois) < 3:
        raise ValueError("3×3 보정에는 패치가 3개 이상 필요하다 (적·황·백 + 회색 카드 권장)")
    lin = bgr_to_linear_rgb(bgr)
    ref = {}
    for name, roi in rois.items():
        m = _patch_lin(lin, roi)
        if m is None:
            raise ValueError(f"패치 '{name}' 영역에 유효 픽셀이 부족하다 — 영역 위치 또는 노출 확인")
        ref[name] = tuple(float(v) for v in m)
    return PatchCalib({k: tuple(int(x) for x in v) for k, v in rois.items()}, ref)


def patch_correction(bgr: np.ndarray, calib: PatchCalib) -> np.ndarray | None:
    """지금 패치 색 → 기준 패치 색 3×3 선형 변환 M (보정 = lin @ M.T). 패치가 하나라도 안 보이면 None."""
    lin = bgr_to_linear_rgb(bgr)
    cur, ref = [], []
    for name, roi in calib.rois.items():
        m = _patch_lin(lin, roi)
        if m is None:
            return None
        cur.append(m)
        ref.append(calib.ref_lin[name])
    C, R = np.asarray(cur), np.asarray(ref)
    if np.linalg.matrix_rank(C, tol=1e-4) < 3:
        return None
    Mt, *_ = np.linalg.lstsq(C, R, rcond=None)                          # C @ M.T ≈ R
    return Mt.T


def apply_matrix(bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    lin = np.clip(bgr_to_linear_rgb(bgr) @ np.asarray(M).T, 0.0, 1.0)
    return (linear_to_srgb(lin)[..., ::-1] * 255.0 + 0.5).astype(np.uint8)


def patch_delta_e(bgr: np.ndarray, calib: PatchCalib) -> float:
    """패치들의 기준 대비 최대 ΔE*ab. 패치가 안 보이면 inf."""
    lin = bgr_to_linear_rgb(bgr)
    worst = 0.0
    for name, roi in calib.rois.items():
        m = _patch_lin(lin, roi)
        if m is None:
            return float("inf")
        de = float(np.linalg.norm(linear_rgb_to_lab(m) - linear_rgb_to_lab(np.asarray(calib.ref_lin[name]))))
        worst = max(worst, de)
    return worst
