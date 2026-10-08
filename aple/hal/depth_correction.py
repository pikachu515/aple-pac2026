"""D435 깊이 보정 — 거리마다 틀어지는 깊이를 회색 카드(크기를 아는 물체)로 재서 바로잡는다. 역할 A.

10/7 실물 D435i(243122071248) 스냅샷에서 확인한 문제 (#29 '직경이 작게 나온다'의 원인):
  회색 카드(130×100 mm, 10/8 자로 실측) 크기로 잰 실제 거리 0.30 / 0.43 / 0.74 m 에서 깊이는 0.22 / 0.29 / 0.41 m 였다.
  멀수록 더 짧게 나온다 — 0.74 m 에서 45% 짧아 사과 직경·높이가 모두 작게 나왔다.
  깊이 0.41 m 를 믿고 픽셀 크기를 미터로 바꾸니 약 85 mm 사과(카드 기준)가 45~50 mm 가 됐다.

스테레오 깊이는 시차(disparity ∝ 1/z)로 계산하므로, 보정이 틀어진 카메라의 오차는 1/z 에서 선형이다:
    1/z_측정 = gain / z_실제 + offset_per_m        →  z_실제 = gain / (1/z_측정 − offset_per_m)
위 세 거리를 이 식 하나로 1% 안에 맞췄다 (gain 1.038, offset 1.052/m). offset 0 이면 z_실제 = gain · z_측정 (배율만).

근본 대책은 카메라 재보정(RealSense On-Chip 보정 + Tare)이다. 이 보정은 재보정을 못 했을 때의 대체이고,
재보정 뒤에도 tools/depth_check.py 로 카드 거리를 다시 확인한다. 값은 카메라마다 다르다 (live_camera.depth_correction).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from aple.schema import SceneSnapshot


@dataclass(frozen=True)
class DepthCorrection:
    gain: float = 1.0
    offset_per_m: float = 0.0

    @property
    def identity(self) -> bool:
        return self.gain == 1.0 and self.offset_per_m == 0.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "DepthCorrection":
        c = cfg.get("live_camera", {}).get("depth_correction") or {}
        return cls(float(c.get("gain", 1.0)), float(c.get("offset_per_m", 0.0)))

    def apply(self, depth_m: np.ndarray) -> np.ndarray:
        """깊이 지도 보정. 결측(0)은 그대로, 보정식이 무한·음수가 되는 픽셀(너무 먼 값)은 결측으로."""
        if self.identity:
            return depth_m
        d = depth_m.astype(np.float64)
        out = np.zeros_like(d)
        v = d > 0
        den = 1.0 / d[v] - self.offset_per_m
        z = np.where(den > 0, self.gain / np.where(den > 0, den, 1.0), 0.0)
        out[v] = z
        return out.astype(np.float32)

    def snapshot(self, snap: SceneSnapshot) -> SceneSnapshot:
        return snap if self.identity else replace(snap, depth_m=self.apply(snap.depth_m))


def fit_correction(z_depth_m, z_true_m) -> DepthCorrection:
    """(측정 깊이, 실제 거리) 쌍 → 보정. 거리가 한 가지면 배율만, 두 가지 이상(5 cm 넘게 차이)이면 배율 + 오프셋."""
    zm = np.asarray(z_depth_m, float)
    zt = np.asarray(z_true_m, float)
    if len(zm) == 0 or len(zm) != len(zt) or np.any(zm <= 0) or np.any(zt <= 0):
        raise ValueError("측정 깊이·실제 거리가 짝이 맞는 양수여야 한다")
    if np.ptp(zt) < 0.05:
        return DepthCorrection(float(np.median(zt / zm)), 0.0)
    gain, offset = np.polyfit(1 / zt, 1 / zm, 1)  # 1/zm = gain/zt + offset
    return DepthCorrection(float(gain), float(offset))


# ================================================================ 회색 카드로 실제 거리 재기

@dataclass
class CardDistance:
    z_depth_m: float              # 카드 안쪽 깊이 중앙값 (보정 전 측정값)
    z_true_m: float               # 카드 픽셀 크기와 실제 크기로 계산한 거리
    size_px: tuple[float, float]  # 긴 변, 짧은 변 (px)
    center_px: tuple[float, float]
    depth_valid: float            # 카드 안쪽 깊이 유효 비율

    @property
    def error(self) -> float:
        """깊이 상대 오차 (음수 = 짧게 나옴)."""
        return self.z_depth_m / self.z_true_m - 1


def find_card(color_bgr: np.ndarray, *, max_sat: int = 32, v_highs=(150, 170, 190, 215), v_low: int = 60,
              min_area_px: int = 1500, aspect_range=(1.1, 1.6), roi: tuple[int, int, int, int] | None = None):
    """무채색·직사각형 덩어리 중 가장 큰 것 → (minAreaRect, 윤곽). 없으면 None.

    카드 밝기는 조명·노출마다 달라 밝기 상한을 몇 가지로 바꿔 본다 (너무 높으면 흰 종이와 붙어 사각형이 아니게 된다).
    roi(x, y, w, h)를 주면 그 안에 중심이 있는 것만 본다 (color_session 의 card.json roi 를 그대로 쓸 수 있다).
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    best = None
    for v_high in v_highs:
        m = ((hsv[..., 1] < max_sat) & (hsv[..., 2] > v_low) & (hsv[..., 2] < v_high)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        for cn in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cn)
            bx, by, bw, bh = cv2.boundingRect(cn)
            if area < min_area_px or bx == 0 or by == 0 or bx + bw >= m.shape[1] or by + bh >= m.shape[0]:
                continue                                  # 화면에 다 들어온 카드만 (잘리면 크기를 못 믿는다)
            rect = cv2.minAreaRect(cn)
            (cx, cy), (w, h), _ = rect
            if min(w, h) <= 0 or area / (w * h) < 0.9 or not aspect_range[0] < max(w, h) / min(w, h) < aspect_range[1]:
                continue
            if roi is not None:
                x, y, rw, rh = roi
                if not (x <= cx <= x + rw and y <= cy <= y + rh):
                    continue
            if best is None or area > best[2]:
                best = (rect, cn, area)
    return None if best is None else best[:2]


def card_distance(snap: SceneSnapshot, card_mm: tuple[float, float], **find_kw) -> CardDistance | None:
    """스냅샷의 회색 카드 → 측정 깊이와 실제 거리. 카드는 카메라를 정면으로 보게(눕혀) 둔다."""
    found = find_card(snap.color, **find_kw)
    if found is None:
        return None
    (cx, cy), (w, h), _ = found[0]
    mask = np.zeros(snap.depth_m.shape, np.uint8)
    cv2.drawContours(mask, [found[1]], -1, 1, -1)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8)).astype(bool)
    valid = mask & (snap.depth_m > 0)
    if valid.sum() < 50:
        return None
    fx, fy, _, _ = snap.intrinsics
    f = (fx + fy) / 2
    long_px, short_px = max(w, h), min(w, h)
    long_mm, short_mm = max(card_mm), min(card_mm)
    z_true = f * (long_mm / long_px + short_mm / short_px) / 2 / 1000
    return CardDistance(float(np.median(snap.depth_m[valid])), float(z_true), (float(long_px), float(short_px)),
                        (float(cx), float(cy)), float(valid.sum() / max(mask.sum(), 1)))
