"""색 공간 변환, 회색 카드 보정, 조명 감시 (설계서 camera_color_setup · color_normalizer · lighting_monitor).

Piper Studio 회색 카드 보정을 쓰지 못하는 경우(Piper Studio 밖에서 D435를 직접 열어 촬영할 때)를 위한
같은 기능의 소프트웨어 구현이다. 판정 모듈은 보정된 영상만 쓴다.

- 입력 영상: H×W×3 uint8 BGR (D435 bgr8, OpenCV 기본)
- 채널 게인은 선형 RGB 공간에서 곱한다 (sRGB 감마 공간에서 곱하면 밝기에 따라 색이 틀어진다)
- 18% 회색 카드의 sRGB 값 ≈ 118/255 ≈ 선형 0.18
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aple.schema import LightingStatus, LightState

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def bgr_to_linear_rgb(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR → float 선형 RGB (0~1). 마지막 축이 R, G, B."""
    return srgb_to_linear(np.asarray(bgr)[..., ::-1] / 255.0)


def linear_rgb_to_lab(rgb_lin: np.ndarray) -> np.ndarray:
    """선형 RGB → CIE L*a*b* (D65). 마지막 축이 L, a, b."""
    xyz = rgb_lin @ _M_RGB2XYZ.T / _WHITE_D65
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    return linear_rgb_to_lab(bgr_to_linear_rgb(bgr))


def hue_chroma(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """색상각 h (도, -180~180, 0 = 적색 방향, 90 = 황색 방향)와 채도 C*."""
    a, b = lab[..., 1], lab[..., 2]
    return np.degrees(np.arctan2(b, a)), np.hypot(a, b)


def colored_mask(lab: np.ndarray, hue_range_deg: tuple[float, float], min_chroma: float) -> np.ndarray:
    """착색 픽셀: 색상각이 범위 안이고 채도가 하한 이상."""
    h, c = hue_chroma(lab)
    return (h >= hue_range_deg[0]) & (h <= hue_range_deg[1]) & (c >= min_chroma)


# ---------------------------------------------------------------- 회색 카드

TARGET_GRAY_SRGB = 118.0


@dataclass
class CardCalib:
    """회색 카드 보정 결과. 같은 조명·같은 카메라 설정이면 계속 재사용한다."""
    roi: tuple[int, int, int, int]       # x, y, w, h (px)
    rgb_gain: tuple[float, float, float] # 선형 공간 채널 게인
    card_ref_lab: tuple[float, float, float]   # 보정 전 카드 L*a*b* (조명 감시 기준)

    def to_dict(self) -> dict:
        return {"roi": list(self.roi), "rgb_gain": list(self.rgb_gain), "card_ref_lab": list(self.card_ref_lab)}

    @staticmethod
    def from_dict(d: dict) -> "CardCalib":
        return CardCalib(tuple(d["roi"]), tuple(d["rgb_gain"]), tuple(d["card_ref_lab"]))


def _roi_pixels(img: np.ndarray, roi: tuple[int, int, int, int], trim: float = 0.15) -> np.ndarray:
    """ROI 가장자리를 trim 비율만큼 잘라낸 안쪽 픽셀 (카드 테두리·그림자 제외)."""
    x, y, w, h = roi
    dx, dy = int(w * trim), int(h * trim)
    return img[y + dy:y + h - dy, x + dx:x + w - dx].reshape(-1, img.shape[-1])


def calibrate_card(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> CardCalib:
    """카드가 R=G=B 이면서 밝기가 18% 회색(sRGB ≈ 118)이 되게 하는 선형 채널 게인을 구한다."""
    lin = _roi_pixels(bgr_to_linear_rgb(bgr), roi)
    lin = lin[(lin.max(axis=1) < 0.98) & (lin.min(axis=1) > 0.005)]   # 포화·암부 제외
    if len(lin) < 50:
        raise ValueError("카드 영역에 유효 픽셀이 부족하다 — ROI 위치 또는 노출 확인")
    mean = np.median(lin, axis=0)
    target = float(srgb_to_linear(TARGET_GRAY_SRGB / 255.0))
    gain = target / mean
    ref = linear_rgb_to_lab(mean)
    return CardCalib(tuple(int(v) for v in roi), tuple(float(g) for g in gain), tuple(float(v) for v in ref))


def apply_gain(bgr: np.ndarray, calib: CardCalib | None) -> np.ndarray:
    """보정 영상 (uint8 BGR). calib 이 없으면 원본을 그대로 돌려준다."""
    if calib is None:
        return bgr
    lin = bgr_to_linear_rgb(bgr) * np.asarray(calib.rgb_gain)
    return (linear_to_srgb(lin)[..., ::-1] * 255.0 + 0.5).astype(np.uint8)


def card_brightness(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """카드 영역 sRGB 평균 밝기 (0~255). 노출을 맞출 때 118 근처가 목표."""
    return float(np.median(_roi_pixels(bgr.astype(np.float64), roi)))


def lighting_status(bgr_raw: np.ndarray, calib: CardCalib, warn: float = 2.5, fault: float = 6.0) -> LightingStatus:
    """보정 시점 대비 카드 색 변화(ΔE*ab, CIE76)로 조명 변화를 판정한다.

    설계서는 ΔE00 을 쓰지만, 같은 카드의 작은 변화를 보는 용도에서는 CIE76 으로 충분하다.
    """
    lin = _roi_pixels(bgr_to_linear_rgb(bgr_raw), calib.roi)
    ok = (lin.max(axis=1) < 0.98) & (lin.min(axis=1) > 0.005)
    if ok.sum() < 50:
        return LightingStatus(LightState.FAULT, float("nan"), False)
    lab = linear_rgb_to_lab(np.median(lin[ok], axis=0))
    de = float(np.linalg.norm(lab - np.asarray(calib.card_ref_lab)))
    state = LightState.OK if de <= warn else LightState.CORRECTING if de <= fault else LightState.FAULT
    return LightingStatus(state, de, True)


# ---------------------------------------------------------------- 임계값 제안

def otsu_threshold(values: np.ndarray, lo: float, hi: float, bins: int = 200) -> float:
    """1차원 Otsu — 두 무리(착색 / 비착색)를 가장 잘 나누는 값."""
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
    between[~np.isfinite(between)] = 0
    best = between >= between.max() * (1 - 1e-9)       # 두 무리 사이 빈 구간은 모두 같은 점수 → 그 가운데
    return float(centers[best].mean())


def suggest_hue_max(hues: np.ndarray, chromas: np.ndarray, min_chroma: float) -> float:
    """사과 픽셀 색상각 분포에서 착색 / 비착색 경계를 제안한다 (적색 무리와 황록색 무리 사이)."""
    h = hues[chromas >= min_chroma]
    if len(h) < 100:
        raise ValueError("판정할 사과 픽셀이 부족하다")
    return otsu_threshold(h, -60.0, 150.0)
