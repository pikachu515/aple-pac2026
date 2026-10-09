"""회색 그리퍼가 회색 카드를 가릴 때 — 깊이로 카드 면 픽셀만 쓴다 (카드 보정·조명 감시)."""

import numpy as np
import pytest

from aple.inspection.card_monitor import CardMonitor
from aple.inspection.color import CardCalib, calibrate_card, lighting_status
from aple.schema import CaptureMode, LightState, SceneSnapshot, Source
from tests.synthetic import INTR, Apple, half_red, render
from tests.test_card_monitor import GC, Clock
from tests.test_inspection import CARD_ROI

APPLE = Apple((0.0, 0.0, 0.5), 0.04, half_red)
FINGER_GRAY = 70                                         # 카드(118)보다 어두운 회색 그리퍼


def scene(finger_frac=0.0, finger_gray=FINGER_GRAY, tint=(1.0, 1.0, 1.0)):
    """카드 왼쪽 finger_frac 만큼을 회색 핑거가 카드 10 cm 위에서 가린 장면."""
    color, depth, _ = render([APPLE], card_rect=CARD_ROI, tint=tint)
    if finger_frac > 0:
        x, y, w, h = CARD_ROI
        fw = int(w * finger_frac)
        color[y:y + h, x:x + fw] = finger_gray
        depth[y:y + h, x:x + fw] -= 0.10
    return color, depth


def card_and_depth(finger_frac=0.0):
    color, depth = scene(finger_frac)
    return color, CARD_ROI, depth


def test_calibration_stores_card_depth_and_round_trips():
    color, depth = scene()
    calib = calibrate_card(color, CARD_ROI, depth)
    x, y, w, h = CARD_ROI
    assert calib.card_depth_m == pytest.approx(float(np.median(depth[y:y + h, x:x + w])), abs=0.005)
    assert CardCalib.from_dict(calib.to_dict()) == calib
    old = {k: v for k, v in calib.to_dict().items() if k != "card_depth_m"}   # 예전 card.json 도 읽힌다
    assert CardCalib.from_dict(old).card_depth_m is None


def test_calibration_ignores_gray_finger_over_half_the_card():
    """중앙값은 절반 미만 가림은 견딘다. 절반을 넘으면 깊이 없이는 게인이 핑거 색으로 바뀐다."""
    clean = calibrate_card(*card_and_depth())
    color, depth = scene(0.6)
    with_depth = calibrate_card(color, CARD_ROI, depth, min_visible=0.3)
    without = calibrate_card(color, CARD_ROI)
    assert with_depth.rgb_gain == pytest.approx(clean.rgb_gain, rel=0.02)
    assert without.rgb_gain[1] > 1.3 * clean.rgb_gain[1]                    # 어두운 핑거를 카드로 보고 밝힌다


def test_calibration_refuses_mostly_covered_card():
    color, depth = scene(0.7)
    with pytest.raises(ValueError, match="가려져"):
        calibrate_card(color, CARD_ROI, depth)


def test_lighting_status_ignores_finger_and_flags_occlusion():
    calib = calibrate_card(*card_and_depth())
    color, depth = scene(0.6)
    assert lighting_status(color, calib).light_state == LightState.FAULT     # 깊이 없으면 핑거를 조명 변화로 오인
    assert lighting_status(color, calib, depth_m=depth, min_visible=0.3).light_state == LightState.OK
    covered = lighting_status(color, calib, depth_m=depth)                   # 기본 min_visible 0.5 → 가려짐
    assert not covered.card_valid and covered.light_state == LightState.FAULT
    dim, dim_depth = scene(0.35, tint=(0.6, 0.6, 0.6))                       # 진짜 조명 변화는 그대로 잡는다
    assert lighting_status(dim, calib, depth_m=dim_depth).light_state != LightState.OK


class Feed:
    def __init__(self):
        self.finger, self.tint, self.calls = 0.0, (1.0, 1.0, 1.0), 0

    def __call__(self):
        self.calls += 1
        color, depth = scene(self.finger, tint=self.tint)
        return SceneSnapshot(self.calls, Source.TOP, CaptureMode.FULL, color, depth, INTR, float(self.calls))


def make_monitor(feed, clock, reuse_s=30.0):
    return CardMonitor(feed, calibrate_card(*card_and_depth()), delta_e_warn=GC["delta_e_warn"],
                       delta_e_fault=GC["delta_e_fault"], occluded_reuse_s=reuse_s, clock=clock)


def test_monitor_reuses_last_visible_status_while_card_is_covered():
    feed, clock = Feed(), Clock()
    monitor = make_monitor(feed, clock)
    top = monitor.capture()                               # scan 자세 — 카드가 보인다
    assert monitor.status_for(top).light_state == LightState.OK
    feed.finger, clock.t = 0.9, 10.0                      # 모드 B — 그리퍼가 카드 위
    held = monitor.capture()
    assert monitor.status_for(held) is monitor.status_for(top)
    clock.t = 45.0                                        # 너무 오래 가려져 있으면 FAULT
    late = monitor.capture()
    assert not monitor.status_for(late).card_valid
    assert monitor.status_for(late).light_state == LightState.FAULT


def test_monitor_without_reuse_window_faults_when_covered():
    feed, clock = Feed(), Clock()
    monitor = make_monitor(feed, clock, reuse_s=0.0)
    monitor.capture()
    feed.finger, clock.t = 0.9, 1.0
    assert monitor.status_for(monitor.capture()).light_state == LightState.FAULT
