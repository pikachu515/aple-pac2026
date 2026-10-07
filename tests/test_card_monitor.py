"""회색 카드 모니터 — 스냅샷 색 보정 + 같은 프레임 기준 조명 상태 (lighting_status_provider)."""

import json

import numpy as np
import pytest

from aple.config import load_config
from aple.hal.piper_studio import PiperStudio
from aple.inspection.card_monitor import CardMonitor
from aple.inspection.color import calibrate_card, card_brightness
from aple.inspection.pipeline import inspect_scene
from aple.schema import AppleDetection2D, CaptureMode, LightState, RejectReason, SceneSnapshot, Source
from tests.synthetic import INTR, Apple, half_red, render
from tests.test_inspection import CARD_ROI, P

CFG = load_config()
GC = CFG["color"]["gray_card"]
APPLE = Apple((0.0, 0.0, 0.5), 0.04, half_red)


class Camera:
    """조명(tint)을 바꿀 수 있는 원본 스냅샷 공급자. 부를 때마다 scene_id 가 1 늘어난다."""

    def __init__(self, tint=(1.0, 1.0, 1.0), card=True):
        self.tint, self.card, self.calls = tint, card, 0

    def __call__(self):
        self.calls += 1
        color, depth, masks = render([APPLE], card_rect=CARD_ROI if self.card else None, tint=self.tint)
        self.masks = masks
        return SceneSnapshot(self.calls, Source.TOP, CaptureMode.FULL, color, depth, INTR, float(self.calls))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def monitor_for(cam, ref_tint=(1.0, 1.0, 1.0), **kw):
    ref = Camera(tint=ref_tint)()                        # 회색 카드 보정 시점의 조명·카메라 상태
    calib = calibrate_card(ref.color, CARD_ROI)
    return CardMonitor(cam, calib, delta_e_warn=GC["delta_e_warn"], delta_e_fault=GC["delta_e_fault"], **kw)


def test_capture_applies_card_gain_and_records_status():
    residual = (1.0, 0.88, 1.0)                          # Piper 보정 뒤에도 남는 녹–자 틀어짐 (카메라 고유)
    cam = Camera(tint=residual)
    monitor = monitor_for(cam, ref_tint=residual)        # 같은 카메라·조명에서 카드 게인을 잡았다
    def card_spread(img):
        x, y, w, h = CARD_ROI
        bgr = [np.median(img[y + 15:y + h - 15, x + 15:x + w - 15, i]) for i in range(3)]
        return max(bgr) - min(bgr)

    snap = monitor.capture()
    status = monitor.status_for(snap)
    assert status.card_valid and status.light_state in (LightState.OK, LightState.CORRECTING)
    assert card_spread(cam().color) > 6                  # 원본: 녹색이 빠져 카드가 자주색 쪽
    assert card_spread(snap.color) <= 3                  # 보정 후: 카드 기준으로 회색
    assert abs(card_brightness(snap.color, CARD_ROI) - GC["target_srgb"]) <= 3


def test_lighting_change_raises_fault_and_holds_grading():
    monitor = monitor_for(Camera(tint=(1.35, 1.0, 0.65)))     # 따뜻한 조명으로 바뀜
    snap = monitor.capture()
    light = monitor.status_for(snap)
    assert light.light_state == LightState.FAULT and light.delta_e > GC["delta_e_fault"]
    det = AppleDetection2D(snap.scene_id, 0, (0, 0, 1, 1), 0.9, monitor._snapshot_fn.masks[0])
    grades = inspect_scene(snap, [det], None, P, light=light)
    assert grades[0].reject_reason == RejectReason.LOW_CONF  # 조명이 틀어지면 판정 보류


def test_missing_or_covered_card_holds_grading():
    """카드 자리에 다른 것이 보이면 조명 변화와 구분하지 않고 FAULT — 어느 쪽이든 판정은 보류한다."""
    monitor = monitor_for(Camera(card=False))
    assert monitor.status_for(monitor.capture()).light_state == LightState.FAULT


def test_provider_reuses_fresh_status_and_remeasures_when_stale():
    cam, clock = Camera(), Clock()
    monitor = monitor_for(cam, max_age_s=2.0, clock=clock)
    monitor.capture()
    calls = cam.calls
    clock.t = 1.0
    assert monitor().light_state == LightState.OK and cam.calls == calls          # 방금 찍은 프레임 기준
    clock.t = 5.0
    cam.tint = (1.35, 1.0, 0.65)
    assert monitor().light_state == LightState.FAULT and cam.calls == calls + 1   # 오래돼서 새로 찍음


def test_plugs_into_piper_studio_providers():
    monitor = monitor_for(Camera())
    studio = PiperStudio(CFG, snapshot_provider=monitor.capture, lighting_status_provider=monitor)
    snap = studio.snapshot()
    assert studio.lighting_status() is monitor.status_for(snap)


def test_from_config_reads_card_json(tmp_path):
    calib = calibrate_card(Camera()().color, CARD_ROI)
    path = tmp_path / "card.json"
    path.write_text(json.dumps(calib.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="card_json"):
        CardMonitor.from_config(CFG, Camera())
    monitor = CardMonitor.from_config(CFG, Camera(), card_path=path)
    assert monitor.calib == calib
    with pytest.raises(KeyError, match="not captured"):
        monitor.status_for(Camera()())
