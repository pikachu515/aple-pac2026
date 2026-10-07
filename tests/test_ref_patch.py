"""레퍼런스 색 패치 조건부 보정 (전략 S3-5) — 채널이 섞이는 조명 변화를 3×3 로 되돌리고 판정을 이어 간다."""

import numpy as np
import pytest

from aple.config import load_config
from aple.inspection.card_monitor import CardMonitor
from aple.inspection.color import calibrate_card
from aple.inspection.pipeline import inspect_scene
from aple.inspection.ref_patch import (PatchCalib, apply_matrix, calibrate_patches, patch_correction,
                                       patch_delta_e)
from aple.schema import AppleDetection2D, CaptureMode, LightState, RejectReason, SceneSnapshot, Source
from tests.synthetic import INTR, Apple, blush, render
from tests.test_inspection import CARD_ROI, P

GC = load_config()["color"]["gray_card"]
PATCHES = [((110, 20, 60, 60), (180, 40, 40)),      # 적
           ((190, 20, 60, 60), (210, 190, 70)),     # 황
           ((270, 20, 60, 60), (200, 200, 200))]    # 백
ROIS = {"red": PATCHES[0][0], "yellow": PATCHES[1][0], "white": PATCHES[2][0], "card": CARD_ROI}
APPLE = Apple((0.0, 0.03, 0.5), 0.04, blush())
# 조명 스펙트럼 변화: 따뜻해지면서 채널이 섞인다 (대각 게인으로는 못 되돌림)
LAMP = np.array([[1.15, 0.20, 0.00],
                 [0.05, 0.95, 0.08],
                 [0.00, 0.25, 0.70]])


class Camera:
    def __init__(self, mix=None, show_patches=True):
        self.mix, self.show_patches, self.calls = mix, show_patches, 0

    def __call__(self):
        self.calls += 1
        color, depth, masks = render([APPLE], card_rect=CARD_ROI, patches=PATCHES if self.show_patches else None,
                                     mix=self.mix)
        self.mask = masks[0]
        return SceneSnapshot(self.calls, Source.TOP, CaptureMode.FULL, color, depth, INTR, float(self.calls))


def baseline():
    ref = Camera()()
    return calibrate_card(ref.color, CARD_ROI), calibrate_patches(ref.color, ROIS)


def judge(monitor, cam):
    snap = monitor.capture()
    light = monitor.status_for(snap)
    det = AppleDetection2D(snap.scene_id, 0, (0, 0, 1, 1), 0.9, cam.mask)
    return light, inspect_scene(snap, [det], None, P, light=light)[0]


def monitor_for(cam, patches):
    card, _ = baseline()
    return CardMonitor(cam, card, delta_e_warn=GC["delta_e_warn"], delta_e_fault=GC["delta_e_fault"],
                       patches=patches)


def test_calibration_round_trip_and_needs_three_patches():
    _, patches = baseline()
    assert PatchCalib.from_dict(patches.to_dict()) == patches
    with pytest.raises(ValueError, match="3개 이상"):
        calibrate_patches(Camera()().color, {"red": ROIS["red"], "card": CARD_ROI})


def test_matrix_restores_patch_colors_under_mixed_lighting():
    _, patches = baseline()
    lamp = Camera(mix=LAMP)().color
    assert patch_delta_e(lamp, patches) > 10                       # 조명이 크게 바뀌었다
    M = patch_correction(lamp, patches)
    assert patch_delta_e(apply_matrix(lamp, M), patches) < 1.0     # 3×3 보정으로 기준 색 복원


def test_monitor_keeps_judging_with_patches_and_holds_without():
    _, patches = baseline()
    cam0 = Camera()
    _, ref = judge(monitor_for(cam0, patches), cam0)               # 기준 조명에서의 판정
    cam = Camera(mix=LAMP)
    light, g = judge(monitor_for(cam, patches), cam)
    assert light.light_state == LightState.CORRECTING              # 틀어짐 탐지 → 보정 → 판정 계속
    assert g.reject_reason != RejectReason.LOW_CONF
    assert abs(g.ratio - ref.ratio) <= 0.02                        # 기준 조명과 같은 착색비율

    cam = Camera(mix=LAMP)
    light, g = judge(monitor_for(cam, None), cam)                  # 패치 없으면
    assert light.light_state == LightState.FAULT and g.reject_reason == RejectReason.LOW_CONF


def test_no_correction_while_light_is_ok():
    _, patches = baseline()
    with_p, without_p = monitor_for(Camera(), patches), monitor_for(Camera(), None)
    a, b = with_p.capture(), without_p.capture()
    assert with_p.status_for(a).light_state == LightState.OK
    assert np.array_equal(a.color, b.color)                        # 평상시에는 패치 보정을 쓰지 않는다


def test_hidden_patches_leave_fault():
    _, patches = baseline()
    cam = Camera(mix=LAMP, show_patches=False)                     # 팔이 패치를 가림
    light, g = judge(monitor_for(cam, patches), cam)
    assert light.light_state == LightState.FAULT and g.reject_reason == RejectReason.LOW_CONF
