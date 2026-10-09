"""카메라로 상자·리젝트 종이 구역·사과 집는 영역 찾기 (aple/perception/workcell_detect.py).

10/9 최종 현장 배치를 본뜬 합성 장면: 영상 왼쪽 골판지 상자 = UPPER (사과 둘), 오른쪽 = MIDDLE,
가운데 파란 평판 트레이 — 로봇 쪽(영상 위) 흰 종이 = REJECT, 나머지 파란 부분 = 집는 영역 (사과 하나).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from aple.config import load_config
from aple.perception import box_occupancy, detector
from aple.perception.sim_scene import camera_looking_down
from aple.perception.workcell_detect import AutoWorkcell, apply_detection, detect_workcell, side_vector
from aple.schema import CaptureMode, Grade, SceneSnapshot, Source

INTR = (600.0, 600.0, 320.0, 240.0)
SIZE = (480, 640)
CARDBOARD, BLUE, WHITE, RED = (120, 140, 160), (170, 160, 90), (228, 228, 228), (40, 40, 170)
T_CAM = camera_looking_down((0.40, 0.0), 0.90)     # 영상 오른쪽 = base −y, 아래 = base −x
WALL_T, WALL_H, FLOOR_Z = 0.01, 0.06, 0.005
UPPER_C, MIDDLE_C, INNER = (0.40, 0.30), (0.40, -0.30), (0.32, 0.22)
TRAY = dict(center=(0.40, 0.0), size=(0.40, 0.30))
PAPER = dict(center=(0.51, 0.0), size=(0.16, 0.24))   # 트레이의 +x 쪽 (영상 위)
TRAY_Z, PAPER_Z = 0.010, 0.011


def _rect(center, size, yaw, z, color, tag=""):
    return dict(center=np.asarray(center, float), size=np.asarray(size, float), yaw=yaw, z=z, color=color, tag=tag)


def box(center, inner, yaw=0.0):
    """골판지 상자 = 바닥 + 네 벽 윗면 (안쪽 크기 inner, 벽 두께 WALL_T)."""
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([c, s]), np.array([-s, c])
    ix, iy = inner
    out = [_rect(center, inner, yaw, FLOOR_Z, CARDBOARD)]
    for sign in (-1, 1):
        out.append(_rect(np.asarray(center) + sign * (ix / 2 + WALL_T / 2) * ex, (WALL_T, iy + 2 * WALL_T), yaw,
                         WALL_H, CARDBOARD))
        out.append(_rect(np.asarray(center) + sign * (iy / 2 + WALL_T / 2) * ey, (ix, WALL_T), yaw, WALL_H, CARDBOARD))
    return out


def scene(*, yaw_upper=math.radians(3), glossy_tray=False, arm=False, scene_id=1):
    rects = (box(UPPER_C, INNER, yaw_upper) + box(MIDDLE_C, INNER)
             + [_rect(TRAY["center"], TRAY["size"], 0.0, TRAY_Z, BLUE, "tray"),
                _rect(PAPER["center"], PAPER["size"], 0.0, PAPER_Z, WHITE)])
    spheres = [((0.33, 0.28, FLOOR_Z + 0.04), 0.04), ((0.47, 0.33, FLOOR_Z + 0.04), 0.04),   # UPPER 안
               ((0.31, -0.02, TRAY_Z + 0.04), 0.04)]                                       # 집는 영역
    if arm:                                           # 로봇 팔이 종이 위로 사과를 든 채 떠 있다
        spheres += [((0.52, 0.0, 0.20 + 0.06 * k), 0.05) for k in range(3)]
    return render(rects, spheres, glossy_tray, scene_id)


def render(rects, spheres, glossy_tray, scene_id):
    fx, fy, cx, cy = INTR
    v, u = np.indices(SIZE)
    dc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(SIZE)], axis=-1)
    d = dc @ T_CAM[:3, :3].T
    o = T_CAM[:3, 3]
    best = (0.0 - o[2]) / d[..., 2]
    color = np.empty(SIZE + (3,), np.uint8)
    color[:] = (232, 232, 232)
    tag = np.full(SIZE, "", object)
    for r in rects:
        s = (r["z"] - o[2]) / d[..., 2]
        p = o[:2] + d[..., :2] * s[..., None] - r["center"]
        c, sn = math.cos(r["yaw"]), math.sin(r["yaw"])
        lx, ly = p[..., 0] * c + p[..., 1] * sn, -p[..., 0] * sn + p[..., 1] * c
        hit = (np.abs(lx) <= r["size"][0] / 2) & (np.abs(ly) <= r["size"][1] / 2) & (s < best)
        best[hit] = s[hit]
        color[hit] = r["color"]
        tag[hit] = r["tag"]
    for center, rad in spheres:
        oc = o - np.asarray(center)
        a = np.einsum("...i,...i", d, d)
        b = 2 * (d @ oc)
        disc = b * b - 4 * a * (oc @ oc - rad * rad)
        s = (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a)
        hit = (disc >= 0) & (s < best)
        best[hit] = s[hit]
        color[hit] = RED if center[2] < 0.15 else (60, 60, 60)
        tag[hit] = ""
    depth = best.astype(np.float32)
    if glossy_tray:
        depth[tag == "tray"] = 0.0                    # 광택 파란 바닥은 깊이가 빈다 (10/9 실기)
    return SceneSnapshot(scene_id, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0)


def auto_cfg(**auto):
    cfg = copy.deepcopy(load_config())
    cfg["workcell"]["auto_detect"] = {"enabled": True, "assign": {"UPPER": "left", "MIDDLE": "right"},
                                      "flat_zone": "REJECT", **auto}
    cfg["workcell"]["T_base_camera"] = T_CAM.tolist()
    return cfg


def test_finds_boxes_reject_paper_and_pick_area():
    det = detect_workcell(scene(), T_CAM, auto_cfg())
    named = det.named()
    assert set(named) == {"UPPER", "MIDDLE", "REJECT"}, det.warnings
    up, mid, rej = named["UPPER"], named["MIDDLE"], named["REJECT"]
    outer = (INNER[0] + 2 * WALL_T, INNER[1] + 2 * WALL_T)
    assert up.center_xy == pytest.approx(UPPER_C, abs=0.008)                 # 영상 왼쪽 = UPPER
    assert mid.center_xy == pytest.approx(MIDDLE_C, abs=0.008)
    assert mid.outer_size_m == pytest.approx(outer, abs=0.015)
    assert math.degrees(up.yaw_rad) == pytest.approx(3.0, abs=2.0)
    assert up.floor_z_m == pytest.approx(FLOOR_Z, abs=0.01)    # 기준 = 카메라 아래 테이블 (트레이 위면 +1 cm — 놓는 쪽은 높게라 안전)
    assert up.wall_height_m == pytest.approx(WALL_H - FLOOR_Z, abs=0.015)
    assert rej.flat and rej.wall_height_m == 0.0
    assert rej.center_xy == pytest.approx(PAPER["center"], abs=0.008)
    assert rej.floor_z_m == pytest.approx(PAPER_Z, abs=0.004)
    for b, inner in ((mid, INNER), (rej, PAPER["size"])):                   # 칸 격자는 안쪽 가운데
        rows, cols = b.grid
        assert rows >= 1 and cols >= 1 and cols * 0.09 <= inner[0] + 1e-6 and rows * 0.09 <= inner[1] + 1e-6
        assert b.origin_m[0] + cols * 0.09 / 2 == pytest.approx(b.center_xy[0], abs=1e-6)
    # 집는 영역 = 트레이에서 종이 반대쪽 (−x), 종이와 겹치지 않는다
    cx, cy, cz = det.pick_center_m
    sx, sy = det.pick_size_m
    assert cx + sx / 2 < PAPER["center"][0] - PAPER["size"][0] / 2
    assert cx - sx / 2 >= TRAY["center"][0] - TRAY["size"][0] / 2 - 0.005
    assert abs(cy) < 0.02 and cz == pytest.approx(PAPER_Z, abs=0.005)


def test_glossy_tray_and_robot_arm_over_paper():
    det = detect_workcell(scene(glossy_tray=True, arm=True), T_CAM, auto_cfg())
    named = det.named()
    assert {"UPPER", "MIDDLE", "REJECT"} <= set(named), det.warnings
    assert named["REJECT"].center_xy == pytest.approx(PAPER["center"], abs=0.012)
    assert det.pick_center_m[0] + det.pick_size_m[0] / 2 < PAPER["center"][0] - PAPER["size"][0] / 2


def test_assignment_rule_sides_and_missing_box():
    assert side_vector(T_CAM, "right") == pytest.approx([0.0, -1.0], abs=1e-9)
    assert side_vector(T_CAM, "bottom") == pytest.approx([-1.0, 0.0], abs=1e-9)
    assert side_vector(T_CAM, "+y") == pytest.approx([0.0, 1.0])
    swapped = detect_workcell(scene(), T_CAM, auto_cfg(assign={"UPPER": "right", "MIDDLE": "left"})).named()
    assert swapped["UPPER"].center_xy[1] < 0 < swapped["MIDDLE"].center_xy[1]     # 한 줄로 반대 배정
    # 왼쪽 상자가 크게 비뚤어 쓰지 못해도 오른쪽 상자가 UPPER 를 받지 않는다
    det = detect_workcell(scene(yaw_upper=math.radians(25)), T_CAM, auto_cfg())
    assert "UPPER" not in det.named() and det.named()["MIDDLE"].center_xy[1] < 0
    with pytest.raises(ValueError):
        side_vector(T_CAM, "sideways")


def test_occupancy_and_pick_detection_with_auto_workcell():
    cfg = auto_cfg()
    auto = AutoWorkcell(cfg, T_CAM)
    snap = scene()
    filled = auto.update(snap)
    occ = {b.box_id: b for b in box_occupancy.estimate(snap, T_CAM, filled)}
    assert set(occ) == {Grade.UPPER, Grade.MIDDLE, Grade.REJECT}
    assert all(b.valid for b in occ.values())
    assert sum(occ[Grade.UPPER].occupied) >= 2 and not any(occ[Grade.MIDDLE].occupied)
    assert occ[Grade.REJECT].wall_height_m == pytest.approx(occ[Grade.REJECT].origin_m[2])      # 벽 없음
    tray = filled["workcell"]["tray"]
    assert tray["wall_height_m"] == 0.0 and tray["support"] == "tray_z"
    dets = detector.detect(snap, filled)                       # 집는 영역의 사과 하나만 (상자 안 사과는 아니다)
    assert len(dets) == 1
    # 다음 장면에서 상자를 못 찾아도 (가림) 직전 값을 이어 쓴다
    blank = SceneSnapshot(2, Source.TOP, CaptureMode.FULL, np.full_like(snap.color, 232), snap.depth_m, INTR, 0.0)
    again = auto.update(blank)
    assert again["workcell"]["boxes"]["UPPER"]["origin_m"] == filled["workcell"]["boxes"]["UPPER"]["origin_m"]


def test_pick_detection_on_glossy_tray_uses_tray_plane():
    cfg = auto_cfg()
    snap = scene(glossy_tray=True)
    filled = AutoWorkcell(cfg, T_CAM).update(snap)
    assert len(detector.detect(snap, filled)) == 1


def test_apply_detection_keeps_measured_tray_and_fallback_origins():
    cfg = auto_cfg()
    cfg["workcell"]["tray"]["center_m"] = [0.3, 0.0, 0.0]
    cfg["workcell"]["boxes"]["REJECT"]["origin_m"] = [0.1, 0.2, 0.0]
    det = detect_workcell(scene(), T_CAM, cfg)
    det.boxes = [b for b in det.boxes if b.name != "REJECT"]
    out = apply_detection(cfg, det)
    assert out["workcell"]["tray"]["center_m"] == [0.3, 0.0, 0.0]                  # 실측 트레이는 그대로
    assert out["workcell"]["boxes"]["REJECT"]["origin_m"] == [0.1, 0.2, 0.0]       # 못 찾으면 예비값
    assert cfg["workcell"]["boxes"]["UPPER"]["origin_m"] is None                     # 원본은 안 바뀐다


def test_live_execution_does_not_require_box_origins_in_auto_mode():
    from aple.hal.live_runtime import missing_for_live_execution
    manual = missing_for_live_execution(load_config())
    auto = missing_for_live_execution(auto_cfg())
    assert any("workcell.tray.center_m" in m for m in manual)
    assert not any("workcell.tray.center_m" in m or "origin_m" in m for m in auto)


def test_scene_runner_uses_auto_workcell():
    from aple.execution.pick_place_pipeline import PickPlacePipeline
    from aple.execution.testing import RecordingMotionAdapter
    from aple.orchestration.scene_runner import SceneRunner
    cfg = auto_cfg()
    runner = SceneRunner.from_config(cfg, PickPlacePipeline(cfg, RecordingMotionAdapter()))
    plan = runner.plan(scene())
    assert {b.box_id for b in plan.boxes} == {Grade.UPPER, Grade.MIDDLE, Grade.REJECT}
    assert len(plan.apples) == 1                                                      # 집는 영역 사과만
    assert runner.auto_workcell.last is not None


def test_cli_on_snapshot_and_save(tmp_path, monkeypatch):
    from tests.conftest import pristine_copy
    from tools import workcell_detect as tool
    snap = scene()
    path = tmp_path / "snap.npz"
    np.savez_compressed(path, color=snap.color, depth_m=snap.depth_m, intrinsics=np.asarray(INTR))
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")
    monkeypatch.setattr(tool, "load_config", lambda: auto_cfg())
    det = tool.main(["--snapshot", str(path), "--out", str(tmp_path / "o.png"), "--save", str(workcell)])
    assert (tmp_path / "o.png").exists() and "UPPER" in det.named()
    import yaml
    saved = yaml.safe_load(workcell.read_text(encoding="utf-8"))["workcell"]["boxes"]
    assert saved["UPPER"]["origin_m"] == pytest.approx(list(det.named()["UPPER"].origin_m), abs=1e-3)
    assert saved["REJECT"]["wall_height_m"] == 0


def test_assign_removes_box_by_identity_when_several_candidates():
    """10/9 실기: 손이 상자 옆에 들어와 '제외' 후보가 생기자 list.remove 가 ndarray 필드 비교로 터졌다."""
    from aple.perception.workcell_detect import BoxDetection, _assign
    import dataclasses
    fields = {f.name for f in dataclasses.fields(BoxDetection)}
    cfg = load_config()
    T = np.asarray(camera_looking_down() if not isinstance(camera_looking_down(), tuple) else camera_looking_down()[0])
    left, right = side_vector(T, "left"), side_vector(T, "right")

    def box(xy):
        base = {name: None for name in fields}
        base.update(center_xy=np.asarray(xy, float), usable=True, flat=False)
        try:
            return BoxDetection(**base)
        except TypeError:
            pytest.skip("BoxDetection 생성자가 바뀌었다")

    near_left, far_left, far_right = box(0.10 * left[:2]), box(0.30 * left[:2]), box(0.30 * right[:2])
    p = {"assign": {"UPPER": "left", "MIDDLE": "right"}}
    _assign([near_left, far_left, far_right], T, p, (0.0, 0.0))             # 터지지 않으면 된다
