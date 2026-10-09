"""실기 연결용 코드 (역할 A): 파지 후보, Piper Studio 실시간 스냅샷, 실기 점검 도구 m1_live."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.hal.depth_align import D435Calib, DepthEncoding, Intrinsics
from aple.hal.live_snapshot import PiperLiveCamera, median_depth
from aple.hal.piper_studio import PiperStudio
from aple.perception import detector, grasp_planner, pose_estimator
from aple.perception.sim_scene import INTR, camera_looking_down, render
from aple.schema import AppleState, CaptureMode, SceneSnapshot
from tests.test_depth_align import piper_encode
from tools import m1_live, rviz_view


def apple(i, x, y, r=0.037, occluded=False):
    return AppleState(1, i, i, (x, y, r), 2 * r, 2 * r, 0.0, 0.001, 1.0, 1.0, occluded=occluded)


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c['piper_web']['gripper_width_range_m'] = [.02, .12]
    c["workcell"]["T_base_camera"] = camera_looking_down().tolist()
    return c


# ================================================================ 파지 후보

def test_isolated_apple_gets_free_vertical_grasp(cfg):
    (g,) = grasp_planner.plan_grasps([apple(1, 0.3, 0.0)], cfg)
    assert g.collision_free and g.apple_id == 1 and g.scene_id == 1
    assert g.grasp_pose.position_m == pytest.approx((0.3, 0.0, 0.037))      # 적도 높이 = 중심 z
    assert g.approach_vector == (0.0, 0.0, 1.0) and g.approach_height_m > 0
    x, y, z, w = g.grasp_pose.orientation_xyzw
    assert math.isclose(x * x + y * y + z * z + w * w, 1.0)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert R[:, 2] == pytest.approx([0, 0, -1])                              # TCP z 축이 아래


def test_fingers_turn_away_from_close_neighbour(cfg):
    a, b = apple(1, 0.30, 0.0), apple(2, 0.38, 0.0)                         # x 방향으로 6 mm 간격
    g = {c.apple_id: c for c in grasp_planner.plan_grasps([a, b], cfg)}
    assert abs(math.degrees(g[1].grasp_yaw_rad) - 90) < 1e-6                # 핑거는 y 방향으로 닫힌다
    assert g[1].collision_free


def test_crowded_apple_is_not_collision_free(cfg):
    ring = [apple(1, 0.3, 0.0)] + [apple(k + 2, 0.3 + 0.075 * math.cos(t), 0.075 * math.sin(t))
                                   for k, t in enumerate(np.linspace(0, 2 * math.pi, 6, endpoint=False))]
    g = {c.apple_id: c for c in grasp_planner.plan_grasps(ring, cfg)}
    assert not g[1].collision_free and g[1].min_clearance_m == 0.0


def test_tray_wall_limits_grasp(cfg):
    cfg["workcell"]["tray"].update(center_m=[0.3, 0.0, 0.0], size_m=[0.2, 0.2])
    (g,) = grasp_planner.plan_grasps([apple(1, 0.3, 0.07)], cfg)          # 벽까지 30 mm − r
    assert abs(math.degrees(g.grasp_yaw_rad)) < 1e-6                         # 벽과 나란히 닫는다 (x 방향)


def test_occluded_apple_has_no_candidate_but_blocks_fingers(cfg):
    a, hidden = apple(1, 0.30, 0.0), apple(2, 0.38, 0.0, occluded=True)
    (g,) = grasp_planner.plan_grasps([a, hidden], cfg)
    assert g.apple_id == 1 and abs(math.degrees(g.grasp_yaw_rad) - 90) < 1e-6


def test_candidates_drive_role_b_plan(cfg):
    """후보가 있으면 역할 B 계획이 DEFER 가 아니라 PICK_PLACE 를 낸다."""
    from aple.schema import Grade, GradeResult, InspectMode, RejectReason, TaskAction
    a = apple(1, 0.3, 0.0)
    grade = GradeResult(1, 1, Grade.UPPER, 0.6, 0.55, 0.65, 0.25, 0.15, RejectReason.NONE, True,
                        InspectMode.TOP_SINGLE, "fuji")
    pipe = PickPlacePipeline(cfg)
    no_cand = pipe.plan_scene([a], [grade], [], [])
    with_cand = PickPlacePipeline(cfg).plan_scene([a], [grade], [], grasp_planner.plan_grasps([a], cfg))
    assert no_cand.items[0].task.action == TaskAction.DEFER
    assert with_cand.items[0].task.action == TaskAction.PICK_PLACE


# ================================================================ 실시간 스냅샷

IDENT = Intrinsics(*INTR, 640, 480)
CALIB = D435Calib(IDENT, IDENT, np.eye(3), np.zeros(3))
ENC = DepthEncoding(150.0, 1200.0)


def fake_reader(frames):
    """[(color, depth_m), ...] → read_pair 처럼 한 쌍씩. 같은 프레임을 두 번 돌려 중복 무시도 시험한다."""
    import time
    def seq():
        for color, depth in frames:
            pair = (color, piper_encode(depth, ENC), time.time_ns())
            yield pair
            yield pair
    it = iter(seq())
    return lambda: next(it, None)


def test_median_depth_needs_half_of_frames():
    a = np.full((2, 2), 0.6, np.float32)
    b, c = a.copy(), a.copy()
    b[0, 0] = 0
    c[0, 0] = 0
    c[1, 1] = 0
    m = median_depth([a, b, c])
    assert m[0, 0] == 0 and m[1, 1] == pytest.approx(0.6) and m[0, 1] == pytest.approx(0.6)


def test_live_camera_builds_snapshots_with_increasing_scene_id():
    T = camera_looking_down()
    depth, _, color = render(T, [((0.3, 0.0, 0.037), 0.037)])
    cam = PiperLiveCamera(CALIB, ENC, fake_reader([(color, depth)] * 6), frames=3)
    s1, s2 = cam(), cam(CaptureMode.POSE_ONLY, tray_clear=False)
    assert (s1.scene_id, s2.scene_id) == (1, 2)
    assert s2.capture_mode == CaptureMode.POSE_ONLY and not s2.tray_clear
    assert isinstance(s1, SceneSnapshot) and s1.depth_m.shape == (480, 640)


def test_live_camera_times_out_without_frames():
    cam = PiperLiveCamera(CALIB, ENC, lambda: None, frames=2, timeout_s=0.05)
    with pytest.raises(TimeoutError):
        cam()


def test_live_snapshot_feeds_m1_with_8bit_depth(cfg):
    """Piper Studio 8비트 깊이(한 단계 약 4 mm)를 3장 중앙값으로 받아도 직경 ±3 mm."""
    T = camera_looking_down()
    truth = [((0.25, 0.08, 0.036), 0.036), ((0.35, -0.08, 0.041), 0.041)]
    frames = []
    for k in range(3):
        depth, _, color = render(T, truth, noise_m=0.002, dropout=0.2, seed=k)
        frames.append((color, depth))
    snap = PiperLiveCamera(CALIB, ENC, fake_reader(frames), frames=3)()
    apples = pose_estimator.estimate(snap, detector.detect(snap, cfg), T, None, cfg)
    assert len(apples) == 2
    for c, r in truth:
        a = min(apples, key=lambda a: np.linalg.norm(np.subtract(a.center_m, c)))
        assert abs(a.diameter_m - 2 * r) < 0.003 and np.linalg.norm(np.subtract(a.center_m, c)) < 0.004


def test_live_camera_plugs_into_piper_studio(cfg):
    depth, _, color = render(camera_looking_down(), [])
    cam = PiperLiveCamera(CALIB, ENC, fake_reader([(color, depth)]), frames=1)
    snap = PiperStudio(cfg, snapshot_provider=cam).snapshot()
    assert snap.scene_id == 1


def test_from_config_explains_missing_depth_calibration(cfg, tmp_path):
    cfg["live_camera"]["depth_calib"] = str(tmp_path / "none.json")
    with pytest.raises(FileNotFoundError, match="rs_check"):
        PiperLiveCamera.from_config(cfg)


# ================================================================ m1_live 도구

@pytest.fixture
def demo_file(tmp_path, monkeypatch):
    snap, cfg, *_ = rviz_view.demo_scene(0)
    path = tmp_path / "scan.npz"
    np.savez_compressed(path, color=snap.color, depth_m=snap.depth_m, intrinsics=np.array(snap.intrinsics),
                        stamp_s=0.0)
    monkeypatch.setattr(m1_live, "load_config", lambda: cfg)
    return path, cfg


def test_m1_live_runs_full_dry_run_on_snapshot(demo_file, capsys):
    path, _ = demo_file
    (r,) = m1_live.main(["--snapshot", str(path), "--plan"])
    out = capsys.readouterr().out
    assert len(r["apples"]) == 5 and len(r["grasps"]) == 5 and r["support"] is None
    assert "드라이런" in out and "PICK_PLACE" in out and "UPPER 칸" in out


def test_m1_live_caliper_table(demo_file, capsys):
    path, _ = demo_file
    (r,) = m1_live.main(["--snapshot", str(path)])
    true_mm = [a.diameter_m * 1000 + 1.0 for a in sorted(r["apples"], key=lambda a: a.det_id)]
    m1_live.main(["--snapshot", str(path), "--calipers", ",".join(f"{v:.1f}" for v in true_mm)])
    out = capsys.readouterr().out
    assert "목표 ±3 mm 충족" in out


def test_m1_live_flags_moved_camera(demo_file, capsys):
    path, cfg = demo_file
    T = np.array(cfg["workcell"]["T_base_camera"])
    T[2, 3] += 0.02                                                  # 카메라가 2 cm 내려앉은 것처럼
    cfg["workcell"]["T_base_camera"] = T.tolist()
    (r,) = m1_live.main(["--snapshot", str(path)])
    assert r["support"] and "받침면 높이" in r["support"]


def test_m1_live_requires_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(m1_live, "load_config", lambda: copy.deepcopy(load_config()))
    with pytest.raises(SystemExit, match="T_base_camera"):
        m1_live.main(["--snapshot", str(tmp_path / "x.npz")])
