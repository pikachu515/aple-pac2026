"""시험은 '현장 실측 전' 설정을 기준으로 한다.

현장에서 configs/*.yaml 에 실측값(field_values.py set, touch_calib --save)을 넣어도 시험 결과가 바뀌지 않게,
저장소 설정을 읽을 때 아래 실측 키를 null 로 되돌린다. 실측값이 필요한 시험은 cfg 에 직접 넣는다.
"""

from __future__ import annotations

import copy
from pathlib import Path

import aple.config as _config

#: 현장에서 채우는 값 — 시험에서는 비어 있는 것으로 본다
FIELD_KEYS = ("workcell.T_base_camera", "workcell.tcp_offset_m", "color.card_json",
              "workcell.poses.home_joint_rad", "workcell.poses.scan_joint_rad",
              "workcell.auto_detect.enabled",  # 자동 검출은 그 시험(test_workcell_detect)에서만 켠다
              "grasp_planner.tilt_start_m",    # 먼 곳 기울여 잡기는 그 시험(test_tilted_grasp)에서만 켠다
              "perception.box.max_object_height_m",   # 팔·손 높이 거르기 (10/9 실기) — 시험은 기본(끔)
              # 역할 B 실기 실행 값 (10/9 제안 PR) — 시험은 실측 전 상태로 본다
              "safety.joint_limits_rad", "safety.workspace_aabb_m",
              *(f"safety.segment_validation.{k}" for k in (
                  "sample_step_rad", "floor_z_m", "clearance_m", "wall_thickness_m", "support_contact_tolerance_m",
                  "obstacle_aabbs_m", "self_collision_pairs", "payload_self_links")),
              *(f"execution.motion.{k}" for k in (
                  "velocity_limits_rad_s", "acceleration_limits_rad_s2", "command_period_s", "tracking_tolerance_rad",
                  "velocity_tolerance_rad_s", "capture_joint_tolerance_rad")),
              *(f"execution.gripper.{k}" for k in (
                  "closing_compression_m", "contact_settle_s", "contact_width_tolerance_m", "contact_stability_m")))

#: null 이 아닌 실측 전 값 (자리표시자)
PRISTINE_VALUES = {"piper_web.iface": "PLACEHOLDER_follower", "piper_web.camera_id": "PLACEHOLDER_d435",
                   # 수제 핑거 실측 (10/9) 전 — 시뮬레이션·시험은 기본 Piper 그리퍼 기준
                   "piper_web.gripper_width_range_m": [0.0, 0.07], "size_bins.gripper_range_mm": [30, 100],
                   "inspection.slip_check.sigma_pose_m": 0.003,
                   # 10/9 현장 검출·파지 조정 — 시험은 기본(깊이 분할·고정 개방)으로, 새 동작은 그 시험에서 켠다
                   "perception.detect.segment": "depth", "perception.detect.roi_margin_m": 0.0,
                   "perception.detect.keep_margin_m": 0.0, "perception.detect.split_by_height": False,
                   "perception.detect.min_apple_color": 0.0,
                   "grasp_planner.adaptive_open": False, "grasp_planner.tray_open_sides": [],
                   "grasp_planner.tilt_yaw_free_below_deg": 0.0, "grasp_planner.tilt_yaw_dev_deg": 0.0,
                   "grasp_planner.min_open_margin_m": 0.01}

_original_load = _config.load_config


def _pristine(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for dotted, value in [*((k, None) for k in FIELD_KEYS), *PRISTINE_VALUES.items()]:
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node.get(key, {})
        if leaf in node:
            node[leaf] = value
    return cfg


def load_config(config_dir=_config.CONFIG_DIR):
    cfg = _original_load(config_dir)
    return _pristine(cfg) if Path(config_dir).resolve() == Path(_config.CONFIG_DIR).resolve() else cfg


_config.load_config = load_config


#: 실측 전 줄 (주석 포함). 설정 파일을 복사해 쓰는 시험이 '실측 전' 상태에서 시작하게 한다
PRISTINE_LINES = {
    "workcell.yaml": {
        "  T_base_camera:": "  T_base_camera: null           # 4x4, eye-to-hand 캘리브레이션 결과 (m)",
        "  tcp_offset_m:": "  tcp_offset_m: null            # 핑거 어댑터 반영",
        "    home_joint_rad:": "    home_joint_rad: null        # 6 rad. 시작·종료·복구 자세",
        "    scan_joint_rad:": "    scan_joint_rad: null        # 6 rad. 탑뷰 촬영 자세 (팔·그림자가 트레이·상자를 가리지 않게)",
        "  auto_detect:": "  auto_detect: {enabled: false}",
    },
    "execution.yaml": {
        '  iface:': '  iface: "PLACEHOLDER_follower"  # PLACEHOLDER: 시뮬/실행 전 Piper Web 등록 follower 이름으로 교체',
        '  camera_id:': '  camera_id: "PLACEHOLDER_d435"  # PLACEHOLDER: 시뮬/실행 전 실제 D435 camera id로 교체',
    },
    "color.yaml": {
        "  card_json:": "  card_json: null                 # tools/color_session.py card 결과 경로 (현장 세션 폴더의 card.json). CardMonitor 가 읽는다",
    },
}


def reset_field_values(config_dir) -> None:
    """복사한 configs 폴더의 실측 키를 실측 전 줄로 되돌린다."""
    for name, lines in PRISTINE_LINES.items():
        path = Path(config_dir) / name
        text = path.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(text):
            for prefix, pristine in lines.items():
                if line.startswith(prefix):
                    text[i] = pristine
        path.write_text("\n".join(text), encoding="utf-8")


def pristine_copy(src_name: str, dest) -> Path:
    """configs/<src_name> 을 dest 로 복사하고 실측 키를 되돌린다."""
    import shutil
    dest = Path(dest)
    shutil.copy(Path(_config.CONFIG_DIR) / src_name, dest)
    lines = PRISTINE_LINES.get(src_name, {})
    text = dest.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(text):
        for prefix, pristine in lines.items():
            if line.startswith(prefix):
                text[i] = pristine
    dest.write_text("\n".join(text), encoding="utf-8")
    return dest
