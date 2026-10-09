"""M1 인식 결과를 RViz2 로 보기 — 점군, 사과 구, 상자 칸 점유, 트레이, 터치 캘리브레이션 점 (역할 A).

로봇 시뮬레이션이 아니라 '역할 A 모듈이 장면을 어떻게 보는가'를 3D 로 확인하는 도구다.
ROS 2 는 보기에만 쓴다 (시스템 본체는 Piper Studio 위 Python — docs/architecture.md D-1).

    source /opt/ros/humble/setup.bash
    python3 tools/rviz_view.py --demo --rviz                     # 합성 장면 (정답 함께 표시)
    python3 tools/rviz_view.py --snapshot data/live/snap_0000.npz --session data/calib/touch.json --rviz
    python3 tools/rviz_view.py --demo --no-ros                   # ROS 없이 요약만 출력

RViz 에 보이는 것 (고정 좌표계 base_link)
  Depth cloud      스냅샷 깊이를 컬러로 칠한 점군 (camera_top_optical_frame, T_base_camera 로 배치)
  apples           검출한 사과 구. 초록 = confidence ≥ 0.5, 주황 = 낮음 (LOW_CONF 대상). 글자: #id 직경 conf
  truth            (데모만) 정답 사과 — 흰 반투명 구. 초록 구와 겹치면 위치·직경이 맞은 것
  boxes            상자 칸: 빨강 = 점유, 초록 = 빈 칸, 회색 = 판단 불가. 높이 = 칸 상면. 선 = 벽 윗면
  tray             트레이 사각형 (workcell.tray.center_m 이 있을 때)
  calib            터치 캘리브레이션: 파랑 = 로봇이 터치한 점, 노랑 = 카메라가 본 점을 변환한 위치, 글자 = 오차
  TF               camera_top_optical_frame (사용 중인 변환), camera_true (데모 정답)
Ctrl-C 로 끝낸다.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.hal.calibration import (T_from_cfg, base_to_pixel, camera_point_at, camera_to_base,  # noqa: E402
                                  fit_touch_points)
from aple.hal.replay import load_snapshot  # noqa: E402
from aple.perception import box_occupancy, detector, pose_estimator  # noqa: E402
from aple.perception.geometry import depth_points  # noqa: E402
from aple.perception.sim_scene import INTR, box_tops, camera_looking_down, render  # noqa: E402
from aple.schema import AppleState, BoxOccupancy, CaptureMode, SceneSnapshot, Source  # noqa: E402

RVIZ_CONFIG = ROOT / "tools" / "rviz" / "m1_view.rviz"
CAM_FRAME = "camera_top_optical_frame"


@dataclass
class View:
    T: np.ndarray                                   # 사용 중인 T_base_camera
    cloud_cam: np.ndarray                           # N×3 카메라 좌표
    cloud_bgr: np.ndarray                           # N×3 uint8
    apples: list[AppleState]
    boxes: list[BoxOccupancy]
    tray: tuple | None = None                       # (center_m, size_m)
    calib: list[dict] = field(default_factory=list) # {name, base, pred, err}
    truth: list[tuple] = field(default_factory=list)   # [(center, r)]
    T_true: np.ndarray | None = None


# ================================================================ 장면 준비

def demo_scene(seed: int = 0) -> tuple[SceneSnapshot, dict, dict, list, np.ndarray]:
    """현장 절차 그대로: ① 표식 장면으로 터치 캘리브레이션 → ② 그 변환으로 트레이·상자 장면을 인식.

    → (작업 스냅샷, cfg, 캘리브레이션 세션, 정답 사과, 정답 T)
    """
    rng = np.random.default_rng(seed)
    T_true = camera_looking_down((0.30, 0.0), 0.65, tilt_rad=0.05)

    # ① 표식 8개 (3개는 6 cm 받침대 위). 클릭 오차 0.7 px, 터치 오차 0.7 mm
    markers = np.array([[0.12, 0.22, 0.0], [0.46, 0.22, 0.0], [0.12, -0.24, 0.0], [0.46, -0.26, 0.0],
                        [0.30, 0.02, 0.0], [0.20, -0.06, 0.06], [0.38, 0.10, 0.06], [0.26, -0.16, 0.06]])
    blocks = [(x - 0.03, x + 0.03, y - 0.03, y + 0.03, z) for x, y, z in markers if z > 0]
    depth_c, _, _ = render(T_true, tops=blocks, noise_m=0.001, seed=seed)
    uv, _ = base_to_pixel(markers, INTR, T_true)
    uv += rng.normal(0, 0.7, uv.shape)
    points = []
    for k, ((u, v), m) in enumerate(zip(uv, markers), start=1):
        points.append({"name": f"P{k}", "pixel": [float(u), float(v)],
                       "cam_m": camera_point_at(depth_c, u, v, INTR).tolist(),
                       "base_m": (m + rng.normal(0, 0.0007, 3)).tolist()})
    session = {"snapshot": "(demo)", "intrinsics": list(INTR), "points": points}
    T_fit = fit_touch_points([p["cam_m"] for p in points], [p["base_m"] for p in points]).T

    # ② 트레이 위 사과 5개 (두 개는 맞닿음) + 상자 UPPER 에 이미 2개, MIDDLE 은 빈 상자
    cfg = copy.deepcopy(load_config())
    cfg['piper_web']['gripper_width_range_m'] = [.02, .12]  # Synthetic demo only.
    cfg["workcell"]["T_base_camera"] = T_fit.tolist()
    cfg["workcell"]["tray"].update(center_m=[0.30, -0.01, 0.0], size_m=[0.28, 0.28])
    upper, middle = (0.21, 0.16, 0.01), (0.21, -0.31, 0.01)
    for name, origin in (("UPPER", upper), ("MIDDLE", middle)):
        cfg["workcell"]["boxes"][name].update(grid=[2, 3], cell_pitch_m=0.06, wall_height_m=0.08,
                                              origin_m=list(origin))
    apples = [((0.22, -0.08), 0.037), ((0.22, -0.006), 0.037), ((0.35, 0.05), 0.041),
              ((0.38, -0.07), 0.033), ((0.27, 0.085), 0.035)]
    spheres = [((x, y, r), r) for (x, y), r in apples]
    for k in (0, 4):
        row, col = divmod(k, 3)
        spheres.append(((upper[0] + (col + 0.5) * 0.06, upper[1] + (row + 0.5) * 0.06, upper[2] + 0.028), 0.028))
    depth, _, color = render(T_true, spheres, box_tops(upper, 2, 3, 0.06, 0.08) + box_tops(middle, 2, 3, 0.06, 0.08),
                             noise_m=0.0015, dropout=0.2, seed=seed + 1)
    snap = SceneSnapshot(1, Source.TOP, CaptureMode.FULL, color, depth, INTR, time.time())
    return snap, cfg, session, spheres[:len(apples)], T_true


def build_view(snap: SceneSnapshot, cfg: dict, session: dict | None = None, truth=(), T_true=None,
               cloud_step: int = 2) -> View:
    T = T_from_cfg(cfg)
    if T is None:
        raise SystemExit("workcell.T_base_camera 가 비어 있다 — tools/touch_calib.py fit --save 를 먼저 하거나 --demo 를 쓴다")
    dets = detector.detect(snap, cfg)
    apples = pose_estimator.estimate(snap, dets, T, None, cfg)
    boxes = box_occupancy.estimate(snap, T, cfg)

    pts = depth_points(snap.depth_m, snap.intrinsics)[::cloud_step, ::cloud_step]
    ok = snap.depth_m[::cloud_step, ::cloud_step] > 0
    cloud, bgr = pts[ok], snap.color[::cloud_step, ::cloud_step][ok]

    tray = cfg["workcell"].get("tray", {})
    tray_view = (tray["center_m"], tray["size_m"]) if tray.get("center_m") is not None else None
    calib = []
    for p in (session or {}).get("points", []):
        if p.get("cam_m") is None or p.get("base_m") is None:
            continue
        pred = camera_to_base(np.asarray(p["cam_m"]), T)
        calib.append({"name": p["name"], "base": np.asarray(p["base_m"]), "pred": pred,
                      "err": float(np.linalg.norm(pred - np.asarray(p["base_m"])))})
    return View(T, cloud, bgr, apples, boxes, tray_view, calib, list(truth), T_true)


def summary(view: View) -> str:
    lines = [f"카메라 위치 (base_link) {np.round(view.T[:3, 3] * 1000, 1).tolist()} mm, 점군 {len(view.cloud_cam)}점"]
    for a in view.apples:
        line = (f"사과 #{a.apple_id}  중심 {np.round(np.array(a.center_m) * 1000, 1).tolist()} mm  "
                f"직경 {a.diameter_m * 1000:.1f} mm  conf {a.confidence:.2f}  간격 {a.clearance_m * 1000:.0f} mm")
        if view.truth:
            (c, r) = min(view.truth, key=lambda t: np.linalg.norm(np.subtract(t[0], a.center_m)))
            line += (f"  | 정답 대비 위치 {np.linalg.norm(np.subtract(c, a.center_m)) * 1000:.1f} mm, "
                     f"직경 {(a.diameter_m - 2 * r) * 1000:+.1f} mm")
        lines.append(line)
    for b in view.boxes:
        cells = " ".join("■" if o else "□" for o in b.occupied)
        lines.append(f"상자 {b.box_id.name:<6} {'유효' if b.valid else '무효'}  칸 {cells}  "
                     f"벽 윗면 {b.wall_height_m * 1000:.0f} mm (측정 {b.wall_height_measured_m * 1000:.0f})")
    if view.calib:
        errs = [c["err"] * 1000 for c in view.calib]
        lines.append(f"캘리브레이션 점 {len(errs)}개  잔차 중앙값 {np.median(errs):.1f} mm / 최대 {max(errs):.1f} mm")
    return "\n".join(lines)


# ================================================================ ROS 2 발행

def _quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """회전 행렬 → (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return (q[0], q[1], q[2], (R[k, j] - R[j, k]) / s)


def publish(view: View, launch_rviz: bool) -> None:
    try:
        import rclpy
        from geometry_msgs.msg import Point, TransformStamped
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import ColorRGBA
        from tf2_ros import StaticTransformBroadcaster
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError as e:
        raise SystemExit("ROS 2 를 찾지 못했다 — 먼저 `source /opt/ros/humble/setup.bash` 를 한다 "
                         "(또는 --no-ros 로 요약만 본다)") from e

    rclpy.init()
    node = rclpy.create_node("aple_m1_view")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub_cloud = node.create_publisher(PointCloud2, "/m1/cloud", qos)
    pub_markers = node.create_publisher(MarkerArray, "/m1/markers", qos)
    tf_pub = StaticTransformBroadcaster(node)

    def tf(child, T):
        m = TransformStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = "base_link", child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, T[:3, 3])
        q = _quat(T[:3, :3])
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = map(float, q)
        return m

    tfs = [tf(CAM_FRAME, view.T)] + ([tf("camera_true", view.T_true)] if view.T_true is not None else [])
    tf_pub.sendTransform(tfs)

    def cloud_msg():
        msg = PointCloud2()
        msg.header.frame_id = CAM_FRAME
        msg.header.stamp = node.get_clock().now().to_msg()
        n = len(view.cloud_cam)
        rgb = (view.cloud_bgr[:, 2].astype(np.uint32) << 16 | view.cloud_bgr[:, 1].astype(np.uint32) << 8
               | view.cloud_bgr[:, 0].astype(np.uint32))
        data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        data["x"], data["y"], data["z"], data["rgb"] = (*view.cloud_cam.T, rgb)
        msg.height, msg.width = 1, n
        msg.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32,
                                 count=1) for i, k in enumerate(("x", "y", "z", "rgb"))]
        msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 16, 16 * n, True
        msg.data = data.tobytes()
        return msg

    markers = MarkerArray()
    ids = iter(range(1, 100000))

    def mk(ns, kind, xyz, scale, rgba, text=""):
        m = Marker()
        m.header.frame_id = "base_link"
        m.ns, m.id, m.type, m.action = ns, next(ids), kind, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = map(float, scale)
        m.color = ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))
        m.text = text
        markers.markers.append(m)
        return m

    def rect(ns, cx, cy, sx, sy, z, rgba, width=0.004):
        m = mk(ns, Marker.LINE_STRIP, (0, 0, 0), (width, 0, 0), rgba)
        for x, y in ((cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2),
                     (cx - sx / 2, cy + sy / 2), (cx - sx / 2, cy - sy / 2)):
            m.points.append(Point(x=float(x), y=float(y), z=float(z)))

    for a in view.apples:
        d = a.diameter_m
        ok = a.confidence >= 0.5
        mk("apples", Marker.SPHERE, a.center_m, (d, d, d), (0.1, 0.9, 0.2, 0.55) if ok else (1.0, 0.55, 0.0, 0.6))
        mk("apple_labels", Marker.TEXT_VIEW_FACING, (a.center_m[0], a.center_m[1], a.top_surface_z_m + 0.03),
           (0, 0, 0.016), (1, 1, 1, 1), f"#{a.apple_id} {d * 1000:.0f}mm c{a.confidence:.2f}")
    for c, r in view.truth:
        mk("truth", Marker.SPHERE, c, (2 * r,) * 3, (1, 1, 1, 0.25))
    for b in view.boxes:
        x0, y0, z0 = b.origin_m
        for k, (h, occ) in enumerate(zip(b.heights_m, b.occupied)):
            row, col = divmod(k, b.cols)
            cx, cy = x0 + (col + 0.5) * b.cell_pitch_m, y0 + (row + 0.5) * b.cell_pitch_m
            unknown = occ and abs(h - b.wall_height_m) < 1e-9
            hz = max(h - z0, 0.002)
            rgba = (0.5, 0.5, 0.5, 0.5) if unknown else (0.9, 0.15, 0.15, 0.6) if occ else (0.1, 0.8, 0.3, 0.35)
            mk("boxes", Marker.CUBE, (cx, cy, z0 + hz / 2), (b.cell_pitch_m * 0.9, b.cell_pitch_m * 0.9, hz), rgba)
        sx, sy = b.cols * b.cell_pitch_m, b.rows * b.cell_pitch_m
        rect("walls", x0 + sx / 2, y0 + sy / 2, sx, sy, b.wall_height_m, (1, 0.8, 0.2, 1))
        mk("box_labels", Marker.TEXT_VIEW_FACING, (x0 + sx / 2, y0 + sy / 2, b.wall_height_m + 0.04), (0, 0, 0.02),
           (1, 0.8, 0.2, 1), b.box_id.name + ("" if b.valid else " (INVALID)"))
    if view.tray:
        (cx, cy, cz), (sx, sy) = view.tray
        rect("tray", cx, cy, sx, sy, cz + 0.002, (0.3, 0.6, 1.0, 1))
    if view.calib:
        lines = mk("calib", Marker.LINE_LIST, (0, 0, 0), (0.002, 0, 0), (1, 0.2, 0.2, 1))
        for c in view.calib:
            mk("calib", Marker.SPHERE, c["base"], (0.012,) * 3, (0.2, 0.4, 1, 1))
            mk("calib", Marker.SPHERE, c["pred"], (0.008,) * 3, (1, 1, 0, 1))
            lines.points += [Point(x=float(c["base"][0]), y=float(c["base"][1]), z=float(c["base"][2])),
                             Point(x=float(c["pred"][0]), y=float(c["pred"][1]), z=float(c["pred"][2]))]
            mk("calib_labels", Marker.TEXT_VIEW_FACING, c["base"] + [0, 0, 0.025], (0, 0, 0.014), (0.6, 0.8, 1, 1),
               f"{c['name']} {c['err'] * 1000:.1f}mm")

    rviz = None
    if launch_rviz:
        if shutil.which("rviz2") is None:
            raise SystemExit("rviz2 를 찾지 못했다 — ros-humble-rviz2 설치 또는 setup.bash source 확인")
        rviz = subprocess.Popen(["rviz2", "-d", str(RVIZ_CONFIG)])

    def tick():
        if not rclpy.ok():                     # Ctrl-C·종료 신호 뒤에는 발행하지 않는다
            return
        pub_cloud.publish(cloud_msg())
        pub_markers.publish(markers)

    tick()
    node.create_timer(1.0, tick)
    print("발행 중: /m1/cloud, /m1/markers, TF base_link → " + CAM_FRAME + "   (Ctrl-C 종료)")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rviz is not None:
            rviz.terminate()
        node.destroy_node()
        rclpy.try_shutdown()


# ================================================================ 실행

def main(argv: list[str] | None = None) -> View:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="합성 장면 (정답 포함). --snapshot 이 없으면 기본")
    ap.add_argument("--snapshot", type=Path, help="스냅샷 .npz (capture_snapshot / judge_live / touch_calib grab)")
    ap.add_argument("--session", type=Path, help="touch_calib 세션 JSON — 캘리브레이션 점 표시")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rviz", action="store_true", help="rviz2 를 설정 파일과 함께 띄운다")
    ap.add_argument("--no-ros", action="store_true", help="ROS 없이 요약만 출력")
    args = ap.parse_args(argv)

    if args.snapshot and not args.demo:
        snap, cfg = load_snapshot(args.snapshot, scene_id=1), load_config()
        session = json.loads(args.session.read_text(encoding="utf-8")) if args.session else None
        view = build_view(snap, cfg, session)
    else:
        snap, cfg, session, truth, T_true = demo_scene(args.seed)
        view = build_view(snap, cfg, session, truth, T_true)
    print(summary(view))
    if not args.no_ros:
        publish(view, args.rviz)
    return view


if __name__ == "__main__":
    main()
