"""ChArUco 보드로 카메라 점검·깊이 오차 확인·eye-to-hand 캘리브레이션 (역할 A).

보드 치수는 configs/charuco.yaml (설계 × print_scale). 이번 인쇄물은 81% 로 나와 칸 28.35 mm, 마커 21.06 mm 다.
스냅샷은 tools/touch_calib.py grab (Piper Studio 공유 메모리) 또는 capture_snapshot.py 로 찍은 .npz 를 쓴다.

1) 보드 한 장 점검 — 보드를 시야에 두고 찍는다 (팔은 시야 밖)
     python tools/charuco_calib.py check data/calib/board.npz --draw data/calib/board_check.png
   모서리 수, 재투영 오차, 카메라→보드 거리·기울기, D435 깊이 − 보드 평면 (깊이 편향)을 보여 준다.
   T_base_camera 가 있으면 보드 평면의 base_link 높이도 보여 준다 (테이블 위에 눕혔다면 ≈ 보드 두께).

2) eye-to-hand — 보드를 그리퍼에 단단히 물리고 자세를 바꿔 가며 10~15장 찍는다
     python tools/touch_calib.py grab --out data/calib/he_01.npz            # 자세마다 한 장
     python tools/charuco_calib.py handeye add data/calib/he_01.npz --pose "x y z rx ry rz" --session data/calib/handeye.json
       --pose: 그 순간의 플랜지 자세 (mm, deg — Piper Studio 말단 자세 RX RY RZ, 고정축 roll-pitch-yaw)
     python tools/charuco_calib.py handeye fit --session data/calib/handeye.json                       # 점검
     python tools/charuco_calib.py handeye fit --session data/calib/handeye.json --save configs/workcell.yaml
   자세는 위치만 옮기지 말고 회전축을 바꿔 가며 (손목을 ±20~30° 기울이고 J6 를 돌려) 찍는다.
   점검: 표본마다 '그리퍼 → 보드' 변환을 따로 계산해 서로 얼마나 일치하는지 본다 (보드는 그리퍼에 고정이므로
   같아야 한다). 어긋난 표본은 --drop 으로 빼고 다시 맞춘다. 경고가 있으면 --force 없이는 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.hal.calibration import T_from_cfg, rpy_to_matrix, save_T_base_camera, solve_eye_to_hand  # noqa: E402
from aple.hal.charuco import BoardPose, board_depth_error, board_spec, estimate_board_pose, make_board  # noqa: E402
from aple.hal.replay import load_snapshot  # noqa: E402


# ================================================================ 점검 1장

def check_snapshot(snap, cfg: dict) -> tuple[BoardPose | None, list[str]]:
    spec = board_spec(cfg)
    pose = estimate_board_pose(snap.color, snap.intrinsics, spec)
    if pose is None:
        return None, ["보드를 찾지 못했다 (모서리 6개 미만) — 보드 전체가 보이는지, 반사·흐림이 없는지 확인"]
    total = (spec.squares_x - 1) * (spec.squares_y - 1)
    lines = [f"모서리 {pose.n_corners}/{total}  재투영 RMS {pose.reproj_rms_px:.2f} px",
             f"카메라 → 보드 평면 {pose.distance_m * 1000:.1f} mm, 기울기 {pose.tilt_deg:.1f}°  "
             f"(칸 {spec.square_m * 1000:.2f} mm · 마커 {spec.marker_m * 1000:.2f} mm 기준)"]
    if pose.reproj_rms_px > spec.max_reproj_px:
        lines.append(f"[경고] 재투영 오차가 {spec.max_reproj_px} px 를 넘는다 — 흐림·구김, 또는 print_scale·내부 파라미터 확인")
    if snap.depth_m is not None and np.any(snap.depth_m > 0):
        err, n = board_depth_error(snap.depth_m, pose, snap.intrinsics)
        if n:
            lines.append(f"깊이 − 보드 평면 = {err * 1000:+.1f} mm (모서리 {n}개 중앙값)"
                         + ("  [경고] 5 mm 넘음 — 깊이 인코딩 창·정렬 보정·print_scale 확인" if abs(err) > 0.005 else ""))
    T = T_from_cfg(cfg)
    if T is not None:
        T_base_board = T @ pose.T_camera_board
        c = T_base_board[:3, :3] @ np.array([spec.squares_x * spec.square_m / 2, spec.squares_y * spec.square_m / 2, 0]) \
            + T_base_board[:3, 3]
        tilt_base = np.degrees(np.arccos(min(1.0, abs(T_base_board[2, 2]))))
        lines.append(f"base_link 에서 보드 중심 ({c[0] * 1000:.1f}, {c[1] * 1000:.1f}, {c[2] * 1000:.1f}) mm, "
                     f"수평 대비 {tilt_base:.1f}°")
    return pose, lines


def draw_pose(color: np.ndarray, pose: BoardPose, intrinsics, axis_m: float = 0.05) -> np.ndarray:
    img = color.copy()
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    for u, v in pose.image_points:
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 255, 255), 1)
    rvec = cv2.Rodrigues(pose.T_camera_board[:3, :3])[0]
    cv2.drawFrameAxes(img, K, np.zeros(5), rvec, pose.T_camera_board[:3, 3], axis_m)
    return img


# ================================================================ eye-to-hand

def flange_pose(text: str) -> np.ndarray:
    """'x y z rx ry rz' (mm, deg) → 4×4 T_base_gripper (m)."""
    v = [float(t) for t in text.replace(",", " ").split()]
    if len(v) != 6:
        raise ValueError("--pose 는 숫자 6개: x y z (mm) rx ry rz (deg)")
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*np.deg2rad(v[3:]))
    T[:3, 3] = np.asarray(v[:3]) / 1000
    return T


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0))))


_METHODS = [("PARK", cv2.CALIB_HAND_EYE_PARK), ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
            ("TSAI", cv2.CALIB_HAND_EYE_TSAI), ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)]


def _consistency(T: np.ndarray, Tbg: list, Tcb: list) -> tuple[np.ndarray, np.ndarray]:
    """표본마다 그리퍼 → 보드 변환을 계산해 평균에서 벗어난 정도 (m, deg). 보드가 고정이면 0 이어야 한다."""
    Tgb = [np.linalg.inv(a) @ T @ b for a, b in zip(Tbg, Tcb)]
    U, _, Vt = np.linalg.svd(sum(t[:3, :3] for t in Tgb))
    R_mean = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    t_mean = np.mean([t[:3, 3] for t in Tgb], axis=0)
    return (np.array([np.linalg.norm(t[:3, 3] - t_mean) for t in Tgb]),
            np.array([_angle_deg(t[:3, :3], R_mean) for t in Tgb]))


def handeye_fit(samples: list[dict], cfg: dict) -> dict:
    """→ {T, method, spread_m, spread_deg (표본별), warnings}.

    OpenCV 해법 네 가지를 모두 풀고 표본 일관성(그리퍼↔보드 편차)이 가장 좋은 것을 고른다.
    기본값 TSAI 는 표본 간 상대 회전이 큰 자세 묶음(권장 촬영법)에서 엉뚱한 해를 내는 경우가 있다 (시험에서 확인).
    """
    he = cfg["charuco"].get("handeye", {})
    Tbg = [np.asarray(s["T_base_gripper"]) for s in samples]
    Tcb = [np.asarray(s["T_camera_board"]) for s in samples]
    best = None
    for name, flag in _METHODS:
        try:
            T = solve_eye_to_hand(Tbg, Tcb, method=flag)
        except (cv2.error, ValueError):
            continue
        sm, sd = _consistency(T, Tbg, Tcb)
        score = float(np.median(sm) + 0.001 * np.median(sd))       # 1° ≈ 1 mm 로 묶어 비교
        if best is None or score < best[0]:
            best = (score, name, T, sm, sd)
    if best is None:
        raise ValueError("eye-to-hand 해를 구하지 못했다 — 표본 자세를 더 다양하게")
    _, method, T, spread_m, spread_deg = best

    warn = []
    n_min = int(he.get("min_samples", 10))
    if len(samples) < n_min:
        warn.append(f"표본이 {len(samples)}개다 — {n_min}개 이상 권장")
    rot_range = max(_angle_deg(a[:3, :3], b[:3, :3]) for a in Tbg for b in Tbg)
    if rot_range < 30:
        warn.append(f"그리퍼 자세 회전 범위가 {rot_range:.0f}° 뿐이다 — 손목을 기울이고 J6 를 돌려 30° 이상 다양하게")
    lim_m, lim_deg = float(he.get("max_spread_m", 0.003)), float(he.get("max_spread_deg", 1.0))
    for s, dm, dd in zip(samples, spread_m, spread_deg):
        if dm > lim_m or dd > lim_deg:
            warn.append(f"{s['name']}: 그리퍼↔보드 편차 {dm * 1000:.1f} mm / {dd:.2f}° — 자세 입력·보드 흔들림 확인 (--drop {s['name']})")
    if T[2, 2] > -0.5:
        warn.append("카메라 광축이 아래를 보지 않는다 — 자세 단위(mm·deg)나 RX RY RZ 순서 확인")
    return {"T": T, "method": method, "spread_m": spread_m, "spread_deg": spread_deg, "warnings": warn}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"samples": []}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ================================================================ 명령

def cmd_check(args):
    cfg = load_config()
    snap = load_snapshot(args.snapshot)
    pose, lines = check_snapshot(snap, cfg)
    print("\n".join(lines))
    if pose is not None and args.draw:
        args.draw.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.draw), draw_pose(snap.color, pose, snap.intrinsics))
        print(f"그림: {args.draw}")
    return pose


def cmd_add(args):
    cfg = load_config()
    snap = load_snapshot(args.snapshot)
    pose = estimate_board_pose(snap.color, snap.intrinsics, board_spec(cfg))
    if pose is None:
        raise SystemExit("보드를 찾지 못했다 — 이 자세는 건너뛰고 다시 찍는다")
    data = _load(args.session)
    names = {s["name"] for s in data["samples"]}
    k = len(data["samples"]) + 1
    while f"S{k}" in names:
        k += 1
    sample = {"name": f"S{k}", "snapshot": str(args.snapshot),
              "T_base_gripper": flange_pose(args.pose).round(6).tolist(),
              "T_camera_board": pose.T_camera_board.round(6).tolist(),
              "n_corners": pose.n_corners, "reproj_px": round(pose.reproj_rms_px, 3)}
    data["samples"].append(sample)
    _save(args.session, data)
    print(f"{sample['name']} 추가: 모서리 {pose.n_corners}, 재투영 {pose.reproj_rms_px:.2f} px, "
          f"보드 거리 {pose.distance_m * 1000:.0f} mm  (표본 {len(data['samples'])}개)")
    return sample


def cmd_fit(args):
    cfg = load_config()
    data = _load(args.session)
    unknown = set(args.drop) - {s["name"] for s in data["samples"]}
    if unknown:
        raise SystemExit(f"세션에 없는 표본: {sorted(unknown)}")
    samples = [s for s in data["samples"] if s["name"] not in args.drop]
    if len(samples) < 3:
        raise SystemExit(f"표본이 {len(samples)}개다 — 3개 이상 (권장 10~15개) 필요")
    r = handeye_fit(samples, cfg)
    print(f"{'표본':<6}{'모서리':>6}{'재투영 px':>10}{'편차 mm':>9}{'편차 °':>8}")
    for s, dm, dd in zip(samples, r["spread_m"], r["spread_deg"]):
        print(f"{s['name']:<6}{s['n_corners']:>6}{s['reproj_px']:>10.2f}{dm * 1000:>9.1f}{dd:>8.2f}")
    c = r["T"][:3, 3]
    print(f"\n표본 {len(samples)}개  해법 {r['method']}  편차 중앙값 {np.median(r['spread_m']) * 1000:.1f} mm / {np.median(r['spread_deg']):.2f}°")
    print(f"카메라 위치 (base_link) ({c[0] * 1000:.1f}, {c[1] * 1000:.1f}, {c[2] * 1000:.1f}) mm, "
          f"광축 방향 ({', '.join(f'{x:.3f}' for x in r['T'][:3, 2])})")
    for w in r["warnings"]:
        print(f"[경고] {w}")
    data["fit"] = {"T_base_camera": r["T"].round(6).tolist(), "used": [s["name"] for s in samples],
                   "drop": list(args.drop)}
    _save(args.session, data)
    if args.save:
        if r["warnings"] and not args.force:
            raise SystemExit("경고가 있어 저장하지 않았다 — 확인 후 --force 로 저장하거나 표본을 다시 찍는다")
        line = save_T_base_camera(str(args.save), r["T"])
        print(f"저장: {args.save}\n  {line[:100]}...")
    return r


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="스냅샷 한 장에서 보드 자세·깊이 편향 점검")
    c.add_argument("snapshot", type=Path)
    c.add_argument("--draw", type=Path, help="모서리·축을 그린 그림 저장")

    he = sub.add_parser("handeye", help="보드를 그리퍼에 물린 eye-to-hand 캘리브레이션")
    hs = he.add_subparsers(dest="he_cmd", required=True)
    a = hs.add_parser("add", help="표본 추가 (스냅샷 + 그 순간의 플랜지 자세)")
    a.add_argument("snapshot", type=Path)
    a.add_argument("--pose", required=True, help="'x y z rx ry rz' — mm, deg")
    a.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    f = hs.add_parser("fit", help="T_base_camera 를 맞추고 점검, --save 로 저장")
    f.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    f.add_argument("--drop", nargs="*", default=[], help="뺄 표본 이름 (예: S3)")
    f.add_argument("--save", type=Path, help="T_base_camera 를 쓸 workcell.yaml")
    f.add_argument("--force", action="store_true", help="경고가 있어도 저장")

    args = ap.parse_args(argv)
    if args.cmd == "check":
        return cmd_check(args)
    return {"add": cmd_add, "fit": cmd_fit}[args.he_cmd](args)


if __name__ == "__main__":
    main()
