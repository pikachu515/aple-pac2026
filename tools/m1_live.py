"""실기에서 M1 (역할 A) 돌려 보기 — Piper Studio 를 켠 채로 D435 장면을 인식하고, 로봇이 할 일을 미리 본다.

팔은 움직이지 않는다. 로봇 실행은 역할 B 의 aple/execution (piper_web.enabled 이후)에서 한다.
여기서는 '실기 카메라 입력 → 사과 3D 위치·직경 → 상자 점유 → 파지 후보 → (역할 C 판정) → 역할 B 계획'
경로가 현장 데이터로 끝까지 맞물리는지 확인한다 (전략 IV장 2단계, S12-4 드라이런).

    python tools/m1_live.py                                   # 실시간 1회 스캔 (configs/live_camera.yaml)
    python tools/m1_live.py --scans 5 --wait                  # Enter 마다 다시 스캔 (사과를 옮겨 가며 추적 확인)
    python tools/m1_live.py --calipers 74.5,68,81.2           # 캘리퍼스 직경(mm, 사과 번호 순)과 비교 → 직경 오차
    python tools/m1_live.py --plan                            # 역할 C 판정 + 역할 B 계획까지 드라이런
    python tools/m1_live.py --save data/live_m1               # 스냅샷 저장 (나중에 --snapshot 으로 다시)
    python tools/m1_live.py --snapshot data/live_m1/scan_0001.npz --plan
    python tools/m1_live.py --rviz                            # 마지막 스캔을 RViz2 로 (source /opt/ros/humble/setup.bash)

사과 번호(#)는 영상에서 위→아래, 왼→오 순서다 (검출 번호 det). 캘리퍼스 값도 그 순서로 넣는다.
준비: tools/touch_calib.py fit --save (T_base_camera), workcell.tray.center_m · boxes.*.origin_m 실측
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.hal.calibration import T_from_cfg, camera_to_base  # noqa: E402
from aple.hal.replay import load_snapshot  # noqa: E402
from aple.perception import box_occupancy, detector, grasp_planner, pose_estimator  # noqa: E402
from aple.perception.geometry import depth_points  # noqa: E402
from aple.schema import SceneSnapshot  # noqa: E402


# ================================================================ 점검

def preflight(cfg: dict) -> list[str]:
    """스캔 전에 실측값이 채워졌는지. 빈 값은 경고만 하고 계속한다 (T 만 필수)."""
    notes = []
    wc = cfg["workcell"]
    if T_from_cfg(cfg) is None:
        raise SystemExit("workcell.T_base_camera 가 비어 있다 — 먼저 tools/touch_calib.py 로 캘리브레이션한다")
    if wc["tray"].get("center_m") is None:
        notes.append("workcell.tray.center_m 미실측 — 영상 전체에서 사과를 찾는다 (상자 밖 물체도 사과로 잡힐 수 있다)")
    missing = [k for k, b in wc["boxes"].items() if b.get("origin_m") is None]
    if missing:
        notes.append(f"상자 origin_m 미실측: {missing} — 그 상자는 점유를 재지 않고 계획에서 빠진다")
    return notes


def support_height_check(snap: SceneSnapshot, cfg: dict, dets) -> str | None:
    """사과·상자 밖 받침면 높이 (base z 중앙값) 가 트레이 높이와 1 cm 넘게 다르면 캘리브레이션을 의심한다."""
    T = T_from_cfg(cfg)
    roi = detector.work_roi(snap, cfg)
    for d in dets:
        roi &= ~d.mask
    sel = roi & (snap.depth_m > 0)
    if sel.sum() < 500:
        return None
    z = camera_to_base(depth_points(snap.depth_m, snap.intrinsics)[sel], T)[:, 2]
    tray = cfg["workcell"]["tray"]
    expect = tray["center_m"][2] if tray.get("center_m") is not None else 0.0
    zm = float(np.median(z))
    if abs(zm - expect) > 0.01:
        return (f"받침면 높이 {zm * 1000:.0f} mm ≠ 기대 {expect * 1000:.0f} mm — T_base_camera 또는 트레이 높이 확인 "
                "(캘리브레이션 이후 카메라가 움직였을 수 있다)")
    return None


# ================================================================ 스캔 1회

def scan(snap: SceneSnapshot, cfg: dict, prev=None) -> dict:
    T = T_from_cfg(cfg)
    dets = detector.detect(snap, cfg)
    apples = pose_estimator.estimate(snap, dets, T, prev, cfg)
    boxes = box_occupancy.estimate(snap, T, cfg)
    grasps = grasp_planner.plan_grasps(apples, cfg)
    return {"snap": snap, "dets": dets, "apples": apples, "boxes": boxes, "grasps": grasps,
            "valid_depth": float((snap.depth_m > 0).mean()), "support": support_height_check(snap, cfg, dets)}


def report(r: dict, calipers_mm: list[float] | None = None) -> str:
    lines = [f"[scene {r['snap'].scene_id}] 유효 깊이 {r['valid_depth'] * 100:.0f}%  검출 {len(r['dets'])}개"]
    if r["support"]:
        lines.append(f"  [경고] {r['support']}")
    grasp = {g.apple_id: g for g in r["grasps"]}
    by_det = sorted(r["apples"], key=lambda a: (a.det_id == 0, a.det_id))
    for a in by_det:
        g = grasp.get(a.apple_id)
        gtxt = (f"  파지 yaw {np.degrees(g.grasp_yaw_rad):3.0f}° 간격 {g.min_clearance_m * 1000:4.0f} mm "
                f"{'OK' if g.collision_free else '좁음→DEFER'}") if g else "  (가려짐)"
        lines.append(f"  #{a.det_id or '-'} id{a.apple_id}  중심 ({a.center_m[0] * 1000:6.1f}, {a.center_m[1] * 1000:6.1f}, "
                     f"{a.center_m[2] * 1000:5.1f}) mm  직경 {a.diameter_m * 1000:5.1f} mm  conf {a.confidence:.2f}  "
                     f"깊이 {a.depth_valid_ratio * 100:3.0f}%{gtxt}")
    for b in r["boxes"]:
        cells = " ".join("■" if o else "□" for o in b.occupied)
        lines.append(f"  상자 {b.box_id.name:<6} {'유효' if b.valid else '무효'}  {cells}  "
                     f"벽 윗면 설정 {b.wall_height_m * 1000:.0f} / 측정 {b.wall_height_measured_m * 1000:.0f} mm")
    if calipers_mm:
        lines.append(caliper_table(r, calipers_mm))
    return "\n".join(lines)


def caliper_table(r: dict, calipers_mm: list[float]) -> str:
    """역할 A 핵심 지표: 직경 오차 (캘리퍼스 대비, 목표 ±3 mm — 전략 4-4)."""
    visible = sorted((a for a in r["apples"] if a.det_id), key=lambda a: a.det_id)
    if len(visible) != len(calipers_mm):
        return f"  [캘리퍼스] 검출 {len(visible)}개 ≠ 입력 {len(calipers_mm)}개 — 사과 번호 순서대로 같은 개수를 넣는다"
    errs = [a.diameter_m * 1000 - c for a, c in zip(visible, calipers_mm)]
    rows = [f"    #{a.det_id}  측정 {a.diameter_m * 1000:5.1f}  캘리퍼스 {c:5.1f}  오차 {e:+5.1f} mm"
            for a, c, e in zip(visible, calipers_mm, errs)]
    ok = max(abs(e) for e in errs) <= 3.0
    rows.append(f"    평균 {np.mean(errs):+.1f} mm, 최대 |오차| {max(abs(e) for e in errs):.1f} mm → "
                + ("목표 ±3 mm 충족" if ok else "목표 ±3 mm 미달"))
    return "  [캘리퍼스]\n" + "\n".join(rows)


def dry_run_plan(r: dict, cfg: dict) -> str:
    """역할 C 판정(inspect_scene) + 역할 B 계획(PickPlacePipeline.plan_scene). 로봇 명령은 보내지 않는다."""
    from aple.execution.pick_place_pipeline import PickPlacePipeline
    from aple.inspection.coloration import params_from_config
    from aple.inspection.pipeline import inspect_scene

    if not r["apples"]:
        return "  [계획] 사과가 없다"
    grades = inspect_scene(r["snap"], r["dets"], r["apples"], params_from_config(cfg))
    plan = PickPlacePipeline(cfg).plan_scene(r["apples"], grades, r["boxes"], r["grasps"])
    gb = {g.apple_id: g for g in grades}
    lines = ["  [계획 — 드라이런, 로봇 명령 없음]  (순서 = 실행 순서)"]
    for item in plan.items:
        a, t = item.apple, item.task
        g = gb.get(a.apple_id)
        grade = f"{g.grade.name:<6} 착색 {g.ratio * 100:3.0f}% {g.reject_reason.name}" if g else "-"
        where = ""
        if item.placement is not None:
            pl = item.placement
            where = (f" → {pl.box_id.name} 칸 {pl.cell_index}" if pl.feasible else " → 놓을 칸 없음 (P1)")
        lines.append(f"    id{a.apple_id}  {t.action.name:<18} {grade}{where}")
    return "\n".join(lines)


# ================================================================ 실행

def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, help="저장한 스냅샷 .npz (없으면 Piper Studio 실시간)")
    ap.add_argument("--scans", type=int, default=1)
    ap.add_argument("--wait", action="store_true", help="스캔마다 Enter 를 기다린다")
    ap.add_argument("--calipers", help="캘리퍼스 직경 mm, 사과 번호 순 (예: 74.5,68,81.2)")
    ap.add_argument("--plan", action="store_true", help="역할 C 판정 + 역할 B 계획 드라이런")
    ap.add_argument("--save", type=Path, help="스냅샷을 저장할 폴더")
    ap.add_argument("--rviz", action="store_true", help="마지막 스캔을 RViz2 로 보기")
    args = ap.parse_args(argv)

    cfg = load_config()
    for n in preflight(cfg):
        print(f"[알림] {n}")
    calipers = [float(v) for v in args.calipers.split(",")] if args.calipers else None

    if args.snapshot:
        frames = iter([load_snapshot(args.snapshot, scene_id=1)])
        grab = lambda: next(frames)                                   # noqa: E731
        args.scans = 1
    else:
        from aple.hal.live_snapshot import PiperLiveCamera
        try:
            grab = PiperLiveCamera.from_config(cfg)
        except (FileNotFoundError, RuntimeError) as e:
            raise SystemExit(str(e))

    results, prev = [], None
    for k in range(args.scans):
        if args.wait and k > 0:
            input("Enter = 다시 스캔 (팔이 시야 밖인지 확인) ")
        r = scan(grab(), cfg, prev)
        prev = r["apples"]
        print(report(r, calipers))
        if args.plan:
            print(dry_run_plan(r, cfg))
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
            s = r["snap"]
            out = args.save / f"scan_{s.scene_id:04d}.npz"
            np.savez_compressed(out, color=s.color, depth_m=s.depth_m, intrinsics=np.array(s.intrinsics),
                                stamp_s=s.stamp_s)
            print(f"  저장 {out}")
        results.append(r)

    if args.rviz and results:
        from tools.rviz_view import View, publish
        r = results[-1]
        s = r["snap"]
        pts = depth_points(s.depth_m, s.intrinsics)[::2, ::2]
        ok = s.depth_m[::2, ::2] > 0
        tray = cfg["workcell"]["tray"]
        publish(View(T_from_cfg(cfg), pts[ok], s.color[::2, ::2][ok], r["apples"], r["boxes"],
                     (tray["center_m"], tray["size_m"]) if tray.get("center_m") is not None else None), True)
    return results


if __name__ == "__main__":
    main()
