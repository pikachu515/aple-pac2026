"""카메라–로봇 변환 간이 캘리브레이션 — 표식 6~10개를 카메라로 찍고 로봇 끝점으로 터치한다 (역할 A).

결과 T_base_camera 를 configs/workcell.yaml 에 써 넣는다. 카메라를 옮기거나 거치대가 흔들리면 다시 한다.
세션 파일(JSON)에 점마다 픽셀·카메라 좌표·로봇 좌표가 쌓이므로 단계를 나눠 해도 된다.

준비물
- 표식 6~10개: 무광 십자 스티커나 납작한 와셔. 트레이·상자 양 끝까지 작업 영역 전체에 흩는다.
  2~3개는 받침대(상자 윗면 등) 위에 두어 높이를 섞는다. 반짝이는 표식은 깊이가 비어 안 된다.
- 끝이 뾰족한 터치 팁(그리퍼에 물린 핀 등). 플랜지 → 팁 끝 거리를 재서 workcell.tcp_offset_m 에 적거나,
  로봇 좌표를 팁 끝 기준으로 직접 입력한다.

절차
  1. 팔을 카메라 시야 밖으로 보내고, 표식을 놓은 장면을 한 장 찍는다 (Piper Studio 켠 채로)
       python tools/touch_calib.py grab --out data/calib/touch.npz --near 150 --far 1200
     (Piper Studio 없이 찍은 스냅샷 .npz 도 된다 — capture_snapshot.py, judge_live.py --once 결과)
  2. 영상에서 표식 중심을 차례로 클릭한다 (확대 창이 따라다닌다)
       python tools/touch_calib.py mark data/calib/touch.npz --session data/calib/touch.json
       키: 클릭 = 점 추가   u = 마지막 점 취소   q / Enter = 끝 (클릭할 때마다 저장된다)
     이 뒤로 표식을 움직이지 않는다.
  3. 로봇 끝점을 표식 중심에 차례로 대고 좌표를 입력한다 (클릭한 순서와 같게)
       python tools/touch_calib.py robot --session data/calib/touch.json
       입력: "x y z"            팁 끝 위치 (mm, base_link)
             "x y z rx ry rz"   플랜지 위치 (mm) + 자세 (deg) → workcell.tcp_offset_m 으로 팁 끝 계산
             빈 줄 = 건너뛰기, q = 중단
       --auto: Piper Web 관절각 + Piper Studio 순기구학(piper_robot.kinematics)으로 자동 기록
               (piper_web.enabled 와 follower iface 가 설정돼 있어야 한다)
  4. 맞추고 점검한 뒤 저장한다
       python tools/touch_calib.py fit --session data/calib/touch.json                  # 결과만 보기
       python tools/touch_calib.py fit --session ... --drop P4                          # 의심 점 빼기
       python tools/touch_calib.py fit --session ... --holdout P8                       # P8 은 검증용으로만
       python tools/touch_calib.py fit --session ... --save configs/workcell.yaml       # 저장
     leave-one-out 오차 = 그 점을 빼고 맞춘 변환으로 그 점을 예측한 오차. 처음 보는 사과 위치에 대한
     정확도 추정이다. 목표는 중앙값 3 mm 이내 (전략 4-4, 스크립트 실행 S9-6).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.hal.calibration import (TouchFit, camera_point_at, fit_touch_points, rpy_to_matrix,  # noqa: E402
                                  save_T_base_camera, tcp_offset_from_cfg, tip_from_flange)
from aple.hal.replay import load_snapshot  # noqa: E402
from aple.hal.shm_frames import SHM_DIR  # noqa: E402


# ================================================================ 세션 파일

def _mm(v) -> str:
    return "(" + ", ".join(f"{x * 1000:.1f}" for x in v) + ") mm"


def load_session(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"snapshot": None, "intrinsics": None, "points": []}


def save_session(path: Path, session: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")


def add_point(session: dict, snap, u: float, v: float) -> dict:
    """클릭한 픽셀 → 카메라 좌표를 구해 세션에 점을 추가한다."""
    cam = camera_point_at(snap.depth_m, u, v, snap.intrinsics)
    names = {p["name"] for p in session["points"]}
    k = len(session["points"]) + 1
    while f"P{k}" in names:
        k += 1
    pt = {"name": f"P{k}", "pixel": [round(float(u), 1), round(float(v), 1)],
          "cam_m": [round(float(x), 5) for x in cam], "base_m": None}
    session["points"].append(pt)
    return pt


def parse_robot_entry(text: str, tcp_offset_m) -> list[float] | None:
    """'x y z' (mm, 팁 끝) 또는 'x y z rx ry rz' (mm, deg, 플랜지) → 팁 끝 base_link 좌표 (m). 빈 줄 → None."""
    vals = [float(t) for t in text.replace(",", " ").split()]
    if not vals:
        return None
    if len(vals) == 3:
        return [v / 1000 for v in vals]
    if len(vals) == 6:
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.deg2rad(vals[3:]))
        T[:3, 3] = np.asarray(vals[:3]) / 1000
        return [float(v) for v in tip_from_flange(T, tcp_offset_m)]
    raise ValueError("숫자 3개(x y z mm) 또는 6개(x y z mm, rx ry rz deg)를 넣는다")


def fit_session(session: dict, drop=(), holdout=()) -> tuple:
    """→ (TouchFit, 맞춤에 쓴 점 목록, [(검증 점, 오차 m)])."""
    unknown = (set(drop) | set(holdout)) - {p["name"] for p in session["points"]}
    if unknown:
        raise SystemExit(f"세션에 없는 점: {sorted(unknown)}")
    done = [p for p in session["points"] if p["cam_m"] is not None and p["base_m"] is not None]
    used = [p for p in done if p["name"] not in drop and p["name"] not in holdout]
    if len(used) < 3:
        raise SystemExit(f"로봇 좌표까지 있는 점이 {len(used)}개다 — 3개 이상 (권장 6~10개) 필요")
    fit = fit_touch_points([p["cam_m"] for p in used], [p["base_m"] for p in used])
    checks = []
    for p in done:
        if p["name"] in holdout:
            pred = fit.T[:3, :3] @ np.asarray(p["cam_m"]) + fit.T[:3, 3]
            checks.append((p, float(np.linalg.norm(pred - np.asarray(p["base_m"])))))
    return fit, used, checks


# ================================================================ 명령

def cmd_grab(args) -> None:
    from aple.hal.depth_align import D435Calib, DepthEncoding, snapshot_from_piper
    from aple.hal.shm_frames import ShmError, ShmReader, find_realsense_serials, read_pair, segment_path

    if not args.calib.exists():
        raise SystemExit(f"깊이→컬러 보정 파일이 없다: {args.calib} — docs/piper_studio_integration.md 2-3절")
    calib = D435Calib.load(args.calib)
    serial = args.serial or next(iter(find_realsense_serials(args.shm_dir)), None)
    if serial is None:
        raise SystemExit("공유 메모리에 RealSense 영상이 없다 — Piper Studio 에서 D435 를 켰는지 확인")
    try:
        color_r = ShmReader(segment_path(f"rs:{serial}:color", args.shm_dir))
        depth_r = ShmReader(segment_path(f"rs:{serial}:depth", args.shm_dir))
    except ShmError as e:
        raise SystemExit(str(e))
    pair = read_pair(color_r, depth_r, timeout_s=3.0)
    if pair is None:
        raise SystemExit("컬러·깊이 프레임을 받지 못했다 — 스트림이 켜져 있는지 확인")
    color, depth_enc, _ = pair
    snap = snapshot_from_piper(color, depth_enc, DepthEncoding(args.near, args.far), calib, 0, time.time())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, color=color, depth_m=snap.depth_m, intrinsics=np.array(snap.intrinsics),
                        stamp_s=snap.stamp_s)
    valid = float((snap.depth_m > 0).mean())
    print(f"저장: {args.out}  (유효 깊이 {valid * 100:.0f}%)")


def cmd_mark(args) -> None:
    from aple.hal.depth_correction import DepthCorrection
    snap = DepthCorrection.from_cfg(load_config()).snapshot(load_snapshot(args.snapshot))   # 저장 파일은 보정 전 깊이
    session = load_session(args.session)
    if session["snapshot"] not in (None, str(args.snapshot)) and session["points"]:
        raise SystemExit(f"이 세션은 다른 스냅샷({session['snapshot']})으로 시작했다 — 새 --session 을 쓴다")
    session["snapshot"], session["intrinsics"] = str(args.snapshot), list(snap.intrinsics)

    if args.pixels:                                       # 화면 없이: "u,v;u,v;..."
        for item in args.pixels.split(";"):
            u, v = (float(t) for t in item.split(","))
            pt = add_point(session, snap, u, v)
            print(f"{pt['name']}  픽셀 ({u:.0f}, {v:.0f})  카메라 {_mm(pt['cam_m'])}")
        save_session(args.session, session)
        return

    import cv2
    win = "touch_calib mark (click / u undo / q done)"
    state = {"mouse": (0, 0)}

    def on_mouse(event, x, y, flags, _):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                pt = add_point(session, snap, x, y)
                save_session(args.session, session)
                print(f"{pt['name']}  픽셀 ({x}, {y})  카메라 {_mm(pt['cam_m'])}")
            except ValueError as e:
                print(f"[무시] {e}")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = snap.color.copy()
        for p in session["points"]:
            u, v = (int(round(c)) for c in p["pixel"])
            cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
            cv2.putText(img, p["name"], (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        mx, my = state["mouse"]
        r, z = 20, 5                                      # 커서 주변 40 px 를 5배 확대
        pad = cv2.copyMakeBorder(snap.color, r, r, r, r, cv2.BORDER_CONSTANT)
        crop = cv2.resize(pad[my:my + 2 * r, mx:mx + 2 * r], (2 * r * z, 2 * r * z), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(crop, (r * z, r * z), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
        img[:crop.shape[0], img.shape[1] - crop.shape[1]:] = crop
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 13):
            break
        if k == ord("u") and session["points"]:
            print(f"취소: {session['points'].pop()['name']}")
            save_session(args.session, session)
    cv2.destroyAllWindows()
    print(f"점 {len(session['points'])}개 → {args.session}")


def _piper_fk():
    """Piper Studio 순기구학 함수 (관절각 rad 6개 → 4×4 플랜지 자세). 함수 이름은 설치본에서 찾는다."""
    from importlib import import_module
    try:
        kin = import_module("piper_robot.kinematics")
    except ImportError as e:
        raise SystemExit("piper_robot.kinematics 를 불러올 수 없다 — Piper Studio 와 같은 Python 환경에서 "
                         "실행하거나 수동 입력을 쓴다") from e
    for name in ("fk", "forward_kinematics", "forward"):
        fn = getattr(kin, name, None)
        if callable(fn):
            def run(q, fn=fn):
                out = fn(np.asarray(q, dtype=float))
                T = np.asarray(out["T"] if isinstance(out, dict) else out, dtype=float)
                if T.shape != (4, 4):
                    raise SystemExit(f"piper_robot.kinematics.{name} 결과가 4×4 가 아니다 ({T.shape})")
                if np.abs(T[:3, 3]).max() > 3.0:          # mm 로 주는 구현 대비
                    T = T.copy()
                    T[:3, 3] /= 1000.0
                return T
            return run
    public = [n for n in dir(kin) if not n.startswith("_")]
    raise SystemExit(f"piper_robot.kinematics 에 순기구학 함수가 없다 (있는 것: {public}) — 수동 입력을 쓴다")


def cmd_robot(args) -> None:
    session = load_session(args.session)
    if not session["points"]:
        raise SystemExit("세션에 점이 없다 — mark 를 먼저 한다")
    cfg = load_config()
    offset = tcp_offset_from_cfg(cfg)
    print(f"tcp_offset_m = {offset.tolist()} (플랜지 자세를 넣을 때만 쓰인다)")
    fk = studio = None
    if args.auto:
        from aple.hal.piper_studio import PiperStudio
        fk, studio = _piper_fk(), PiperStudio(cfg)

    for p in session["points"]:
        if p["base_m"] is not None and not args.redo:
            continue
        while True:
            label = f"{p['name']} (영상 {p['pixel'][0]:.0f}, {p['pixel'][1]:.0f})"
            text = input(f"{label}: 팁을 표식 중심에 대고 " + ("Enter" if args.auto else "좌표 입력") + " > ").strip()
            if text.lower() == "q":
                save_session(args.session, session)
                return
            try:
                if args.auto:
                    if text:                          # 자동 모드에서도 숫자를 넣으면 그 값을 쓴다
                        base = parse_robot_entry(text, offset)
                    else:
                        base = [float(v) for v in tip_from_flange(fk(studio.joint_state()), offset)]
                else:
                    base = parse_robot_entry(text, offset)
            except ValueError as e:
                print(f"  [다시] {e}")
                continue
            break
        if base is None:
            print("  건너뜀")
            continue
        p["base_m"] = [round(v, 5) for v in base]
        save_session(args.session, session)
        print(f"  기록 {_mm(base)}")
    print("완료 — 다음: fit")


def cmd_fit(args) -> TouchFit:
    session = load_session(args.session)
    fit, used, checks = fit_session(session, args.drop, args.holdout)
    print(f"{'점':<5}{'픽셀':>14}{'잔차 mm':>10}{'LOO mm':>10}")
    for p, res, loo in zip(used, fit.residuals_m, fit.loo_errors_m):
        loo_s = f"{loo * 1000:.1f}" if np.isfinite(loo) else "-"
        print(f"{p['name']:<5}{p['pixel'][0]:>8.0f},{p['pixel'][1]:<5.0f}{res * 1000:>10.1f}{loo_s:>10}")
    for p, err in checks:
        print(f"{p['name']:<5}{'(검증)':>14}{'':>10}{err * 1000:>10.1f}")
    finite = fit.loo_errors_m[np.isfinite(fit.loo_errors_m)]
    print(f"\n점 {len(used)}개  RMS {fit.rms_m * 1000:.2f} mm"
          + (f"  LOO 중앙값 {np.median(finite) * 1000:.2f} mm / 최대 {finite.max() * 1000:.2f} mm" if len(finite) else ""))
    cam_pos = fit.T[:3, 3]
    print(f"카메라 위치 (base_link) {_mm(cam_pos)}, 광축 방향 ({', '.join(f'{x:.3f}' for x in fit.T[:3, 2])})")
    for w in fit.warnings:
        print(f"[경고] {w}")

    session["fit"] = {"T_base_camera": fit.T.round(6).tolist(), "rms_m": fit.rms_m,
                      "used": [p["name"] for p in used], "drop": list(args.drop), "holdout": list(args.holdout)}
    save_session(args.session, session)
    if args.save:
        if fit.warnings and not args.force:
            raise SystemExit("경고가 있어 저장하지 않았다 — 확인 후 --force 로 저장하거나 점을 다시 잰다")
        line = save_T_base_camera(str(args.save), fit.T)
        print(f"저장: {args.save}\n  {line[:100]}...")
    return fit



def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grab", help="Piper Studio 공유 메모리에서 스냅샷 한 장 저장")
    g.add_argument("--out", type=Path, default=ROOT / "data" / "calib" / "touch.npz")
    g.add_argument("--calib", type=Path, default=ROOT / "configs" / "calib" / "d435.json")
    g.add_argument("--near", type=float, default=150.0, help="Piper Studio 깊이 인코딩 near_mm")
    g.add_argument("--far", type=float, default=1200.0, help="Piper Studio 깊이 인코딩 far_mm")
    g.add_argument("--serial")
    g.add_argument("--shm-dir", type=Path, default=SHM_DIR)

    m = sub.add_parser("mark", help="스냅샷에서 표식 중심 클릭")
    m.add_argument("snapshot", type=Path)
    m.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "touch.json")
    m.add_argument("--pixels", help="화면 없이 픽셀 지정: 'u,v;u,v;...'")

    r = sub.add_parser("robot", help="점마다 로봇 끝점 좌표 기록")
    r.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "touch.json")
    r.add_argument("--auto", action="store_true", help="관절각 + Piper 순기구학으로 자동 기록")
    r.add_argument("--redo", action="store_true", help="이미 기록한 점도 다시 묻는다")

    f = sub.add_parser("fit", help="변환을 맞추고 점검, --save 로 workcell.yaml 에 저장")
    f.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "touch.json")
    f.add_argument("--drop", nargs="*", default=[], help="맞춤에서 뺄 점 이름 (예: P4)")
    f.add_argument("--holdout", nargs="*", default=[], help="맞춤에 쓰지 않고 검증 오차만 볼 점")
    f.add_argument("--save", type=Path, help="T_base_camera 를 쓸 workcell.yaml")
    f.add_argument("--force", action="store_true", help="경고가 있어도 저장")

    args = ap.parse_args(argv)
    return {"grab": cmd_grab, "mark": cmd_mark, "robot": cmd_robot, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
