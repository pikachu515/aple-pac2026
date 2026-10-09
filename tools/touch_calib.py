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
  2-1. (자로 재기 어려우면) 피벗 보정으로 플랜지 → 핑거 끝 오프셋을 구한다 — 그리퍼를 닫고 핑거 끝을
       표식 하나에 댄 채, 끝은 그 자리에 두고 손목 각도만 바꿔 4~6 자세를 기록한다 (각도 차 20° 이상)
       python tools/touch_calib.py pivot --session data/calib/touch.json --iface can0 [--save configs/workcell.yaml]
       결과는 세션에 저장돼 다음 robot 단계가 쓴다. --save 는 workcell.tcp_offset_m 에도 쓴다
  3. 로봇 끝점을 표식 중심에 차례로 대고 좌표를 입력한다 (클릭한 순서와 같게)
       python tools/touch_calib.py robot --session data/calib/touch.json
       입력: "x y z"            팁 끝 위치 (mm, base_link)
             "x y z rx ry rz"   플랜지 위치 (mm) + 자세 (deg) → workcell.tcp_offset_m 으로 팁 끝 계산
             빈 줄 = 건너뛰기, q = 중단
       --auto: Piper Web 관절각 + Piper Studio 순기구학(piper_robot.kinematics)으로 자동 기록
               --iface can0 으로 팔로워를 주거나, piper_web.enabled 와 iface 를 설정해 둔다.
               팔로워는 리더암 텔레오퍼레이션(Studio 릴레이)으로 표식에 댄다.
               piper_robot 은 PYTHONPATH 로 (docs/simulation.md)
  4. 맞추고 점검한 뒤 저장한다 (--auto 로 찍었으면 플랜지 자세가 남아 있어 끝점 오프셋을 함께 풀 수 있다:
       fit --solve-tcp --tcp-guess 0,0,143 — 점마다 손목 방향이 조금씩 달라야 오프셋이 정해진다)
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


def fit_with_tcp(points_cam, flanges, t_prior, prior_weight: float = 0.001, iters: int = 200):
    """T_base_camera 와 끝점 오프셋 t (플랜지 좌표계) 를 함께 맞춘다. → (T, t).

    끝점 base = R_i · t + p_i (R_i, p_i: 점 i 를 찍을 때의 플랜지 자세) = R_c · c_i + p_c.
    R_c 를 고정하면 (p_c, t) 는 선형이므로 Kabsch ↔ 선형 최소제곱을 번갈아 푼다.
    점마다 손목 방향이 같으면 t 는 p_c 와 구분되지 않는다 — prior_weight 로 t_prior 쪽에 묶는다.
    """
    from aple.hal.calibration import fit_rigid_transform
    C = np.asarray(points_cam, dtype=np.float64)
    F = [np.asarray(T, dtype=np.float64) for T in flanges]
    t = np.asarray(t_prior, dtype=np.float64).copy()
    w = np.sqrt(prior_weight * len(C))
    for _ in range(iters):
        tips = np.array([T[:3, :3] @ t + T[:3, 3] for T in F])
        T_bc, _ = fit_rigid_transform(C, tips)
        Rc = T_bc[:3, :3]
        A = np.zeros((3 * len(C) + 3, 6))
        b = np.zeros(3 * len(C) + 3)
        for i, (c, T) in enumerate(zip(C, F)):
            A[3 * i:3 * i + 3, :3] = np.eye(3)               # p_c
            A[3 * i:3 * i + 3, 3:] = -T[:3, :3]              # −R_i t
            b[3 * i:3 * i + 3] = T[:3, 3] - Rc @ c
        A[-3:, 3:] = w * np.eye(3)
        b[-3:] = w * np.asarray(t_prior, dtype=np.float64)
        t_new = np.linalg.lstsq(A, b, rcond=None)[0][3:]
        if np.linalg.norm(t_new - t) < 1e-7:
            t = t_new
            break
        t = t_new
    tips = np.array([T[:3, :3] @ t + T[:3, 3] for T in F])
    T_bc, _ = fit_rigid_transform(C, tips)
    return T_bc, t


def tcp_for_fit(session: dict, cfg_offset, guess=None) -> tuple[np.ndarray | None, str]:
    """맞춤에 쓸 끝점 오프셋과 출처. 세션 피벗 결과 > workcell.yaml > --tcp-guess(함께 풀 때 시작값)."""
    if session.get("tcp_offset_m") is not None:
        return np.asarray(session["tcp_offset_m"], dtype=float), "피벗 보정"
    if cfg_offset is not None and np.any(np.asarray(cfg_offset) != 0):
        return np.asarray(cfg_offset, dtype=float), "workcell.yaml"
    if guess is not None:
        return np.asarray(guess, dtype=float), "--tcp-guess"
    return None, ""


def fit_session(session: dict, drop=(), holdout=(), *, tcp=None, solve_tcp: bool = False) -> tuple:
    """→ (TouchFit, 맞춤에 쓴 점 목록, [(검증 점, 오차 m)]).

    모든 점에 플랜지 자세(flange_T, robot --auto 가 기록)가 있으면 끝점 = R·tcp + p 로 다시 계산한다.
    solve_tcp 면 tcp 를 시작값으로 T_base_camera 와 함께 푼다 (session["fit"]["tcp_offset_m"]).
    """
    unknown = (set(drop) | set(holdout)) - {p["name"] for p in session["points"]}
    if unknown:
        raise SystemExit(f"세션에 없는 점: {sorted(unknown)}")
    done = [p for p in session["points"] if p["cam_m"] is not None and p["base_m"] is not None]
    used = [p for p in done if p["name"] not in drop and p["name"] not in holdout]
    if len(used) < 3:
        raise SystemExit(f"로봇 좌표까지 있는 점이 {len(used)}개다 — 3개 이상 (권장 6~10개) 필요")
    if all(p.get("flange_T") for p in done) and (tcp is not None or solve_tcp):
        t = np.zeros(3) if tcp is None else np.asarray(tcp, dtype=float)
        if solve_tcp:
            _, t = fit_with_tcp([p["cam_m"] for p in used], [p["flange_T"] for p in used], t)
        for p in done:
            T = np.asarray(p["flange_T"], dtype=float)
            p["base_m"] = [round(float(v), 5) for v in T[:3, :3] @ t + T[:3, 3]]
        session["fit_tcp_offset_m"] = [round(float(v), 5) for v in t]
    elif solve_tcp:
        raise SystemExit("--solve-tcp 는 robot --auto 로 플랜지 자세(flange_T)를 기록한 점에서만 된다")
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


def _piper_fk(kin=None):
    """Piper Studio 순기구학 함수 (관절각 rad 6개 → 4×4 플랜지 자세). 함수 이름은 설치본에서 찾는다.

    v0.5.12 설치본에는 fk 함수가 없고 link_transforms(q (N,6), geometry) → (N, 링크, 4, 4) 만 있다.
    그때는 실행 코드(aple/execution/trajectory.build_piper_kinematics)와 같게 link6 플랜지를 꺼낸다.
    """
    from importlib import import_module
    if kin is None:
        try:
            kin = import_module("piper_robot.kinematics")
        except ImportError as e:
            raise SystemExit("piper_robot.kinematics 를 불러올 수 없다 — PYTHONPATH 에 Piper Studio 의 piper_robot 을 "
                             "넣거나(docs/simulation.md) 수동 입력을 쓴다") from e
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
    if callable(getattr(kin, "link_transforms", None)) and callable(getattr(kin, "geometry", None)):
        geom = kin.geometry()
        if geom.tip != "link6":
            raise SystemExit(f"piper_robot 기구학의 끝 링크가 link6 이 아니다 ({geom.tip}) — 플랜지 기준이 다르다")

        def run_links(q):
            links = np.asarray(kin.link_transforms(np.asarray(q, dtype=float)[None, :], geom), dtype=float)
            return links[0, geom.tip_index]
        return run_links
    public = [n for n in dir(kin) if not n.startswith("_")]
    raise SystemExit(f"piper_robot.kinematics 에 순기구학 함수가 없다 (있는 것: {public}) — 수동 입력을 쓴다")


def pivot_fit(flanges: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """피벗 보정 — 끝점을 한 점에 고정하고 자세만 바꾼 플랜지 자세들 → (플랜지 좌표계 끝점 오프셋,
    base 좌표계 고정점, 자세별 잔차 m, 자세 간 최대 회전각 deg).

    R_i · t + p_i = x  (모든 i)  →  [R_i  −I] [t; x] = −p_i 를 최소제곱으로 푼다.
    회전이 다양하지 않으면 t 가 정해지지 않는다 (최대 회전각 20° 이상 권장).
    """
    if len(flanges) < 4:
        raise ValueError("자세가 4개 이상 필요하다")
    A = np.vstack([np.hstack([T[:3, :3], -np.eye(3)]) for T in flanges])
    b = np.concatenate([-T[:3, 3] for T in flanges])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    t, x = sol[:3], sol[3:]
    res = np.array([np.linalg.norm(T[:3, :3] @ t + T[:3, 3] - x) for T in flanges])
    spread = max(np.degrees(np.arccos(np.clip((np.trace(Ti[:3, :3].T @ Tj[:3, :3]) - 1) / 2, -1, 1)))
                 for i, Ti in enumerate(flanges) for Tj in flanges[i + 1:])
    return t, x, res, float(spread)


def _auto_reader(cfg: dict, iface: str | None):
    """→ (fk, studio). 관절각 읽기만 한다 — iface 를 주면 이 실행에서만 켜고 설정 파일은 그대로."""
    from aple.hal.piper_studio import PiperStudio
    if iface:
        cfg["piper_web"] = {**cfg["piper_web"], "enabled": True, "iface": iface}
    return _piper_fk(), PiperStudio(cfg)


def cmd_pivot(args) -> None:
    session = load_session(args.session)
    fk, studio = _auto_reader(load_config(), args.iface)
    flanges: list[np.ndarray] = []
    print("그리퍼를 닫고 핑거 끝을 표식 하나에 댄다. 끝은 그 자리에 두고 손목 각도만 바꿔 가며 Enter (4~6번, q = 끝)")
    while True:
        text = input(f"자세 {len(flanges) + 1} > ").strip().lower()
        if text == "q":
            break
        if text == "u" and flanges:
            flanges.pop()
            print("  마지막 자세 취소")
            continue
        flanges.append(np.asarray(fk(studio.joint_state()), dtype=float))
        print(f"  기록 — 플랜지 {_mm(flanges[-1][:3, 3])}")
    try:
        t, x, res, spread = pivot_fit(flanges)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"끝점 오프셋 (플랜지 좌표계) {_mm(t)}  — 길이 {np.linalg.norm(t) * 1000:.1f} mm")
    print(f"고정점 (base_link) {_mm(x)}")
    print(f"자세별 잔차 {', '.join(f'{r * 1000:.1f}' for r in res)} mm (중앙값 {np.median(res) * 1000:.1f}),"
          f" 자세 간 최대 회전 {spread:.0f}°")
    warn = []
    if spread < 20:
        warn.append("자세 간 회전이 20° 미만 — 손목을 더 크게 돌려 다시 한다")
    if np.median(res) > 0.003:
        warn.append("잔차 중앙값이 3 mm 를 넘는다 — 기록 중에 끝이 미끄러졌을 수 있다")
    for w in warn:
        print(f"[경고] {w}")
    session["pivot"] = {"flanges": [T.round(6).tolist() for T in flanges], "residuals_m": res.round(5).tolist(),
                        "offset_m": [round(float(v), 5) for v in t], "warnings": warn}
    if warn and not args.force:
        save_session(args.session, session)          # 기록은 남긴다 — 오프셋은 쓰지 않는다
        print("오프셋은 저장하지 않았다 (--force 로 저장). 자세 기록은 세션 pivot 에 남겼다")
        return
    session["tcp_offset_m"] = [round(float(v), 5) for v in t]
    save_session(args.session, session)
    print(f"세션에 저장: {args.session} (robot 단계가 이 오프셋을 쓴다)")
    if args.save:
        from aple.config import set_config_value
        line = set_config_value(str(args.save), "workcell.tcp_offset_m", session["tcp_offset_m"],
                                note="touch_calib pivot")
        print(f"{args.save}: {line.strip()}")


def cmd_robot(args) -> None:
    session = load_session(args.session)
    if not session["points"]:
        raise SystemExit("세션에 점이 없다 — mark 를 먼저 한다")
    cfg = load_config()
    if session.get("tcp_offset_m") is not None:       # 피벗 보정 결과가 우선
        offset = np.asarray(session["tcp_offset_m"], dtype=float)
        print(f"tcp_offset_m = {offset.tolist()} (세션의 피벗 보정 결과)")
    else:
        offset = tcp_offset_from_cfg(cfg)
        print(f"tcp_offset_m = {offset.tolist()} (workcell.yaml)")
    fk = studio = None
    if args.auto:
        fk, studio = _auto_reader(cfg, args.iface)

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
                        p.pop("flange_T", None)
                    else:
                        T = np.asarray(fk(studio.joint_state()), dtype=float)
                        base = [float(v) for v in tip_from_flange(T, offset)]
                        # 끝점 오프셋은 나중에(fit --solve-tcp) 바뀔 수 있다 — 플랜지 자세 전체를 남긴다
                        p["flange_T"] = T.round(6).tolist()
                        tilt = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
                        print(f"  툴 기울기 {tilt:.0f}° (수직 0°)")
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
    guess = [float(v) / 1000 for v in args.tcp_guess.split(",")] if args.tcp_guess else None
    tcp, source = tcp_for_fit(session, load_config()["workcell"].get("tcp_offset_m"), guess)
    if args.solve_tcp and tcp is None:
        tcp, source = np.zeros(3), "0"
    fit, used, checks = fit_session(session, args.drop, args.holdout, tcp=tcp, solve_tcp=args.solve_tcp)
    if session.get("fit_tcp_offset_m") is not None:
        t = np.asarray(session["fit_tcp_offset_m"])
        how = f"함께 풂 (시작값 {source})" if args.solve_tcp else source
        print(f"끝점 오프셋 {_mm(t)} — 길이 {np.linalg.norm(t) * 1000:.1f} mm [{how}]")
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
        if args.save_tcp and session.get("fit_tcp_offset_m") is not None:
            from aple.config import set_config_value
            print("  " + set_config_value(str(args.save), "workcell.tcp_offset_m", session["fit_tcp_offset_m"],
                                          note="touch_calib fit").strip())
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
    r.add_argument("--iface", help="--auto 에서 관절각을 읽을 팔로워 (예: can0). 주면 piper_web.enabled 없이 읽기만 한다")

    pv = sub.add_parser("pivot", help="피벗 보정으로 플랜지 → 핑거 끝 오프셋 구하기 (자로 재지 않을 때)")
    pv.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "touch.json")
    pv.add_argument("--iface", help="관절각을 읽을 팔로워 (예: can0)")
    pv.add_argument("--save", type=Path, help="workcell.tcp_offset_m 을 쓸 workcell.yaml")
    pv.add_argument("--force", action="store_true", help="경고가 있어도 저장")

    f = sub.add_parser("fit", help="변환을 맞추고 점검, --save 로 workcell.yaml 에 저장")
    f.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "touch.json")
    f.add_argument("--drop", nargs="*", default=[], help="맞춤에서 뺄 점 이름 (예: P4)")
    f.add_argument("--holdout", nargs="*", default=[], help="맞춤에 쓰지 않고 검증 오차만 볼 점")
    f.add_argument("--save", type=Path, help="T_base_camera 를 쓸 workcell.yaml")
    f.add_argument("--force", action="store_true", help="경고가 있어도 저장")
    f.add_argument("--solve-tcp", action="store_true",
                   help="끝점 오프셋을 T_base_camera 와 함께 푼다 (robot --auto 로 플랜지 자세를 기록한 경우)")
    f.add_argument("--tcp-guess", help="끝점 오프셋 시작값/가정 'x,y,z' mm (예: 0,0,143)")
    f.add_argument("--save-tcp", action="store_true", help="--save 때 맞춘 끝점 오프셋도 workcell.tcp_offset_m 에 쓴다")

    args = ap.parse_args(argv)
    return {"grab": cmd_grab, "mark": cmd_mark, "robot": cmd_robot, "pivot": cmd_pivot, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
