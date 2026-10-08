"""D435 깊이 재보정 — On-Chip 보정(평탄도) + Tare(절대 거리). 역할 A.

#29 원인: 10/7 D435i 는 실제 0.74 m 를 0.41 m 로 읽었다 (aple/hal/depth_correction.py). 카메라 보정이 틀어진 것이므로
RealSense 자체 보정으로 바로잡는다. Piper Studio(rsd)가 카메라를 점유하므로 rsd 를 멈추고 쓴다 (rs_check.py 와 같다).

    systemctl --user stop piper-rsd
    python tools/rs_calibrate.py check                    # 지금 상태: 회색 카드 거리 오차, 평면 평탄도
    python tools/rs_calibrate.py occ                      # On-Chip 보정 → 임시 적용 → 다시 check (저장 안 함)
    python tools/rs_calibrate.py occ --write              # 좋아졌으면 카메라에 저장
    python tools/rs_calibrate.py tare                     # 회색 카드로 실제 거리 계산 → Tare → 임시 적용 → check
    python tools/rs_calibrate.py tare --height-mm 780     # 줄자로 잰 렌즈–바닥 수직 높이 + IMU 기울기 (카메라가 기울어도 된다)
    python tools/rs_calibrate.py tare --gt-mm 650 --write # 광축 방향 거리를 직접 줄 때 (카메라가 바닥에 수직일 때만 = 높이)
    python tools/rs_calibrate.py restore logs/rs_calibrate/backup_....json --write   # 되돌리기
    python tools/rs_calibrate.py restore --factory --write
    systemctl --user start piper-rsd

준비: 카메라를 고정하고, 질감이 있는 평평한 면(책상·바닥)이 화면 가운데를 넉넉히 채우게 0.5~0.7 m 위에서 내려다본다.
회색 카드(live_camera.depth_check.card_mm)를 그 면에 눕혀 화면 안에 둔다. 손·팔·사과는 화면 가운데에서 치운다.
순서는 occ → tare (평탄도를 먼저 맞추고 거리를 맞춘다). 바꾸기 전 보정표는 항상 logs/rs_calibrate/ 에 백업한다.

카메라가 기울면 Tare 기준 거리는 수직 높이가 아니라 '광축(화면 가운데)이 바닥에 닿는 거리'다. D435i IMU(가속도)로
중력 방향을 읽어 바닥 법선 n 을 얻고, 수직 높이 h 에서 픽셀 시선 r=(x, y, 1) 의 깊이 z = h / (n·r) 를 계산한다.
가운데(r = (0, 0, 1))가 Tare 기준이고, 카드 위치의 예상 깊이로 카드 측정과 서로 확인한다.
IMU 권한이 없으면 (Permission denied) librealsense udev 규칙의 IIO 줄이 필요하다 (docs/field_tuning.md 2-1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from aple.config import load_config  # noqa: E402
from aple.hal.depth_correction import card_distance  # noqa: E402
from aple.perception.geometry import depth_points, fit_plane_ransac  # noqa: E402
from aple.schema import CaptureMode, SceneSnapshot, Source  # noqa: E402

BACKUP_DIR = ROOT / "logs/rs_calibrate"
# librealsense 권장 설정 (On-Chip: 256×144 @ 90 fps 깊이, speed 3 = medium)
OCC_JSON = {"calib type": 0, "speed": 3, "scan parameter": 0, "data sampling": 0}
TARE_JSON = {"average step count": 20, "step count": 20, "accuracy": 2, "scan parameter": 0, "data sampling": 0}

rs = None


def open_device():
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        raise SystemExit("RealSense 가 없다 — USB 연결 확인, Piper Studio 를 멈췄는지 (systemctl --user stop piper-rsd)")
    dev = devs[0]
    return dev, dev.get_info(rs.camera_info.serial_number)


def backup(acd, serial: str, tag: str) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"backup_{serial}_{time.strftime('%m%d_%H%M%S')}_{tag}.json"
    path.write_text(json.dumps({"serial": serial, "table": list(acd.get_calibration_table())}), encoding="utf-8")
    return path


# ================================================================ 측정

def capture(frames: int = 15, width: int = 848, height: int = 480, table: list[int] | None = None, acd=None,
            write: bool = False):
    """컬러 + 컬러에 정렬한 깊이 (frames 장 중앙값) → SceneSnapshot.

    table 을 주면 스트림을 켠 뒤 그 보정표를 적용하고 찍는다 — 보정표는 스트리밍 중에 넣어야 반영되고,
    스트림을 새로 켜면 카메라에 저장된 값으로 돌아간다 (10/8 실측). write 면 같은 스트림 안에서 카메라에 저장한다.
    """
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)
        for _ in range(10):
            pipe.wait_for_frames()
        if table is not None:
            acd.set_calibration_table(table)
            if write:
                acd.write_calibration()
        for _ in range(30):                                  # 자동 노출 수렴 · 새 보정표 반영
            pipe.wait_for_frames()
        depths, color = [], None
        for _ in range(frames):
            fs = align.process(pipe.wait_for_frames())
            depths.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * scale)
            color = np.asanyarray(fs.get_color_frame().get_data()).copy()
    finally:
        pipe.stop()
    stack = np.stack(depths)
    valid = (stack > 0).sum(axis=0) >= frames // 2 + 1
    with np.errstate(invalid="ignore"):
        depth = np.where(valid, np.nanmedian(np.where(stack > 0, stack, np.nan), axis=0), 0).astype(np.float32)
    return SceneSnapshot(0, Source.TOP, CaptureMode.FULL, color, depth,
                         (intr.fx, intr.fy, intr.ppx, intr.ppy), time.time())


def flatness(snap: SceneSnapshot, frac: float = 0.5) -> dict:
    """화면 가운데 frac 영역 평면: 기울기·잔차 RMS, 가운데 대비 좌우 끝 띠의 휨 (mm)."""
    H, W = snap.depth_m.shape
    mh, mw = int(H * (1 - frac) / 2), int(W * (1 - frac) / 2)
    roi = np.zeros((H, W), bool)
    roi[mh:H - mh, mw:W - mw] = True
    valid = roi & (snap.depth_m > 0)
    pts = depth_points(snap.depth_m, snap.intrinsics)
    n, d = fit_plane_ransac(pts[valid][::7], iters=300, tol_m=0.006)
    h = pts @ n + d
    inl = valid & (np.abs(h) < 0.06)
    cols = np.indices((H, W))[1]
    band = (W - 2 * mw) // 5
    def band_median(sel):
        v = h[inl & sel]
        return float(np.median(v)) if len(v) > 50 else float("nan")
    left = band_median(cols < mw + band)
    mid = band_median(np.abs(cols - W / 2) < band / 2)
    right = band_median(cols >= W - mw - band)
    return {"z_center_m": float(np.median(snap.depth_m[valid])), "tilt_deg": float(np.degrees(np.arccos(abs(n[2])))),
            "rms_mm": float(np.std(h[inl]) * 1000), "bow_mm": float(((left + right) / 2 - mid) * 1000),
            "valid": float(valid.sum() / roi.sum())}


def report(snap: SceneSnapshot, card_mm, title: str, down: np.ndarray | None = None,
           height_m: float | None = None) -> dict:
    out = {"flat": flatness(snap)}
    f = out["flat"]
    print(f"[{title}] 가운데 깊이 {f['z_center_m']:.3f} m, 유효 {f['valid'] * 100:.0f}%, 평면 잔차 {f['rms_mm']:.1f} mm, "
          f"가장자리 휨 {f['bow_mm']:+.1f} mm, 깊이로 본 기울기 {f['tilt_deg']:.1f}°")
    H, W = snap.depth_m.shape
    if down is not None and height_m is not None:
        win = snap.depth_m[H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10]
        z_meas = float(np.median(win[win > 0])) if (win > 0).sum() > 20 else float("nan")
        z_imu = floor_depth(down, height_m, snap.intrinsics[2], snap.intrinsics[3], snap.intrinsics)
        out["imu"] = {"z_center_true_m": z_imu, "z_center_depth_m": z_meas, "error": z_meas / z_imu - 1}
        print(f"[{title}] IMU+높이: 화면 가운데 바닥까지 {z_imu:.3f} m / 깊이 {z_meas:.3f} m → 오차 {(z_meas / z_imu - 1) * 100:+.1f}%")
    cd = card_distance(snap, card_mm)
    if cd is None:
        print(f"[{title}] 회색 카드({card_mm[0]:g}×{card_mm[1]:g} mm)를 못 찾았다 — 눕혀서 화면 안에, 다 보이게")
    else:
        line = f"[{title}] 카드: 깊이 {cd.z_depth_m:.3f} m / 카드 크기로 잰 거리 {cd.z_true_m:.3f} m → 오차 {cd.error * 100:+.1f}%"
        if down is not None and height_m is not None:
            line += f"  (IMU+높이로 본 카드 위치 깊이 {floor_depth(down, height_m, *cd.center_px, snap.intrinsics):.3f} m)"
        print(line)
        out["card"] = {"z_depth_m": cd.z_depth_m, "z_true_m": cd.z_true_m, "error": cd.error}
    return out


def read_gravity(samples: int = 200) -> np.ndarray:
    """IMU 가속도 평균 → 카메라 좌표(x 오른쪽, y 아래, z 앞)의 '아래' 방향 단위 벡터. 카메라를 움직이지 않는다."""
    cfg = rs.config()
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
    pipe = rs.pipeline()
    try:
        pipe.start(cfg)
    except RuntimeError as e:
        raise SystemExit(f"IMU 를 열 수 없다: {e}\n  → udev 규칙에 IIO 줄 추가 (docs/field_tuning.md 2-1)") from e
    acc = []
    try:
        while len(acc) < samples + 50:
            m = pipe.wait_for_frames().first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
            acc.append((m.x, m.y, m.z))
    finally:
        pipe.stop()
    a = np.array(acc[50:])
    if np.linalg.norm(a.std(axis=0)) > 0.3:
        raise SystemExit("IMU 값이 흔들린다 — 카메라를 고정하고 다시")
    g = a.mean(axis=0)
    # 가속도계는 중력의 반작용(위쪽)을 잰다. 카메라가 바닥을 보면 '아래'는 +z 쪽이므로 그쪽을 향하게 맞춘다
    down = g / np.linalg.norm(g)
    return down if down[2] > 0 else -down


def floor_depth(down: np.ndarray, height_m: float, u: float, v: float, intrinsics) -> float:
    """수직 높이 height_m 인 바닥에서 픽셀 (u, v) 시선이 닿는 점의 깊이(z)."""
    fx, fy, cx, cy = intrinsics
    r = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return float(height_m / (down @ r))


# ================================================================ 보정

def run_occ(acd) -> tuple[list[int], float]:
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
    pipe = rs.pipeline()
    pipe.start(cfg)
    try:
        for _ in range(30):
            pipe.wait_for_frames()
        print("On-Chip 보정 중 (최대 30 s, 카메라·장면을 움직이지 않는다)...")
        table, health = acd.run_on_chip_calibration(json.dumps(OCC_JSON), lambda p: None, 30000)
    finally:
        pipe.stop()
    return list(table), float(health[0])


def run_tare(acd, gt_mm: float) -> tuple[list[int], float]:
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
    pipe = rs.pipeline()
    pipe.start(cfg)
    try:
        for _ in range(30):
            pipe.wait_for_frames()
        print(f"Tare 보정 중 (기준 거리 {gt_mm:.0f} mm, 최대 30 s)...")
        table, health = acd.run_tare_calibration(float(gt_mm), json.dumps(TARE_JSON), lambda p: None, 30000)
    finally:
        pipe.stop()
    return list(table), float(health[0])


def apply_and_check(acd, serial: str, table: list[int], card_mm, write: bool, before: dict,
                    down=None, height_m=None) -> int:
    after = report(capture(table=table, acd=acd), card_mm, "적용 후", down, height_m)   # 이 스트림 동안만 적용
    better = True
    for key in ("card", "imu"):
        if key in before and key in after:
            better &= abs(after[key]["error"]) < abs(before[key]["error"]) + 0.005
    if not write:
        print("저장하지 않았다 (위 '적용 후' 측정 동안만 적용). 좋아졌으면 같은 명령에 --write")
        return 0
    if not better:
        print("카드 오차가 줄지 않아 저장하지 않는다 — 장면(평평한 면·카드)을 확인하고 다시")
        return 2
    report(capture(table=table, acd=acd, write=True), card_mm, "저장", down, height_m)
    report(capture(), card_mm, "저장 후 새 스트림", down, height_m)
    print(f"카메라에 저장했다 ({serial}). 되돌리기: python tools/rs_calibrate.py restore <백업 파일> --write")
    return 0


def main(argv: list[str] | None = None) -> int:
    global rs
    import pyrealsense2
    rs = pyrealsense2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "occ", "tare", "restore"])
    ap.add_argument("backup", nargs="?", type=Path, help="restore: 백업 파일")
    ap.add_argument("--write", action="store_true", help="카메라에 저장 (없으면 임시 적용만)")
    ap.add_argument("--gt-mm", type=float, help="tare: 광축 방향 카메라–평면 거리 (카메라가 바닥에 수직이면 = 높이)")
    ap.add_argument("--height-mm", type=float, help="줄자로 잰 렌즈–바닥 수직 높이 — IMU 기울기로 광축 거리를 계산")
    ap.add_argument("--factory", action="store_true", help="restore: 공장 보정값으로")
    ap.add_argument("--card-mm", help="카드 긴 변,짧은 변 (기본 live_camera.depth_check.card_mm)")
    args = ap.parse_args(argv)

    cfg = load_config()
    dc = cfg["live_camera"].get("depth_check", {})
    card_mm = tuple(float(v) for v in (args.card_mm.split(",") if args.card_mm else dc.get("card_mm", (130, 100))))

    dev, serial = open_device()
    acd = rs.auto_calibrated_device(dev)
    print(f"D435 {serial}, 펌웨어 {dev.get_info(rs.camera_info.firmware_version)}")

    if args.cmd == "restore":
        print(f"백업: {backup(acd, serial, 'before_restore')}")
        if args.factory:
            if not args.write:
                print("공장값 복원은 바로 저장된다 — --write 를 함께 준다")
                return 2
            acd.reset_to_factory_calibration()
            print("공장 보정값으로 되돌렸다")
            return 0
        if args.backup is None:
            ap.error("restore 는 백업 파일 또는 --factory")
        data = json.loads(args.backup.read_text(encoding="utf-8"))
        if data["serial"] != serial:
            raise SystemExit(f"백업 카메라({data['serial']})와 연결된 카메라({serial})가 다르다")
        if not args.write:
            print("저장하지 않으면 되돌릴 수 없다 (보정표는 스트림 동안만 적용된다) — --write 를 함께 준다")
            return 2
        report(capture(table=data["table"], acd=acd, write=True), card_mm, "복원")
        report(capture(), card_mm, "복원 후 새 스트림")
        print("백업 보정표를 카메라에 저장했다")
        return 0

    down = height_m = None
    if args.height_mm:
        down, height_m = read_gravity(), args.height_mm / 1000
        print(f"IMU: 광축이 수직에서 {np.degrees(np.arccos(down[2])):.1f}° 기울어 있다 (수직 높이 {args.height_mm:.0f} mm)")
    before = report(capture(), card_mm, "지금", down, height_m)
    if args.cmd == "check":
        return 0

    print(f"백업: {backup(acd, serial, 'before_' + args.cmd)}")
    if args.cmd == "occ":
        table, health = run_occ(acd)
        verdict = "좋음" if abs(health) < 0.25 else "보통" if abs(health) < 0.75 else "나쁨 — 보정이 필요했다"
        print(f"On-Chip 건강도 {health:+.3f} (보정 전 상태: {verdict})")
    else:
        gt_mm = args.gt_mm
        if gt_mm is None and "imu" in before:
            gt_mm = before["imu"]["z_center_true_m"] * 1000
            print(f"기준 거리: IMU 기울기 + 수직 높이로 계산한 화면 가운데 바닥 거리 {gt_mm:.0f} mm")
        if gt_mm is None:
            if "card" not in before:
                raise SystemExit("기준 거리가 없다 — 회색 카드를 화면에 두거나 --gt-mm 으로 줄자 거리를 준다")
            gt_mm = before["card"]["z_true_m"] * 1000
            print(f"기준 거리: 회색 카드 크기로 잰 {gt_mm:.0f} mm (카드가 놓인 평면까지)")
        table, health = run_tare(acd, gt_mm)
        print(f"Tare 결과 {health:+.3f}")
    return apply_and_check(acd, serial, table, card_mm, args.write, before, down, height_m)


if __name__ == "__main__":
    raise SystemExit(main())
